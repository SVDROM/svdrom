"""Lazy FIF readers used by the Alljoined EEG notebook."""

from __future__ import annotations

from pathlib import Path

import dask.array as da
import numpy as np
from dask import delayed


def read_eeg_block(
    path: Path,
    start: int,
    stop: int,
    channels: tuple[str, ...],
) -> np.ndarray:
    """Read a FIF/EDF time block in microvolts without preloading a recording.

    EDF exports include equipment telemetry, which is excluded explicitly.
    Alljoined-1.6M sometimes spells the AFz electrode as Afz.
    """
    import mne

    if path.suffix.lower() == ".edf":
        include = list(channels) + (["Afz"] if "AFz" in channels else [])
        raw = mne.io.read_raw_edf(path, preload=False, include=include, verbose="ERROR")
    else:
        raw = mne.io.read_raw_fif(path, preload=False, verbose="ERROR")
    try:
        if path.suffix.lower() == ".edf":
            if "Afz" in raw.ch_names:
                raw.rename_channels({"Afz": "AFz"})
            raw.pick(list(channels))
        else:
            raw.pick("eeg")
        if tuple(raw.ch_names) != channels:
            msg = f"EEG channel order changed in {path.name}."
            raise ValueError(msg)
        data = (raw.get_data(start=start, stop=stop) * 1e6).astype(np.float32)
    finally:
        raw.close()

    if data.shape != (len(channels), stop - start) or not np.isfinite(data).all():
        msg = f"Invalid EEG block in {path.name}, samples {start}:{stop}."
        raise ValueError(msg)
    return data


def lazy_eeg_file(
    path: Path,
    n_samples: int,
    channels: tuple[str, ...],
    chunk_samples: int,
) -> da.Array:
    """Represent a whole FIF recording as time-chunked, lazy Dask blocks."""
    if n_samples <= 0 or chunk_samples <= 0 or not channels:
        msg = "A positive recording length, chunk size, and EEG channels are required."
        raise ValueError(msg)
    blocks = [
        da.from_delayed(
            delayed(read_eeg_block)(
                path, start, min(start + chunk_samples, n_samples), channels
            ),
            shape=(len(channels), min(chunk_samples, n_samples - start)),
            dtype=np.float32,
        )
        for start in range(0, n_samples, chunk_samples)
    ]
    return da.concatenate(blocks, axis=1)


def alljoined_paths(directory: Path) -> list[Path]:
    """Return the fixed 12-training/one-held-out Alljoined-1 file order."""
    sessions = {
        1: (1, 2),
        2: (1,),
        3: (1, 2),
        4: (1, 2),
        5: (1, 2),
        6: (1, 2),
        7: (1,),
        8: (1,),
    }
    return [
        directory / f"subj{subject:02d}_session{session}_eeg.fif"
        for subject, numbers in sessions.items()
        for session in numbers
    ]


def recording_manifest(paths: list[Path]) -> list[dict]:
    """Inspect FIF metadata, enforcing matching channels, rates and volt units.

    File sizes identify the local inputs, but are not content checksums.
    A reference flag does not identify the physical acquisition reference.
    """
    import mne
    from mne._fiff.constants import FIFF

    records = []
    for path in paths:
        raw = mne.io.read_raw_fif(path, preload=False, verbose="ERROR")
        try:
            raw.pick("eeg")
            channels = list(raw.ch_names)
            rate = float(raw.info["sfreq"])
            if not channels or any(
                ch["unit"] != FIFF.FIFF_UNIT_V for ch in raw.info["chs"]
            ):
                msg = f"Expected EEG channels stored in volts in {path.name}."
                raise ValueError(msg)
            if records and (
                channels != records[0]["channels"] or rate != records[0]["sfreq"]
            ):
                msg = f"Inconsistent EEG channels or sampling rate in {path.name}."
                raise ValueError(msg)
            records.append(
                {
                    "file": path.name,
                    "bytes": path.stat().st_size,
                    "n_samples": int(raw.n_times),
                    "sfreq": rate,
                    "channels": channels,
                    "unit": "V",
                    "unit_multipliers": [int(ch["unit_mul"]) for ch in raw.info["chs"]],
                    "calibration": [float(ch["cal"]) for ch in raw.info["chs"]],
                    "range": [float(ch["range"]) for ch in raw.info["chs"]],
                    "custom_ref_applied": int(raw.info["custom_ref_applied"]),
                    "bad_channels": list(raw.info["bads"]),
                    "projectors": [str(p["desc"]) for p in raw.info["projs"]],
                }
            )
        finally:
            raw.close()
    return records


def reconstruction_diagnostics(
    observed: da.Array, reconstructed: da.Array
) -> dict[str, da.Array]:
    """Separate channel mean error from fluctuation error, without changing a fit.

    Arrays have shape (channels, samples). Test means are descriptive evaluation
    statistics only. MSE = offset MSE + fluctuation MSE, up to roundoff.
    """
    if (
        observed.ndim != 2
        or observed.shape != reconstructed.shape
        or observed.size == 0
    ):
        msg = "Expected matching, nonempty channel-by-sample arrays."
        raise ValueError(msg)
    observed = observed.astype(np.float64)
    residual = observed - reconstructed.astype(np.float64)
    offset = residual.mean(axis=1)
    fluctuation = residual - offset[:, None]
    return {
        "channel_mean": observed.mean(axis=1),
        "channel_std": observed.std(axis=1),
        "offset_rmse": da.sqrt(da.mean(offset**2)),
        "fluctuation_rmse": da.sqrt(da.mean(fluctuation**2)),
        "total_rmse": da.sqrt(da.mean(residual**2)),
        "fluctuation_energy": 1
        - da.sum(fluctuation**2)
        / da.sum((observed - observed.mean(axis=1, keepdims=True)) ** 2),
    }
