"""Disk-backed preparation, encoding and verification of Alljoined-1.6M EEG."""

from __future__ import annotations

import json
from hashlib import sha256
from itertools import product
from pathlib import Path

import dask.array as da
import h5py
import numpy as np
import xarray as xr
from dask import delayed
from scipy.signal import welch

from demos.eeg_analysis import filtered_eeg_file
from svdrom import POD

CHANNELS = tuple(
    "Cz FCz AFz Fp1 F5 F1 CP5 CP3 CP1 P1 P3 P5 P7 PO7 PO3 O1 "
    "Pz POz Oz O2 PO4 PO8 P8 P6 P4 P2 CP2 CP4 CP6 F2 F6 Fp2".split()
)
PREPROCESSING = {
    "high_pass_hz": 0.5,
    "halo_seconds": 32,
    "filter_order": 4,
    "zero_phase": True,
    "per_recording_center": True,
    "dtype": "float32",
}


def edf_metadata(path: Path) -> dict:
    """Validate the EDF header and payload length without reading signal data.

    Only the 32 named electrodes count toward the EEG matrix size. Equipment
    telemetry and annotations are preserved in the source, outside this demo.
    Calibration endpoints are recorded; no physical reference is inferred.
    """
    with path.open("rb") as stream:
        header = stream.read(256)
        if len(header) != 256 or header[:8].strip() != b"0":
            msg = f"Invalid EDF header: {path}."
            raise ValueError(msg)
        header_bytes = int(header[184:192])
        records = int(header[236:244])
        duration = float(header[244:252])
        signals = int(header[252:256])
        fields = stream.read(header_bytes - 256)
    if header_bytes != 256 * (signals + 1) or records <= 0 or duration <= 0:
        msg = f"Invalid EDF dimensions: {path}."
        raise ValueError(msg)

    def strings(offset: int, width: int) -> list[str]:
        return [
            fields[offset * signals + width * i : offset * signals + width * (i + 1)]
            .decode("ascii")
            .strip()
            for i in range(signals)
        ]

    labels = [label.replace("Afz", "AFz") for label in strings(0, 16)]
    if any(labels.count(channel) != 1 for channel in CHANNELS):
        msg = f"Missing or duplicate EEG electrode in {path}."
        raise ValueError(msg)
    indexes = [labels.index(channel) for channel in CHANNELS]
    samples = [int(value) for value in strings(216, 8)]
    units = strings(96, 8)
    if any(samples[i] / duration != 256 or units[i] != "uV" for i in indexes):
        msg = f"Expected 256 Hz EEG with uV EDF calibration in {path}."
        raise ValueError(msg)
    size = path.stat().st_size
    if size != header_bytes + records * 2 * sum(samples):
        msg = f"EDF payload length disagrees with header: {path}."
        raise ValueError(msg)
    calibration = {
        name: [float(strings(offset, 8)[i]) for i in indexes]
        for name, offset in (
            ("physical_min", 104),
            ("physical_max", 112),
            ("digital_min", 120),
            ("digital_max", 128),
        )
    }
    if any(
        calibration["physical_max"][i] <= calibration["physical_min"][i]
        or calibration["digital_max"][i] <= calibration["digital_min"][i]
        for i in range(len(CHANNELS))
    ):
        msg = f"Invalid EEG calibration endpoints in {path}."
        raise ValueError(msg)
    return {
        "bytes": size,
        "mtime_ns": path.stat().st_mtime_ns,
        "n_samples": records * samples[indexes[0]],
        "sfreq": 256.0,
        "channels": list(CHANNELS),
        "edf_unit": "uV",
        "physical_reference": "not established by this audit",
        **calibration,
    }


def alljoined16_manifest(directory: Path) -> dict:
    """Audit all 20 x 4 x 19 scheduled blocks, handling known filename errors.

    Directory participant IDs take precedence over subject 8's mislabeled EDF
    filenames, as in the installed NeuralFetch study. Unexpected EDF files are
    listed separately so nested download copies cannot inflate the corpus size.
    Header/payload checks and size/mtime provenance are not content checksums.
    """
    records = []
    selected = set()
    profiles: list[dict] = []
    for subject, session, block in product(range(1, 21), range(1, 5), range(1, 20)):
        folder = directory / f"sub-{subject:02d}" / f"session_{session:02d}"
        folder /= f"block_{block:02d}"
        name_subject = subject
        if subject == 8 and ((session == 1 and block >= 2) or session in (3, 4)):
            name_subject = 19
        matches = list(
            folder.glob(
                f"Subject {name_subject}, Session {session}, Block {block}*.edf"
            )
        )
        if len(matches) != 1:
            msg = f"Expected one scheduled EDF in {folder}; found {len(matches)}."
            raise ValueError(msg)
        path = matches[0]
        item = edf_metadata(path)
        profile = {
            name: item.pop(name)
            for name in (
                "sfreq",
                "channels",
                "edf_unit",
                "physical_reference",
                "physical_min",
                "physical_max",
                "digital_min",
                "digital_max",
            )
        }
        if profile not in profiles:
            profiles.append(profile)
        item["metadata_profile"] = profiles.index(profile)
        item.update(
            file=str(path.relative_to(directory)),
            subject=subject,
            session=session,
            block=block,
            key=f"sub{subject:02d}_s{session:02d}_b{block:02d}",
        )
        records.append(item)
        selected.add(path)
    return {
        "dataset": "Alljoined-1.6M",
        "source_root": str(directory.resolve()),
        "records": records,
        "metadata_profiles": profiles,
        "noncanonical_edf_files": [
            str(path.relative_to(directory))
            for path in sorted(set(directory.rglob("*.edf")) - selected)
        ],
    }


def prepare_cache(directory: Path, cache: Path, chunk_samples: int = 16_384) -> dict:
    """Condition every scheduled block into a resumable, chunked HDF5 cache.

    Signal reads and filtering are bounded by chunk_samples plus filter halos.
    Record means are estimated before a separate bounded centering pass. No
    recordings, electrodes or time samples are rejected or downsampled.
    Only incomplete groups in this demo's matching cache may be replaced.
    """
    if chunk_samples <= 0:
        msg = "chunk_samples must be positive."
        raise ValueError(msg)
    manifest = alljoined16_manifest(directory)
    metadata = json.dumps(
        {
            "manifest": manifest,
            "preprocessing": PREPROCESSING,
            "chunk_samples": chunk_samples,
        },
        sort_keys=True,
    )
    cache.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(cache, "r+" if cache.exists() else "x") as store:
        if "metadata" in store.attrs and store.attrs["metadata"] != metadata:
            msg = "Cache provenance/settings changed; use a new cache path."
            raise ValueError(msg)
        if "metadata" not in store.attrs:
            if len(store):
                msg = "Refusing to overwrite an unrelated HDF5 file."
                raise ValueError(msg)
            store.attrs["metadata"] = metadata
        for index, item in enumerate(manifest["records"]):
            key, n_samples = item["key"], item["n_samples"]
            if key in store and store[key].attrs.get("complete", False):
                continue
            if key in store:
                del store[key]
            group = store.create_group(key)
            group.attrs["record"] = json.dumps(item)
            signal = group.create_dataset(
                "signal",
                shape=(len(CHANNELS), n_samples),
                dtype="f4",
                chunks=(len(CHANNELS), min(chunk_samples, n_samples)),
            )
            filtered = filtered_eeg_file(
                directory / item["file"], n_samples, CHANNELS, chunk_samples, 256
            )
            total = np.zeros(len(CHANNELS), dtype=np.float64)
            for start in range(0, n_samples, chunk_samples):
                stop = min(start + chunk_samples, n_samples)
                values = filtered[:, start:stop].compute(scheduler="single-threaded")
                signal[:, start:stop] = values
                total += values.sum(axis=1, dtype=np.float64)
            mean = total / n_samples
            group.create_dataset("mean", data=mean.astype(np.float32))
            covariance = np.zeros((len(CHANNELS), len(CHANNELS)), dtype=np.float64)
            for start in range(0, n_samples, chunk_samples):
                stop = min(start + chunk_samples, n_samples)
                values = (
                    signal[:, start:stop].astype(np.float64) - mean[:, None]
                ).astype(np.float32)
                signal[:, start:stop] = values
                precise = values.astype(np.float64)
                covariance += precise @ precise.T
            group.create_dataset("covariance", data=covariance / n_samples)
            group.attrs["complete"] = True
            store.flush()
            if (index + 1) % 76 == 0:
                print(f"Cached {index + 1}/1520 recordings", flush=True)
    return manifest


def cache_manifest(cache: Path) -> dict:
    """Read cache provenance and require every scheduled group to be complete."""
    with h5py.File(cache, "r") as store:
        metadata = json.loads(store.attrs["metadata"])
        if metadata["preprocessing"] != PREPROCESSING:
            msg = "Incompatible EEG conditioning settings."
            raise ValueError(msg)
        records = metadata["manifest"]["records"]
        for item in records:
            group = store[item["key"]]
            if (
                not group.attrs.get("complete", False)
                or group["signal"].shape != (len(CHANNELS), item["n_samples"])
                or group["signal"].dtype != np.dtype("float32")
            ):
                msg = f"Incomplete or malformed cached recording {item['key']}."
                raise ValueError(msg)
    return metadata["manifest"]


def read_h5_block(path: Path, key: str, start: int, stop: int) -> np.ndarray:
    """Read only the requested signal/coefficient block, closing the file."""
    with h5py.File(path, "r") as store:
        return store[key][:, start:stop]


def cached_recording(cache: Path, item: dict, chunk_samples: int) -> da.Array:
    """Build a lazy recording whose tasks open/close independent HDF5 slices."""
    if chunk_samples <= 0:
        msg = "chunk_samples must be positive."
        raise ValueError(msg)
    blocks = [
        da.from_delayed(
            delayed(read_h5_block)(cache, f"{item['key']}/signal", start, stop),
            shape=(len(CHANNELS), stop - start),
            dtype=np.float32,
        )
        for start in range(0, item["n_samples"], chunk_samples)
        for stop in [min(start + chunk_samples, item["n_samples"])]
    ]
    return da.concatenate(blocks, axis=1)


def fit_cached(cache: Path, count: int, chunk_samples: int, n_modes: int = 24) -> POD:
    """Fit SVD-ROM to participants 1..count, always excluding participant 20."""
    if not 1 <= count <= 19:
        msg = "Use 1-19 training participants; participant 20 is held out."
        raise ValueError(msg)
    manifest = cache_manifest(cache)
    train = da.concatenate(
        [
            cached_recording(cache, item, chunk_samples)
            for item in manifest["records"]
            if item["subject"] <= count
        ],
        axis=1,
    )
    return POD(
        n_modes=n_modes,
        svd_algorithm="covariance",
        time_dimension="sample",
        remove_mean=False,
        compute_time_coeffs=False,
        compute_energy_ratio=True,
    ).fit(xr.DataArray(train, dims=("channel", "sample")))


def encode_cached(
    cache: Path, model: POD, output: Path, rank: int, chunk_samples: int
) -> dict:
    """Stream float32 coefficients for every recording to a new HDF5 file.

    Coefficients absorb POD's training-length normalization: decoding is simply
    modes @ coefficients. Per-record filtered means are saved separately. The
    target is the conditioned signal, not raw EEG or stimulus information.
    """
    if not 0 < rank <= model.n_components or chunk_samples <= 0:
        msg = "Use a retained model rank and positive chunk size."
        raise ValueError(msg)
    manifest = cache_manifest(cache)
    training_samples = sum(
        item["n_samples"] for item in manifest["records"] if item["subject"] <= 19
    )
    if model.time_coeffs.sizes["sample"] != training_samples:
        msg = "Encoding requires the model fitted to participants 1-19."
        raise ValueError(msg)
    output.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(cache, "r") as source, h5py.File(output, "x") as target:
        target.attrs["preprocessing"] = json.dumps(PREPROCESSING)
        target.attrs["manifest"] = json.dumps(manifest)
        target.attrs["training_subjects"] = json.dumps(
            sorted(
                {
                    item["subject"]
                    for item in manifest["records"]
                    if item["subject"] <= 19
                }
            )
        )
        target.attrs["decode"] = "modes @ coefficients; add mean for filtered level"
        target.create_dataset("modes", data=model.modes.values[:, :rank].astype("f4"))
        scale = np.sqrt(training_samples)
        for index, item in enumerate(manifest["records"]):
            group = target.create_group(item["key"])
            group.attrs["record"] = json.dumps(item)
            group.create_dataset("mean", data=source[f"{item['key']}/mean"][:])
            coefficients = group.create_dataset(
                "coefficients",
                shape=(rank, item["n_samples"]),
                dtype="f4",
                chunks=(rank, min(chunk_samples, item["n_samples"])),
            )
            for start in range(0, item["n_samples"], chunk_samples):
                stop = min(start + chunk_samples, item["n_samples"])
                block = source[f"{item['key']}/signal"][:, start:stop]
                array = xr.DataArray(da.from_array(block), dims=("channel", "sample"))
                values = model.transform(array, compute=False).data.compute(
                    scheduler="single-threaded"
                )[:rank]
                coefficients[:, start:stop] = (values * scale).astype("f4")
            if (index + 1) % 380 == 0:
                print(f"Encoded rank {rank}: {index + 1}/1520 recordings", flush=True)
        target.attrs["complete"] = True
    samples = sum(item["n_samples"] for item in manifest["records"])
    payload = (
        len(CHANNELS) * rank + rank * samples + len(manifest["records"]) * len(CHANNELS)
    ) * 4
    return {
        "rank": rank,
        "file": str(output),
        "payload_bytes": payload,
        "file_bytes": output.stat().st_size,
        "matrix_bytes": len(CHANNELS) * samples * 4,
    }


def validate_encoding(cache: Path, encoded: Path, rank: int | None = None) -> dict:
    """Check factor provenance, rank, dtypes and all recording dimensions.

    The basis fingerprint identifies the shared decoder. It does not checksum
    every coefficient or establish source-file integrity; full decoding remains
    a separate check. Validation reads only metadata, means and the small basis.
    """
    manifest = cache_manifest(cache)
    subjects = sorted({r["subject"] for r in manifest["records"] if r["subject"] <= 19})
    with h5py.File(encoded, "r") as store:
        if (
            not store.attrs.get("complete", False)
            or json.loads(store.attrs.get("manifest", "null")) != manifest
            or json.loads(store.attrs.get("preprocessing", "null")) != PREPROCESSING
            or json.loads(store.attrs.get("training_subjects", "null")) != subjects
        ):
            msg = "Incomplete encoding or mismatched input provenance/settings."
            raise ValueError(msg)
        basis = store["modes"][:]
        if (
            basis.ndim != 2
            or basis.shape[0] != len(CHANNELS)
            or not 0 < basis.shape[1] <= len(CHANNELS)
            or (rank is not None and basis.shape[1] != rank)
            or basis.dtype != np.dtype("float32")
            or not np.isfinite(basis).all()
            or not np.allclose(basis.T @ basis, np.eye(basis.shape[1]), atol=1e-6)
        ):
            msg = "Malformed encoding basis or unexpected rank."
            raise ValueError(msg)
        for item in manifest["records"]:
            group = store[item["key"]]
            if (
                json.loads(group.attrs.get("record", "null")) != item
                or group["mean"].shape != (len(CHANNELS),)
                or group["mean"].dtype != np.dtype("float32")
                or not np.isfinite(group["mean"][:]).all()
                or group["coefficients"].shape != (basis.shape[1], item["n_samples"])
                or group["coefficients"].dtype != np.dtype("float32")
            ):
                msg = f"Malformed encoded recording {item['key']}."
                raise ValueError(msg)
    return {
        "rank": basis.shape[1],
        "basis_sha256": sha256(basis.tobytes()).hexdigest(),
        "file_bytes": encoded.stat().st_size,
    }


def verify_encoding(cache: Path, encoded: Path, chunk_samples: int) -> dict:
    """Decode saved factors in bounded blocks and score every participant/sample.

    This fresh read checks actual float32 disk factors, not an analytical or
    in-memory projection. Error is relative to the conditioned cache. Covariance
    statistics here use a zero baseline (per-record centering precedes encoding).
    """
    if chunk_samples <= 0:
        msg = "chunk_samples must be positive."
        raise ValueError(msg)
    validate_encoding(cache, encoded)
    manifest = cache_manifest(cache)
    scores = {}
    with h5py.File(cache, "r") as source, h5py.File(encoded, "r") as target:
        modes = target["modes"][:]
        for item in manifest["records"]:
            score = scores.setdefault(
                item["subject"],
                {
                    "sse": 0.0,
                    "baseline": 0.0,
                    "samples": 0,
                    "original_covariance": np.zeros((len(CHANNELS), len(CHANNELS))),
                    "decoded_covariance": np.zeros((len(CHANNELS), len(CHANNELS))),
                },
            )
            for start in range(0, item["n_samples"], chunk_samples):
                stop = min(start + chunk_samples, item["n_samples"])
                original = source[f"{item['key']}/signal"][:, start:stop].astype("f8")
                coefficients = target[f"{item['key']}/coefficients"][:, start:stop]
                decoded = (modes @ coefficients).astype("f8")
                if not np.isfinite(decoded).all():
                    msg = "Non-finite decoded signal."
                    raise ValueError(msg)
                score["sse"] += float(np.sum((original - decoded) ** 2))
                score["baseline"] += float(np.sum(original**2))
                score["samples"] += stop - start
                score["original_covariance"] += original @ original.T
                score["decoded_covariance"] += decoded @ decoded.T
    from demos.eeg_analysis import correlation_matrix

    upper = np.triu_indices(len(CHANNELS), k=1)
    for score in scores.values():
        original = score.pop("original_covariance")
        decoded = score.pop("decoded_covariance")
        difference = correlation_matrix(original) - correlation_matrix(decoded)
        valid = difference[upper][np.isfinite(difference[upper])]
        score["correlation_rmse"] = (
            float(np.sqrt(np.mean(valid**2))) if valid.size else float("nan")
        )
        if score["baseline"] <= 0:
            msg = "Cannot score a participant with zero conditioned energy."
            raise ValueError(msg)
        score["energy_retained"] = 1 - score["sse"] / score["baseline"]
        score["rmse_uv"] = np.sqrt(score["sse"] / (len(CHANNELS) * score["samples"]))
    return {
        "rank": int(modes.shape[1]),
        "participants": scores,
        "all_sample_energy_retained": 1
        - sum(s["sse"] for s in scores.values())
        / sum(s["baseline"] for s in scores.values()),
    }


def _encoded_spectral_block(
    cache: Path,
    encoded: tuple[Path, ...],
    key: str,
    start: int,
    stop: int,
    nperseg: int,
) -> tuple[int, np.ndarray]:
    """Read aligned Welch windows and decode the saved factors for those windows."""
    values = [read_h5_block(cache, f"{key}/signal", start, stop)]
    for path in encoded:
        with h5py.File(path, "r") as store:
            values.append(
                store["modes"][:] @ store[f"{key}/coefficients"][:, start:stop]
            )
    spectra = np.stack(
        [
            welch(value, fs=256, nperseg=nperseg, noverlap=nperseg // 2, axis=1)[1]
            for value in values
        ]
    )
    windows = 1 + (stop - start - nperseg) // (nperseg // 2)
    return windows, spectra * windows


def encoded_spectra(
    cache: Path,
    encoded: tuple[Path, ...],
    subject: int = 20,
    nperseg: int = 2048,
    windows_per_task: int = 16,
) -> dict:
    """Compute full-participant spectra from disk factors with a bounded reduction.

    Every complete Welch window is included, with 50% overlap and no windows
    spanning recording boundaries. The final incomplete window is omitted.
    Weighting is by window count. Spectra measure the conditioned signal.
    """
    if nperseg < 2 or nperseg % 2 or windows_per_task <= 0 or not encoded:
        msg = "Use an even window length, positive task size, and encoded files."
        raise ValueError(msg)
    from demos.eeg_analysis import _merge_spectra

    inputs = [validate_encoding(cache, path) for path in encoded]
    manifest = cache_manifest(cache)
    hop = nperseg // 2
    tasks = []
    for item in manifest["records"]:
        if item["subject"] != subject:
            continue
        count = max(0, 1 + (item["n_samples"] - nperseg) // hop)
        for first in range(0, count, windows_per_task):
            windows = min(windows_per_task, count - first)
            start = first * hop
            stop = start + (windows - 1) * hop + nperseg
            tasks.append(
                delayed(_encoded_spectral_block)(
                    cache, encoded, item["key"], start, stop, nperseg
                )
            )
    if not tasks:
        msg = "No complete spectral windows for this participant."
        raise ValueError(msg)
    while len(tasks) > 1:
        tasks = [
            delayed(_merge_spectra)(tasks[index], tasks[index + 1])
            if index + 1 < len(tasks)
            else tasks[index]
            for index in range(0, len(tasks), 2)
        ]
    count, sums = tasks[0].compute()
    return {
        "frequencies": np.fft.rfftfreq(nperseg, 1 / 256),
        "psd": sums / count,
        "n_windows": count,
        "provenance": {
            "manifest": manifest,
            "preprocessing": PREPROCESSING,
            "subject": subject,
            "nperseg": nperseg,
            "encodings": inputs,
        },
    }


def save_encoded_spectra(output: Path, spectra: dict) -> None:
    """Write a new spectral artifact together with its actual input provenance."""
    with h5py.File(output, "x") as store:
        store.create_dataset("frequencies", data=spectra["frequencies"])
        store.create_dataset("psd", data=spectra["psd"])
        store.attrs["n_windows"] = spectra["n_windows"]
        store.attrs["provenance"] = json.dumps(spectra["provenance"])


def load_encoded_spectra(
    cache: Path,
    encoded: tuple[Path, ...],
    path: Path,
    subject: int = 20,
    nperseg: int = 2048,
) -> dict:
    """Reject stale spectral artifacts and check curve ordering and window count.

    Source metadata, conditioning, participant, window length, basis fingerprints
    and encoded file sizes must match. These checks do not checksum the full
    coefficient arrays; numerical verification is a separate disk round trip.
    """
    if nperseg < 2 or nperseg % 2 or not encoded:
        msg = "Use an even window length and encoded files."
        raise ValueError(msg)
    inputs = [validate_encoding(cache, item) for item in encoded]
    manifest = cache_manifest(cache)
    expected = {
        "manifest": manifest,
        "preprocessing": PREPROCESSING,
        "subject": subject,
        "nperseg": nperseg,
        "encodings": inputs,
    }
    with h5py.File(path, "r") as store:
        if json.loads(store.attrs.get("provenance", "null")) != expected:
            msg = (
                "Spectral artifact provenance changed; recompute saved-factor spectra."
            )
            raise ValueError(msg)
        frequencies, psd = store["frequencies"][:], store["psd"][:]
        count = int(store.attrs["n_windows"])
    windows = sum(
        max(0, 1 + (item["n_samples"] - nperseg) // (nperseg // 2))
        for item in manifest["records"]
        if item["subject"] == subject
    )
    if (
        count != windows
        or count <= 0
        or not np.array_equal(frequencies, np.fft.rfftfreq(nperseg, 1 / 256))
        or psd.shape != (1 + len(encoded), len(CHANNELS), nperseg // 2 + 1)
        or not np.isfinite(psd).all()
        or np.any(psd < 0)
    ):
        msg = "Malformed spectral artifact or incorrect Welch window count."
        raise ValueError(msg)
    return {"frequencies": frequencies, "psd": psd, "n_windows": count}
