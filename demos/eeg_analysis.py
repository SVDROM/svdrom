"""Bounded EEG conditioning and quantitative reconstruction checks for the demo."""

from __future__ import annotations

from pathlib import Path

import dask
import dask.array as da
import numpy as np
from dask import delayed
from scipy.integrate import trapezoid
from scipy.signal import butter, sosfiltfilt, welch

from demos.eeg_utils import read_eeg_block


def _filtered_block(
    path: Path,
    start: int,
    stop: int,
    n_samples: int,
    channels: tuple[str, ...],
    sfreq: float,
    high_pass: float,
    halo_seconds: float,
) -> np.ndarray:
    """Filter with real neighboring samples, discarding the overlap afterward."""
    halo = round(halo_seconds * sfreq)
    left, right = max(0, start - halo), min(n_samples, stop + halo)
    values = read_eeg_block(path, left, right, channels).astype(np.float64)
    sos = butter(4, high_pass, btype="highpass", fs=sfreq, output="sos")
    filtered = sosfiltfilt(sos, values, axis=1)
    return filtered[:, start - left : stop - left].astype(np.float32)


def filtered_eeg_file(
    path: Path,
    n_samples: int,
    channels: tuple[str, ...],
    chunk_samples: int,
    sfreq: float,
    high_pass: float = 0.5,
    halo_seconds: float = 32,
) -> da.Array:
    """Read a continuous recording with a zero-phase, fourth-order high-pass.

    Time and channel counts are preserved. Each block includes real context on
    either side; only the first/last recording boundaries use filter padding.
    The overlap approximates whole-recording filtering and is checked in tests.
    This conditions drift, without rereferencing or rejecting artifacts.
    """
    if (
        n_samples < 16
        or chunk_samples <= 0
        or not channels
        or not 0 < high_pass < sfreq / 2
        or halo_seconds < 16 / high_pass
    ):
        msg = "Use a valid cutoff, positive chunks, >=16 samples, and sufficient halo."
        raise ValueError(msg)
    blocks = [
        da.from_delayed(
            delayed(_filtered_block)(
                path,
                start,
                min(start + chunk_samples, n_samples),
                n_samples,
                channels,
                sfreq,
                high_pass,
                halo_seconds,
            ),
            shape=(len(channels), min(chunk_samples, n_samples - start)),
            dtype=np.float32,
        )
        for start in range(0, n_samples, chunk_samples)
    ]
    return da.concatenate(blocks, axis=1)


def recording_statistics(values: da.Array) -> dict[str, np.ndarray | int]:
    """Compute a stable mean and centered covariance using two bounded passes.

    The mean is computed first so Dask need not retain all input blocks while
    waiting for the global mean. Covariance normalization uses N, not N-1.
    """
    if values.ndim != 2 or min(values.shape) <= 0:
        msg = "Expected a nonempty channel-by-sample matrix."
        raise ValueError(msg)
    mean = values.astype(np.float64).mean(axis=1).compute()
    centered = values.astype(np.float64) - mean[:, None]
    covariance = (centered @ centered.T / values.shape[1]).compute()
    return {"mean": mean, "covariance": covariance, "n_samples": values.shape[1]}


def covariance_modes(covariance: np.ndarray, n_modes: int) -> np.ndarray:
    """Return leading orthonormal modes of a small symmetric spatial covariance."""
    if (
        covariance.ndim != 2
        or covariance.shape[0] != covariance.shape[1]
        or not 0 < n_modes < covariance.shape[0]
        or not np.isfinite(covariance).all()
    ):
        msg = "Expected finite square covariance and a valid truncation rank."
        raise ValueError(msg)
    _, modes = np.linalg.eigh((covariance + covariance.T) / 2)
    return modes[:, ::-1][:, :n_modes]


def correlation_matrix(covariance: np.ndarray) -> np.ndarray:
    """Normalize covariance, marking correlations of constant channels undefined."""
    std = np.sqrt(np.maximum(np.diag(covariance), 0))
    denominator = std[:, None] * std[None, :]
    return np.divide(
        covariance,
        denominator,
        out=np.full_like(covariance, np.nan),
        where=denominator > 0,
    )


def covariance_scores(
    covariance: np.ndarray, modes: np.ndarray, ranks: tuple[int, ...]
) -> dict[int, dict[str, float | np.ndarray]]:
    """Score exact linear projections using full-recording sufficient statistics.

    MSE, energy and reconstructed channel covariance are determined by the
    spatial covariance. Correlation RMSE excludes diagonal/undefined entries.
    """
    if (
        covariance.ndim != 2
        or covariance.shape != (modes.shape[0], modes.shape[0])
        or not ranks
        or any(not 0 < rank <= modes.shape[1] for rank in ranks)
        or not np.isfinite(covariance).all()
        or not np.allclose(modes.T @ modes, np.eye(modes.shape[1]), atol=1e-8)
    ):
        msg = "Expected finite covariance, orthonormal modes and valid ranks."
        raise ValueError(msg)
    energy = float(np.trace(covariance))
    if energy <= 0:
        msg = "Fluctuation energy must be positive."
        raise ValueError(msg)
    reference_correlation = correlation_matrix(covariance)
    scores = {}
    for rank in ranks:
        basis = modes[:, :rank]
        projection = basis @ basis.T
        residual_operator = np.eye(len(covariance)) - projection
        error_covariance = residual_operator @ covariance @ residual_operator.T
        reconstructed_covariance = projection @ covariance @ projection.T
        predicted_correlation = correlation_matrix(reconstructed_covariance)
        valid = np.triu(np.ones_like(covariance, dtype=bool), k=1)
        valid &= np.isfinite(reference_correlation) & np.isfinite(predicted_correlation)
        correlation_error = predicted_correlation - reference_correlation
        error_energy = max(float(np.trace(error_covariance)), 0)
        scores[rank] = {
            "energy": 1 - error_energy / energy,
            "rmse": np.sqrt(error_energy / len(covariance)),
            "channel_rmse": np.sqrt(np.maximum(np.diag(error_covariance), 0)),
            "correlation_rmse": float(np.sqrt(np.mean(correlation_error[valid] ** 2)))
            if valid.any()
            else float("nan"),
            "reconstructed_covariance": reconstructed_covariance,
        }
    return scores


def participant_validation(
    statistics: list[dict], subjects: list[int], ranks: tuple[int, ...]
) -> list[dict]:
    """Exclude every session of each participant, using cached spatial covariances.

    Each covariance has already seen every sample of its recording. Summing
    N*C reproduces the covariance backend on the concatenated centered files,
    avoiding eight repeated disk scans and never projecting one session across
    a temporal boundary. NumPy solves only the small covariance eigenproblem.
    """
    if len(statistics) != len(subjects) or len(set(subjects)) < 2:
        msg = "Provide matching recording statistics and at least two participants."
        raise ValueError(msg)
    results = []
    for subject in sorted(set(subjects)):
        train = [
            s for s, label in zip(statistics, subjects, strict=True) if label != subject
        ]
        test = [
            s for s, label in zip(statistics, subjects, strict=True) if label == subject
        ]
        n_train = sum(s["n_samples"] for s in train)
        n_test = sum(s["n_samples"] for s in test)
        train_cov = sum(s["n_samples"] * s["covariance"] for s in train) / n_train
        test_cov = sum(s["n_samples"] * s["covariance"] for s in test) / n_test
        modes = covariance_modes(train_cov, max(ranks))
        train_scores = covariance_scores(train_cov, modes, ranks)
        test_scores = covariance_scores(test_cov, modes, ranks)
        optimal_modes = covariance_modes(test_cov, max(ranks))
        optimal_scores = covariance_scores(test_cov, optimal_modes, ranks)
        results.append(
            {
                "subject": subject,
                "training_samples": n_train,
                "test_samples": n_test,
                "ranks": {
                    rank: {
                        "training_energy": train_scores[rank]["energy"],
                        "test_energy": test_scores[rank]["energy"],
                        "test_rmse": test_scores[rank]["rmse"],
                        "correlation_rmse": test_scores[rank]["correlation_rmse"],
                        "test_optimal_energy": optimal_scores[rank]["energy"],
                    }
                    for rank in ranks
                },
            }
        )
    return results


def compression_ratio(n_channels: int, n_samples: int, rank: int) -> float:
    """Return nominal float32 ratio including modes, coefficients and channel means.

    Time coordinates, formats, preprocessing settings and singular values are
    excluded. Coefficients may absorb singular-value scaling when stored.
    """
    if min(n_channels, n_samples, rank) <= 0 or rank > n_channels:
        msg = "Use positive dimensions and rank no greater than the channel count."
        raise ValueError(msg)
    return n_channels * n_samples / (n_channels * rank + rank * n_samples + n_channels)


def _spectral_block(
    values: np.ndarray,
    modes: np.ndarray,
    ranks: tuple[int, ...],
    sfreq: float,
    nperseg: int,
) -> tuple[int, np.ndarray]:
    """Average identically aligned Welch windows for input and each projection."""
    count = 1 + (values.shape[1] - nperseg) // (nperseg // 2)
    _, original = welch(values, fs=sfreq, nperseg=nperseg, noverlap=nperseg // 2)
    coefficients = modes.T @ values
    spectra = [original]
    for rank in ranks:
        reconstruction = modes[:, :rank] @ coefficients[:rank]
        _, power = welch(
            reconstruction, fs=sfreq, nperseg=nperseg, noverlap=nperseg // 2
        )
        spectra.append(power)
    return count, np.stack(spectra) * count


def _merge_spectra(
    first: tuple[int, np.ndarray], second: tuple[int, np.ndarray]
) -> tuple[int, np.ndarray]:
    """Merge weighted PSD summaries without retaining every block's spectrum."""
    return first[0] + second[0], first[1] + second[1]


def reconstruction_spectra(
    values: da.Array,
    modes: np.ndarray,
    ranks: tuple[int, ...],
    sfreq: float,
    nperseg: int = 4096,
    windows_per_task: int = 16,
) -> tuple[np.ndarray, np.ndarray, int]:
    """Compute whole-recording Welch PSD with aligned overlapping bounded windows.

    Output shape is (1 + number of ranks, channels, frequencies); input comes
    first. All complete 50%-overlapping windows are used, with no cross-session
    windows. At most windows_per_task windows are processed in one task.
    """
    if (
        values.ndim != 2
        or values.shape[0] != modes.shape[0]
        or nperseg < 2
        or nperseg % 2
        or values.shape[1] < nperseg
        or sfreq <= 0
        or windows_per_task <= 0
        or not ranks
        or any(not 0 < rank <= modes.shape[1] for rank in ranks)
    ):
        msg = "Use compatible arrays, valid ranks, even window size and positive rate."
        raise ValueError(msg)
    hop = nperseg // 2
    n_windows = 1 + (values.shape[1] - nperseg) // hop
    tasks = []
    for first in range(0, n_windows, windows_per_task):
        count = min(windows_per_task, n_windows - first)
        start = first * hop
        stop = start + (count - 1) * hop + nperseg
        tasks.append(
            delayed(_spectral_block)(
                values[:, start:stop], modes, ranks, sfreq, nperseg
            )
        )
    # Reduce summaries in a balanced tree, rather than retaining one PSD array
    # per task in the driver as recording duration increases.
    while len(tasks) > 1:
        paired = [
            delayed(_merge_spectra)(tasks[index], tasks[index + 1])
            for index in range(0, len(tasks) - 1, 2)
        ]
        if len(tasks) % 2:
            paired.append(tasks[-1])
        tasks = paired
    count, total = dask.compute(tasks[0])[0]
    power = total / count
    return np.fft.rfftfreq(nperseg, 1 / sfreq), power, n_windows


def band_power_errors(
    frequencies: np.ndarray,
    spectra: np.ndarray,
    bands: dict[str, tuple[float, float]],
) -> dict[str, np.ndarray]:
    """Return per-channel relative band-power errors for each reconstructed rank.

    Undefined zero-power baselines produce NaN, rather than inflated ratios.
    Bands require at least two FFT bins to define the trapezoidal integral.
    """
    if spectra.ndim != 3 or spectra.shape[2] != len(frequencies) or not bands:
        msg = "Expected input/reconstruction PSD arrays and named frequency bands."
        raise ValueError(msg)
    errors = {}
    for name, (low, high) in bands.items():
        mask = (frequencies >= low) & (frequencies <= high)
        if mask.sum() < 2 or low >= high:
            msg = f"Band {name} needs at least two bins and increasing bounds."
            raise ValueError(msg)
        powers = trapezoid(spectra[:, :, mask], frequencies[mask], axis=-1)
        errors[name] = np.divide(
            powers[1:] - powers[0],
            powers[0],
            out=np.full_like(powers[1:], np.nan),
            where=powers[0] > 0,
        )
    return errors
