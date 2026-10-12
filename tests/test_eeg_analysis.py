"""Scientific checks for EEG conditioning, compression and validation helpers."""

from __future__ import annotations

import sys
from importlib import import_module
from pathlib import Path

import dask.array as da
import numpy as np
import pytest
from scipy.signal import butter, sosfiltfilt, welch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
analysis = import_module("demos.eeg_analysis")


def test_filter_blocks_match_whole_recording_and_preserve_oscillations(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Overlap avoids seams while removing offset/drift and retaining a 10 Hz tone."""
    sfreq = 64
    time = np.arange(sfreq * 100) / sfreq
    values = np.stack(
        [
            1000
            + 30 * np.sin(2 * np.pi * 0.05 * time)
            + 4 * np.sin(2 * np.pi * 10 * time),
            2000
            + 20 * np.sin(2 * np.pi * 0.1 * time)
            + 3 * np.sin(2 * np.pi * 8 * time),
        ]
    ).astype(np.float32)
    calls = []

    def reader(
        path: Path, start: int, stop: int, channels: tuple[str, ...]
    ) -> np.ndarray:
        assert path.name == "synthetic.fif"
        assert channels == ("A", "B")
        calls.append((start, stop))
        return values[:, start:stop]

    monkeypatch.setattr(analysis, "read_eeg_block", reader)
    lazy = analysis.filtered_eeg_file(
        Path("synthetic.fif"), values.shape[1], ("A", "B"), 700, sfreq
    )
    assert not calls
    result = lazy.compute(scheduler="synchronous")
    expected = sosfiltfilt(
        butter(4, 0.5, btype="highpass", fs=sfreq, output="sos"),
        values.astype(np.float64),
        axis=1,
    )
    np.testing.assert_allclose(result, expected, atol=1e-4)
    assert result.shape == values.shape
    assert result.dtype == np.float32
    assert max(stop - start for start, stop in calls) <= 700 + 2 * 32 * sfreq
    inner = slice(32 * sfreq, -32 * sfreq)
    np.testing.assert_allclose(result[0, inner].std(), 4 / np.sqrt(2), rtol=0.01)
    assert abs(result[0, inner].mean()) < 0.001


@pytest.mark.parametrize(
    "kwargs",
    [
        {"high_pass": 0},
        {"chunk_samples": 0},
        {"halo_seconds": 1},
        {"n_samples": 15},
        {"channels": ()},
    ],
)
def test_filter_rejects_invalid_configuration(kwargs: dict) -> None:
    """An unstable cutoff or insufficient context fails before reading files."""
    config = {
        "path": Path("unused"),
        "n_samples": 100,
        "channels": ("A",),
        "chunk_samples": 30,
        "sfreq": 64,
    }
    config.update(kwargs)
    with pytest.raises(ValueError, match="valid cutoff"):
        analysis.filtered_eeg_file(**config)


def test_recording_statistics_handles_large_offsets() -> None:
    """Centered covariance matches NumPy despite large constant channel offsets."""
    rng = np.random.default_rng(2)
    values = rng.normal(size=(4, 173)) + np.arange(4)[:, None] * 1e6
    result = analysis.recording_statistics(da.from_array(values, chunks=(4, 23)))
    np.testing.assert_allclose(result["mean"], values.mean(axis=1), atol=1e-9)
    np.testing.assert_allclose(
        result["covariance"], np.cov(values, bias=True), atol=1e-10
    )
    assert result["n_samples"] == values.shape[1]
    with pytest.raises(ValueError, match="nonempty"):
        analysis.recording_statistics(da.zeros((3, 0)))


def test_covariance_scores_match_explicit_waveform_reconstruction() -> None:
    """Summary-based energy, RMSE, channel errors and correlations are exact."""
    rng = np.random.default_rng(12)
    values = rng.normal(size=(5, 53)) * np.arange(1, 6)[:, None]
    values -= values.mean(axis=1, keepdims=True)
    covariance = values @ values.T / values.shape[1]
    modes = analysis.covariance_modes(covariance, 4)
    scores = analysis.covariance_scores(covariance, modes, (1, 2, 4))
    for rank in (1, 2, 4):
        reconstruction = modes[:, :rank] @ (modes[:, :rank].T @ values)
        error = values - reconstruction
        np.testing.assert_allclose(
            scores[rank]["energy"], 1 - np.sum(error**2) / np.sum(values**2)
        )
        np.testing.assert_allclose(scores[rank]["rmse"], np.sqrt(np.mean(error**2)))
        np.testing.assert_allclose(
            scores[rank]["channel_rmse"], np.sqrt(np.mean(error**2, axis=1))
        )
        np.testing.assert_allclose(
            scores[rank]["reconstructed_covariance"],
            reconstruction @ reconstruction.T / values.shape[1],
        )
    np.testing.assert_allclose(
        analysis.correlation_matrix(covariance), np.corrcoef(values)
    )
    constant_covariance = covariance.copy()
    constant_covariance[0] = constant_covariance[:, 0] = 0
    assert np.isnan(analysis.correlation_matrix(constant_covariance)[0]).all()


def test_participant_validation_excludes_all_sessions_and_matches_direct_svd() -> None:
    """Cached covariance folding agrees with a fresh concatenated training SVD."""
    rng = np.random.default_rng(8)
    subjects = [1, 1, 2, 3]
    arrays = [rng.normal(size=(5, length)) for length in (17, 23, 19, 29)]
    arrays = [array - array.mean(axis=1, keepdims=True) for array in arrays]
    statistics = [
        {"covariance": array @ array.T / array.shape[1], "n_samples": array.shape[1]}
        for array in arrays
    ]
    folds = analysis.participant_validation(statistics, subjects, (1, 3))
    for fold in folds:
        subject = fold["subject"]
        train = np.concatenate(
            [a for a, s in zip(arrays, subjects, strict=True) if s != subject], axis=1
        )
        test = np.concatenate(
            [a for a, s in zip(arrays, subjects, strict=True) if s == subject], axis=1
        )
        modes, _, _ = np.linalg.svd(train, full_matrices=False)
        assert fold["training_samples"] == train.shape[1]
        assert fold["test_samples"] == test.shape[1]
        for rank in (1, 3):
            prediction = modes[:, :rank] @ (modes[:, :rank].T @ test)
            expected = 1 - np.sum((test - prediction) ** 2) / np.sum(test**2)
            np.testing.assert_allclose(fold["ranks"][rank]["test_energy"], expected)
            assert fold["ranks"][rank]["test_optimal_energy"] >= expected - 1e-12
    with pytest.raises(ValueError, match="two participants"):
        analysis.participant_validation(statistics[:2], subjects[:2], (1,))


def test_spectral_reconstruction_matches_full_welch_including_chunk_boundaries() -> (
    None
):
    """Streaming window alignment gives exactly the full-array Welch PSD."""
    rng = np.random.default_rng(32)
    values = rng.normal(size=(4, 4123))
    modes, _ = np.linalg.qr(rng.normal(size=(4, 3)))
    frequency, spectra, count = analysis.reconstruction_spectra(
        da.from_array(values, chunks=(4, 137)),
        modes,
        (1, 3),
        128,
        nperseg=256,
        windows_per_task=5,
    )
    assert count == 1 + (values.shape[1] - 256) // 128
    expected_frequency, original = welch(values, fs=128, nperseg=256, noverlap=128)
    np.testing.assert_array_equal(frequency, expected_frequency)
    np.testing.assert_allclose(spectra[0], original, rtol=1e-12)
    for index, rank in enumerate((1, 3), start=1):
        reconstructed = modes[:, :rank] @ (modes[:, :rank].T @ values)
        _, expected = welch(reconstructed, fs=128, nperseg=256, noverlap=128)
        np.testing.assert_allclose(spectra[index], expected, rtol=1e-12)
    with pytest.raises(ValueError, match="compatible arrays"):
        analysis.reconstruction_spectra(da.from_array(values), modes, (4,), 128)


def test_band_power_errors_use_power_rather_than_amplitude() -> None:
    """Halving waveform amplitude quarters band power; zero baselines are undefined."""
    frequency = np.arange(65)
    spectra = np.ones((3, 2, 65))
    spectra[1] *= 0.25
    spectra[2] *= 4
    spectra[:, 1] = 0
    errors = analysis.band_power_errors(frequency, spectra, {"alpha": (8, 13)})["alpha"]
    np.testing.assert_allclose(errors[:, 0], [-0.75, 3])
    assert np.isnan(errors[:, 1]).all()
    with pytest.raises(ValueError, match="two bins"):
        analysis.band_power_errors(frequency, spectra, {"narrow": (8, 8.1)})
    with pytest.raises(ValueError, match="PSD arrays"):
        analysis.band_power_errors(frequency, spectra[0], {"alpha": (8, 13)})


def test_compression_ratio_includes_offset_metadata() -> None:
    """The nominal ratio counts coefficients, spatial modes and channel means."""
    ratio = analysis.compression_ratio(64, 10000, 8)
    np.testing.assert_allclose(ratio, 640000 / (80000 + 512 + 64))
    assert ratio < 8
    with pytest.raises(ValueError, match="positive dimensions"):
        analysis.compression_ratio(64, 100, 65)


def test_covariance_helpers_reject_undefined_energy_and_invalid_modes() -> None:
    """Undefined scores and nonorthogonal bases fail explicitly."""
    with pytest.raises(ValueError, match="positive"):
        analysis.covariance_scores(np.zeros((3, 3)), np.eye(3)[:, :2], (1,))
    with pytest.raises(ValueError, match="orthonormal"):
        analysis.covariance_scores(np.eye(3), np.ones((3, 2)), (1,))
    with pytest.raises(ValueError, match="square covariance"):
        analysis.covariance_modes(np.ones((2, 3)), 1)
