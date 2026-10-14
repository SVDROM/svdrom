"""Tests for the out-of-core reader used by the EEG notebook."""

from __future__ import annotations

import sys
from importlib import import_module
from pathlib import Path
from types import SimpleNamespace
from typing import ClassVar

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
eeg_utils = import_module("demos.eeg_utils")


def test_read_eeg_block_converts_units_and_closes_file(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A requested block is read in channel order and converted to microvolts."""
    signal = np.arange(20, dtype=np.float64).reshape(2, 10) * 1e-6

    class FakeRaw:
        """Minimal MNE Raw stand-in for a deterministic block read."""

        ch_names: ClassVar[list[str]] = ["Fz", "Cz"]

        def __init__(self) -> None:
            self.closed = False

        def pick(self, kind: str) -> None:
            assert kind == "eeg"

        def get_data(self, start: int, stop: int) -> np.ndarray:
            return signal[:, start:stop]

        def close(self) -> None:
            self.closed = True

    raw = FakeRaw()
    reader = SimpleNamespace(read_raw_fif=lambda *args, **kwargs: raw)
    monkeypatch.setitem(sys.modules, "mne", SimpleNamespace(io=reader))

    block = eeg_utils.read_eeg_block(Path("recording.fif"), 2, 5, ("Fz", "Cz"))

    np.testing.assert_array_equal(block, signal[:, 2:5] * 1e6)
    assert block.dtype == np.float32
    assert raw.closed


def test_lazy_eeg_file_reads_only_when_computed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The reader builds independent time blocks and defers I/O."""
    calls: list[tuple[int, int]] = []

    def fake_read(
        path: Path, start: int, stop: int, channels: tuple[str, ...]
    ) -> np.ndarray:
        assert path.name == "recording.fif"
        assert channels == ("Fz", "Cz")
        calls.append((start, stop))
        return np.full((2, stop - start), start, dtype=np.float32)

    monkeypatch.setattr(eeg_utils, "read_eeg_block", fake_read)
    array = eeg_utils.lazy_eeg_file(Path("recording.fif"), 10, ("Fz", "Cz"), 4)

    assert array.shape == (2, 10)
    assert not calls
    np.testing.assert_array_equal(
        array.compute(),
        np.tile(np.array([0, 0, 0, 0, 4, 4, 4, 4, 8, 8]), (2, 1)),
    )
    assert sorted(calls) == [(0, 4), (4, 8), (8, 10)]


@pytest.mark.parametrize("bad_block", [np.full((2, 3), np.nan), np.zeros((1, 3))])
def test_read_eeg_block_rejects_invalid_data(
    monkeypatch: pytest.MonkeyPatch, bad_block: np.ndarray
) -> None:
    """Non-finite values and an unexpected channel count fail early."""

    class FakeRaw:
        """Minimal MNE Raw stand-in returning a malformed block."""

        ch_names: ClassVar[list[str]] = ["Fz", "Cz"]

        def pick(self, kind: str) -> None:
            assert kind == "eeg"

        def get_data(self, start: int, stop: int) -> np.ndarray:
            assert 0 <= start < stop
            return bad_block

        def close(self) -> None:
            pass

    reader = SimpleNamespace(read_raw_fif=lambda *args, **kwargs: FakeRaw())
    monkeypatch.setitem(sys.modules, "mne", SimpleNamespace(io=reader))

    with pytest.raises(ValueError, match="Invalid EEG block"):
        eeg_utils.read_eeg_block(Path("recording.fif"), 2, 5, ("Fz", "Cz"))


@pytest.mark.parametrize(
    ("n_samples", "chunk_samples", "channels"),
    [(0, 4, ("Fz",)), (10, 0, ("Fz",)), (10, 4, ())],
)
def test_lazy_eeg_file_rejects_invalid_sizes(
    n_samples: int, chunk_samples: int, channels: tuple[str, ...]
) -> None:
    """Invalid shape parameters fail before building a Dask graph."""
    with pytest.raises(ValueError, match="positive recording length"):
        eeg_utils.lazy_eeg_file(
            Path("recording.fif"), n_samples, channels, chunk_samples
        )


def test_alljoined_paths_fixed_split() -> None:
    """The corpus contains 12 training paths followed by subject 08."""
    paths = eeg_utils.alljoined_paths(Path("data"))
    assert len(paths) == len(set(paths)) == 13
    assert paths[0] == Path("data/subj01_session1_eeg.fif")
    assert paths[-1] == Path("data/subj08_session1_eeg.fif")


@pytest.mark.parametrize("mismatch", [None, "channels", "sfreq", "unit"])
def test_recording_manifest_validates_headers(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    mismatch: str | None,
) -> None:
    """Header inconsistencies fail before signal reads and always close files."""
    raws = []

    class FakeRaw:
        """Metadata-only FIF stand-in."""

        def __init__(self, second: bool) -> None:
            self.closed = False
            self.ch_names = ["Cz"] if second and mismatch == "channels" else ["Fz"]
            self.n_times = 100
            self.info = {
                "sfreq": 128 if second and mismatch == "sfreq" else 512,
                "chs": [
                    {
                        "unit": 0 if second and mismatch == "unit" else 107,
                        "unit_mul": 0,
                        "cal": 1.0,
                        "range": 1.0,
                    }
                ],
                "custom_ref_applied": 0,
                "bads": [],
                "projs": [],
            }

        def pick(self, kind: str) -> None:
            assert kind == "eeg"

        def close(self) -> None:
            self.closed = True

    def read(path: Path, **kwargs: object) -> FakeRaw:
        raw = FakeRaw(path.name == "second.fif")
        raws.append(raw)
        return raw

    monkeypatch.setitem(
        sys.modules, "mne", SimpleNamespace(io=SimpleNamespace(read_raw_fif=read))
    )
    monkeypatch.setitem(
        sys.modules,
        "mne._fiff.constants",
        SimpleNamespace(FIFF=SimpleNamespace(FIFF_UNIT_V=107)),
    )
    paths = [tmp_path / "first.fif", tmp_path / "second.fif"]
    for path in paths:
        path.write_bytes(b"header")
    if mismatch:
        with pytest.raises(ValueError, match="Expected EEG|Inconsistent EEG"):
            eeg_utils.recording_manifest(paths)
    else:
        manifest = eeg_utils.recording_manifest(paths)
        assert manifest[0]["bytes"] == 6
        assert manifest[1]["unit"] == "V"
        assert manifest[1]["n_samples"] == 100
    assert len(raws) == 2
    assert all(raw.closed for raw in raws)


def test_reconstruction_diagnostics_separates_offset_and_fluctuations() -> None:
    """A fixed offset leaves fluctuation reconstruction exact and obeys MSE identity."""
    import dask
    import dask.array as da

    observed = da.from_array(np.arange(24).reshape(3, 8), chunks=(3, 3))
    reconstructed = observed + np.array([3, 4, 0])[:, None]
    metrics = dask.compute(
        eeg_utils.reconstruction_diagnostics(observed, reconstructed)
    )[0]
    np.testing.assert_allclose(metrics["offset_rmse"], np.sqrt(25 / 3))
    np.testing.assert_allclose(metrics["fluctuation_rmse"], 0, atol=1e-12)
    np.testing.assert_allclose(metrics["fluctuation_energy"], 1)
    np.testing.assert_allclose(
        metrics["total_rmse"] ** 2,
        metrics["offset_rmse"] ** 2 + metrics["fluctuation_rmse"] ** 2,
    )
    np.testing.assert_allclose(metrics["channel_mean"], [3.5, 11.5, 19.5])


@pytest.mark.parametrize("shape", [(3,), (3, 4), (3, 0)])
def test_reconstruction_diagnostics_rejects_shapes(shape: tuple[int, ...]) -> None:
    """A malformed reconstruction cannot silently broadcast into valid metrics."""
    import dask.array as da

    with pytest.raises(ValueError, match="matching, nonempty"):
        eeg_utils.reconstruction_diagnostics(da.zeros(shape), da.zeros((3, 0)))


def test_benchmark_case_scores_full_synthetic_recording(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Benchmark computation agrees with direct projection for synthetic inputs."""
    import dask.array as da

    benchmark = import_module("demos.benchmark_eeg")
    rng = np.random.default_rng(9)
    signal = (rng.normal(size=(10, 3)) @ rng.normal(size=(3, 40))).astype(np.float32)
    monkeypatch.setattr(
        benchmark,
        "recording_manifest",
        lambda paths: [
            {"channels": list("ABCDEFGHIJ"), "n_samples": 40, "sfreq": 512}
            for _ in paths
        ],
    )
    monkeypatch.setattr(
        benchmark,
        "filtered_eeg_file",
        lambda *args: da.from_array(signal, chunks=(10, 7)),
    )
    result = benchmark.run_case(Path("unused"), 2, 7, 1)
    assert result["training_samples"] == 80
    assert result["training_files"] == 2
    np.testing.assert_allclose(result["held_out_energy"], 1, atol=1e-10)
    assert result["fit_seconds"] > 0
    with pytest.raises(ValueError, match="1-12"):
        benchmark.run_case(Path("unused"), 13, 7, 1)


def test_benchmark_cli_runs_fresh_workers(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """The parent launches independent workers and records their environment."""
    import json

    benchmark = import_module("demos.benchmark_eeg")
    calls = []

    def fake_run(command: list[str], **kwargs: object) -> SimpleNamespace:
        calls.append(command)
        assert kwargs["env"]["OPENBLAS_NUM_THREADS"] == "1"
        return SimpleNamespace(
            returncode=0,
            stdout=json.dumps(
                {"training_files": int(command[command.index("--worker") + 1])}
            ),
        )

    output = tmp_path / "benchmark.json"
    monkeypatch.setattr(
        sys, "argv", ["benchmark", "--counts", "1", "4", "--output", str(output)]
    )
    monkeypatch.setattr(benchmark.platform, "platform", lambda: "test-platform")
    monkeypatch.setattr(benchmark.platform, "processor", lambda: "test-processor")
    monkeypatch.setattr(benchmark.subprocess, "run", fake_run)
    monkeypatch.setattr(benchmark.importlib.metadata, "version", lambda _name: "test")
    benchmark.main()
    report = json.loads(output.read_text())
    assert [case["training_files"] for case in report["cases"]] == [1, 4]
    assert len(calls) == 2
    assert report["gpu_used"] is False
