"""Full-schedule auditing and actual disk round trips for the large EEG demo."""

from __future__ import annotations

import json
import sys
from importlib import import_module
from itertools import product
from pathlib import Path
from types import SimpleNamespace

import dask
import dask.array as da
import h5py
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
large = import_module("demos.eeg_large")
benchmark = import_module("demos.benchmark_eeg_large")


def write_edf(path: Path, *, rate: int = 256, unit: str = "uV") -> None:
    """Write a minimal valid EDF with independently specified calibration."""
    channels = list(large.CHANNELS)
    channels[2] = "Afz"
    n = len(channels)
    header = b"".join(
        str(value).encode().ljust(width)
        for value, width in (
            (0, 8),
            ("patient", 80),
            ("recording", 80),
            ("01.01.26", 8),
            ("12.00.00", 8),
            (256 * (n + 1), 8),
            ("", 44),
            (1, 8),
            (1, 8),
            (n, 4),
        )
    )
    fields = b"".join(
        str(value).encode().ljust(width)
        for values, width in (
            (channels, 16),
            ([""] * n, 80),
            ([unit] * n, 8),
            ([-100] * n, 8),
            ([100] * n, 8),
            ([-32768] * n, 8),
            ([32767] * n, 8),
            ([""] * n, 80),
            ([rate] * n, 8),
            ([""] * n, 32),
        )
        for value in values
    )
    path.write_bytes(header + fields + bytes(n * rate * 2))


def test_edf_header_counts_only_eeg_and_checks_payload(tmp_path: Path) -> None:
    """Header calibration, native rate and sample count are independently checked."""
    path = tmp_path / "sample.edf"
    write_edf(path)
    item = large.edf_metadata(path)
    assert item["n_samples"] == 256
    assert item["channels"] == list(large.CHANNELS)
    assert item["physical_min"] == [-100] * 32
    assert item["digital_max"] == [32767] * 32
    path.write_bytes(path.read_bytes()[:-2])
    with pytest.raises(ValueError, match="payload length"):
        large.edf_metadata(path)
    write_edf(path, rate=128)
    with pytest.raises(ValueError, match="256 Hz"):
        large.edf_metadata(path)
    write_edf(path, unit="mV")
    with pytest.raises(ValueError, match="uV"):
        large.edf_metadata(path)
    path.write_bytes(b"bad")
    with pytest.raises(ValueError, match="Invalid EDF"):
        large.edf_metadata(path)


def test_full_manifest_matches_schedule_and_reports_extra_files(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """All canonical blocks are required, including the mislabeled subject 8 files."""
    example = tmp_path / "example.edf"
    write_edf(example)
    metadata = large.edf_metadata(example)
    monkeypatch.setattr(large, "edf_metadata", lambda _path: metadata.copy())
    paths = []
    for subject, session, block in product(range(1, 21), range(1, 5), range(1, 20)):
        name = (
            19
            if subject == 8 and ((session == 1 and block >= 2) or session in (3, 4))
            else subject
        )
        directory = (
            tmp_path
            / f"sub-{subject:02d}"
            / f"session_{session:02d}"
            / f"block_{block:02d}"
        )
        directory.mkdir(parents=True)
        path = (
            directory
            / f"Subject {name}, Session {session}, Block {block} recording.edf"
        )
        path.touch()
        paths.append(path)
    manifest = large.alljoined16_manifest(tmp_path)
    assert len(manifest["records"]) == 1520
    assert len(manifest["metadata_profiles"]) == 1
    assert manifest["noncanonical_edf_files"] == ["example.edf"]
    subject8 = [i for i in manifest["records"] if i["subject"] == 8]
    assert len(subject8) == 76
    assert "Subject 19" in subject8[1]["file"]
    paths[-1].unlink()
    with pytest.raises(ValueError, match="Expected one scheduled EDF"):
        large.alljoined16_manifest(tmp_path)


@pytest.fixture()
def synthetic_cache(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> tuple:
    """Create a small disk corpus through the same streamed preparation path."""
    rng = np.random.default_rng(4)
    modes = rng.normal(size=(32, 6))
    records, signals = [], {}
    for subject in (1, 2, 20):
        for block in (1, 2):
            key = f"sub{subject:02d}_s01_b{block:02d}"
            n = 61 + block
            records.append(
                {
                    "key": key,
                    "file": key + ".edf",
                    "subject": subject,
                    "session": 1,
                    "block": block,
                    "n_samples": n,
                }
            )
            signals[key] = (
                modes @ rng.normal(size=(6, n)) + np.arange(32)[:, None]
            ).astype("f4")
    manifest = {
        "records": records,
        "source_root": str(tmp_path),
        "dataset": "synthetic",
        "noncanonical_edf_files": [],
    }
    monkeypatch.setattr(large, "alljoined16_manifest", lambda _path: manifest)
    monkeypatch.setattr(
        large,
        "filtered_eeg_file",
        lambda path, *_args: da.from_array(signals[path.stem], chunks=(32, 11)),
    )
    cache = tmp_path / "cache.h5"
    large.prepare_cache(tmp_path, cache, 17)
    return cache, manifest, signals


def test_cache_centering_covariance_resume_and_lazy_reads(
    synthetic_cache: tuple,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Cached statistics match NumPy and completed records are reused safely."""
    cache, manifest, signals = synthetic_cache
    assert large.cache_manifest(cache) == manifest
    item = manifest["records"][0]
    centered = (
        signals[item["key"]] - signals[item["key"]].mean(axis=1, dtype="f8")[:, None]
    )
    centered = centered.astype("f4")
    lazy = large.cached_recording(cache, item, 13)
    assert lazy.chunks[1][0] == 13
    np.testing.assert_array_equal(lazy.compute(), centered)
    with h5py.File(cache, "r") as store:
        covariance = store[f"{item['key']}/covariance"][:]
    np.testing.assert_allclose(
        covariance, centered.astype("f8") @ centered.astype("f8").T / centered.shape[1]
    )

    def unexpected(*_args: object) -> None:
        msg = "A complete cache must not re-read source data"
        raise AssertionError(msg)

    monkeypatch.setattr(large, "filtered_eeg_file", unexpected)
    large.prepare_cache(cache.parent, cache, 17)
    with pytest.raises(ValueError, match="provenance/settings"):
        large.prepare_cache(cache.parent, cache, 18)
    with pytest.raises(ValueError, match="positive"):
        large.cached_recording(cache, item, 0)
    with h5py.File(cache, "r+") as store:
        store[item["key"]].attrs["complete"] = False
    with pytest.raises(ValueError, match="Incomplete"):
        large.cache_manifest(cache)


def test_cache_resumes_incomplete_group_and_rejects_unrelated_files(
    synthetic_cache: tuple,
    tmp_path: Path,
) -> None:
    """An interrupted record is rebuilt; another user's HDF5 data is protected."""
    cache, manifest, _signals = synthetic_cache
    with h5py.File(cache, "r+") as store:
        group = store[manifest["records"][0]["key"]]
        group.attrs["complete"] = False
        group["signal"][:] = 99
    large.prepare_cache(cache.parent, cache, 17)
    large.cache_manifest(cache)
    other = tmp_path / "unrelated.h5"
    with h5py.File(other, "w") as store:
        store.create_dataset("precious", data=[1, 2])
    with pytest.raises(ValueError, match="unrelated"):
        large.prepare_cache(tmp_path, other, 17)
    with pytest.raises(ValueError, match="positive"):
        large.prepare_cache(tmp_path, cache, 0)


def test_public_pod_disk_encoding_and_full_reconstruction(
    synthetic_cache: tuple,
) -> None:
    """Saved float32 factors reconstruct every sample without held-out fitting."""
    cache, manifest, _signals = synthetic_cache
    with dask.config.set(scheduler="single-threaded"):
        model = large.fit_cached(cache, 19, 17, n_modes=16)
    assert model.time_coeffs.sizes["sample"] == sum(
        i["n_samples"] for i in manifest["records"] if i["subject"] != 20
    )
    output = cache.parent / "encoded.h5"
    storage = large.encode_cached(cache, model, output, 8, 17)
    verified = large.verify_encoding(cache, output, 13)
    assert set(verified["participants"]) == {1, 2, 20}
    assert verified["all_sample_energy_retained"] > 1 - 1e-10
    assert storage["file_bytes"] == output.stat().st_size
    assert (
        storage["payload_bytes"]
        == (32 * 8 + 8 * sum(i["n_samples"] for i in manifest["records"]) + 32 * 6) * 4
    )
    with h5py.File(cache, "r") as source, h5py.File(output, "r") as target:
        assert target["modes"].dtype == np.dtype("f4")
        item = manifest["records"][0]
        decoded = target["modes"][:] @ target[f"{item['key']}/coefficients"][:]
        np.testing.assert_allclose(
            decoded, source[f"{item['key']}/signal"][:], atol=2e-6
        )
    with pytest.raises(FileExistsError):
        large.encode_cached(cache, model, output, 8, 17)
    with pytest.raises(ValueError, match="retained model rank"):
        large.encode_cached(cache, model, output, 17, 17)
    with pytest.raises(ValueError, match="positive"):
        large.verify_encoding(cache, output, 0)
    with h5py.File(output, "r+") as store:
        store.attrs["complete"] = False
    with pytest.raises(ValueError, match="Incomplete encoding"):
        large.verify_encoding(cache, output, 13)
    with pytest.raises(ValueError, match="1-19"):
        large.fit_cached(cache, 20, 17)


def test_edf_reader_selects_electrodes_and_closes_on_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Telemetry is excluded and the filename's Afz spelling is normalized."""
    utils = import_module("demos.eeg_utils")
    raw = SimpleNamespace(ch_names=["Afz", "Cz"])
    raw.rename_channels = lambda mapping: setattr(
        raw, "ch_names", [mapping.get(c, c) for c in raw.ch_names]
    )
    raw.pick = lambda channels: setattr(raw, "ch_names", channels)
    raw.get_data = (
        lambda **kwargs: np.ones((2, kwargs["stop"] - kwargs["start"])) * 1e-6
    )
    raw.close = lambda: setattr(raw, "closed", True)

    def read(_path: Path, **kwargs: object) -> SimpleNamespace:
        assert kwargs["preload"] is False
        assert kwargs["include"] == ["AFz", "Cz", "Afz"]
        return raw

    monkeypatch.setitem(
        sys.modules, "mne", SimpleNamespace(io=SimpleNamespace(read_raw_edf=read))
    )
    values = utils.read_eeg_block(Path("example.edf"), 1, 4, ("AFz", "Cz"))
    np.testing.assert_array_equal(values, np.ones((2, 3)))
    assert raw.closed


def test_large_benchmark_scores_and_encodes_synthetic_corpus(
    synthetic_cache: tuple,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Benchmark fit and encoding stages exercise the public POD pipeline."""
    cache, _manifest, _signals = synthetic_cache
    case = benchmark.run_case(cache, 19, 17, 1)
    assert case["training_recordings"] == 4
    assert case["held_out_energy"]["8"] > 0.99999
    monkeypatch.setattr(
        benchmark,
        "encoded_spectra",
        lambda source, paths: {
            "frequencies": np.arange(1025) / 8,
            "psd": np.ones((3, 32, 1025)),
            "n_windows": 2,
            "provenance": {
                "manifest": large.cache_manifest(source),
                "preprocessing": large.PREPROCESSING,
                "subject": 20,
                "nperseg": 2048,
                "encodings": [large.validate_encoding(source, path) for path in paths],
            },
        },
    )
    result = benchmark.run_encoding(cache, cache.parent / "factors", 17, 1)
    assert len(result["encodings"]) == 2
    assert len(result["participant_folds"]) == 3


def test_disk_spectra_match_whole_recording_welch(synthetic_cache: tuple) -> None:
    """Aligned spectral tasks agree with SciPy, including recording boundaries."""
    from scipy.signal import welch

    cache, manifest, _signals = synthetic_cache
    with dask.config.set(scheduler="single-threaded"):
        model = large.fit_cached(cache, 19, 17, n_modes=16)
    output = cache.parent / "spectral-encoding.h5"
    large.encode_cached(cache, model, output, 8, 17)
    with dask.config.set(scheduler="single-threaded"):
        result = large.encoded_spectra(cache, (output,), nperseg=16, windows_per_task=2)
    sums = np.zeros((2, 32, 9))
    windows = 0
    with h5py.File(cache, "r") as source, h5py.File(output, "r") as target:
        for item in manifest["records"]:
            if item["subject"] != 20:
                continue
            original = source[f"{item['key']}/signal"][:]
            decoded = target["modes"][:] @ target[f"{item['key']}/coefficients"][:]
            count = 1 + (item["n_samples"] - 16) // 8
            powers = [
                welch(values, fs=256, nperseg=16, noverlap=8)[1]
                for values in (original, decoded)
            ]
            sums += np.stack(powers) * count
            windows += count
    assert result["n_windows"] == windows
    np.testing.assert_allclose(result["psd"], sums / windows, rtol=1e-6)
    spectral_path = cache.parent / "spectra.h5"
    large.save_encoded_spectra(spectral_path, result)
    loaded = large.load_encoded_spectra(cache, (output,), spectral_path, nperseg=16)
    np.testing.assert_array_equal(loaded["psd"], result["psd"])
    with pytest.raises(ValueError, match="provenance changed"):
        large.load_encoded_spectra(
            cache, (output,), spectral_path, subject=1, nperseg=16
        )
    with pytest.raises(ValueError, match="even window"):
        large.load_encoded_spectra(cache, (output,), spectral_path, nperseg=15)
    with h5py.File(spectral_path, "r+") as store:
        store.attrs["n_windows"] = windows + 1
    with pytest.raises(ValueError, match="Welch window count"):
        large.load_encoded_spectra(cache, (output,), spectral_path, nperseg=16)
    with h5py.File(spectral_path, "r+") as store:
        store.attrs["n_windows"] = windows
    # Flipping a basis sign preserves orthonormality but changes the disk decoder.
    # Reusing the previous PSD artifact must therefore fail provenance checks.
    with h5py.File(output, "r+") as store:
        store["modes"][:, 0] *= -1
    with pytest.raises(ValueError, match="provenance changed"):
        large.load_encoded_spectra(cache, (output,), spectral_path, nperseg=16)
    with pytest.raises(ValueError, match="even window"):
        large.encoded_spectra(cache, (output,), nperseg=15)
    with pytest.raises(ValueError, match="No complete spectral windows"):
        large.encoded_spectra(cache, (output,), nperseg=2048)


@pytest.mark.parametrize(
    "problem", ["conditioning", "subjects", "rank", "coefficients"]
)
def test_factor_validation_rejects_stale_or_malformed_inputs(
    synthetic_cache: tuple,
    problem: str,
) -> None:
    """Same-size files cannot bypass provenance, basis rank or coefficient checks."""
    cache, _manifest, _signals = synthetic_cache
    with dask.config.set(scheduler="single-threaded"):
        model = large.fit_cached(cache, 19, 17, n_modes=16)
    output = cache.parent / "factors.h5"
    large.encode_cached(cache, model, output, 8, 17)
    assert large.validate_encoding(cache, output, rank=8)["rank"] == 8
    if problem == "rank":
        with pytest.raises(ValueError, match="unexpected rank"):
            large.validate_encoding(cache, output, rank=16)
        return
    with h5py.File(output, "r+") as store:
        if problem == "conditioning":
            store.attrs["preprocessing"] = json.dumps({"high_pass_hz": 1})
        elif problem == "subjects":
            store.attrs["training_subjects"] = json.dumps([1, 2, 20])
        else:
            key = "sub01_s01_b01/coefficients"
            del store[key]
            store.create_dataset(key, shape=(8, 1), dtype="f4")
    with pytest.raises(ValueError, match="provenance/settings|Malformed encoded"):
        large.verify_encoding(cache, output, 17)


def test_large_benchmark_cli_launches_sequential_fresh_workers(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """The CLI records separate preparation, fitting and encoding memory peaks."""
    calls = []

    def run(command: list[str], **kwargs: object) -> SimpleNamespace:
        stage = command[command.index("--worker") + 1]
        calls.append(stage)
        assert kwargs["env"]["OPENBLAS_NUM_THREADS"] == "1"
        return SimpleNamespace(returncode=0, stdout=json.dumps({"peak_rss_gib": 0.25}))

    output = tmp_path / "report.json"
    monkeypatch.setattr(
        sys, "argv", ["benchmark", "--counts", "1", "19", "--output", str(output)]
    )
    monkeypatch.setattr(benchmark.platform, "platform", lambda: "test")
    monkeypatch.setattr(benchmark.subprocess, "run", run)
    monkeypatch.setattr(benchmark.importlib.metadata, "version", lambda _name: "test")
    benchmark.main()
    assert calls == ["prepare", "fit", "fit", "encode"]
    assert json.loads(output.read_text())["encode"]["peak_rss_gib"] == 0.25


@pytest.mark.parametrize("stage", ["prepare", "fit", "encode"])
def test_large_cli_worker_outputs_json(
    synthetic_cache: tuple,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    stage: str,
) -> None:
    """Each fresh worker returns parseable JSON with its own OS peak memory."""
    cache, _manifest, _signals = synthetic_cache
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "benchmark",
            "--worker",
            stage,
            "--cache",
            str(cache),
            "--chunk-samples",
            "17",
        ],
    )
    monkeypatch.setattr(
        benchmark, "run_case", lambda *_args: {"training_participants": 19}
    )
    monkeypatch.setattr(benchmark, "run_encoding", lambda *_args: {"encodings": []})
    benchmark.main()
    result = json.loads(capsys.readouterr().out)
    assert result["peak_rss_gib"] > 0
    if stage == "prepare":
        assert result["recordings_reused"] == 6
        assert result["recordings_prepared"] == 0


def test_preparation_report_reuse_checks_cache_provenance(
    synthetic_cache: tuple,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """A measured preparation can be reused only with its exact cache snapshot."""
    cache, manifest, _signals = synthetic_cache
    preparation = tmp_path / "preparation.json"
    preparation.write_text(
        json.dumps(
            {
                "manifest": manifest,
                "cache_bytes": cache.stat().st_size,
                "seconds": 12,
                "peak_rss_gib": 0.2,
            }
        )
    )
    report = tmp_path / "imported.json"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "benchmark",
            "--counts",
            "19",
            "--cache",
            str(cache),
            "--preparation-report",
            str(preparation),
            "--output",
            str(report),
        ],
    )
    monkeypatch.setattr(benchmark.platform, "platform", lambda: "test")
    monkeypatch.setattr(benchmark.importlib.metadata, "version", lambda _name: "test")
    stages = []

    def run(command: list[str], **_kwargs: object) -> SimpleNamespace:
        stages.append(command[command.index("--worker") + 1])
        return SimpleNamespace(returncode=0, stdout=json.dumps({"peak_rss_gib": 0.3}))

    monkeypatch.setattr(benchmark.subprocess, "run", run)
    benchmark.main()
    assert stages == ["fit", "encode"]
    assert json.loads(report.read_text())["prepare"]["seconds"] == 12
    preparation.write_text(json.dumps({"manifest": manifest, "cache_bytes": 0}))
    with pytest.raises(ValueError, match="does not describe"):
        benchmark.main()
