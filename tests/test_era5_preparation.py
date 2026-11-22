"""Offline tests for native-precision, resumable ERA5 data preparation."""

from __future__ import annotations

import io
import json
import runpy
from pathlib import Path
from types import SimpleNamespace
from urllib.error import URLError

import numcodecs
import numpy as np
import pytest
import xarray as xr


@pytest.fixture()
def preparation() -> dict:
    """Load the standalone acquisition script without making network requests."""
    return runpy.run_path(
        str(Path(__file__).resolve().parents[1] / "benchmarks/prepare_era5.py")
    )


def test_chunk_validation_and_resume(
    preparation: dict, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Raw chunks retain dtype, resume without downloads, and replace corruption."""
    codec = numcodecs.Blosc(cname="lz4")
    values = np.arange(6, dtype="float32")
    payload = codec.encode(values)
    metadata = {"compressor": codec.get_config(), "dtype": "<f4", "chunks": [1, 2, 3]}
    variable = "2m_temperature"
    (tmp_path / variable).mkdir()
    copy = preparation["copy_chunk"]
    calls = []

    def fetch(url: str) -> bytes:
        calls.append(url)
        return payload

    monkeypatch.setitem(copy.__globals__, "fetch", fetch)
    first = copy(tmp_path, variable, 100, 0, metadata)
    second = copy(tmp_path, variable, 100, 0, metadata)
    assert not first["reused"], "Expected the first chunk to be downloaded."
    assert second["reused"], "Expected the verified chunk to be reused."
    assert len(calls) == 1, "Expected no redundant network read for a verified chunk."
    assert (
        first["sha256"] == second["sha256"]
    ), "Expected stable compressed-content hashes."
    (tmp_path / variable / "0.0.0").write_bytes(b"corrupt")
    replaced = copy(tmp_path, variable, 100, 0, metadata)
    assert not replaced["reused"], "Expected corrupted data to be downloaded again."
    assert len(calls) == 2, "Expected exactly one repair download."
    with pytest.raises(ValueError, match="non-finite"):
        preparation["validate_chunk"](
            codec.encode(np.full(6, np.nan, dtype="float32")), metadata
        )


def test_fetch_retries_transient_failures(
    preparation: dict, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A transient network failure is retried, with bounded attempts."""
    fetch = preparation["fetch"]
    attempts = []

    def open_url(_url: str, timeout: int) -> io.BytesIO:
        assert timeout > 0, "Expected a bounded network timeout."
        attempts.append(1)
        if len(attempts) == 1:
            message = "temporary"
            raise URLError(message)
        return io.BytesIO(b"metadata")

    monkeypatch.setitem(fetch.__globals__, "urlopen", open_url)
    monkeypatch.setattr(preparation["time"], "sleep", lambda _seconds: None)
    assert (
        fetch("https://example.test", attempts=2) == b"metadata"
    ), "Expected recovery after a transient failure."
    assert len(attempts) == 2, "Expected one retry."


def test_subset_preserves_real_samples(preparation: dict, tmp_path: Path) -> None:
    """Linked subsets preserve selected values and coordinates without resampling."""
    source, target = tmp_path / "source.zarr", tmp_path / "subset.zarr"
    values = np.arange(36, dtype="float32").reshape(6, 2, 3)
    field = xr.DataArray(
        values,
        dims=("time", "latitude", "longitude"),
        coords={"time": np.arange(6), "latitude": [1, 2], "longitude": [3, 4, 5]},
        name="2m_temperature",
    )
    field.to_zarr(
        source, encoding={"2m_temperature": {"chunks": (1, 2, 3)}}, zarr_format=2
    )
    (source / "acquisition.json").write_text(
        json.dumps(
            {
                "complete": True,
                "shape": [6, 2, 3],
                "dtype": "<f4",
                "variable": "2m_temperature",
            }
        )
    )
    preparation["create_subset"](source, target, "2m_temperature", 2)
    preparation["create_subset"](source, target, "2m_temperature", 2)
    actual = xr.open_zarr(target)["2m_temperature"]
    xr.testing.assert_equal(actual, field.isel(time=slice(None, None, 2)))
    assert actual.dtype == np.dtype("float32"), "Expected native precision unchanged."
    assert (target / "_SUCCESS").exists(), "Expected a completed subset marker."
    manifest = json.loads((target / "acquisition.json").read_text())
    assert (
        manifest["decoded_bytes"] == values[::2].nbytes
    ), "Expected the genuine selected-data size."


def test_initialization_rejects_incompatible_source(
    preparation: dict, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Raw encoded copies cannot silently reinterpret incompatible source layouts."""
    initialize = preparation["initialize"]
    metadata = {"chunks": [4, 2, 3]}
    monkeypatch.setitem(
        initialize.__globals__,
        "fetch",
        lambda url: json.dumps(
            metadata
            if url.endswith(".zarray")
            else {"_ARRAY_DIMENSIONS": ["time", "latitude", "longitude"]}
        ).encode(),
    )
    with pytest.raises(ValueError, match="hourly full-grid"):
        initialize(tmp_path, "2m_temperature", 2018)


def test_pilot_resumes_to_complete_store(
    preparation: dict, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A pilot verifies chunks incrementally before marking a download complete."""
    target = tmp_path / "weather.zarr"
    (target / "2m_temperature").mkdir(parents=True)
    manifest_path = target / "acquisition.json"
    manifest_path.write_text(
        json.dumps({"shape": [2, 2, 3], "decoded_bytes": 48, "complete": False})
    )
    codec = numcodecs.Blosc()
    payload = codec.encode(np.arange(6, dtype="float32"))
    metadata = {"dtype": "<f4", "chunks": [1, 2, 3], "compressor": codec.get_config()}
    main = preparation["main"]
    monkeypatch.setitem(main.__globals__, "initialize", lambda *_args: (metadata, 100))
    monkeypatch.setitem(main.__globals__, "fetch", lambda _url: payload)
    monkeypatch.setattr(
        preparation["shutil"], "disk_usage", lambda _path: SimpleNamespace(free=2**40)
    )
    assert main([str(target), "--stop-after", "1"]) == 0, "Expected a successful pilot."
    assert not json.loads(manifest_path.read_text())[
        "complete"
    ], "Expected an incomplete pilot manifest."
    assert not (
        target / "_SUCCESS"
    ).exists(), "Expected no marker before all chunks arrive."
    assert main([str(target)]) == 0, "Expected successful resumed acquisition."
    assert json.loads(manifest_path.read_text())[
        "complete"
    ], "Expected complete acquisition."
    assert (target / "_SUCCESS").exists(), "Expected a completed-store marker."
    assert (
        main([str(target), "--stop-after", "1"]) == 0
    ), "Expected a successful repeated pilot."
    assert json.loads(manifest_path.read_text())[
        "complete"
    ], "Expected completion to be preserved."


def test_invalid_chunk_is_rejected(preparation: dict) -> None:
    """Missing weather values are caught before accepting a source chunk."""
    codec = numcodecs.Blosc()
    values = np.array([1.0, np.nan], dtype="float32")
    metadata = {"dtype": "<f4", "chunks": [1, 1, 2], "compressor": codec.get_config()}
    with pytest.raises(ValueError, match="missing/non-finite"):
        preparation["validate_chunk"](codec.encode(values), metadata)


def test_subset_rejects_incomplete_or_different_source(
    preparation: dict, tmp_path: Path
) -> None:
    """Subset preparation cannot conceal missing chunks or select another field."""
    source, target = tmp_path / "source.zarr", tmp_path / "subset.zarr"
    source.mkdir()
    manifest = source / "acquisition.json"
    manifest.write_text(json.dumps({"complete": False}))
    with pytest.raises(ValueError, match="incomplete"):
        preparation["create_subset"](source, target, "2m_temperature", 2)
    manifest.write_text(json.dumps({"complete": True, "variable": "temperature"}))
    with pytest.raises(ValueError, match="differs"):
        preparation["create_subset"](source, target, "2m_temperature", 2)
    assert not target.exists(), "Expected rejection before writing a subset."
