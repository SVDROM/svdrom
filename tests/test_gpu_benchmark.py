"""Exercise the standalone benchmark against a small offline weather fixture."""

from __future__ import annotations

import json
import runpy
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import dask.array as da
import numpy as np
import pytest
import xarray as xr


@pytest.mark.parametrize("backend", ["numpy", "cupy"])
@pytest.mark.parametrize("model", ["dask-svd", "svdrom-svd", "svdrom-pod"])
def test_benchmark_weather_roundtrip(tmp_path: Path, backend: str, model: str) -> None:
    """The CLI times every case and writes reproducible machine-readable data."""
    if backend == "cupy":
        cp = pytest.importorskip("cupy")
        try:
            available = cp.is_available()
        except Exception as exc:
            pytest.skip(f"CUDA is unavailable: {exc}")
        if not available:
            pytest.skip("CuPy is installed but no CUDA device is available.")
    rng = np.random.default_rng(42)
    field = xr.DataArray(
        rng.normal(size=(24, 4, 4)),
        dims=("time", "latitude", "longitude"),
        name="air",
    )
    dataset = tmp_path / "weather.nc"
    field.to_netcdf(dataset, engine="scipy")
    output = tmp_path / "results.json"
    script = Path(__file__).resolve().parents[1] / "benchmarks" / "gpu_svd.py"
    result = subprocess.run(
        [
            sys.executable,
            str(script),
            str(dataset),
            "--backend",
            backend,
            "--model",
            model,
            "--engine",
            "scipy",
            "--rank",
            "3",
            "--warmups",
            "0",
            "--repeats",
            "1",
            "--output",
            str(output),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, f"Expected successful CPU benchmark: {result.stderr}"
    report = json.loads(output.read_text())
    assert report["environment"]["shape"] == [
        16,
        24,
    ], "Expected spatial stacking with time along columns."
    assert (
        len(report["results"]) == 12
    ), "Expected both dtypes, residencies and all three SVD settings."
    for row in report["results"]:
        assert row["status"] == "ok", f"Expected a successful benchmark case: {row}"
        assert row["seconds_median"] > 0, "Expected measured computation time."
        assert row["model"] == model, "Expected the requested numerical workflow."
        assert row["output_backend"] == "numpy", "Expected CPU-backed final factors."
        assert row["retained_factor_bytes"] > 0, "Expected retained-factor accounting."
        if model != "dask-svd":
            assert (
                row["download_seconds_median"] is None
            ), "Expected unprofiled transfers to be explicit."
        elif backend == "numpy":
            assert row["upload_seconds_median"] == 0, "Expected no upload on CPU."
            assert row["download_seconds_median"] == 0, "Expected no download on CPU."
        else:
            assert row["peak_gpu_bytes_sampled"] > 0, "Expected sampled device memory."
        assert (
            len(row["singular_values"]) == 3
        ), "Expected only the requested retained factors."
    assert (
        f"| {backend} |" in result.stdout
    ), "Expected a Markdown table suitable for the issue."


def test_device_persistence_budget_is_enforced(monkeypatch: pytest.MonkeyPatch) -> None:
    """Oversized GPU persistence is skipped before device allocation or compute."""
    script = Path(__file__).resolve().parents[1] / "benchmarks" / "gpu_svd.py"
    namespace = runpy.run_path(str(script))
    matrix = da.from_array(np.ones((32, 64)), chunks=(8, 16))
    fake_cupy = SimpleNamespace(
        is_available=lambda: True,
        get_default_memory_pool=lambda: SimpleNamespace(free_all_blocks=lambda: None),
        cuda=SimpleNamespace(runtime=SimpleNamespace(memGetInfo=lambda: (1024, 2048))),
    )
    monkeypatch.setattr(
        namespace["importlib"], "import_module", lambda _name: fake_cupy
    )
    result = namespace["run_case"](
        matrix,
        "cupy",
        "float64",
        "persisted",
        "tsqr",
        0,
        SimpleNamespace(device_memory_fraction=0.5, model="dask-svd"),
    )
    assert (
        result["status"] == "skipped"
    ), "Expected oversized persistence to be skipped."
    assert "budget" in result["reason"], "Expected an explicit residency-budget reason."


def test_host_persistence_budget_is_enforced(monkeypatch: pytest.MonkeyPatch) -> None:
    """Large inputs must not be eagerly persisted into limited WSL system RAM."""
    script = Path(__file__).resolve().parents[1] / "benchmarks/gpu_svd.py"
    namespace = runpy.run_path(str(script))
    matrix = da.from_array(np.ones((32, 64)), chunks=(8, 16))
    monkeypatch.setattr(
        namespace["psutil"], "virtual_memory", lambda: SimpleNamespace(available=1024)
    )
    result = namespace["run_case"](
        matrix,
        "numpy",
        "float32",
        "persisted",
        "randomized",
        0,
        SimpleNamespace(model="dask-svd", host_memory_fraction=0.5),
    )
    assert result["status"] == "skipped", "Expected host persistence to be skipped."
    assert "host" in result["reason"], "Expected the host-budget reason."


def test_host_memory_guard_aborts_and_unregisters() -> None:
    """The monitor enforces a budget and removes its Dask callback on shutdown."""
    script = Path(__file__).resolve().parents[1] / "benchmarks/gpu_svd.py"
    namespace = runpy.run_path(str(script))
    monitor = namespace["DeviceMemory"](None, host_limit=1)
    callbacks = namespace["Callback"].active.copy()
    monitor.start()
    try:
        with pytest.raises(MemoryError, match="host-memory budget"):
            monitor.check_budget()
    finally:
        monitor.stop()
    assert namespace["Callback"].active == callbacks, "Expected callback cleanup."
    assert monitor.peak_host_rss > 0, "Expected host RSS measurements."


@pytest.mark.parametrize("sync_before", [False, True])
def test_transfer_clock_synchronization(sync_before: bool) -> None:
    """Pure-copy clocks exclude queued work; the historical clock stays reproducible."""
    script = Path(__file__).resolve().parents[1] / "benchmarks/gpu_svd.py"
    namespace = runpy.run_path(str(script))
    events = []

    def copy(values: np.ndarray) -> np.ndarray:
        events.append("copy")
        return values

    xp = SimpleNamespace(
        asarray=copy,
        cuda=SimpleNamespace(
            runtime=SimpleNamespace(deviceSynchronize=lambda: events.append("sync"))
        ),
    )
    transfers = namespace["Transfers"](xp, sync_before=sync_before)
    values = np.arange(4)
    result = transfers.upload(values)
    expected = ["sync", "copy", "sync"] if sync_before else ["copy", "sync"]
    assert events == expected, "Expected the requested copy-clock boundaries."
    assert result is values, "Expected the backend's copied output unchanged."
    assert transfers.upload_seconds >= 0, "Expected elapsed copy-window accounting."


def test_incomplete_acquisition_is_rejected(tmp_path: Path) -> None:
    """Missing source chunks cannot be mistaken for a valid benchmark dataset."""
    script = Path(__file__).resolve().parents[1] / "benchmarks/gpu_svd.py"
    namespace = runpy.run_path(str(script))
    (tmp_path / "acquisition.json").write_text(json.dumps({"complete": False}))
    with pytest.raises(ValueError, match="acquisition is incomplete"):
        namespace["load_matrix"](SimpleNamespace(path=str(tmp_path)))


@pytest.mark.parametrize("model", ["dask-svd", "svdrom-svd", "svdrom-pod"])
@pytest.mark.parametrize("iterator", ["power", "QR"])
def test_staged_execution_and_validation(
    tmp_path: Path, model: str, iterator: str
) -> None:
    """Staged SVD/POD preserve low-rank reconstruction and report bounded memory."""
    script = Path(__file__).resolve().parents[1] / "benchmarks/gpu_svd.py"
    namespace = runpy.run_path(str(script))
    rng = np.random.default_rng(42)
    matrix = da.from_array(
        rng.normal(size=(32, 2)) @ rng.normal(size=(2, 24)), chunks=(8, 6)
    )
    host_limit = namespace["psutil"].Process().memory_info().rss + 512 * 1024**2
    args = namespace["parse_args"](
        [
            str(tmp_path),
            "--rank",
            "2",
            "--model",
            model,
            "--staged",
            "--iterator",
            iterator,
            "--warmups",
            "0",
            "--repeats",
            "1",
            "--validation-snapshots",
            "3",
            "--host-memory-limit",
            str(host_limit),
        ]
    )
    with namespace["dask"].config.set(
        {"array.backend": "numpy", "scheduler": "synchronous"}
    ):
        row = namespace["run_case"](
            matrix, "numpy", "float64", "disk", "randomized", 2, args
        )
    assert row["staged"], "Expected staged execution to be recorded."
    assert row["iterator"] == iterator, "Expected the requested iteration method."
    assert row["status"] == "ok", f"Expected successful bounded execution: {row}"
    assert row["validation"]["rmse"] < 1e-10, "Expected exact low-rank recovery."
    assert (
        row["peak_host_rss_bytes_sampled"] < host_limit
    ), "Expected bounded RSS growth."
