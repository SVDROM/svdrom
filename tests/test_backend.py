"""CPU-only coverage of optional backend selection and lazy conversion."""

from __future__ import annotations

from types import ModuleType

import dask
import dask.array as da
import numpy as np
import pytest
import xarray as xr
from dask.callbacks import Callback

from svdrom import _backend


def test_numpy_transfers_are_lazy_noops() -> None:
    """Default conversions preserve array identity without executing tasks."""
    values = np.arange(24).reshape(4, 6)
    lazy = da.from_array(values, chunks=(2, 3))
    tasks = []
    with (
        dask.config.set({"array.backend": "numpy"}),
        Callback(posttask=lambda *args: tasks.append(args[0])),
    ):
        assert _backend.get_xp() is np, "Expected NumPy by default."
        assert _backend.to_device(values) is values, "Expected no eager CPU copy."
        assert _backend.to_host(values) is values, "Expected no host copy."
        assert _backend.to_device(lazy) is lazy, "Expected no lazy CPU conversion."
        assert _backend.to_host(lazy) is lazy, "Expected no lazy host conversion."
    assert not tasks, "Expected conversion to avoid executing Dask tasks."


@pytest.mark.parametrize("as_dataset", [False, True])
def test_xarray_transfers_preserve_metadata(as_dataset: bool) -> None:
    """Both Xarray containers retain coordinates, attributes and laziness."""
    original = xr.DataArray(
        da.arange(12, chunks=4).reshape((3, 4)),
        dims=("space", "time"),
        coords={
            "space": ["a", "b", "c"],
            "time": np.arange(4),
            "latitude": ("space", np.arange(3, dtype=float)),
        },
        name="temperature",
        attrs={"units": "K"},
    )
    if as_dataset:
        original = original.to_dataset()
        original.attrs["source"] = "test"
    with dask.config.set({"array.backend": "numpy"}):
        device = _backend.to_device(original)
        result = _backend.to_host(original)
    assert device is original, "Expected NumPy backend to reuse the Xarray object."
    assert result is original, "Expected host transfer to reuse the Xarray object."
    assert result.chunks, "Expected the result to stay Dask-backed."
    xr.testing.assert_identical(result, original)


def test_host_conversion_does_not_depend_on_selected_backend() -> None:
    """Host conversion works even when CuPy is selected but not installed."""
    values = np.arange(4)
    lazy = da.from_array(values, chunks=2)
    with dask.config.set({"array.backend": "cupy"}):
        assert _backend.to_host(values) is values, "Expected NumPy input unchanged."
        assert _backend.to_host(lazy) is lazy, "Expected CPU chunks unchanged."
        np.testing.assert_array_equal(
            _backend.to_host([1, 2]), [1, 2], err_msg="Expected host array conversion."
        )


def test_missing_cupy_has_installation_guidance(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Selecting unavailable optional dependencies gives an actionable error."""

    def missing(_name: str) -> ModuleType:
        msg = "No module named cupy"
        raise ModuleNotFoundError(msg)

    monkeypatch.setattr(_backend.importlib, "import_module", missing)
    with (
        dask.config.set({"array.backend": "cupy"}),
        pytest.raises(ImportError, match="uv sync --extra gpu"),
    ):
        _backend.get_xp()


@pytest.mark.parametrize("driver_error", [False, True])
def test_unavailable_gpu_has_runtime_guidance(
    monkeypatch: pytest.MonkeyPatch, driver_error: bool
) -> None:
    """Installed CuPy without a device or working driver reports a clear error."""
    cupy = ModuleType("cupy")

    def available() -> bool:
        if driver_error:
            msg = "Driver unavailable"
            raise RuntimeError(msg)
        return False

    cupy.is_available = available
    monkeypatch.setattr(_backend.importlib, "import_module", lambda _name: cupy)
    with (
        dask.config.set({"array.backend": "cupy"}),
        pytest.raises(RuntimeError, match="CUDA"),
    ):
        _backend.get_xp()


def test_unsupported_dask_backend() -> None:
    """An unsupported external backend is rejected rather than silently ignored."""
    with (
        dask.config.set({"array.backend": "unknown"}),
        pytest.raises(ValueError, match="Unsupported array backend"),
    ):
        _backend.get_xp()
