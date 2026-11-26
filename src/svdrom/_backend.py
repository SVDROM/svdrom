from __future__ import annotations

import importlib
from collections.abc import Callable
from types import ModuleType
from typing import Any

import dask
import dask.array as da
import numpy as np
import xarray as xr


def get_xp() -> ModuleType:
    """Return Dask's configured array module, validating optional GPU support."""
    backend = dask.config.get("array.backend", "numpy")
    if backend == "numpy":
        return np
    if backend != "cupy":
        msg = f"Unsupported array backend: {backend!r}. Use 'numpy' or 'cupy'."
        raise ValueError(msg)
    try:
        cupy = importlib.import_module("cupy")
    except ImportError as exc:
        msg = (
            "The CuPy backend requires CuPy. Install it with "
            "'uv sync --extra gpu' on a CUDA 12 machine."
        )
        raise ImportError(msg) from exc
    try:
        available = cupy.is_available()
    except Exception as exc:
        msg = "CuPy cannot access CUDA. Check the NVIDIA driver and CUDA runtime."
        raise RuntimeError(msg) from exc
    if not available:
        msg = "The CuPy backend requires an available NVIDIA GPU and CUDA runtime."
        raise RuntimeError(msg)
    return cupy


def _convert_xarray(
    arr: xr.DataArray | xr.Dataset, convert: Callable[[Any], Any]
) -> xr.DataArray | xr.Dataset:
    """Replace only data buffers, sharing coordinates and indexes."""
    if isinstance(arr, xr.DataArray):
        original = arr.data
        converted = convert(original)
        return arr if converted is original else arr.copy(deep=False, data=converted)

    original_data = {name: var.data for name, var in arr.data_vars.items()}
    converted_data = {name: convert(data) for name, data in original_data.items()}
    if all(converted_data[name] is data for name, data in original_data.items()):
        return arr
    return arr.copy(deep=False, data=converted_data)


def to_device(arr: Any, *, lazy: bool = False) -> Any:
    """Move numerical data to Dask's selected backend without computing chunks.

    Xarray coordinates and attributes stay on the CPU. NumPy inputs are returned
    unchanged under the default backend. Dask transfers are lazy and idempotent.
    With ``lazy=True``, eager host arrays that need uploading are wrapped in Dask
    arrays first. CuPy-incompatible variable dtypes stay on the CPU.
    """
    if isinstance(arr, xr.DataArray | xr.Dataset):
        return _convert_xarray(arr, lambda data: to_device(data, lazy=lazy))
    xp = get_xp()
    if xp is np:
        return to_host(arr)

    dtype = getattr(arr, "dtype", None)
    if dtype is not None and (
        dtype.kind not in "biufc" or dtype.itemsize > (16 if dtype.kind == "c" else 8)
    ):
        return arr
    if isinstance(arr, da.Array):
        return arr.to_backend(xp.__name__)
    if lazy and isinstance(arr, np.ndarray):
        return da.from_array(arr).to_backend(xp.__name__)
    return xp.asarray(arr)


def to_host(arr: Any) -> Any:
    """Return CPU data, transferring Dask output blocks before their final gather.

    This conversion is independent of the current backend setting. It preserves
    lazy collections and Xarray metadata and does not import CuPy for NumPy data.
    """
    if isinstance(arr, xr.DataArray | xr.Dataset):
        return _convert_xarray(arr, to_host)
    if isinstance(arr, da.Array):
        return arr.to_backend("numpy")
    if isinstance(arr, np.ndarray):
        return arr
    if hasattr(arr, "__cuda_array_interface__"):
        return importlib.import_module("cupy").asnumpy(arr)
    return np.asarray(arr)
