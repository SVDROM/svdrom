from __future__ import annotations

import importlib
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


def to_device(arr: Any) -> Any:
    """Move numerical data to Dask's selected backend without computing chunks.

    Xarray coordinates and attributes stay on the CPU. NumPy inputs are returned
    unchanged under the default backend; Dask transfers are lazy and idempotent.
    """
    if isinstance(arr, xr.Dataset):
        return arr.map(to_device, keep_attrs=True)
    if isinstance(arr, xr.DataArray):
        return arr.copy(data=to_device(arr.data))
    xp = get_xp()
    if isinstance(arr, da.Array):
        return arr.to_backend(xp.__name__)
    return to_host(arr) if xp is np else xp.asarray(arr)


def to_host(arr: Any) -> Any:
    """Return CPU data, transferring Dask output blocks before their final gather.

    This conversion is independent of the current backend setting. It preserves
    lazy collections and Xarray metadata and does not import CuPy for NumPy data.
    """
    if isinstance(arr, xr.Dataset):
        return arr.map(to_host, keep_attrs=True)
    if isinstance(arr, xr.DataArray):
        return arr.copy(data=to_host(arr.data))
    if isinstance(arr, da.Array):
        return arr.to_backend("numpy")
    if isinstance(arr, np.ndarray):
        return arr
    if hasattr(arr, "__cuda_array_interface__"):
        return importlib.import_module("cupy").asnumpy(arr)
    return np.asarray(arr)
