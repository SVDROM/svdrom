from __future__ import annotations

import xarray as xr

from svdrom.logger import setup_logger

logger = setup_logger("I/O", "io.log")


def open_dataset(filename: str, chunks: str | dict | int = "auto"):
    """
    Opens a compatible file as a Xarray.DataSet (can contain multiple variables).

    Parameters
    ----------
    filename : str
        Path to the file to be opened.
    chunks : str, dict, or int, optional
        Chunking strategy for dask-backed arrays.
        Can be one of:
            - "auto" (default): will use Dask auto chunking.
            - a dictionary mapping dimension names to chunk sizes,
            e.g. `chunks={"time": 10}`.
            - `-1`: loads the data with Dask using a single chunk.

    Returns
    -------
    xr.Dataset
        The opened Xarray DataSet.
    """
    try:
        logger.info("Opening Xarray.DataSet from %s.", filename)
        return xr.open_dataset(filename, chunks=chunks)
    except Exception as e:
        msg = f"Error opening {filename} as Xarray.DataSet."
        logger.exception(msg)
        raise RuntimeError(msg) from e


def open_dataarray(filename: str, chunks: str | dict | int = "auto"):
    """
    Opens a compatible file as a Xarray.DataArray (containing a single variable).

    Parameters
    ----------
    filename : str
        Path to the file to be opened.
    chunks : str, dict, or int, optional
        Chunking strategy for dask-backed arrays.
        Can be one of:
            - "auto" (default): will use Dask auto chunking.
            - a dictionary mapping dimension names to chunk sizes,
            e.g. `chunks={"time": 10}`.
            - `-1`: loads the data with Dask using a single chunk.

    Returns
    -------
    xr.DataArray
        The opened Xarray DataArray.
    """
    try:
        logger.info("Opening Xarray.DataArray from %s.", filename)
        return xr.open_dataarray(filename, chunks=chunks)
    except Exception as e:
        msg = f"Error opening {filename} as Xarray.DataArray."
        logger.exception(msg)
        raise RuntimeError(msg) from e
