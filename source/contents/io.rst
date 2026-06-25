I/O Module
==========

The I/O module provides convenience functions for opening datasets and data arrays from file formats compatible with Xarray (e.g. NetCDF, Zarr).

.. py:module:: io

.. py:function:: open_dataset(filename, chunks='auto')

    Open a compatible file as an ``xarray.Dataset`` (can contain multiple variables).

    :param filename: Path to the file to be opened.
    :type filename: str
    :param chunks: Chunking strategy for Dask-backed arrays. Can be one of:

        - ``"auto"`` (default): use Dask auto chunking.
        - A dictionary mapping dimension names to chunk sizes, e.g. ``{"time": 10}``.
        - ``-1``: load the data with Dask using a single chunk.
    :type chunks: str, dict, or int
    :returns: The opened Xarray Dataset.
    :rtype: xarray.Dataset
    :raises RuntimeError: If the file cannot be opened.

    .. rubric:: Examples

    .. code-block:: python

        from svdrom.io import open_dataset

        ds = open_dataset("era5.zarr")
        ds = open_dataset("era5.zarr", chunks={"time": 100})

.. py:function:: open_dataarray(filename, chunks='auto')

    Open a compatible file as an ``xarray.DataArray`` (containing a single variable).

    :param filename: Path to the file to be opened.
    :type filename: str
    :param chunks: Chunking strategy for Dask-backed arrays. Can be one of:

        - ``"auto"`` (default): use Dask auto chunking.
        - A dictionary mapping dimension names to chunk sizes, e.g. ``{"time": 10}``.
        - ``-1``: load the data with Dask using a single chunk.
    :type chunks: str, dict, or int
    :returns: The opened Xarray DataArray.
    :rtype: xarray.DataArray
    :raises RuntimeError: If the file cannot be opened.

    .. rubric:: Examples

    .. code-block:: python

        from svdrom.io import open_dataarray

        da = open_dataarray("temperature.zarr")
        da = open_dataarray("temperature.zarr", chunks=-1)
