Loading Data with Xarray
========================

SVD-ROM works with `Xarray <https://docs.xarray.dev/>`__ objects backed by `Dask <https://www.dask.org/>`__ arrays.
Data is loaded with Xarray's own I/O functions, which support all the file formats used across SVD-ROM (Zarr, NetCDF4, HDF5) and expose the full set of backend options.

Pass ``chunks`` to obtain a Dask-backed (lazy, out-of-core) array:

.. code-block:: python

    import xarray as xr

    # a dataset with multiple variables
    ds = xr.open_dataset("era5.zarr", chunks="auto")
    ds = xr.open_dataset("era5.zarr", chunks={"time": 100})

    # a single variable
    da = xr.open_dataarray("temperature.zarr", chunks="auto")
    da = xr.open_dataarray("temperature.zarr", chunks=-1)

The ``chunks`` argument can be:

- ``"auto"``: let Dask pick the chunk sizes.
- A dictionary mapping dimension names to chunk sizes, e.g. ``{"time": 10}``.
- ``-1``: load the data with Dask using a single chunk.

Because SVD-ROM operates on tall-and-skinny matrices, chunking along the temporal dimension only (e.g. ``chunks={"time": 100}``) usually gives the best performance.

See the `Xarray reading and writing files guide <https://docs.xarray.dev/en/stable/user-guide/io.html>`__ for the complete list of arguments, including ``engine``, ``decode_times``, ``mask_and_scale`` and ``drop_variables``.

.. note::

    Earlier versions of SVD-ROM provided ``svdrom.io.open_dataset`` and ``svdrom.io.open_dataarray``.
    These wrappers have been removed; use ``xr.open_dataset`` and ``xr.open_dataarray`` directly instead.
