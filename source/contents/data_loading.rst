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

It is generally most efficient to preserve the chunking of the data as stored on disk, or to pick multiples of the on-disk chunk sizes. For example, if an ERA5 Zarr store is chunked along the time dimension with chunks of 10, using ``chunks={"time": 100}`` reads well from it.

Note that this is separate from the chunking that the decompositions themselves prefer: see the ``rechunk`` flag of :py:class:`svd.TruncatedSVD` in :doc:`svd`, where a single chunk along the smallest dimension improves the accuracy of the randomised SVD at the cost of a rechunking step.

See the `Xarray reading and writing files guide <https://docs.xarray.dev/en/stable/user-guide/io.html>`__ for the complete list of arguments, including ``engine``, ``decode_times``, ``mask_and_scale`` and ``drop_variables``.

.. note::

    Earlier versions of SVD-ROM provided ``svdrom.io.open_dataset`` and ``svdrom.io.open_dataarray``.
    These wrappers have been removed; use ``xr.open_dataset`` and ``xr.open_dataarray`` directly instead.
