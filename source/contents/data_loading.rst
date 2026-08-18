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

Loading Multiple Files
-----------------------

To combine many files (e.g. one file per timestep) into a single lazy, Dask-backed dataset, use ``xr.open_mfdataset``:

.. code-block:: python

    import xarray as xr

    ds = xr.open_mfdataset(
        files,
        combine="nested",
        concat_dim="time",
        parallel=True,
        engine="h5netcdf",
        data_vars="minimal",
        coords="minimal",
        compat="override",
        chunks={"time": time_chunk},
    )

- ``combine="nested"`` with ``concat_dim="time"``: concatenates the files in the order given along the ``time`` dimension, rather than inferring the arrangement from coordinate values (as ``combine="by_coords"`` would).
- ``parallel=True``: opens the files concurrently using Dask.
- ``engine``: using ``"h5netcdf"`` backend.
- ``data_vars="minimal"`` and ``coords="minimal"``: only concatenate variables/coordinates that already vary along ``concat_dim``, leaving the rest to be taken from the first file. This avoids unnecessarily broadcasting static fields across the new dimension.
- ``compat="override"``: skips comparing non-concatenated variables/coordinates across files for equality, taking them from the first file. Combined with ``data_vars`` and ``coords`` set to ``"minimal"``, this keeps opening many files fast.
- ``chunks``: sets the Dask chunk size along the newly created ``time`` dimension. Note that ``open_mfdataset`` applies ``chunks`` *per file, before concatenation* — if each file covers only one timestep, ``time`` chunks are always size 1 regardless of the value given here. To get coarser chunks along ``time``, rechunk afterwards with ``ds.chunk({"time": desired_chunk_size})``.

