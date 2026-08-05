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

Setting up Dask
---------------

SVD-ROM does not create or manage a Dask cluster for you. For anything beyond
Dask's default threaded scheduler, create a `dask.distributed
<https://distributed.dask.org/>`__ ``Client`` yourself:

.. code-block:: python

    from dask.distributed import Client

    # single process, multiple threads (recommended on one machine)
    client = Client(processes=False)

    # multi-process cluster
    client = Client(processes=True, n_workers=4, threads_per_worker=2)

    # connect to an existing cluster
    client = Client("tcp://192.168.1.100:8786")

The dashboard is available at http://localhost:8787/status by default.

Because SVD-ROM operates on tall-and-skinny Dask arrays with linear algebra
routines that communicate heavily between chunks, a single-process,
multi-threaded ``LocalCluster`` (``processes=False``) is usually the best
choice on a single machine, as it avoids serialisation overhead.

See the `Dask distributed documentation <https://distributed.dask.org/en/stable/client.html>`__
for the full set of ``Client`` and ``LocalCluster`` options.

.. note::

    Earlier versions of SVD-ROM provided ``svdrom.dask_utils.init_dask``.
    This wrapper has been removed; create a ``dask.distributed.Client`` directly instead.
