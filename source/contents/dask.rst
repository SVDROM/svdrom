Setting up Dask
===============

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
