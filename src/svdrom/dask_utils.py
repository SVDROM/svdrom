from __future__ import annotations

from dask.distributed import Client

from svdrom.logger import setup_logger

logger = setup_logger("Dask Setup", "dask_setup.log")


def init_dask(
    dashboard: bool = False, processes: bool = False, address: str | None = None
) -> Client | None:
    """
    Initialize and configure a Dask distributed client.

    Parameters
    ----------
    dashboard : bool, optional
        If True, starts a local Dask cluster with dashboard support. Default is False.
    processes : bool, optional
        If True, uses multiple processes for the local cluster (when dashboard is True).
        If False, uses threads. Default is False.
    address : str or None, optional
        If provided, connects to an external Dask cluster at the given address.
        If None, starts a local cluster or uses the default threaded scheduler.

    Returns
    -------
    Client or None
        Returns a Dask distributed Client if a cluster is started or connected to.
        Returns None if using the default threaded scheduler (no Dask Client).

    Notes
    -----
    - If `address` is provided, connects to the specified Dask cluster.
    - If `dashboard` is True, starts a local cluster with dashboard support.
    - If neither `address` nor `dashboard` is set, uses Dask's default
      threaded scheduler.
    """
    if address:
        client = Client(address)
        logger.info("Connected to external Dask cluster at %s.", address)
    elif dashboard:
        client = Client(processes=processes)
        if processes:
            logger.info(
                "Started a local multi-process Dask cluster with dashboard support."
            )
        else:
            logger.info(
                "Started a local multi-thread Dask cluster with dashboard support."
            )
        logger.info("Dashboard link: http://localhost:8787/status")
    else:
        logger.info("Using default threaded scheduler (no Dask Client).")
        return None

    return client
