Utilities
=========

SVD-ROM provides utility modules for Dask cluster management and logging.

Dask Utilities
--------------

.. py:module:: dask_utils

.. py:function:: init_dask(dashboard=False, processes=False, address=None)

    Initialize and configure a Dask distributed client.

    :param dashboard: If ``True``, starts a local Dask cluster with dashboard support. Default is ``False``.
    :type dashboard: bool
    :param processes: If ``True``, uses multiple processes for the local cluster (when *dashboard* is ``True``). If ``False``, uses threads. Default is ``False``.
    :type processes: bool
    :param address: If provided, connects to an external Dask cluster at the given address. If ``None``, starts a local cluster or uses the default threaded scheduler.
    :type address: str or None
    :returns: A Dask distributed ``Client`` if a cluster is started or connected to, or ``None`` if using the default threaded scheduler.
    :rtype: dask.distributed.Client or None

    .. rubric:: Notes

    - If *address* is provided, connects to the specified Dask cluster.
    - If *dashboard* is ``True``, starts a local cluster with a dashboard at ``http://localhost:8787/status``.
    - If neither *address* nor *dashboard* is set, uses Dask's default threaded scheduler (no ``Client`` is created).

    .. rubric:: Examples

    .. code-block:: python

        from svdrom.dask_utils import init_dask

        # Use default threaded scheduler
        client = init_dask()

        # Start a local multi-threading cluster with dashboard
        client = init_dask(dashboard=True)

        # Connect to an existing cluster
        client = init_dask(address="tcp://192.168.1.100:8786")

Logger
------

.. py:module:: logger

.. py:function:: setup_logger(name, log_file, log_path='logs', level=logging.INFO)

    Set up and return a logger with both file and console handlers.

    This function creates a logger with the specified name, logging level, and log file. Log messages are formatted with timestamp, logger name, log level, and message. Logs are written both to a file (in the specified directory) and to the console.

    :param name: Name of the logger.
    :type name: str
    :param log_file: Name of the log file to write logs to.
    :type log_file: str
    :param log_path: Directory path where the log file will be stored. Default is ``"logs"``.
    :type log_path: str
    :param level: Logging level (e.g. ``logging.INFO``, ``logging.DEBUG``). Default is ``logging.INFO``.
    :type level: int
    :returns: Configured logger instance.
    :rtype: logging.Logger

    .. rubric:: Examples

    .. code-block:: python

        from svdrom.logger import setup_logger

        logger = setup_logger("MyModule", "my_module.log")
        logger.info("This is a log message.")
