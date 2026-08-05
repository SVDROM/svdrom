Utilities
=========

SVD-ROM provides a utility module for logging.

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
