from __future__ import annotations

import logging
import os


def setup_logger(
    name: str, log_file: str, log_path: str = "logs", level=logging.INFO
) -> logging.Logger:
    """
    Sets up and returns a logger with both file and console handlers.

    This function creates a logger with the specified name, logging level, and log file.
    Log messages are formatted with timestamp, logger name, log level, and message.
    Logs are written both to a file (in the specified directory) and to the console.

    Args:
        name (str): Name of the logger.
        log_file (str): Name of the log file to write logs to.
        log_path (str, optional): Directory path where the log file will be stored.
            Defaults to "logs".
        level (int, optional): Logging level (e.g., logging.INFO, logging.DEBUG).
            Defaults to logging.INFO.

    Returns:
        logging.Logger: Configured logger instance.
    """

    # formatter for the log messages
    formatter = logging.Formatter(
        "{asctime} - {name} - {levelname} - {message}",
        style="{",
        datefmt="%Y-%m-%d %H:%M",
    )

    # file handler for logging to a file
    os.makedirs(log_path, exist_ok=True)
    log_file = os.path.join(log_path, log_file)
    file_handler = logging.FileHandler(log_file, mode="a", encoding="utf-8")
    file_handler.setFormatter(formatter)

    # console handler for printing to the console
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)

    # return a logger for the specified logger name and set level
    logger = logging.getLogger(name)
    logger.setLevel(level)

    # if there are no existing handlers, add file and console handlers
    if not logger.handlers:
        logger.addHandler(file_handler)
        logger.addHandler(console_handler)

    return logger
