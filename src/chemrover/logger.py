#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Author: O. Bayley
Description: function to get a Logger object that can stream to files and the console
"""
import os
import logging
import sys
import time
from pathlib import Path
from typing import Optional

LOG_FMT="%(levelname)s %(asctime)s %(location)s %(message)s"
DATE_FMT="%H:%M:%S"

class CustomFormatter(logging.Formatter):
    def __init__(self, fmt=None, datefmt=None):
        super().__init__(fmt, datefmt)
        # Define the maximum length of log level names
        self.level_lengths = {
            "INFO": 4,
            "WARNING": 7,
            "ERROR": 5,
            "DEBUG": 5,
            "CRITICAL": 8,
        }

    def format(self, record):
        # Calculate the padding required for the log level
        levelname = record.levelname
        max_length = max(self.level_lengths.values())
        padding = max_length - self.level_lengths.get(levelname, 0)

        # Add the padding to the log level
        record.levelname = " " * padding + f"[{levelname}]"

        # Shorten the file name to just the models name (without extension)
        record.filename = os.path.splitext(os.path.basename(record.filename))[0]

        # Only include location for DEBUG and ERROR logs
        if record.levelno in (logging.WARN, logging.WARNING, logging.ERROR):
            record.location = f"[File:{record.filename}, Function:{record.funcName}, Line:{record.lineno}]"
        else:
            record.location = f"[{record.funcName}]"

        # Call the parent class's format method
        return super().format(record)


def get_logger(
        name: str,
        log_path: Optional[Path] = None,
        lowest_level: int = logging.DEBUG,
) -> logging.Logger:
    """Function to set up Logger objects sharing the same configuration.

    Args:
        name(str): Name of the Logger object
        log_path(Path): Path to the og files
        lowest_level(int): lowest logging level to be registered in the log (Default: logging.DEBUG)
    Returns:
        logging.Logger: Logger object for the corresponding file
    """
    # Create the logger
    logger:  logging.Logger = logging.Logger(name, level=lowest_level)

    # Remove existing handlers for the logger to avoid duplicating logs
    if logger.hasHandlers():
        logger.handlers.clear()

    # If no path is provided, use the default project directory
    log_path = log_path if log_path else Path(__file__).parent.parent.parent / "logs"

    # Build Path to the log file
    time_hours = time.strftime("%H-%M-%S")
    date = time.strftime("%Y_%m_%d")
    file_path = log_path / name / date / f"T_{time_hours}_{name}.log"
    file_path.parent.mkdir(parents=True, exist_ok=True)

    # Get formatter
    formatter = CustomFormatter(fmt=LOG_FMT, datefmt=DATE_FMT)

    # Set up the file logging handler with the desired formatting
    file_handler = logging.FileHandler(file_path, encoding="utf-8")
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    # Set up the streaming handler with the desired formatting
    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setLevel(logging.INFO)
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    return logger
