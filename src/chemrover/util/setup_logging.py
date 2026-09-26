#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Author: O. Bayley
Description: Logging setup for ChemRover.
             Modules create loggers at import with `logging.getLogger(__name__)`; this is free and
             needs no paths. `setup_logging()` is called once at the start of an experiment and
             attaches the console and (optionally) a run.log file handler to the parent 'chemrover'
             logger, so every module's LOG.xx() calls reach both.
"""
import sys
import logging
from pathlib import Path
from typing import Optional

LOGGER_NAME = "chemrover"            # parent of every `chemrover.*` module logger
LOG_FMT = "%(levelname)s %(asctime)s %(location)s %(message)s"
DATE_FMT = "%H:%M:%S"


class CustomFormatter(logging.Formatter):
    """Pads the level name, and adds the file/function/line location for WARNING and above."""

    def format(self, record):
        # work on a copy: the same record is passed to every handler
        record = logging.makeLogRecord(record.__dict__)
        record.levelname = f"[{record.levelname}]".rjust(10)
        if record.levelno >= logging.WARNING:
            record.location = f"[Md:{record.module}, Fn:{record.funcName}, Line:{record.lineno}]"
        else:
            record.location = f"[Md:{record.module} Fn:{record.funcName}]"
        return super().format(record)


def setup_logging(
        out_path: Optional[Path] = None,
        console_level: int = logging.INFO,
        file_level: int = logging.DEBUG,
) -> None:
    """
    Routes all ChemRover log messages to the console and, if ``run_dir`` is given, to <run_dir>/run.log.
    Call once at the start of an experiment. Calling it again replaces the previous handlers
    (closing the previous log file), so each run's log only contains that run.
    Args:
        out_path: Path | None - folder for run.log (created if missing); None = console only
        console_level: int - lowest level shown on the console
        file_level: int - lowest level written to run.log
    """
    logger = logging.getLogger(LOGGER_NAME)
    logger.setLevel(logging.DEBUG)       # handlers decide what they show
    logger.propagate = False             # don't print a second time via root/Jupyter handlers

    # drop and close any handlers from a previous call
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
        handler.close()

    formatter = CustomFormatter(LOG_FMT, DATE_FMT)

    console = logging.StreamHandler(sys.stdout)
    console.setLevel(console_level)
    console.setFormatter(formatter)
    logger.addHandler(console)

    if out_path is not None:
        out_path = Path(out_path)
        out_path.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(out_path / "run.log", encoding="utf-8")
        file_handler.setLevel(file_level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
