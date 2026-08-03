# coding: utf-8
"""Console and file logging for CTGTRec experiments."""

from __future__ import annotations

import logging
from pathlib import Path

from utils.utils import get_local_time


def _log_level(state):
    if state is None:
        return logging.INFO
    return {
        "debug": logging.DEBUG,
        "info": logging.INFO,
        "warning": logging.WARNING,
        "error": logging.ERROR,
        "critical": logging.CRITICAL,
    }.get(str(state).lower(), logging.INFO)


def init_logger(config):
    """Initialize one root logger anchored to the configured project log path."""
    log_root = Path(str(config["log_dir"]))
    log_root.mkdir(parents=True, exist_ok=True)

    log_filename = "{}-{}-{}.log".format(
        config["model"],
        config["dataset"],
        get_local_time(),
    )
    log_path = log_root / log_filename
    config["log_file"] = str(log_path)

    level = _log_level(config["state"])

    file_handler = logging.FileHandler(
        log_path,
        mode="w",
        encoding="utf-8",
    )
    file_handler.setLevel(level)
    file_handler.setFormatter(
        logging.Formatter(
            "%(asctime)-15s %(levelname)s %(message)s",
            "%a %d %b %Y %H:%M:%S",
        )
    )

    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(level)
    stream_handler.setFormatter(
        logging.Formatter(
            "%(asctime)-15s %(levelname)s %(message)s",
            "%d %b %H:%M",
        )
    )

    # ``force=True`` prevents duplicate handlers in repeated in-process runs.
    logging.basicConfig(
        level=level,
        handlers=[stream_handler, file_handler],
        force=True,
    )
