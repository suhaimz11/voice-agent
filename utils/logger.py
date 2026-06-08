"""
Project logger.

Writes readable logs to the console and to logs/voice_agent.log.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Optional


DEBUG = False

LOG_DIR = Path(__file__).resolve().parents[1] / "logs"
LOG_FILE = LOG_DIR / "voice_agent.log"

_LOGGER_NAME = "voice_agent"
_LOGGER: Optional[logging.Logger] = None


def _get_logger() -> logging.Logger:
    global _LOGGER

    if _LOGGER is not None:
        return _LOGGER

    LOG_DIR.mkdir(exist_ok=True)

    logger = logging.getLogger(_LOGGER_NAME)
    logger.setLevel(logging.DEBUG)
    logger.propagate = False

    if not logger.handlers:
        formatter = logging.Formatter(
            fmt="%(asctime)s [%(levelname)s] %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )

        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.DEBUG if DEBUG else logging.INFO)
        console_handler.setFormatter(formatter)

        file_handler = logging.FileHandler(
            LOG_FILE,
            encoding="utf-8"
        )
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(formatter)

        logger.addHandler(console_handler)
        logger.addHandler(file_handler)

    _LOGGER = logger

    return logger


def log(msg: str, level: str = "info", exc_info: bool = False):
    """
    Log a message to both console and logs/voice_agent.log.
    """

    if level == "debug" and not DEBUG:
        return

    logger = _get_logger()

    log_method = getattr(
        logger,
        level.lower(),
        logger.info
    )

    log_method(
        msg,
        exc_info=exc_info
    )


def monotonic_seconds() -> float:
    """
    Return a monotonic timestamp for elapsed-time measurement.
    """

    return time.perf_counter()


def elapsed_seconds(started_at: float) -> float:
    """
    Return elapsed seconds from a monotonic start timestamp.
    """

    return monotonic_seconds() - started_at


def format_duration(seconds: float) -> str:
    """
    Format elapsed time consistently for human-readable timing logs.
    """

    if seconds < 1:
        return f"{seconds * 1000:.0f}ms"

    return f"{seconds:.2f}s"


def log_timing(
    stage: str,
    started_at: float,
    level: str = "info",
    details: str | None = None,
):
    """
    Log a consistently formatted elapsed-time message for a pipeline stage.
    """

    suffix = f" ({details})" if details else ""

    log(
        f"Timing | {stage}: {format_duration(elapsed_seconds(started_at))}{suffix}",
        level=level,
    )
