"""
Lightweight console logger used across the project.
"""

import datetime


# Enable verbose logs during development
DEBUG = False


def log(msg: str, level: str = "info"):
    """
    Print formatted log messages with timestamps.
    """

    # Skip debug logs unless enabled
    if level == "debug" and not DEBUG:
        return

    prefixes = {
        "info": "ℹ️",
        "debug": "🔍",
        "warning": "⚠️",
        "error": "❌",
    }

    prefix = prefixes.get(level, "•")

    timestamp = datetime.datetime.now().strftime(
        "%H:%M:%S"
    )

    print(
        f"[{timestamp}] "
        f"{prefix} {msg}"
    )