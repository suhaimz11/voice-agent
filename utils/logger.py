"""
utils/logger.py
Simple console logger with levels.
Set DEBUG=True for verbose output during development.
"""

import datetime

DEBUG = False   # set True to see all debug logs


def log(msg: str, level: str = "info"):
    if level == "debug" and not DEBUG:
        return
    prefix = {
        "info":    "  ℹ️ ",
        "debug":   "  🔍",
        "warning": "  ⚠️ ",
        "error":   "  ❌",
    }.get(level, "  ")
    timestamp = datetime.datetime.now().strftime("%H:%M:%S")
    print(f"[{timestamp}] {prefix} {msg}")
