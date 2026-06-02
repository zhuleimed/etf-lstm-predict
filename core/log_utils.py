"""
Simple logging utilities for the ETF prediction project.
"""

import logging
import sys

_RESET = "\033[0m"
_LEVEL_COLORS = {
    logging.DEBUG: "\033[36m",
    logging.INFO: "\033[32m",
    logging.WARNING: "\033[33m",
    logging.ERROR: "\033[31m",
    logging.CRITICAL: "\033[41m",
}


class _ColoredFormatter(logging.Formatter):
    def __init__(self):
        super().__init__("[%(asctime)s] %(message)s", datefmt="%H:%M:%S")

    def format(self, record: logging.LogRecord) -> str:
        asctime = self.formatTime(record, "%H:%M:%S")
        color = _LEVEL_COLORS.get(record.levelno, _RESET)
        prefix = f"{color}[{asctime}]{_RESET}"
        return f"{prefix} {record.getMessage()}"


_loggers: dict[str, logging.Logger] = {}


def get_logger(name: str, level: int = logging.INFO) -> logging.Logger:
    if name in _loggers:
        return _loggers[name]
    logger = logging.getLogger(name)
    logger.setLevel(level)
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(_ColoredFormatter())
        logger.addHandler(handler)
    logger.propagate = False
    _loggers[name] = logger
    return logger


def console_out(*args, **kwargs) -> None:
    print(*args, **kwargs)
