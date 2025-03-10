""""""

from enum import Enum, auto
import os
import logging
from datetime import datetime


class LoggingStreamType(Enum):
    """All supported stream types for logging"""

    CONSOLE = auto()
    FILE = auto()


def get_logger(
    name: str,
    level: int = logging.INFO,
    log_dir: str | None = "logs",
    streams: (
        list[tuple[LoggingStreamType, int]] | tuple[LoggingStreamType, int] | None
    ) = None,
):
    """"""

    logger = logging.getLogger(name)
    logger.setLevel(level)

    if logger.hasHandlers():
        return logger

    if streams is None:
        streams = [
            (LoggingStreamType.CONSOLE, logging.INFO),
            (LoggingStreamType.FILE, logging.INFO),
        ]
    elif not isinstance(streams, list):
        streams = [streams]

    formatter = logging.Formatter(
        "[%(asctime)s] [%(levelname)s] [%(name)s] %(message)s", "%Y-%m-%d %H:%M:%S"
    )

    for stream, log_level in streams:
        handler: logging.Handler
        if stream is LoggingStreamType.CONSOLE:
            handler = logging.StreamHandler()
        elif stream is LoggingStreamType.FILE:
            if log_dir is None:
                raise ValueError("File Handler expect a valid log_dir")
            os.makedirs(log_dir, exist_ok=True)
            log_filename = os.path.join(
                log_dir, f"{datetime.now().strftime('%Y-%m-%d')}.log"
            )
            handler = logging.FileHandler(log_filename, mode="a")
        else:
            raise ValueError("Invalid Logging Stream Type")
        handler.setLevel(log_level)
        handler.setFormatter(formatter)
        logger.addHandler(handler)

    return logger
