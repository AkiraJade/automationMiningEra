"""Thread-safe logging system with Qt Signal broadcasting for Graal Mining Macro."""

import logging
import sys
from typing import Callable, Optional
from PySide6.QtCore import QObject, Signal


class QtLogSignalEmitter(QObject):
    log_emitted = Signal(str, str, str)  # timestamp, level, message


class QtLogHandler(logging.Handler):
    """Custom logging handler that emits Qt signals for GUI display."""

    def __init__(self, emitter: QtLogSignalEmitter):
        super().__init__()
        self.emitter = emitter

    def emit(self, record: logging.LogRecord) -> None:
        try:
            msg = self.format(record)
            level = record.levelname
            timestamp = self.formatter.formatTime(record, "%H:%M:%S") if self.formatter else ""
            self.emitter.log_emitted.emit(timestamp, level, msg)
        except Exception:
            self.handleError(record)


_qt_emitter = QtLogSignalEmitter()
_logger_initialized = False


def get_log_emitter() -> QtLogSignalEmitter:
    return _qt_emitter


def setup_logger(name: str = "GraalMiningMacro", log_file: Optional[str] = "app.log", level: int = logging.INFO) -> logging.Logger:
    global _logger_initialized
    logger = logging.getLogger(name)
    logger.setLevel(level)

    if not _logger_initialized:
        formatter = logging.Formatter("[%(asctime)s] [%(levelname)s] [%(filename)s:%(lineno)d] %(message)s", datefmt="%H:%M:%S")

        # Console Handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

        # Qt GUI Handler
        qt_handler = QtLogHandler(_qt_emitter)
        qt_handler.setFormatter(formatter)
        logger.addHandler(qt_handler)

        # Optional File Handler
        if log_file:
            try:
                file_handler = logging.FileHandler(log_file, encoding="utf-8")
                file_handler.setFormatter(formatter)
                logger.addHandler(file_handler)
            except Exception as e:
                print(f"Failed to setup log file handler: {e}")

        _logger_initialized = True

    return logger
