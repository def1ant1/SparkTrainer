"""
Central logging system for SparkTrainer.
Logs are written to runs/<timestamp>/spark_trainer.log
"""
import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Optional

_logger_instance: Optional[logging.Logger] = None
_log_dir: Optional[Path] = None


def setup_logger(run_name: Optional[str] = None, log_dir: Optional[str] = None) -> logging.Logger:
    """
    Set up the central logger for SparkTrainer.

    Args:
        run_name: Optional run name. If None, uses timestamp.
        log_dir: Optional log directory. If None, uses runs/<timestamp>/

    Returns:
        Configured logger instance
    """
    global _logger_instance, _log_dir

    if _logger_instance is not None:
        return _logger_instance

    # Determine log directory
    if log_dir is None:
        timestamp = run_name or datetime.now().strftime("%Y%m%d_%H%M%S")
        _log_dir = Path("runs") / timestamp
    else:
        _log_dir = Path(log_dir)

    _log_dir.mkdir(parents=True, exist_ok=True)
    log_file = _log_dir / "spark_trainer.log"

    # Create logger
    logger = logging.getLogger("spark_trainer")
    logger.setLevel(logging.INFO)
    logger.propagate = False

    # Remove existing handlers
    logger.handlers.clear()

    # File handler
    file_handler = logging.FileHandler(log_file, mode="a")
    file_handler.setLevel(logging.DEBUG)
    file_formatter = logging.Formatter(
        "%(asctime)s | %(levelname)-8s | %(name)s:%(lineno)d | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    file_handler.setFormatter(file_formatter)
    logger.addHandler(file_handler)

    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_formatter = logging.Formatter(
        "%(asctime)s | %(levelname)-8s | %(message)s",
        datefmt="%H:%M:%S"
    )
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)

    _logger_instance = logger
    logger.info(f"Logger initialized. Logs writing to: {log_file}")

    return logger


def get_logger() -> logging.Logger:
    """
    Get the current logger instance. If not initialized, sets up a default logger.

    Returns:
        Logger instance
    """
    global _logger_instance
    if _logger_instance is None:
        return setup_logger()
    return _logger_instance


def get_log_dir() -> Optional[Path]:
    """
    Get the current log directory.

    Returns:
        Path to log directory, or None if logger not initialized
    """
    return _log_dir
