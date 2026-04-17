"""
logs.py
=======
Logging configuration for the ZHSF anomaly-detection pipeline.

Provides centralized logging setup with both console and file output.
"""

import logging
import os
from datetime import datetime
from config import LOGS_DIR


def setup_logging(
    log_level: str = 'INFO',
    log_to_file: bool = True,
    log_filename: str = None
) -> logging.Logger:
    """
    Set up logging configuration for the anomaly detection pipeline.

    Parameters
    ----------
    log_level : str
        Logging level ('DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL')
    log_to_file : bool
        Whether to also log to a file
    log_filename : str, optional
        Custom log filename. If None, uses timestamp-based name.

    Returns
    -------
    logging.Logger
        Configured logger instance
    """
    # Create logger
    logger = logging.getLogger('anomaly_detection')
    logger.setLevel(getattr(logging, log_level.upper()))

    # Remove existing handlers to avoid duplicates
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)

    # Create formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    # Console handler (always enabled)
    console_handler = logging.StreamHandler()
    console_handler.setLevel(getattr(logging, log_level.upper()))
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # File handler (optional)
    if log_to_file:
        if log_filename is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            log_filename = f'training_{timestamp}.log'

        log_path = os.path.join(LOGS_DIR, log_filename)
        file_handler = logging.FileHandler(log_path)
        file_handler.setLevel(getattr(logging, log_level.upper()))
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

        logger.info(f"Logging to file: {log_path}")

    return logger


def get_logger() -> logging.Logger:
    """
    Get the configured logger instance.

    Returns
    -------
    logging.Logger
        The anomaly detection logger
    """
    return logging.getLogger('anomaly_detection')