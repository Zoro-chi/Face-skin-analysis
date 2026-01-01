"""
Logger Configuration
Sets up logging for the project.
"""

import logging
import sys
from pathlib import Path
from datetime import datetime


def setup_logger(log_dir="outputs/logs", log_level=logging.INFO):
    """
    Setup logger with file and console handlers.

    Args:
        log_dir: Directory to save log files
        log_level: Logging level
    """
    # Create log directory
    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)

    # Create log filename with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_dir / f"skin_analysis_{timestamp}.log"

    # Configure root logger
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[logging.FileHandler(log_file), logging.StreamHandler(sys.stdout)],
    )

    logger = logging.getLogger(__name__)
    logger.info(f"Logging to {log_file}")

    return logger
