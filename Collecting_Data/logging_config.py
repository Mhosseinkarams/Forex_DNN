import logging
import os
from logging.handlers import RotatingFileHandler

def setup_logging(log_dir: str, level=logging.INFO):
    """
    Purpose:
        Initializes the centralized logging system for the entire framework.
        Ensures consistent log formatting and unified output to both console and files.

    Arguments:
        log_dir (str): Directory where the log file will be saved.
        level (int): Logging level (e.g., logging.INFO, logging.DEBUG). Defaults to logging.INFO.

    Side Effects:
        Creates the log directory if it does not exist.
        Configures the root logger with handlers for console and a rotating file.

    Notes:
        Individual modules should not call logging.basicConfig(). Instead, they should use:
        `logger = logging.getLogger("ModuleName")`
    """
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    log_format = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
    
    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(level)

    # Remove existing handlers if any
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)

    # Console Handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(logging.Formatter(log_format))
    root_logger.addHandler(console_handler)

    # Rotating File Handler (10MB * 5 backups)
    log_file = os.path.join(log_dir, "trading_system.log")
    file_handler = RotatingFileHandler(log_file, maxBytes=10*1024*1024, backupCount=5)
    file_handler.setFormatter(logging.Formatter(log_format))
    root_logger.addHandler(file_handler)
