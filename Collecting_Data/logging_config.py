import logging
import os
from logging.handlers import RotatingFileHandler

def setup_logging(log_dir: str, level=logging.INFO):
    """
    Configures the root logger with a console handler and a RotatingFileHandler.
    
    Args:
        log_dir (str): Directory where the log file will be saved.
        level (int): Logging level. Defaults to logging.INFO.
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
