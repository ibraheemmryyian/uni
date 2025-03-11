import logging
from logging.handlers import RotatingFileHandler
import os
from pathlib import Path

def setup_logger(name: str, log_file: str, level=logging.INFO) -> logging.Logger:
    """Set up logger with rotating file handler"""
    # Create base log directory
    base_log_dir = Path("venv/logs")
    
    # Create specific log directory based on module name
    module_log_dir = base_log_dir / name.split('.')[-1]
    module_log_dir.mkdir(parents=True, exist_ok=True)
    
    # Full path to log file
    log_file_path = module_log_dir / "logfile.log"
    
    # Create logger
    logger = logging.getLogger(name)
    logger.setLevel(level)

    # Create handlers
    file_handler = RotatingFileHandler(
        str(log_file_path), 
        maxBytes=10485760,  # 10MB
        backupCount=5
    )
    console_handler = logging.StreamHandler()

    # Create formatters
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)

    # Add handlers
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger