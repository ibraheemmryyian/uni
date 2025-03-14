import logging
import os
from logging.handlers import RotatingFileHandler

def setup_logger(name: str, log_file: str, level=logging.INFO) -> logging.Logger:
    """Set up logger with file and console handlers"""
    
    # Create logs directory if it doesn't exist
    os.makedirs(os.path.dirname(log_file), exist_ok=True)
    
    # Create logger
    logger = logging.getLogger(name)
    
    # Set level directly using the level parameter
    logger.setLevel(level)
    
    # Create handlers
    file_handler = RotatingFileHandler(
        log_file,
        maxBytes=1024 * 1024,  # 1MB
        backupCount=3
    )
    console_handler = logging.StreamHandler()
    
    # Create formatters and add it to handlers
    log_format = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S,%f'
    )
    file_handler.setFormatter(log_format)
    console_handler.setFormatter(log_format)
    
    # Add handlers to the logger
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logger
