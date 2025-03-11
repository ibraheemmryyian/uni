import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path

def setup_logger(name: str, log_file: str, level=logging.INFO) -> logging.Logger:
    """Set up a logger with a rotating file handler."""
    log_dir = Path(log_file).parent
    log_dir.mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger(name)
    logger.setLevel(level)

    # Create a console handler to output logs to the console
    console_handler = logging.StreamHandler()
    console_handler.setLevel(level)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # Set the log file path to the specified directory
    log_file_path = Path("C:/Users/amrey/Desktop/bolt multi file - Copy (2)/venv/logs/app.log")
    handler = RotatingFileHandler(log_file_path, maxBytes=10*1024*1024, backupCount=5)  # 10 MB
    handler.setFormatter(formatter)

    logger.addHandler(handler)

    return logger 