import logging
from pathlib import Path

def setup_logger(name: str, log_file: str, level: str = "INFO") -> logging.Logger:
    """
    Set up a logger that logs messages to a file and console.
    
    :param name: Name of the logger (usually set to __name__).
    :param log_file: Path to the log file.
    :param level: Logging level (e.g., 'INFO', 'DEBUG', 'ERROR').
    :return: Logger instance.
    """
    # Create logs directory if it doesn't exist
    log_dir = Path(log_file).parent
    log_dir.mkdir(parents=True, exist_ok=True)

    # Create a logger
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, level))

    # Create a file handler for logging to file
    file_handler = logging.FileHandler(log_file, encoding='utf-8')
    file_handler.setLevel(getattr(logging, level))

    # Create a console handler for logging to console
    console_handler = logging.StreamHandler()
    console_handler.setLevel(getattr(logging, level))

    # Create a formatter
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    # Set formatter for both handlers
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)

    # Add handlers to the logger
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger
