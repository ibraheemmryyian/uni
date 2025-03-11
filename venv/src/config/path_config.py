from pathlib import Path
from src.faq_data import faq_data

class PathConfig:
    """Central configuration for all paths in the application"""
    
    # Base paths
    BASE_DIR = Path("venv")
    LOGS_DIR = BASE_DIR / "logs"
    
    # Specific log directories
    PROCESS_LOGS = LOGS_DIR / "process"
    CHAT_LOGS = LOGS_DIR / "chat_history"
    FAQ_LOGS = LOGS_DIR / "faq"
    MODEL_LOGS = LOGS_DIR / "model"
    
    # Ensure all directories exist
    @classmethod
    def create_directories(cls):
        """Create all necessary directories"""
        for attr_name in dir(cls):
            if attr_name.endswith('_DIR') or attr_name.endswith('_LOGS'):
                path = getattr(cls, attr_name)
                if isinstance(path, Path):
                    path.mkdir(parents=True, exist_ok=True) 