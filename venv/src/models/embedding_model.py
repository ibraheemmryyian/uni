from sentence_transformers import SentenceTransformer
from ..utils.exceptions import ModelInitializationError
from ..config import MODEL_CONFIG
from src.logger import setup_logger

class EmbeddingModel:
    def __init__(self, log_file=r'C:\Users\amrey\Desktop\bolt multi file\venv\logs\logfile.log'):
        # Pass the log_file argument to the setup_logger function
        self.logger = setup_logger(__name__, log_file)
        
    def initialize(self):
        """Initialize the embedding model."""
        try:
            return SentenceTransformer(MODEL_CONFIG["embedding_model"]["name"])
        except Exception as e:
            raise ModelInitializationError(f"Embedding model initialization failed: {str(e)}")
