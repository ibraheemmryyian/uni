import numpy as np
from typing import List
from sentence_transformers import SentenceTransformer
from src.logger import setup_logger

class VectorStore:
    def __init__(self):
        self.logger = setup_logger(__name__, "logs/vector_store.log")
        self.model = SentenceTransformer('all-MiniLM-L6-v2', device='cpu')
        self.embeddings = None

    def store_embeddings(self, texts: List[str]):
        try:
            self.logger.info("Generating CPU embeddings")
            self.embeddings = self.model.encode(
                texts, 
                convert_to_numpy=True,
                show_progress_bar=False
            )
        except Exception as e:
            self.logger.error(f"Embedding failed: {str(e)}")
            raise

    def get_embeddings(self):
        return self.embeddings

    def embed(self, text: str):
        return self.model.encode([text], convert_to_numpy=True)[0]