from typing import List, Dict
import chromadb
from chromadb.utils import embedding_functions
from src.logger import setup_logger
import numpy as np

class FAQDatabase:
    def __init__(self):
        self.client = chromadb.PersistentClient(path="./faq_db")
        self.embedding_function = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name="all-MiniLM-L6-v2"
        )
        self.collection = self.client.get_or_create_collection(
            name="faqs",
            embedding_function=self.embedding_function
        )
        self.logger = setup_logger(__name__, "logs/faq_database.log")
        self.faqs = []

    def setup(self, faq_data: List[Dict[str, str]]):
        try:
            documents = []
            metadatas = []
            ids = []
            
            for idx, item in enumerate(faq_data):
                documents.append(item["question"])
                metadatas.append({"response": item["response"], "tag": item.get("tag", "")})
                ids.append(str(idx))
                self.faqs.append(item)

            self.collection.upsert(
                documents=documents,
                metadatas=metadatas,
                ids=ids
            )
            self.logger.info(f"Loaded {len(faq_data)} FAQs")

        except Exception as e:
            self.logger.error(f"Setup failed: {str(e)}")
            raise

    def get_response_by_index(self, index: int) -> str:
        try:
            return self.faqs[index]["response"]
        except IndexError:
            self.logger.warning(f"Invalid FAQ index: {index}")
            return None

    def embed(self, text: str) -> np.ndarray:
        """Generate embedding for a single text"""
        return self.embedding_function([text])[0]

    def get_embeddings(self) -> np.ndarray:
        """Get all FAQ embeddings"""
        return np.array([
            self.embedding_function([faq["question"]])[0]
            for faq in self.faqs
        ])

    def query_similar(self, query: str, threshold: float = 0.7) -> List[Dict]:
        try:
            results = self.collection.query(
                query_texts=[query],
                n_results=3,
                include=["metadatas", "distances"]
            )

            # Extract responses and filter based on threshold
            relevant_responses = []
            for meta, distance in zip(results["metadatas"][0], results["distances"][0]):
                if distance <= threshold:  # Lower distance = higher similarity
                    relevant_responses.append({
                        "response": meta["response"],
                        "tag": meta.get("tag", ""),
                        "similarity": 1 - distance  # Convert distance to similarity score
                    })

            return relevant_responses

        except Exception as e:
            self.logger.error(f"Query failed: {str(e)}")
            return []
