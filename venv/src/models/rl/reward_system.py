"""Reward System for Customer Support RL."""

from typing import Dict
import numpy as np
from sentence_transformers import SentenceTransformer

class RewardSystem:
    def __init__(self, embedding_model: str = 'all-mpnet-base-v2'):
        self.embedding_model = SentenceTransformer(embedding_model)
        
    def calculate_reward(self, query: str, response: str, expected_response: str, tag_match: bool, feedback: float = None) -> float:
        """Calculate reward based on multiple factors including user feedback."""
        # Calculate semantic similarity
        query_embedding = self.embedding_model.encode([query])[0]
        response_embedding = self.embedding_model.encode([response])[0]
        expected_embedding = self.embedding_model.encode([expected_response])[0]
        
        # Response similarity with query
        query_similarity = np.dot(query_embedding, response_embedding) / (
            np.linalg.norm(query_embedding) * np.linalg.norm(response_embedding)
        )
        
        # Response similarity with expected response
        expected_similarity = np.dot(response_embedding, expected_embedding) / (
            np.linalg.norm(response_embedding) * np.linalg.norm(expected_embedding)
        )
        
        # Combine rewards
        reward = (
            0.4 * query_similarity +  # Weight for query relevance
            0.4 * expected_similarity +  # Weight for response accuracy
            0.2 * float(tag_match)  # Weight for tag matching
        )
        
        # If feedback is provided, adjust the reward accordingly (0-5 feedback scale)
        if feedback is not None:
            reward += feedback
        
        return reward
