"""State processing utilities for RL model."""

import numpy as np
import torch
from sentence_transformers import SentenceTransformer
from typing import Union

class StateProcessor:
    def __init__(self, embedding_model: str = 'all-mpnet-base-v2'):
        self.embedding_model = SentenceTransformer(embedding_model)
        
    def process_state(self, query: str) -> np.ndarray:
        """Convert query to state representation."""
        return self.embedding_model.encode([query])[0]
        
    def convert_to_tensor(self, state: np.ndarray, device: torch.device) -> torch.Tensor:
        """Convert state to tensor safely."""
        if isinstance(state, np.ndarray):
            state = np.array([state])  # Ensure 2D array
        return torch.FloatTensor(state).to(device)

    def batch_states(self, states: list) -> np.ndarray:
        """Convert list of states to numpy array safely."""
        return np.stack(states)