"""Customer Support Environment for Reinforcement Learning."""
import numpy as np
from typing import Dict, Tuple, Any
from sentence_transformers import SentenceTransformer

class CustomerSupportEnvironment:
    def __init__(self, support_data: Dict[str, Dict[str, str]], embedding_model: str = 'all-mpnet-base-v2'):
        """Initialize the environment with support ticket data."""
        self.support_data = support_data
        self.embedding_model = SentenceTransformer(embedding_model)
        self.current_query = None
        self.current_state = None
        self.feedback = None  # Store feedback (0 to 5)

    def reset(self, query: str) -> np.ndarray:
        """Reset environment with a new query."""
        self.current_query = query
        # Create state embedding from query
        self.current_state = self.embedding_model.encode([query])[0]
        return self.current_state

    def step(self, action: int, feedback: float) -> Tuple[np.ndarray, float, bool, Dict[str, Any]]:
        """Take action and return new state, reward, done flag, and info."""
        selected_response = list(self.support_data.values())[action]
        reward = self._calculate_reward(selected_response, feedback)
        
        return self.current_state, reward, True, {
            'response': selected_response['response'],
            'tag': selected_response['tag']
        }

    def request_feedback(self) -> float:
        """Request feedback from the user on a scale of 0 to 5."""
        while True:
            try:
                feedback = float(input("Was the response accurate? Rate from 0 to 5: "))
                if 0 <= feedback <= 5:
                    self.feedback = feedback
                    return feedback
                else:
                    print("Please enter a valid score between 0 and 5.")
            except ValueError:
                print("Invalid input. Please enter a numeric value between 0 and 5.")

    def _calculate_reward(self, selected_response: Dict[str, str], feedback: float) -> float:
        """Calculate reward based on response similarity and tag matching, adjusted by feedback."""
        # Encode selected response
        response_embedding = self.embedding_model.encode([selected_response['response']])[0]
        
        # Calculate cosine similarity between query and selected response
        similarity = np.dot(self.current_state, response_embedding) / (
            np.linalg.norm(self.current_state) * np.linalg.norm(response_embedding)
        )
        
        # Scale similarity to reward range [-1, 1]
        reward = 2 * similarity - 1
        
        # Adjust the reward based on user feedback (0-5 scale)
        reward += (feedback - 2.5) / 2.5  # Feedback normalized to the range [-1, 1]
        
        return reward
