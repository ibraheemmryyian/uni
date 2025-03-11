from typing import Dict, List, Tuple
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
import torch

class MetricsCalculator:
    @staticmethod
    def calculate_metrics(predictions: List[str], targets: List[str]) -> Dict[str, float]:
        """Calculate various metrics for model evaluation."""
        # Convert to binary format for metric calculation
        pred_binary = np.array([1 if p == t else 0 for p, t in zip(predictions, targets)])
        target_binary = np.array([1 if target == target else 0 for target in targets])  # assuming target is binary
        
        return {
            'accuracy': accuracy_score(target_binary, pred_binary),
            'f1': f1_score(target_binary, pred_binary, average='weighted'),
            'precision': precision_score(target_binary, pred_binary, average='weighted'),
            'recall': recall_score(target_binary, pred_binary, average='weighted')
        }
    
    @staticmethod
    def calculate_perplexity(model_output: torch.Tensor) -> float:
        """Calculate model perplexity."""
        if hasattr(model_output, 'loss'):  # Ensure 'loss' exists in model_output
            return torch.exp(model_output.loss).item()
        else:
            raise ValueError("Model output does not have 'loss' attribute.")
    
    @staticmethod
    def calculate_response_confidence(
        logits: torch.Tensor,
        threshold: float = 0.7
    ) -> Tuple[bool, float]:
        """Calculate confidence score for model response."""
        probs = torch.softmax(logits, dim=-1)
        confidence = torch.max(probs).item()
        return confidence >= threshold, confidence
