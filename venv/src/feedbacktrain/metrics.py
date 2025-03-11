from typing import Dict, List, Tuple
import numpy as np
import torch
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


class MetricsCalculator:
    @staticmethod
    def calculate_metrics(predictions: List[float], targets: List[float]) -> Dict[str, float]:
        """Calculate various regression metrics for model evaluation."""
        # Calculate MAE, MSE, RMSE, and R^2 for feedback ratings (continuous values from 1 to 5)
        mae = mean_absolute_error(targets, predictions)
        mse = mean_squared_error(targets, predictions)
        rmse = np.sqrt(mse)  # Root Mean Squared Error
        r2 = r2_score(targets, predictions)  # R-squared (good for evaluating regression tasks)

        return {
            'mae': mae,
            'mse': mse,
            'rmse': rmse,
            'r2': r2
        }

    @staticmethod
    def calculate_perplexity(model_output: torch.Tensor) -> float:
        """Calculate model perplexity (not typically used for regression, but provided as-is)."""
        if hasattr(model_output, 'loss'):  # Ensure 'loss' exists in model_output
            return torch.exp(model_output.loss).item()
        else:
            raise ValueError("Model output does not have 'loss' attribute.")

    @staticmethod
    def calculate_response_confidence(
        logits: torch.Tensor,
        threshold: float = 0.7
    ) -> Tuple[bool, float]:
        """Calculate confidence score for model response (useful for classification but not regression)."""
        # In the case of regression, confidence can be based on the absolute error or prediction confidence.
        # This method may not be directly applicable for regression, but could be adapted if needed.
        probs = torch.softmax(logits, dim=-1)
        confidence = torch.max(probs).item()
        return confidence >= threshold, confidence
