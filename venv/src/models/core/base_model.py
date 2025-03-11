"""Base model interface for all AI models."""

from abc import ABC, abstractmethod
import torch
from typing import Dict, Any

class BaseModel(ABC):
    def __init__(self, device: torch.device):
        self.device = device
        
    @abstractmethod
    def initialize(self) -> Any:
        """Initialize the model."""
        pass
        
    @abstractmethod
    def predict(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Make predictions."""
        pass