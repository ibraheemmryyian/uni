"""Response generation and processing."""

from typing import Dict, Any
from src.training.config import TRAINING_CONFIG
from src.utils import clean_response

class ResponseHandler:
    @staticmethod
    def process_response(
        response_data: Dict[str, Any],
        confidence_threshold: float = 0.8
    ) -> Dict[str, Any]:
        """Process and validate response."""
        confidence = response_data.get('confidence', 0)
        
        if confidence < confidence_threshold:
            return {
                'response': 'I need more information to provide an accurate response. Could you please provide more details?',
                'confidence': confidence,
                'valid': False
            }
            
        return {
            'response': response_data['response'],
            'confidence': confidence,
            'valid': True
        }