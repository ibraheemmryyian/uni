"""Enhanced validation utilities."""

from typing import Dict, Any, Optional
import re

def validate_query(query: str) -> tuple[bool, Optional[str]]:
    """Validate user query."""
    if not query or not query.strip():
        return False, "Query cannot be empty"
        
    if len(query) > 500:
        return False, "Query is too long"
        
    return True, None

def validate_response(response: Dict[str, Any]) -> tuple[bool, Optional[str]]:
    """Validate response data."""
    required_fields = {'response', 'confidence'}
    
    if not all(field in response for field in required_fields):
        return False, "Missing required response fields"
        
    if not isinstance(response['confidence'], (int, float)):
        return False, "Invalid confidence value"
        
    if not 0 <= response['confidence'] <= 1:
        return False, "Confidence must be between 0 and 1"
        
    return True, None

def sanitize_input(text: str) -> str:
    """Sanitize user input."""
    # Remove any potential harmful characters
    text = re.sub(r'[^\w\s\-.,?!]', '', text)
    return text.strip()