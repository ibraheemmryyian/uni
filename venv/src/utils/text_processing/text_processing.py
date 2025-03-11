"""Text processing utilities."""

import re
from typing import Optional

def clean_response(text: str) -> str:
    """Clean and format response text."""
    # Remove multiple newlines
    text = re.sub(r'\n+', '\n', text)
    
    # Remove AI/Assistant prefixes
    text = re.sub(r'^(AI:|Assistant:)\s*', '', text)
    
    # Remove extra whitespace
    text = ' '.join(text.split())
    
    return text.strip()

def normalize_query(text: str) -> str:
    """Normalize query text for processing."""
    # Convert to lowercase and strip whitespace
    text = text.lower().strip()
    
    # Remove extra whitespace
    text = ' '.join(text.split())
    
    return text

def extract_intent(text: str) -> Optional[str]:
    """Extract primary intent from text."""
    # Common intent patterns
    intent_patterns = {
        'payment_plan': r'payment plan|installments?|pay in parts',
        'time_extension': r'more time|extend|delay|later',
        'dispute': r'dispute|wrong|incorrect|not mine',
        'payment_method': r'how.+pay|payment method|where.+pay',
        'general_inquiry': r'how|what|when|where|why|who'
    }
    
    # Normalize text for pattern matching
    normalized_text = normalize_query(text)
    
    # Check patterns
    for intent, pattern in intent_patterns.items():
        if re.search(pattern, normalized_text):
            return intent
            
    return None