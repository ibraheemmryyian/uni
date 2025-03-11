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

def extract_intent(text: str) -> Optional[str]:
    """Extract primary intent from text."""
    # Add intent extraction logic
    pass

def normalize_query(text: str) -> str:
    """Normalize query text."""
    return text.lower().strip()