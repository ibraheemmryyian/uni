import re
from typing import Optional, Dict
from .normalizers import normalize_query

# Define common greeting patterns for easier maintenance and extension
GREETING_PATTERNS = [
    r'^hi$', r'^hello$', r'^hey$', r'^good morning$', 
    r'^good afternoon$', r'^good evening$'
]

def is_greeting(text: str) -> bool:
    """Check if the input is a greeting."""
    normalized_text = normalize_query(text)  # Ensure the query is normalized to avoid case-sensitivity issues
    return any(re.match(pattern, normalized_text) for pattern in GREETING_PATTERNS)

# Define intent patterns as a module-level constant
INTENT_PATTERNS: Dict[str, str] = {
    'payment_plan': r'payment plan|installments?|pay in parts',
    'time_extension': r'more time|extend|delay|later',
    'dispute': r'dispute|wrong|incorrect|not mine',
    'payment_method': r'how.+pay|payment method|where.+pay',
    'general_inquiry': r'how|what|when|where|why|who'
}

def extract_intent(text: str) -> Optional[str]:
    """Extract primary intent from text."""
    normalized_text = normalize_query(text)  # Normalize input text to account for case insensitivity

    # Iterate through intent patterns and check if any pattern matches the text
    for intent, pattern in INTENT_PATTERNS.items():
        if re.search(pattern, normalized_text, re.IGNORECASE):  # Use re.IGNORECASE for case-insensitive matching
            return intent
            
    return None
