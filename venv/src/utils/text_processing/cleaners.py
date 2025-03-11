"""Text cleaning utilities."""

import re

def clean_response(text: str) -> str:
    """Clean and format response text."""
    text = clean_whitespace(text)
    text = remove_prefixes(text)
    return text.strip()

def clean_whitespace(text: str) -> str:
    """Remove extra whitespace and normalize newlines."""
    # Remove multiple newlines
    text = re.sub(r'\n+', '\n', text)
    # Normalize whitespace
    return ' '.join(text.split())

def remove_prefixes(text: str) -> str:
    """Remove common prefixes like 'AI:' or 'Assistant:'."""
    return re.sub(r'^(AI:|Assistant:)\s*', '', text)