"""Text normalization utilities."""

def normalize_query(text: str) -> str:
    """Normalize query text for processing."""
    return normalize_text(text.lower())

def normalize_text(text: str) -> str:
    """General text normalization."""
    # Remove leading/trailing whitespace
    text = text.strip()
    # Normalize internal whitespace
    return ' '.join(text.split())