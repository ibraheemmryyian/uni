"""Text processing package initialization."""

from .cleaners import clean_response, clean_whitespace
from .normalizers import normalize_query, normalize_text
from .intent_extractor import extract_intent
from .intent_extractor import is_greeting  
import re
from typing import List

def normalize_query(text: str) -> str:
    """Normalize input text by removing extra whitespace and converting to lowercase"""
    return ' '.join(text.lower().split())

def is_greeting(text: str) -> bool:
    """Check if the input text is a greeting"""
    greetings = {
        'hi', 'hello', 'hey', 'good morning', 'good afternoon', 
        'good evening', 'greetings', 'howdy', 'hi there', 'hello there'
    }
    normalized_text = normalize_query(text)
    return any(greeting in normalized_text for greeting in greetings)

def extract_keywords(text: str) -> List[str]:
    """Extract important keywords from text"""
    # Remove special characters and extra spaces
    cleaned = re.sub(r'[^\w\s]', ' ', text)
    words = cleaned.lower().split()
    
    # Remove common stop words (you can expand this list)
    stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'is', 'are'}
    return [word for word in words if word not in stop_words]

def standardize_text(text: str) -> str:
    """Standardize text format"""
    # Remove multiple spaces
    text = re.sub(r'\s+', ' ', text)
    
    # Ensure proper sentence capitalization
    sentences = re.split(r'([.!?]+)', text)
    formatted = []
    for i in range(0, len(sentences)-1, 2):
        sentence = sentences[i].strip()
        punctuation = sentences[i+1] if i+1 < len(sentences) else "."
        
        if sentence:
            sentence = sentence[0].upper() + sentence[1:].lower()
            formatted.append(sentence + punctuation)
    
    return " ".join(formatted)

__all__ = [
    'clean_response',
    'clean_whitespace',
    'normalize_query', 
    'normalize_text',
    'extract_intent',
    'is_greeting'
]
