from typing import TypedDict, Dict, Any, List

class ModelConfig(TypedDict):
    """Configuration for a language model."""
    name: str  # Model name
    max_length: int  # Maximum input length
    temperature: float  # Sampling temperature for generation
    top_p: float  # Top-p (nucleus) sampling
    top_k: int  # Top-k sampling

class FAQDocument(TypedDict):
    """Represents a document in the FAQ database."""
    page_content: str  # The content of the FAQ page
    metadata: Dict[str, Any]  # Metadata related to the FAQ (e.g., category, author)

class ResponseContext(TypedDict):
    """Context required to generate a response."""
    conversation_history: Dict[str, Any]  # History of the conversation
    faq_matches: List[FAQDocument]  # List of matching FAQ documents for reference
