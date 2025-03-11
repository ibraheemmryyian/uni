"""Custom exceptions for the application."""

class ModelInitializationError(Exception):
    """Raised when there's an error initializing AI models."""
    pass

class FAQDatabaseError(Exception):
    """Raised when there's an error with the FAQ database operations."""
    pass

class ResponseGenerationError(Exception):
    """Raised when there's an error generating responses."""
    pass