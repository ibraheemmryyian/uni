import traceback
from typing import Dict, Any, Optional
from src.logger import setup_logger

class ErrorHandler:
    def __init__(self):
        self.logger = setup_logger(__name__, "error_handler")
        self.error_counts = {}

    def handle_error(self, error: Exception, context: Optional[Dict] = None) -> Dict[str, Any]:
        """Handle errors gracefully"""
        error_type = type(error).__name__
        self.error_counts[error_type] = self.error_counts.get(error_type, 0) + 1

        # Log error with context
        self.logger.error(
            f"Error: {str(error)}\n"
            f"Type: {error_type}\n"
            f"Context: {context}\n"
            f"Traceback: {traceback.format_exc()}"
        )

        # Determine appropriate response
        if self.error_counts[error_type] > 10:
            self.logger.critical(f"Critical error frequency for {error_type}")

        return {
            "error_type": error_type,
            "message": str(error),
            "user_message": self._get_user_message(error_type),
            "should_retry": self._should_retry(error_type)
        }

    def _get_user_message(self, error_type: str) -> str:
        """Get user-friendly error message"""
        messages = {
            "TokenizationError": "I'm having trouble understanding that. Could you rephrase?",
            "ModelNotFoundError": "I'm temporarily unavailable. Please try again shortly.",
            "OutOfMemoryError": "I'm processing too many requests. Please try again in a moment."
        }
        return messages.get(error_type, "An unexpected error occurred. Please try again.")

    def _should_retry(self, error_type: str) -> bool:
        """Determine if operation should be retried"""
        retry_errors = {"ConnectionError", "TimeoutError", "ResourceExhaustedError"}
        return error_type in retry_errors 