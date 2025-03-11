from collections import deque
from typing import Dict, Any

class ContextManager:
    def __init__(self):
        self.context = []

    def update_context(self, user_input: str, ai_response: str):
        """Update the context with user input and AI response."""
        self.context.append({"user": user_input, "ai": ai_response})

    def get_context(self) -> str:
        """Get the current context as a string."""
        return "\n".join([f"User: {entry['user']}\nAI: {entry['ai']}" for entry in self.context])

    def clear(self):
        """Clear the conversation context."""
        self.context = []
