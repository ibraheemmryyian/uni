from typing import Dict, Any
import re

class ResponseEnhancer:
    def __init__(self):
        self.placeholders = {
            "{reference ID}": "your unique reference ID",
            "{Lender}": "Tamara",
            "{outstanding balance}": "your current balance",
            "{Registered Number}": "your registered phone number"
        }
        
    def enhance_response(self, response: str, context: Dict[str, Any]) -> str:
        """Replace placeholders with actual values if available in context"""
        # Create a map of actual values for placeholders
        replacement_map = {placeholder: str(context.get(placeholder.strip('{}'), default))
                           for placeholder, default in self.placeholders.items()}

        # Perform all replacements in one go using regex
        enhanced_response = response
        for placeholder, actual_value in replacement_map.items():
            enhanced_response = enhanced_response.replace(placeholder, actual_value)

        return enhanced_response

    def format_currency(self, amount: float, currency: str = "AED") -> str:
        """Format currency amounts"""
        return f"{currency} {amount:,.2f}"
