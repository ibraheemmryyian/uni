from typing import Dict, Any, List
from src.logger import setup_logger

class PromptBuilder:
    def __init__(self, log_file=r"C:\Users\amrey\Desktop\bolt multi file\venv\logs\logfile.log"):
        """
        Initializes the PromptBuilder class with a logger.
        
        Args:
            log_file (str): The path to the log file. Defaults to a sample path.
        """
        # Ensure the log file path is passed to the logger setup
        self.logger = setup_logger(__name__, log_file)
        
    def build_prompt(
        self,
        query: str,
        faq_matches: List[Any],
        conversation_history: Dict[str, Any]
    ) -> str:
        """Build a complete prompt with all context."""
        try:
            prompt_parts = [
                "You are a helpful customer support assistant.",
                "Provide clear and concise responses.",
                f"\nUser query: {query}"
            ]
            
            # Add FAQ matches
            if faq_matches:
                prompt_parts.append("\nRelevant FAQ information:")
                for doc in faq_matches:
                    if hasattr(doc, 'page_content'):
                        prompt_parts.append(f"- {doc.page_content}")
            
            # Add conversation history
            if conversation_history:
                history = conversation_history.get("history", "")
                if history:
                    prompt_parts.append("\nPrevious conversation:")
                    prompt_parts.append(history)
            
            prompt_parts.append("\nResponse:")
            return "\n".join(prompt_parts)
            
        except Exception as e:
            self.logger.error(f"Error building prompt: {str(e)}")
            raise  # Re-raise the exception after logging
