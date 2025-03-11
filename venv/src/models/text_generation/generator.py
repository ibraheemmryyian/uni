from typing import Dict, Any
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from ..config.model_settings import LANGUAGE_MODEL_CONFIG
from src.logger import setup_logger

class TextGenerator:
    def __init__(self, device: torch.device, log_file=r"C:\Users\amrey\Desktop\bolt multi file\venv\logs\logfile.log"):
        """Initialize the text generator with a specified device and log file."""
        # Initialize logger with proper log file
        self.logger = setup_logger(__name__, src.logs.log_file.log)
        self.device = device
        self._initialize_model()
        
    def _initialize_model(self):
        """Initialize the language model and tokenizer."""
        try:
            # Load the model and tokenizer from the pre-trained configuration
            self.model = AutoModelForCausalLM.from_pretrained(
                LANGUAGE_MODEL_CONFIG["name"]
            ).to(self.device)
            
            self.tokenizer = AutoTokenizer.from_pretrained(
                LANGUAGE_MODEL_CONFIG["name"]
            )
            
            # Update the generation configuration with tokenizer's pad token
            self.generation_config = LANGUAGE_MODEL_CONFIG["generation_config"].copy()
            self.generation_config["pad_token_id"] = self.tokenizer.eos_token_id
            
        except Exception as e:
            # Log initialization errors and raise them
            self.logger.error(f"Error initializing text generator: {str(e)}")
            raise
            
    def generate(self, prompt: str) -> str:
        """Generate text response from a given prompt."""
        try:
            # Tokenize the prompt with truncation and padding
            inputs = self.tokenizer(
                prompt,
                return_tensors="pt",
                truncation=True,
                max_length=512,
                padding="longest"  # Ensure consistent input lengths
            ).to(self.device)
            
            # Generate output from the model
            outputs = self.model.generate(
                inputs["input_ids"],
                attention_mask=inputs.get("attention_mask", None),  # Use attention mask if available
                **self.generation_config
            )
            
            # Decode the output into human-readable text
            return self.tokenizer.decode(outputs[0], skip_special_tokens=True)
    
        except Exception as e:
            # Log errors during text generation
            self.logger.error(f"Error generating text: {str(e)}")
            raise
