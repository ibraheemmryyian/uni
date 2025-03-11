import torch
from typing import Dict, Any
from transformers import AutoTokenizer, AutoModelForCausalLM
from .core.base_model import BaseModel
from .core.response_handler import ResponseHandler
from .conversation.context_manager import ContextManager
from pathlib import Path
from src.config.model_config import MODEL_CONFIG
from src.logger import setup_logger

class ResponseGenerator(BaseModel):
    _model = None
    _tokenizer = None

    def __init__(self, device: torch.device, vector_store):
        super().__init__(device)
        self.vector_store = vector_store
        
        # Set up logger
        self.logger = setup_logger(__name__, "response_generator")

        # Initialize model if not already loaded
        if not ResponseGenerator._model:
            self._initialize_model()

    def _initialize_model(self):
        try:
            self.model_name = "HuggingFaceH4/zephyr-7b-beta"
            print(f"Initializing {self.model_name}...")

            # Load tokenizer and model only once
            ResponseGenerator._tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            ResponseGenerator._model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                torch_dtype=torch.float16,
                device_map="auto"
            ).half()  # Load model in half precision for efficiency
            self.logger.info("Model initialized successfully!")

        except Exception as e:
            self.logger.error(f"Error initializing model: {str(e)}")
            raise

    def _create_prompt(self, query: str, faq_data: Dict[str, Dict[str, str]], context: Dict[str, Any] = None) -> str:
        """Create a prompt for the model based on the query, FAQ data, and context."""
        system_prompt = """You are a professional customer support AI assistant. 
Your responses should be:
1. Accurate and based on the provided FAQ information
2. Professional and empathetic
3. Clear and solution-oriented
4. Concise yet complete

FAQ Information:
"""
        # Add FAQ data
        for question, data in faq_data.items():
            system_prompt += f"Q: {question}\nA: {data['response']}\n\n"

        messages = [{"role": "system", "content": system_prompt}]

        # Add conversation history (if available)
        if context and 'conversation_history' in context:
            history = context['conversation_history'].get('history', [])
            for turn in history:
                messages.append({"role": "user", "content": turn.get('input', '')})
                messages.append({"role": "assistant", "content": turn.get('output', '')})

        # Add current query
        messages.append({"role": "user", "content": query})

        # Return formatted prompt
        return self._tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )

    def _tokenize_input(self, prompt: str):
        """Tokenize the input and prepare for model inference."""
        return self._tokenizer(prompt, return_tensors="pt", padding=True, truncation=True, max_length=MODEL_CONFIG['language_model']['max_length'])

    def _generate(self, input_ids: torch.Tensor, attention_mask: torch.Tensor):
        """Generate response using the model."""
        return self._model.generate(
            input_ids.to(self.device),
            attention_mask=attention_mask.to(self.device),
            max_new_tokens=150,
            temperature=0.3,
            top_p=0.85,
            repetition_penalty=1.0,
            do_sample=False,  # Greedy decoding for faster response
            pad_token_id=self._tokenizer.eos_token_id
        )

    def generate_response(self, query: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """Generate a response based on the query, context, and FAQ data."""
        try:
            # Get relevant FAQ data from the vector store
            faq_matches = self.vector_store.similarity_search(query, k=2)
            faq_data = {match['question']: {'response': match['answer']} for match in faq_matches}

            # Create the prompt using the query, FAQ data, and context
            prompt = self._create_prompt(query, faq_data, context)

            # Tokenize input and prepare for model inference
            inputs = self._tokenize_input(prompt)
            attention_mask = inputs.get("attention_mask", torch.ones(inputs["input_ids"].shape, device=self.device))

            # Generate response using the model
            outputs = self._generate(inputs["input_ids"], attention_mask)

            # Decode the model's output
            response = self._tokenizer.decode(outputs[0], skip_special_tokens=True)
            response = response.split("<|assistant|>")[-1].strip()

            # Set confidence based on FAQ data availability
            confidence = 0.9 if faq_data else 0.7

            return {
                "response": response,
                "confidence": confidence,
                "faq_matched": bool(faq_data),
                "model_used": self.model_name
            }

        except Exception as e:
            self.logger.error(f"Error generating response: {str(e)}")
            return {
                "response": "I apologize, but I encountered an error. Please try again or contact our support team.",
                "confidence": 0.0,
                "faq_matched": False,
                "error": str(e)
            }

    def predict(self, query: str, context: Dict[str, Any]) -> str:
        """Simple prediction interface for generating responses."""
        response_data = self.generate_response(query, context)
        return response_data["response"]

    def initialize(self) -> bool:
        """Initialize the response generator."""
        try:
            self.logger.info("Initializing ResponseGenerator with HuggingFaceH4/zephyr-7b-beta")
            return True
        except Exception as e:
            self.logger.error(f"Failed to initialize: {str(e)}")
            return False
