import torch
import logging
import asyncio
import numpy as np
from typing import Dict, Any, List, Optional
from transformers import AutoTokenizer, AutoModelForCausalLM
from src.logger import setup_logger
from src.utils.sentiment_analyzer import SentimentAnalyzer
from src.model_settings import MODEL_NAME, MAX_LENGTH

class ResponseGenerator:
    def __init__(self, faq_database, vector_store=None):
        self.faq_database = faq_database
        self.vector_store = vector_store
        self.sentiment_analyzer = SentimentAnalyzer()
        self.logger = setup_logger(__name__, "logs/response_generator.log")
        self.ratings = []
        self.device = torch.device("cpu")

        try:
            self.logger.info(f"Initializing {MODEL_NAME} for CPU")
            self.tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
            
            # Ensure pad_token is set properly
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            # CPU-specific model loading
            self.model = AutoModelForCausalLM.from_pretrained(
                MODEL_NAME,
                torch_dtype=torch.float32,
                device_map=None,
                low_cpu_mem_usage=True
            ).cpu()
            
            self.logger.info("CPU model ready")

        except Exception as e:
            self.logger.error(f"CPU model init failed: {str(e)}")
            raise

    async def generate_response(self, user_input: str) -> str:
        try:
            # Async sentiment analysis
            sentiment = await asyncio.to_thread(
                self.sentiment_analyzer.analyze_sentiment, 
                user_input
            )
            
            faq_response = self.get_best_faq_response(user_input)
            if faq_response:
                return self._apply_sentiment(faq_response, sentiment)
                
            # Generate response on CPU
            inputs = self.tokenizer(
                user_input,
                return_tensors='pt',
                max_length=MAX_LENGTH,
                truncation=True
            )
            
            outputs = self.model.generate(
                inputs.input_ids,
                attention_mask=inputs.attention_mask,
                max_length=150,
                temperature=0.7,
                top_k=50,
                top_p=0.9,  # Optional: controlled randomness
                do_sample=True
            )
            
            response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            return self._apply_sentiment(response, sentiment)

        except Exception as e:
            self.logger.error(f"Generation error for input '{user_input}': {str(e)}")
            return "Please try rephrasing your question."

    def _apply_sentiment(self, response: str, sentiment: dict) -> str:
        if sentiment["compound"] <= -0.3:
            return f"I'm sorry to hear that. {response}"
        elif sentiment["compound"] >= 0.3:
            return f"That's great! {response}"
        return response

    def get_best_faq_response(self, user_input: str) -> Optional[str]:
        if not self.vector_store:
            return None

        try:
            # CPU-optimized similarity search
            query_embed = self.vector_store.embed(user_input)
            faq_embeddings = self.vector_store.get_embeddings()
            
            similarities = np.dot(faq_embeddings, query_embed)
            best_match_idx = np.argmax(similarities)
            
            if similarities[best_match_idx] > 0.7:  # Higher threshold for better match
                return self.faq_database.get_response_by_index(best_match_idx)
                
        except Exception as e:
            self.logger.warning(f"FAQ lookup error: {str(e)}")
            
        return None
