"""Model configuration settings."""

from typing import Dict, Any

LANGUAGE_MODEL_CONFIG = {
    "name": "fine_tuned_phi2_model",  # Updated to use Phi-2 model
    "generation_config": {
        "max_new_tokens": 150,  # Control new tokens instead of total length
        "temperature": 0.3,  # Set temperature to 0.3
        "top_p": 0.95,  # Keep this if you want to use top-p sampling
        "top_k": 50,
        "do_sample": True,  # Ensure sampling is enabled
        "pad_token_id": 0,  # Will be set during initialization
        "eos_token_id": 2 
    }
}

EMBEDDING_MODEL_CONFIG = {
    "name": "all-mpnet-base-v2"
}

INTENT_CLASSIFIER_CONFIG = {
    "name": "facebook/bart-large-mnli",
    "labels": ["question", "complaint", "request", "greeting"]
}

SENTIMENT_ANALYZER_CONFIG = {
    "name": "distilbert-base-uncased-finetuned-sst-2-english"
}