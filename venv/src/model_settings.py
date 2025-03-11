

import torch  


MODEL_NAME = "fine_tuned_phi2_model"  
MAX_LENGTH = 150  
NUM_RETURN_SEQUENCES = 1  
DEVICE = "cuda" if torch.cuda.is_available() else "cpu" 


TEMPERATURE = 0.3 
TOP_K = 50  

"""Model configuration settings."""

from typing import Dict, Any

LANGUAGE_MODEL_CONFIG = {
    "name": "fine_tuned_phi2_model",  
    "generation_config": {
        "max_new_tokens": 150,  
        "temperature": 0.3, 
        "top_p": 0.95,
        "top_k": 50,
        "do_sample": True,  
        "pad_token_id": 0,  
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
