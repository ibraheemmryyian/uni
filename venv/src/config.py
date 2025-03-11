"""Configuration settings for the Customer Support AI system."""

MODEL_CONFIG = {
    "language_model": {
        "name": "facebook/opt-2.7b",
        "max_length": 200,
        "temperature": 0.7,
        "top_p": 0.95,
        "top_k": 50
    },
    "embedding_model": {
        "name": "all-mpnet-base-v2"
    },
    "intent_classifier": {
        "name": "facebook/bart-large-mnli",
        "labels": ["question", "complaint", "request", "greeting"]
    },
    "sentiment_analyzer": {
        "name": "distilbert-base-uncased-finetuned-sst-2-english"
    }
}