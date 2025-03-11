"""Model configuration settings."""

MODEL_CONFIG = {
    'language_model': {
        'name': 'HuggingFaceH4/zephyr-7b-beta',  # Updated to Zephyr-7B
        'max_length': 512,                      # Max tokens for the model
        'temperature': 0.4,                     # Sampling temperature (higher for more creativity)
        'top_p': 0.9,                           # Top-p sampling for nucleus sampling
        'top_k': 50,                            # Top-k for sampling to limit the choices
        'repetition_penalty': 1.15             # Prevent repetition in responses
    },
    'embedding': {
        'name': 'sentence-transformers/all-mpnet-base-v2',  # Embedding model for vector search
        'max_length': 256                         # Max tokens for embeddings (shorter context)
    },
    'retrieval': {
        'top_k': 3,                              # Number of relevant responses to retrieve from DB
        'similarity_threshold': 0.75             # Threshold for relevance when retrieving responses
    },
    'response': {
        'min_confidence': 0.5,                   # Minimum confidence score for response selection
        'max_length': 150                        # Max tokens for generated responses
    }
}
