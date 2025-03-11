"""Training configuration settings."""

TRAINING_CONFIG = {
    'model': {
        'name': 'facebook/opt-2.7b',  # Smaller but more focused model
        'max_length': 512,
        'temperature': 0.3,  # Lower temperature for more focused outputs
        'top_p': 0.85,
        'top_k': 30,
        'repetition_penalty': 1.2  # Prevent repetitive outputs
    },
    'training': {
        'batch_size': 8,
        'learning_rate': 2e-5,
        'epochs': 10,
        'warmup_steps': 100,
        'max_grad_norm': 1.0,
        'early_stopping_patience': 3,
        'validation_split': 0.1,
        'accumulation_steps': 4,  # Gradient accumulation for stability
        'weight_decay': 0.01
    },
    'evaluation': {
        'metrics': ['accuracy', 'f1', 'precision', 'recall'],
        'threshold': 0.7  # Confidence threshold for responses
    }
}