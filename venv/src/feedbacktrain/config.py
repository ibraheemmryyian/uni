"""Training configuration settings."""

TRAINING_CONFIG = {
    'model': {
        'name': 'facebook/opt-2.7b',  # Model to fine-tune
        'max_length': 512,  # Length of input sequences
        # These parameters are usually used for generation tasks, but for regression, we likely do not need them
        # Removing 'temperature', 'top_p', 'top_k', 'repetition_penalty' as they are not applicable for regression
    },
    'training': {
        'batch_size': 8,  # You can adjust depending on your hardware capabilities
        'learning_rate': 2e-5,  # Learning rate for fine-tuning
        'epochs': 10,  # Number of epochs to train for
        'warmup_steps': 100,  # Steps for learning rate warm-up
        'max_grad_norm': 1.0,  # Gradient clipping to avoid explosion
        'early_stopping_patience': 3,  # Early stopping to prevent overfitting
        'validation_split': 0.1,  # Fraction of data to be used for validation
        'accumulation_steps': 4,  # Gradient accumulation steps for stability (useful with large models)
        'weight_decay': 0.01,  # Regularization to avoid overfitting
    },
    'evaluation': {
        # For regression, we use metrics like MAE, RMSE, and MSE
        'metrics': ['mae', 'rmse', 'mse'],  # Evaluating using Mean Absolute Error, Root Mean Squared Error, and Mean Squared Error
        'threshold': 0.7  # Optional confidence threshold if needed (can be ignored in regression tasks)
    }
}
