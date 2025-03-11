import torch
from .language_model import LanguageModel
from .embedding_model import EmbeddingModel
from src.utils.sentiment_analyzer import SentimentAnalyzer  # Correct import
from src.logger import setup_logger
from src.utils.exceptions import ModelInitializationError

class ModelManager:
    def __init__(self, log_file):
        """Ensure that log_file is passed correctly for logging setup."""
        self.logger = setup_logger(__name__, log_file)
        
        # Automatically set the device to GPU if available, else fallback to CPU
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.logger.info(f"Device set to use {self.device}")

    def initialize_models(self):
        """Initialize all AI models."""
        try:
            self.logger.info("Initializing models...")
            models = {}

            # Initialize each model and store them in the 'models' dictionary
            models["language_model"], models["language_tokenizer"] = self._initialize_language_model()
            models["embedding_model"] = self._initialize_embedding_model()
            models["intent_classifier"] = self._initialize_intent_classifier()
            models["sentiment_analyzer"] = self._initialize_sentiment_analyzer()
            
            self.logger.info("All models initialized successfully")
            return models
            
        except Exception as e:
            self.logger.error(f"Failed to initialize models: {str(e)}")
            raise ModelInitializationError(f"Failed to initialize models: {str(e)}")

    def _initialize_language_model(self):
        """Helper function to initialize the language model."""
        try:
            language_model = LanguageModel(self.device)
            model, tokenizer = language_model.initialize()  # Initialize separately in case of complex setup
            self.logger.info("Language model initialized successfully")
            return model, tokenizer
        except Exception as e:
            self.logger.error(f"Failed to initialize language model: {str(e)}")
            raise ModelInitializationError(f"Failed to initialize language model: {str(e)}")

    def _initialize_embedding_model(self):
        """Helper function to initialize the embedding model."""
        try:
            embedding_model = EmbeddingModel()
            embedding_model.initialize()  # Initialize separately
            self.logger.info("Embedding model initialized successfully")
            return embedding_model
        except Exception as e:
            self.logger.error(f"Failed to initialize embedding model: {str(e)}")
            raise ModelInitializationError(f"Failed to initialize embedding model: {str(e)}")
    
    def _initialize_intent_classifier(self):
        """Helper function to initialize the intent classifier."""
        try:
            # Moved the import inside this function to avoid circular import issues
            from .intent_classifier import IntentClassifier
            intent_classifier = IntentClassifier()
            intent_classifier.initialize()  # Initialize separately
            self.logger.info("Intent classifier initialized successfully")
            return intent_classifier
        except Exception as e:
            self.logger.error(f"Failed to initialize intent classifier: {str(e)}")
            raise ModelInitializationError(f"Failed to initialize intent classifier: {str(e)}")
    
    def _initialize_sentiment_analyzer(self):
        """Helper function to initialize the sentiment analyzer."""
        try:
            sentiment_analyzer = SentimentAnalyzer()
            sentiment_analyzer.initialize()  # Initialize separately
            self.logger.info("Sentiment analyzer initialized successfully")
            return sentiment_analyzer
        except Exception as e:
            self.logger.error(f"Failed to initialize sentiment analyzer: {str(e)}")
            raise ModelInitializationError(f"Failed to initialize sentiment analyzer: {str(e)}")