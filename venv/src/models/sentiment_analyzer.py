from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from src.logger import setup_logger  # Adjust the import as needed

class SentimentAnalyzer:
    def __init__(self, log_file="logs/sentiment_analysis.log"):  # Use relative path
        """Initialize the sentiment analyzer with logging."""
        from pathlib import Path
        Path(log_file).parent.mkdir(parents=True, exist_ok=True)  # Create logs directory if it doesn't exist
        self.logger = setup_logger(__name__, log_file)
        self.analyzer = SentimentIntensityAnalyzer()  # Initialize VADER sentiment analyzer

    def analyze_sentiment(self, text: str) -> dict:
        """Analyze the sentiment of the given text and return sentiment scores."""
        try:
            sentiment_score = self.analyzer.polarity_scores(text)
            self.logger.info(f"Sentiment analysis result: {sentiment_score}")
            return sentiment_score
        except Exception as e:
            self.logger.error(f"Error during sentiment analysis: {str(e)}")
            raise
