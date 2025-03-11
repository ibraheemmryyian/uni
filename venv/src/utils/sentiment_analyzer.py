from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from src.logger import setup_logger  # Adjust the import as needed

class SentimentAnalyzer:
    def __init__(self, log_file=r'C:\Users\amrey\Desktop\bolt multi file\venv\logs\logfile.log'):
        """Initialize the sentiment analyzer with logging."""
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
