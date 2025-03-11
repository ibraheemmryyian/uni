from typing import Dict, Any
import re
from textblob import TextBlob

class QualityControl:
    def __init__(self):
        self.profanity_list = set(['word1', 'word2'])  # Add your profanity list
        self.required_patterns = {
            'greeting': r'(hello|hi|hey)',
            'politeness': r'(please|thank|sorry)',
            'contact': r'(email|phone|contact)'
        }

    def check_response(self, response: str) -> Dict[str, Any]:
        """Check response quality"""
        quality_metrics = {
            "profanity_free": self._check_profanity(response),
            "grammar_score": self._check_grammar(response),
            "required_elements": self._check_required_elements(response),
            "sentiment_score": self._analyze_sentiment(response)
        }
        return quality_metrics

    def _check_profanity(self, text: str) -> bool:
        """Check for profanity"""
        words = set(text.lower().split())
        return not bool(words.intersection(self.profanity_list))

    def _check_grammar(self, text: str) -> float:
        """Check grammar quality"""
        blob = TextBlob(text)
        return blob.correct()

    def _check_required_elements(self, text: str) -> Dict[str, bool]:
        """Check for required response elements"""
        results = {}
        for name, pattern in self.required_patterns.items():
            results[name] = bool(re.search(pattern, text.lower()))
        return results 