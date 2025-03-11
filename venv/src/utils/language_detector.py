from langdetect import detect, DetectorFactory
DetectorFactory.seed = 0  # For consistent results

class LanguageDetector:
    @staticmethod
    def detect_language(text: str) -> str:
        """Detect language of input text"""
        try:
            return detect(text)
        except:
            return "en"  # Default to English 