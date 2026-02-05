"""Intent Classifier Agent - Routes input to appropriate model based on context using Gemini."""

import os
from typing import Any, Optional
from enum import Enum

from dotenv import load_dotenv
from pydantic import BaseModel, Field

from .base_agent import BaseAgent

# Load environment variables
load_dotenv()


class Language(str, Enum):
    """Supported languages."""
    ENGLISH = "english"
    TURKISH = "turkish"


class Domain(str, Enum):
    """Supported domains/tasks."""
    SENTIMENT = "sentiment"
    NEWS = "news"


class Dataset(str, Enum):
    """Available datasets."""
    IMDB = "imdb"
    TURKISH_SENTIMENT = "turkish_sentiment"
    AG_NEWS = "ag_news"
    TURKISH_NEWS = "turkish_news"


class IntentResult(BaseModel):
    """Structured output for intent classification."""
    language: Language = Field(description="Detected language of the text")
    domain: Domain = Field(description="Detected domain/task type of the text")
    dataset: Dataset = Field(description="Recommended dataset to use for classification")
    confidence: float = Field(description="Confidence score between 0 and 1")
    reasoning: str = Field(description="Brief explanation of why this classification was chosen")


class IntentClassifierAgent(BaseAgent):
    """
    Agent responsible for analyzing input text and determining
    which classification model/dataset to use.

    Uses Google Gemini with structured output for accurate detection.
    Detects: language (English/Turkish), domain (sentiment/news), and recommends dataset.
    """

    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize Intent Classifier Agent.

        Args:
            api_key: Google Gemini API key. If not provided, will look for GEMINI_API_KEY env var.
        """
        super().__init__(name="IntentClassifier")
        self.api_key = api_key or os.getenv("GEMINI_API_KEY")
        self.client = None
        self._initialized = False

    def _initialize_gemini(self) -> bool:
        """Initialize Gemini client lazily."""
        if self._initialized:
            return True

        if not self.api_key:
            return False

        try:
            from google import genai

            self.client = genai.Client(api_key=self.api_key)
            self._initialized = True
            return True
        except Exception as e:
            print(f"Failed to initialize Gemini: {e}")
            return False

    def process(self, input_data: Any) -> dict:
        """
        Analyze input text and determine the appropriate model to use.

        Args:
            input_data: Raw text input from user (str) or dict with 'text' key

        Returns:
            dict with 'language', 'domain', 'dataset', 'confidence', 'reasoning', 'gemini_available'
        """
        # Handle both string and dict input
        if isinstance(input_data, str):
            text = input_data
        elif isinstance(input_data, dict):
            text = input_data.get("text", "")
        else:
            text = str(input_data)

        # Try Gemini first
        gemini_available = self._initialize_gemini()

        if gemini_available:
            result = self._detect_with_gemini(text)
            if result:
                result["gemini_available"] = True
                return result

        # Fallback to static detection
        fallback_result = self._detect_static(text)
        fallback_result["gemini_available"] = False
        return fallback_result

    def _detect_with_gemini(self, text: str) -> Optional[dict]:
        """Detect intent using Gemini with structured output."""
        if not self.client:
            return None

        prompt = f"""Analyze the following text and determine:
1. The language (English or Turkish)
2. The domain/type (sentiment analysis for reviews, or news classification)
3. The most appropriate dataset to use

Available datasets:
- imdb: English movie reviews (sentiment: positive/negative)
- turkish_sentiment: Turkish product/service reviews (sentiment: positive/neutral/negative)
- ag_news: English news articles (categories: World, Sports, Business, Sci/Tech)
- turkish_news: Turkish news articles (categories: politics, world, economy, culture, health, sports, technology)

Text to analyze:
"{text[:1000]}"

Provide your analysis with confidence score (0.0 to 1.0) and brief reasoning."""

        try:
            response = self.client.models.generate_content(
                model="gemini-2.0-flash",
                contents=prompt,
                config={
                    "response_mime_type": "application/json",
                    "response_schema": IntentResult,
                }
            )

            # Parse the structured response
            import json
            result_data = json.loads(response.text)

            return {
                "language": result_data.get("language", "english"),
                "domain": result_data.get("domain", "sentiment"),
                "dataset": result_data.get("dataset", "imdb"),
                "confidence": result_data.get("confidence", 0.5),
                "reasoning": result_data.get("reasoning", ""),
            }

        except Exception as e:
            print(f"Gemini intent detection error: {e}")
            return None

    def _detect_static(self, text: str) -> dict:
        """Fallback static detection based on keywords and characters."""
        # Detect language
        language = self._detect_language_static(text)

        # Detect domain
        domain, dataset = self._detect_domain_static(text, language)

        return {
            "language": language,
            "domain": domain,
            "dataset": dataset,
            "confidence": 0.7,  # Lower confidence for static detection
            "reasoning": "Detected using keyword-based rules (Gemini unavailable)",
        }

    def _detect_language_static(self, text: str) -> str:
        """Static language detection based on character analysis."""
        turkish_chars = set("çğıöşüÇĞİÖŞÜ")
        text_chars = set(text)

        if turkish_chars & text_chars:
            return "turkish"

        # Check for common Turkish words
        turkish_words = {"ve", "bir", "bu", "için", "ile", "çok", "ama", "değil", "var", "olan",
                         "daha", "gibi", "kadar", "sonra", "önce", "olarak", "şu", "her", "ne", "de", "da"}
        words = set(text.lower().split())

        if len(turkish_words & words) >= 2:
            return "turkish"

        return "english"

    def _detect_domain_static(self, text: str, language: str) -> tuple:
        """Static domain detection based on keywords."""
        text_lower = text.lower()

        if language == "english":
            # Check for movie review indicators
            movie_words = {"movie", "film", "actor", "actress", "director", "scene", "plot",
                          "character", "cinema", "watch", "watching", "watched", "acting",
                          "screenplay", "cast", "performance", "ending", "story"}
            if any(word in text_lower for word in movie_words):
                return "sentiment", "imdb"

            # Check for news indicators
            news_words = {"reuters", "ap", "announced", "reported", "official", "government",
                         "company", "president", "minister", "according", "statement",
                         "market", "economy", "stock", "shares", "million", "billion"}
            if any(word in text_lower for word in news_words):
                return "news", "ag_news"

            # Default to sentiment for English
            return "sentiment", "imdb"

        else:  # Turkish
            # Check for product review indicators
            product_words = {"ürün", "kargo", "teslimat", "sipariş", "satıcı", "fiyat",
                            "kalite", "memnun", "tavsiye", "aldım", "kullanıyorum",
                            "güzel", "kötü", "harika", "berbat", "idare"}
            if any(word in text_lower for word in product_words):
                return "sentiment", "turkish_sentiment"

            # Check for news indicators
            news_words = {"haber", "açıkladı", "dedi", "bildirdi", "başbakan", "cumhurbaşkanı",
                         "meclis", "bakan", "hükümet", "parti", "seçim", "ekonomi", "piyasa",
                         "dolar", "euro", "borsa", "spor", "maç", "takım", "lig"}
            if any(word in text_lower for word in news_words):
                return "news", "turkish_news"

            # Default to sentiment for Turkish
            return "sentiment", "turkish_sentiment"

    def detect_language(self, text: str) -> str:
        """Detect if text is Turkish or English."""
        result = self.process(text)
        return result["language"]

    def detect_domain(self, text: str) -> str:
        """Detect the domain/context of the text (sentiment, news, etc.)."""
        result = self.process(text)
        return result["domain"]

    def get_recommended_dataset(self, text: str) -> str:
        """Get the recommended dataset for the given text."""
        result = self.process(text)
        return result["dataset"]
