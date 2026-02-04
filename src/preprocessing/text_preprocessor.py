"""Text preprocessing utilities for cleaning and normalizing text."""

import re
import string
from typing import List

# Turkish stopwords
TURKISH_STOPWORDS = {
    "acaba", "ama", "aslında", "az", "bazı", "belki", "beri", "bile", "bir", "birçok",
    "biri", "birkaç", "birkez", "birşey", "birşeyi", "biz", "bize", "bizden", "bizi",
    "bizim", "bu", "buna", "bunda", "bundan", "bunlar", "bunları", "bunların", "bunu",
    "bunun", "burada", "çok", "çünkü", "da", "daha", "dahi", "de", "defa", "değil",
    "diğer", "diye", "dolayı", "dolayısıyla", "için", "ile", "ilgili", "ise", "işte",
    "kadar", "karşın", "kendi", "kendine", "kendini", "ki", "kim", "kime", "kimi",
    "kimse", "madem", "mı", "mi", "mu", "mü", "nasıl", "ne", "neden", "nedenle",
    "nerde", "nerede", "nereye", "niye", "o", "olan", "olmak", "olması", "olmayan",
    "olsa", "olsun", "olup", "olur", "olursa", "oluyor", "ona", "ondan", "onlar",
    "onlara", "onlardan", "onları", "onların", "onu", "onun", "orada", "öyle", "pek",
    "rağmen", "sadece", "sanki", "şayet", "şey", "şeyi", "şeyler", "şimdi", "şöyle",
    "şu", "şuna", "şunda", "şundan", "şunu", "tabi", "tamam", "tüm", "tümü", "üzere",
    "var", "vardı", "ve", "veya", "ya", "yani", "yapacak", "yapılan", "yapılması",
    "yapıyor", "yapmak", "yaptı", "yaptığı", "yaptığını", "yaptıkları", "yine", "yoksa",
    "zaten", "zira",
}

# English stopwords
ENGLISH_STOPWORDS = {
    "a", "about", "above", "after", "again", "against", "all", "am", "an", "and",
    "any", "are", "aren't", "as", "at", "be", "because", "been", "before", "being",
    "below", "between", "both", "but", "by", "can't", "cannot", "could", "couldn't",
    "did", "didn't", "do", "does", "doesn't", "doing", "don't", "down", "during",
    "each", "few", "for", "from", "further", "had", "hadn't", "has", "hasn't", "have",
    "haven't", "having", "he", "he'd", "he'll", "he's", "her", "here", "here's", "hers",
    "herself", "him", "himself", "his", "how", "how's", "i", "i'd", "i'll", "i'm",
    "i've", "if", "in", "into", "is", "isn't", "it", "it's", "its", "itself", "let's",
    "me", "more", "most", "mustn't", "my", "myself", "no", "nor", "not", "of", "off",
    "on", "once", "only", "or", "other", "ought", "our", "ours", "ourselves", "out",
    "over", "own", "same", "shan't", "she", "she'd", "she'll", "she's", "should",
    "shouldn't", "so", "some", "such", "than", "that", "that's", "the", "their",
    "theirs", "them", "themselves", "then", "there", "there's", "these", "they",
    "they'd", "they'll", "they're", "they've", "this", "those", "through", "to", "too",
    "under", "until", "up", "very", "was", "wasn't", "we", "we'd", "we'll", "we're",
    "we've", "were", "weren't", "what", "what's", "when", "when's", "where", "where's",
    "which", "while", "who", "who's", "whom", "why", "why's", "with", "won't", "would",
    "wouldn't", "you", "you'd", "you'll", "you're", "you've", "your", "yours",
    "yourself", "yourselves",
}


class TextPreprocessor:
    """
    Text preprocessor for cleaning and normalizing text data.

    Steps:
    - HTML tag removal
    - URL removal
    - Lowercase conversion
    - Punctuation removal
    - Number removal (optional)
    - Stopwords filtering
    """

    def __init__(
        self,
        language: str = "english",
        remove_stopwords: bool = True,
        remove_punctuation: bool = True,
        remove_numbers: bool = False,
        remove_urls: bool = True,
        remove_html: bool = True,
        lowercase: bool = True,
        min_word_length: int = 2,
    ):
        self.language = language.lower()
        self.remove_stopwords_flag = remove_stopwords
        self.remove_punctuation_flag = remove_punctuation
        self.remove_numbers_flag = remove_numbers
        self.remove_urls_flag = remove_urls
        self.remove_html_flag = remove_html
        self.lowercase_flag = lowercase
        self.min_word_length = min_word_length
        self.stopwords = self._load_stopwords()

    def _load_stopwords(self) -> set:
        """Load stopwords for the specified language."""
        if self.language == "turkish":
            return TURKISH_STOPWORDS
        elif self.language == "english":
            return ENGLISH_STOPWORDS
        else:
            return set()

    def preprocess(self, text: str) -> str:
        """Apply all preprocessing steps to text."""
        if not isinstance(text, str):
            text = str(text)

        # Remove HTML tags first
        if self.remove_html_flag:
            text = self._remove_html_tags(text)

        # Remove URLs
        if self.remove_urls_flag:
            text = self._remove_urls(text)

        # Lowercase
        if self.lowercase_flag:
            text = text.lower()

        # Remove punctuation
        if self.remove_punctuation_flag:
            text = self._remove_punctuation(text)

        # Remove numbers
        if self.remove_numbers_flag:
            text = self._remove_numbers(text)

        # Tokenize and filter
        words = text.split()

        # Remove stopwords
        if self.remove_stopwords_flag:
            words = [w for w in words if w not in self.stopwords]

        # Filter by minimum word length
        if self.min_word_length > 0:
            words = [w for w in words if len(w) >= self.min_word_length]

        # Join back and remove extra whitespace
        text = " ".join(words)

        return text

    def preprocess_batch(self, texts: List[str], show_progress: bool = False) -> List[str]:
        """Preprocess a batch of texts."""
        if show_progress:
            try:
                from tqdm import tqdm
                return [self.preprocess(text) for text in tqdm(texts, desc="Preprocessing")]
            except ImportError:
                pass
        return [self.preprocess(text) for text in texts]

    def _remove_punctuation(self, text: str) -> str:
        """Remove punctuation from text, preserving Turkish characters."""
        # Keep Turkish special characters: ç, ğ, ı, ö, ş, ü, Ç, Ğ, İ, Ö, Ş, Ü
        # Remove standard punctuation
        translator = str.maketrans("", "", string.punctuation)
        return text.translate(translator)

    def _remove_stopwords(self, text: str) -> str:
        """Remove stopwords from text."""
        words = text.split()
        filtered_words = [w for w in words if w not in self.stopwords]
        return " ".join(filtered_words)

    def _remove_urls(self, text: str) -> str:
        """Remove URLs from text."""
        # Match http, https, www URLs
        url_pattern = re.compile(r"https?://\S+|www\.\S+")
        return url_pattern.sub(" ", text)

    def _remove_html_tags(self, text: str) -> str:
        """Remove HTML tags from text."""
        # Remove HTML tags
        html_pattern = re.compile(r"<[^>]+>")
        text = html_pattern.sub(" ", text)
        # Also handle HTML entities
        text = re.sub(r"&[a-zA-Z]+;", " ", text)
        return text

    def _remove_numbers(self, text: str) -> str:
        """Remove numbers from text."""
        return re.sub(r"\d+", " ", text)

    def get_config(self) -> dict:
        """Return current preprocessing configuration."""
        return {
            "language": self.language,
            "remove_stopwords": self.remove_stopwords_flag,
            "remove_punctuation": self.remove_punctuation_flag,
            "remove_numbers": self.remove_numbers_flag,
            "remove_urls": self.remove_urls_flag,
            "remove_html": self.remove_html_flag,
            "lowercase": self.lowercase_flag,
            "min_word_length": self.min_word_length,
            "num_stopwords": len(self.stopwords),
        }
