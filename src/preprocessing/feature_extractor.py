"""Feature extraction methods: TF-IDF and Transformer embeddings."""

import pickle
from pathlib import Path
from typing import List
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer


class FeatureExtractor:
    """
    Feature extractor supporting TF-IDF and Transformer embeddings.

    Methods:
    - TF-IDF: Statistical word frequency representation
    - Sentence-BERT: Dense semantic embeddings
    """

    def __init__(self, method: str = "tfidf", **kwargs):
        """
        Initialize feature extractor.

        Args:
            method: 'tfidf' or 'transformer'
            **kwargs: Additional arguments for the vectorizer
        """
        self.method = method
        self.vectorizer = None
        self.transformer_model = None
        self._initialize(kwargs)

    def _initialize(self, kwargs: dict) -> None:
        """Initialize the appropriate vectorizer."""
        if self.method == "tfidf":
            self.vectorizer = TfidfVectorizer(
                max_features=kwargs.get("max_features", 10000),
                ngram_range=kwargs.get("ngram_range", (1, 2)),
                min_df=kwargs.get("min_df", 2),
                max_df=kwargs.get("max_df", 0.95),
                sublinear_tf=kwargs.get("sublinear_tf", True),
            )
        elif self.method == "transformer":
            # Will be initialized lazily when needed
            self.model_name = kwargs.get("model_name", "sentence-transformers/all-MiniLM-L6-v2")
        else:
            raise ValueError(f"Unknown method: {self.method}")

    def fit(self, texts: List[str]) -> "FeatureExtractor":
        """Fit the vectorizer on the given texts."""
        if self.method == "tfidf":
            self.vectorizer.fit(texts)
        elif self.method == "transformer":
            self._load_transformer_model()
        return self

    def transform(self, texts: List[str], sparse: bool = True):
        """
        Transform texts to feature vectors.

        Args:
            texts: List of texts to transform
            sparse: If True, return sparse matrix (memory efficient for large datasets).
                    If False, return dense numpy array.

        Returns:
            Sparse matrix or dense numpy array of features
        """
        if self.method == "tfidf":
            features = self.vectorizer.transform(texts)
            if sparse:
                return features  # Keep as sparse matrix
            return features.toarray()  # Convert to dense only when requested
        elif self.method == "transformer":
            return self._encode_with_transformer(texts)

    def fit_transform(self, texts: List[str], sparse: bool = True):
        """
        Fit and transform in one step.

        Args:
            texts: List of texts to fit and transform
            sparse: If True, return sparse matrix (memory efficient)

        Returns:
            Sparse matrix or dense numpy array of features
        """
        self.fit(texts)
        return self.transform(texts, sparse=sparse)

    def _load_transformer_model(self) -> None:
        """Load the Sentence-BERT model."""
        # TODO: Implement with sentence-transformers
        # from sentence_transformers import SentenceTransformer
        # self.transformer_model = SentenceTransformer(self.model_name)
        raise NotImplementedError("Transformer model loading not yet implemented")

    def _encode_with_transformer(self, texts: List[str]) -> np.ndarray:
        """Encode texts using transformer model."""
        if self.transformer_model is None:
            self._load_transformer_model()
        # TODO: Implement encoding
        # return self.transformer_model.encode(texts)
        raise NotImplementedError("Transformer encoding not yet implemented")

    def get_feature_names(self) -> List[str]:
        """Get feature names (only for TF-IDF)."""
        if self.method == "tfidf" and self.vectorizer is not None:
            return self.vectorizer.get_feature_names_out().tolist()
        return []

    def get_vocabulary_size(self) -> int:
        """Get the size of the vocabulary."""
        if self.method == "tfidf" and self.vectorizer is not None:
            return len(self.vectorizer.vocabulary_)
        return 0

    def save(self, path: str) -> None:
        """Save the feature extractor to disk."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        with open(path, "wb") as f:
            pickle.dump(self, f)
        print(f"Feature extractor saved to {path}")

    @classmethod
    def load(cls, path: str) -> "FeatureExtractor":
        """Load a feature extractor from disk."""
        with open(path, "rb") as f:
            extractor = pickle.load(f)
        print(f"Feature extractor loaded from {path}")
        return extractor
