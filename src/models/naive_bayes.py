"""Naive Bayes classifier for text classification."""

import numpy as np
from sklearn.naive_bayes import MultinomialNB

from .base_model import BaseModel


class NaiveBayesClassifier(BaseModel):
    """
    Naive Bayes classifier using sklearn's MultinomialNB.

    Best for: Text classification with TF-IDF features
    Pros: Fast training, handles high-dimensional sparse data well
    Cons: Assumes feature independence
    """

    def __init__(self, alpha: float = 1.0):
        """
        Initialize Naive Bayes classifier.

        Args:
            alpha: Additive (Laplace) smoothing parameter (0 for no smoothing)
        """
        super().__init__(name="NaiveBayes")
        self.alpha = alpha
        self.model = MultinomialNB(alpha=alpha)

    def fit(self, X: np.ndarray, y: np.ndarray) -> "NaiveBayesClassifier":
        """Train the Naive Bayes model."""
        self.model.fit(X, y)
        self.is_fitted = True
        self.classes_ = self.model.classes_
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions."""
        return self.model.predict(X)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Return prediction probabilities."""
        return self.model.predict_proba(X)
