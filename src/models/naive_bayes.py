"""Naive Bayes classifier for text classification."""

import numpy as np
from sklearn.naive_bayes import MultinomialNB

from .base_model import BaseModel


class NaiveBayesClassifier(BaseModel):
    """Naive Bayes classifier using sklearn's MultinomialNB."""

    def __init__(self, alpha: float = 1.0):
        super().__init__(name="NaiveBayes")
        self.alpha = alpha
        self.model = MultinomialNB(alpha=alpha)

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """Train the Naive Bayes model."""
        self.model.fit(X, y)
        self.is_fitted = True

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions."""
        return self.model.predict(X)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Return prediction probabilities."""
        return self.model.predict_proba(X)
