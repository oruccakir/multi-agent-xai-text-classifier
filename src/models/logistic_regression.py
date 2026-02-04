"""Logistic Regression classifier for text classification."""

import numpy as np
from sklearn.linear_model import LogisticRegression as SklearnLR

from .base_model import BaseModel


class LogisticRegressionClassifier(BaseModel):
    """Logistic Regression classifier using sklearn."""

    def __init__(self, C: float = 1.0, max_iter: int = 1000, random_state: int = 42):
        super().__init__(name="LogisticRegression")
        self.C = C
        self.max_iter = max_iter
        self.model = SklearnLR(C=C, max_iter=max_iter, random_state=random_state)

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """Train the Logistic Regression model."""
        self.model.fit(X, y)
        self.is_fitted = True

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions."""
        return self.model.predict(X)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Return prediction probabilities."""
        return self.model.predict_proba(X)
