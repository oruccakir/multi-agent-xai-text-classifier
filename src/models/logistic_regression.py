"""Logistic Regression classifier for text classification."""

import numpy as np
from sklearn.linear_model import LogisticRegression as SklearnLR

from .base_model import BaseModel


class LogisticRegressionClassifier(BaseModel):
    """
    Logistic Regression classifier using sklearn.

    Best for: When you need interpretable linear model with probabilities
    Pros: Fast, interpretable coefficients, works well with sparse data
    Cons: Assumes linear decision boundary
    """

    def __init__(
        self,
        C: float = 1.0,
        max_iter: int = 1000,
        solver: str = "lbfgs",
        random_state: int = 42,
    ):
        """
        Initialize Logistic Regression classifier.

        Args:
            C: Inverse of regularization strength (smaller = stronger)
            max_iter: Maximum number of iterations
            solver: Algorithm to use ('lbfgs', 'liblinear', 'saga')
            random_state: Random seed for reproducibility
        """
        super().__init__(name="LogisticRegression")
        self.C = C
        self.max_iter = max_iter
        self.model = SklearnLR(
            C=C,
            max_iter=max_iter,
            solver=solver,
            random_state=random_state,
        )

    def fit(self, X: np.ndarray, y: np.ndarray) -> "LogisticRegressionClassifier":
        """Train the Logistic Regression model."""
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

    def get_coefficients(self) -> np.ndarray:
        """Get model coefficients (feature weights)."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted first!")
        return self.model.coef_
