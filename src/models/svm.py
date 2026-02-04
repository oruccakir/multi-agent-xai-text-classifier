"""Support Vector Machine classifier for text classification."""

import numpy as np
from sklearn.svm import SVC

from .base_model import BaseModel


class SVMClassifier(BaseModel):
    """SVM classifier using sklearn's SVC."""

    def __init__(self, kernel: str = "rbf", C: float = 1.0, probability: bool = True):
        super().__init__(name="SVM")
        self.kernel = kernel
        self.C = C
        self.model = SVC(kernel=kernel, C=C, probability=probability)

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """Train the SVM model."""
        self.model.fit(X, y)
        self.is_fitted = True

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions."""
        return self.model.predict(X)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Return prediction probabilities."""
        return self.model.predict_proba(X)
