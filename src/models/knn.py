"""K-Nearest Neighbors classifier for text classification."""

import numpy as np
from sklearn.neighbors import KNeighborsClassifier

from .base_model import BaseModel


class KNNClassifier(BaseModel):
    """KNN classifier using sklearn."""

    def __init__(self, n_neighbors: int = 5, metric: str = "cosine"):
        super().__init__(name="KNN")
        self.n_neighbors = n_neighbors
        self.metric = metric
        self.model = KNeighborsClassifier(n_neighbors=n_neighbors, metric=metric)

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """Train the KNN model."""
        self.model.fit(X, y)
        self.is_fitted = True

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions."""
        return self.model.predict(X)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Return prediction probabilities."""
        return self.model.predict_proba(X)
