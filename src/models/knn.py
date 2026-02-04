"""K-Nearest Neighbors classifier for text classification."""

import numpy as np
from sklearn.neighbors import KNeighborsClassifier

from .base_model import BaseModel


class KNNClassifier(BaseModel):
    """
    K-Nearest Neighbors classifier using sklearn.

    Best for: When you want instance-based learning
    Pros: Simple, no training phase, naturally handles multi-class
    Cons: Slow at prediction time, sensitive to irrelevant features
    """

    def __init__(
        self,
        n_neighbors: int = 5,
        metric: str = "cosine",
        weights: str = "distance",
        n_jobs: int = -1,
    ):
        """
        Initialize KNN classifier.

        Args:
            n_neighbors: Number of neighbors to use
            metric: Distance metric ('cosine', 'euclidean', 'manhattan')
            weights: Weight function ('uniform', 'distance')
            n_jobs: Number of parallel jobs (-1 for all cores)
        """
        super().__init__(name="KNN")
        self.n_neighbors = n_neighbors
        self.metric = metric
        self.model = KNeighborsClassifier(
            n_neighbors=n_neighbors,
            metric=metric,
            weights=weights,
            n_jobs=n_jobs,
        )

    def fit(self, X: np.ndarray, y: np.ndarray) -> "KNNClassifier":
        """Train the KNN model (stores training data)."""
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
