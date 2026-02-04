"""Random Forest classifier for text classification."""

import numpy as np
from sklearn.ensemble import RandomForestClassifier as SklearnRF

from .base_model import BaseModel


class RandomForestClassifier(BaseModel):
    """
    Random Forest classifier using sklearn.

    Best for: When you need robust predictions with feature importance
    Pros: Handles non-linear relationships, provides feature importance
    Cons: Can be slow and memory-intensive with many trees
    """

    def __init__(
        self,
        n_estimators: int = 100,
        max_depth: int = None,
        min_samples_split: int = 2,
        n_jobs: int = -1,
        random_state: int = 42,
    ):
        """
        Initialize Random Forest classifier.

        Args:
            n_estimators: Number of trees in the forest
            max_depth: Maximum depth of trees (None for unlimited)
            min_samples_split: Minimum samples required to split a node
            n_jobs: Number of parallel jobs (-1 for all cores)
            random_state: Random seed for reproducibility
        """
        super().__init__(name="RandomForest")
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.model = SklearnRF(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            n_jobs=n_jobs,
            random_state=random_state,
        )

    def fit(self, X: np.ndarray, y: np.ndarray) -> "RandomForestClassifier":
        """Train the Random Forest model."""
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

    def get_feature_importance(self) -> np.ndarray:
        """Get feature importance scores."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted first!")
        return self.model.feature_importances_
