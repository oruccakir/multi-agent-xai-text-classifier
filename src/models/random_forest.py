"""Random Forest classifier for text classification."""

import numpy as np
from sklearn.ensemble import RandomForestClassifier as SklearnRF

from .base_model import BaseModel


class RandomForestClassifier(BaseModel):
    """Random Forest classifier using sklearn."""

    def __init__(self, n_estimators: int = 100, max_depth: int = None, random_state: int = 42):
        super().__init__(name="RandomForest")
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.model = SklearnRF(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=random_state,
        )

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """Train the Random Forest model."""
        self.model.fit(X, y)
        self.is_fitted = True

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions."""
        return self.model.predict(X)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Return prediction probabilities."""
        return self.model.predict_proba(X)
