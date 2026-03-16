"""Decision Tree classifier for text classification."""

import numpy as np
from sklearn.tree import DecisionTreeClassifier as SklearnDT

from .base_model import BaseModel


class DecisionTreeClassifier(BaseModel):
    """
    Decision Tree classifier using sklearn.

    Best for: Interpretable models, understanding decision logic
    Pros: Fully interpretable, fast inference, no feature scaling needed
    Cons: Prone to overfitting, unstable (high variance)
    """

    def __init__(
        self,
        max_depth: int = None,
        min_samples_split: int = 2,
        min_samples_leaf: int = 1,
        criterion: str = "gini",
        random_state: int = 42,
    ):
        super().__init__(name="DecisionTree")
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.criterion = criterion
        self.random_state = random_state
        self.model = SklearnDT(
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            criterion=criterion,
            random_state=random_state,
        )

    def fit(self, X: np.ndarray, y: np.ndarray) -> "DecisionTreeClassifier":
        """Train the Decision Tree model."""
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
