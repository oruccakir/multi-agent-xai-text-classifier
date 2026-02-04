"""Support Vector Machine classifier for text classification."""

import numpy as np
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV

from .base_model import BaseModel


class SVMClassifier(BaseModel):
    """
    SVM classifier using sklearn's LinearSVC with probability calibration.

    Best for: High-dimensional sparse data like TF-IDF
    Pros: Effective in high-dimensional spaces, memory efficient
    Cons: Slower on very large datasets
    """

    def __init__(self, C: float = 1.0, max_iter: int = 1000, random_state: int = 42):
        """
        Initialize SVM classifier.

        Args:
            C: Regularization parameter (smaller = stronger regularization)
            max_iter: Maximum number of iterations
            random_state: Random seed for reproducibility
        """
        super().__init__(name="SVM")
        self.C = C
        self.max_iter = max_iter
        self.random_state = random_state
        # Use LinearSVC for efficiency with high-dimensional data
        self._base_model = LinearSVC(
            C=C,
            max_iter=max_iter,
            random_state=random_state,
            dual="auto",
        )
        # Wrap with calibration for probability estimates
        self.model = CalibratedClassifierCV(self._base_model, cv=3)

    def fit(self, X: np.ndarray, y: np.ndarray) -> "SVMClassifier":
        """Train the SVM model."""
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
