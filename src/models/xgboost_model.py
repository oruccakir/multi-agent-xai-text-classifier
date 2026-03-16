"""XGBoost classifier for text classification."""

import numpy as np
from xgboost import XGBClassifier
from sklearn.preprocessing import LabelEncoder

from .base_model import BaseModel


class XGBoostClassifier(BaseModel):
    """
    XGBoost gradient boosting classifier.

    Best for: Structured/tabular features, competitions, high accuracy
    Pros: Often best-in-class accuracy, handles missing values, feature importance
    Cons: Slower to train than linear models, many hyperparameters
    """

    def __init__(
        self,
        n_estimators: int = 100,
        max_depth: int = 6,
        learning_rate: float = 0.1,
        subsample: float = 0.8,
        colsample_bytree: float = 0.8,
        n_jobs: int = -1,
        random_state: int = 42,
    ):
        super().__init__(name="XGBoost")
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.subsample = subsample
        self.colsample_bytree = colsample_bytree
        self.n_jobs = n_jobs
        self.random_state = random_state
        self._label_encoder = LabelEncoder()
        self.model = XGBClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            subsample=subsample,
            colsample_bytree=colsample_bytree,
            n_jobs=n_jobs,
            random_state=random_state,
            eval_metric="mlogloss",
            verbosity=0,
        )

    def fit(self, X: np.ndarray, y: np.ndarray) -> "XGBoostClassifier":
        """Train the XGBoost model."""
        y_encoded = self._label_encoder.fit_transform(y)
        self.model.fit(X, y_encoded)
        self.is_fitted = True
        self.classes_ = self._label_encoder.classes_
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions."""
        y_encoded = self.model.predict(X)
        return self._label_encoder.inverse_transform(y_encoded)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Return prediction probabilities."""
        return self.model.predict_proba(X)

    def get_feature_importance(self) -> np.ndarray:
        """Get feature importance scores."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted first!")
        return self.model.feature_importances_
