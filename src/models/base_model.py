"""Base model class for all classification models."""

import pickle
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, List
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)


class BaseModel(ABC):
    """Abstract base class for all classification models."""

    def __init__(self, name: str):
        self.name = name
        self.model = None
        self.is_fitted = False
        self.classes_ = None

    @abstractmethod
    def fit(self, X: np.ndarray, y: np.ndarray) -> "BaseModel":
        """Train the model on the given data."""
        pass

    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions on the given data."""
        pass

    @abstractmethod
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Return prediction probabilities."""
        pass

    def evaluate(self, X: np.ndarray, y: np.ndarray) -> Dict[str, Any]:
        """
        Evaluate the model on test data.

        Args:
            X: Feature matrix
            y: True labels

        Returns:
            Dictionary containing evaluation metrics
        """
        y_pred = self.predict(X)

        # Calculate metrics
        results = {
            "model_name": self.name,
            "accuracy": accuracy_score(y, y_pred),
            "f1_macro": f1_score(y, y_pred, average="macro"),
            "f1_weighted": f1_score(y, y_pred, average="weighted"),
            "precision": precision_score(y, y_pred, average="weighted", zero_division=0),
            "recall": recall_score(y, y_pred, average="weighted", zero_division=0),
            "confusion_matrix": confusion_matrix(y, y_pred),
            "classification_report": classification_report(y, y_pred, output_dict=True, zero_division=0),
        }

        return results

    def save(self, path: str) -> None:
        """Save model to disk."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        with open(path, "wb") as f:
            pickle.dump(self, f)
        print(f"Model saved to {path}")

    @classmethod
    def load(cls, path: str) -> "BaseModel":
        """Load model from disk."""
        with open(path, "rb") as f:
            model = pickle.load(f)
        print(f"Model loaded from {path}")
        return model

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name={self.name}, fitted={self.is_fitted})"
