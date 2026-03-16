"""Base model class for all classification models."""

import pickle
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, List
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    auc,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_curve,
)
from sklearn.preprocessing import label_binarize


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
        y_proba = self.predict_proba(X)
        classes = list(self.classes_)

        # Calculate metrics
        # precision and recall use macro averaging to be consistent with f1_macro
        results = {
            "model_name": self.name,
            "accuracy": accuracy_score(y, y_pred),
            "f1_macro": f1_score(y, y_pred, average="macro"),
            "f1_weighted": f1_score(y, y_pred, average="weighted"),
            "precision": precision_score(y, y_pred, average="macro", zero_division=0),
            "recall": recall_score(y, y_pred, average="macro", zero_division=0),
            "confusion_matrix": confusion_matrix(y, y_pred),
            "classification_report": classification_report(y, y_pred, output_dict=True, zero_division=0),
            "roc_curves": self._compute_roc_curves(y, y_proba, classes),
        }

        return results

    def _compute_roc_curves(self, y, y_proba: np.ndarray, classes: list) -> dict:
        """Compute per-class ROC curves (One-vs-Rest for multiclass)."""
        roc_data = {}
        try:
            if len(classes) == 2:
                # Binary: one curve for the positive class
                pos_cls = classes[1]
                y_binary = (np.asarray(y) == pos_cls).astype(int)
                fpr, tpr, _ = roc_curve(y_binary, y_proba[:, 1])
                roc_data[str(pos_cls)] = {
                    "fpr": fpr.tolist(),
                    "tpr": tpr.tolist(),
                    "auc": float(auc(fpr, tpr)),
                }
            else:
                # Multiclass OvR
                y_bin = label_binarize(np.asarray(y), classes=classes)
                for i, cls in enumerate(classes):
                    fpr, tpr, _ = roc_curve(y_bin[:, i], y_proba[:, i])
                    roc_data[str(cls)] = {
                        "fpr": fpr.tolist(),
                        "tpr": tpr.tolist(),
                        "auc": float(auc(fpr, tpr)),
                    }
        except Exception:
            pass  # Non-critical; return empty dict on failure
        return roc_data

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
