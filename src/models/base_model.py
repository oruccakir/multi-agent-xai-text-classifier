"""Base model class for all classification models."""

from abc import ABC, abstractmethod
from typing import Any
import numpy as np


class BaseModel(ABC):
    """Abstract base class for all classification models."""

    def __init__(self, name: str):
        self.name = name
        self.model = None
        self.is_fitted = False

    @abstractmethod
    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
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

    def save(self, path: str) -> None:
        """Save model to disk."""
        # TODO: Implement model saving
        raise NotImplementedError("Model saving not yet implemented")

    def load(self, path: str) -> None:
        """Load model from disk."""
        # TODO: Implement model loading
        raise NotImplementedError("Model loading not yet implemented")

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name={self.name}, fitted={self.is_fitted})"
