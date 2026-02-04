"""Base agent class for all agents in the system."""

from abc import ABC, abstractmethod
from typing import Any


class BaseAgent(ABC):
    """Abstract base class for all agents."""

    def __init__(self, name: str):
        self.name = name

    @abstractmethod
    def process(self, input_data: Any) -> Any:
        """Process input data and return result."""
        pass

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name={self.name})"
