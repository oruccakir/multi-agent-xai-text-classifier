"""Configuration management for the project."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional
import yaml


@dataclass
class ModelConfig:
    """Configuration for a classification model."""

    name: str
    params: Dict = field(default_factory=dict)


@dataclass
class Config:
    """Main configuration class for the project."""

    # Paths
    data_dir: str = "data"
    models_dir: str = "data/models"
    reports_dir: str = "reports"

    # Preprocessing
    language: str = "english"
    remove_stopwords: bool = True
    remove_punctuation: bool = True
    lowercase: bool = True

    # Feature extraction
    feature_method: str = "tfidf"  # 'tfidf' or 'transformer'
    tfidf_max_features: int = 10000
    tfidf_ngram_range: tuple = (1, 2)
    transformer_model: str = "sentence-transformers/all-MiniLM-L6-v2"

    # Training
    test_size: float = 0.2
    random_state: int = 42
    cv_folds: int = 5

    # Models to use
    models: List[str] = field(
        default_factory=lambda: [
            "naive_bayes",
            "svm",
            "random_forest",
            "knn",
            "logistic_regression",
            "transformer",
        ]
    )

    # XAI settings
    lime_num_features: int = 10
    lime_num_samples: int = 1000
    shap_max_evals: int = 500

    # LLM settings
    llm_model: str = "gpt-3.5-turbo"
    llm_api_key: Optional[str] = None

    @classmethod
    def from_yaml(cls, path: str) -> "Config":
        """Load configuration from YAML file."""
        with open(path, "r") as f:
            data = yaml.safe_load(f)
        return cls(**data)

    def to_yaml(self, path: str) -> None:
        """Save configuration to YAML file."""
        with open(path, "w") as f:
            yaml.dump(self.__dict__, f, default_flow_style=False)

    def get_model_config(self, model_name: str) -> ModelConfig:
        """Get configuration for a specific model."""
        model_configs = {
            "naive_bayes": ModelConfig("NaiveBayes", {"alpha": 1.0}),
            "svm": ModelConfig("SVM", {"kernel": "rbf", "C": 1.0}),
            "random_forest": ModelConfig(
                "RandomForest", {"n_estimators": 100, "max_depth": None}
            ),
            "knn": ModelConfig("KNN", {"n_neighbors": 5, "metric": "cosine"}),
            "logistic_regression": ModelConfig(
                "LogisticRegression", {"C": 1.0, "max_iter": 1000}
            ),
            "transformer": ModelConfig(
                "Transformer",
                {"model_name": "distilbert-base-uncased", "num_labels": 2},
            ),
        }
        return model_configs.get(model_name)
