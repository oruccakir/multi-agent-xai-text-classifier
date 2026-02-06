"""Classification Agent - Performs text classification using trained models."""

from pathlib import Path
from typing import Any, Dict, List, Optional

from .base_agent import BaseAgent


class ClassificationAgent(BaseAgent):
    """
    Agent responsible for text classification using trained models.

    Pipeline:
    1. Load trained model and feature extractor
    2. Preprocess input text
    3. Extract features using TF-IDF
    4. Classify using the loaded model
    5. Return prediction, confidence, and probabilities
    """

    def __init__(self):
        super().__init__(name="ClassificationAgent")
        self._models_cache: Dict[str, Any] = {}
        self._feature_extractors_cache: Dict[str, Any] = {}
        self._preprocessors_cache: Dict[str, Any] = {}

    def process(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Classify input text using the specified model.

        Args:
            input_data: dict with:
                - 'text': Raw input text
                - 'experiment_path': Path to experiment directory
                - 'dataset': Dataset name (e.g., 'imdb', 'turkish_sentiment')
                - 'model_name': Model to use (e.g., 'logistic_regression', 'svm')
                - 'language': Language of the text ('english' or 'turkish')

        Returns:
            dict with:
                - 'prediction': Predicted class label
                - 'confidence': Confidence score (max probability)
                - 'probabilities': Dict of class -> probability
                - 'processed_text': Preprocessed text
                - 'model_name': Model used
                - 'dataset': Dataset used
        """
        text = input_data.get("text", "")
        experiment_path = input_data.get("experiment_path", "")
        dataset = input_data.get("dataset", "")
        model_name = input_data.get("model_name", "logistic_regression")
        language = input_data.get("language", "english")

        if not text:
            return {
                "error": "No text provided",
                "prediction": None,
                "confidence": 0.0,
                "probabilities": {},
            }

        if not experiment_path or not dataset:
            return {
                "error": "Missing experiment_path or dataset",
                "prediction": None,
                "confidence": 0.0,
                "probabilities": {},
            }

        # Load model
        device = input_data.get("device", None)
        model = self._load_model(experiment_path, dataset, model_name, device=device)

        if model is None:
            return {
                "error": f"Failed to load model: {model_name}",
                "prediction": None,
                "confidence": 0.0,
                "probabilities": {},
            }

        is_transformer = model_name == "transformer"

        if is_transformer:
            # Transformer uses raw text directly
            prediction = model.predict([text])[0]
            probabilities = model.predict_proba([text])[0]
            confidence = float(max(probabilities))
            prob_dict = dict(zip(model.classes_, [float(p) for p in probabilities]))

            preprocessor = self._get_preprocessor(language)
            processed_text = preprocessor.preprocess(text)
        else:
            feature_extractor = self._load_feature_extractor(experiment_path, dataset)

            if feature_extractor is None:
                return {
                    "error": "Failed to load feature extractor",
                    "prediction": None,
                    "confidence": 0.0,
                    "probabilities": {},
                }

            # Preprocess text
            preprocessor = self._get_preprocessor(language)
            processed_text = preprocessor.preprocess(text)

            # Extract features
            features = feature_extractor.transform([processed_text])

            # Classify
            prediction = model.predict(features)[0]
            probabilities = model.predict_proba(features)[0]
            confidence = float(max(probabilities))
            prob_dict = dict(zip(model.classes_, [float(p) for p in probabilities]))

        return {
            "prediction": prediction,
            "confidence": confidence,
            "probabilities": prob_dict,
            "processed_text": processed_text,
            "model_name": model_name,
            "dataset": dataset,
            "classes": list(model.classes_),
        }

    def classify_text(
        self,
        text: str,
        experiment_path: str,
        dataset: str,
        model_name: str = "logistic_regression",
        language: str = "english",
        device: str = None,
    ) -> Dict[str, Any]:
        """
        Convenience method to classify text.

        Args:
            text: Raw input text
            experiment_path: Path to experiment directory
            dataset: Dataset name
            model_name: Model to use
            language: Language of the text
            device: Device for transformer model inference

        Returns:
            Classification result dict
        """
        return self.process({
            "text": text,
            "experiment_path": experiment_path,
            "dataset": dataset,
            "model_name": model_name,
            "language": language,
            "device": device,
        })

    def classify_batch(
        self,
        texts: List[str],
        experiment_path: str,
        dataset: str,
        model_name: str = "logistic_regression",
        language: str = "english",
    ) -> List[Dict[str, Any]]:
        """
        Classify multiple texts efficiently.

        Args:
            texts: List of raw input texts
            experiment_path: Path to experiment directory
            dataset: Dataset name
            model_name: Model to use
            language: Language of the texts

        Returns:
            List of classification result dicts
        """
        if not texts:
            return []

        # Load model and feature extractor once
        model = self._load_model(experiment_path, dataset, model_name)
        feature_extractor = self._load_feature_extractor(experiment_path, dataset)

        if model is None or feature_extractor is None:
            return [{"error": "Failed to load model or feature extractor"} for _ in texts]

        # Preprocess all texts
        preprocessor = self._get_preprocessor(language)
        processed_texts = [preprocessor.preprocess(text) for text in texts]

        # Extract features for all texts
        features = feature_extractor.transform(processed_texts)

        # Classify all texts
        predictions = model.predict(features)
        probabilities_batch = model.predict_proba(features)

        # Build results
        results = []
        for i, (text, processed_text, prediction, probs) in enumerate(
            zip(texts, processed_texts, predictions, probabilities_batch)
        ):
            confidence = float(max(probs))
            prob_dict = dict(zip(model.classes_, [float(p) for p in probs]))

            results.append({
                "prediction": prediction,
                "confidence": confidence,
                "probabilities": prob_dict,
                "processed_text": processed_text,
                "model_name": model_name,
                "dataset": dataset,
                "classes": list(model.classes_),
            })

        return results

    def get_available_models(self, experiment_path: str, dataset: str) -> List[str]:
        """
        Get list of available trained models for a dataset.

        Args:
            experiment_path: Path to experiment directory
            dataset: Dataset name

        Returns:
            List of available model names
        """
        sklearn_models = [
            "naive_bayes", "svm", "random_forest", "knn", "logistic_regression"
        ]
        dataset_path = Path(experiment_path) / dataset

        if not dataset_path.exists():
            return []

        found = [
            model_name for model_name in sklearn_models
            if (dataset_path / f"{model_name}.pkl").exists()
        ]

        # Check for transformer model (saved as directory)
        if (dataset_path / "transformer.dir").exists():
            found.append("transformer")

        return found

    def _load_model(
        self, experiment_path: str, dataset: str, model_name: str, device: str = None
    ) -> Optional[Any]:
        """Load a trained model from disk with caching."""
        cache_key = f"{experiment_path}/{dataset}/{model_name}"
        if device and model_name == "transformer":
            cache_key += f"/{device}"

        if cache_key in self._models_cache:
            return self._models_cache[cache_key]

        if model_name == "transformer":
            model_path = Path(experiment_path) / dataset / "transformer.dir"
            if not model_path.exists():
                print(f"Transformer model not found: {model_path}")
                return None
            try:
                from src.models.transformer import TransformerClassifier
                model = TransformerClassifier.load(str(model_path), device=device)
                self._models_cache[cache_key] = model
                return model
            except Exception as e:
                print(f"Failed to load transformer model: {e}")
                return None
        else:
            model_path = Path(experiment_path) / dataset / f"{model_name}.pkl"
            if not model_path.exists():
                print(f"Model not found: {model_path}")
                return None
            try:
                from src.models.base_model import BaseModel
                model = BaseModel.load(str(model_path))
                self._models_cache[cache_key] = model
                return model
            except Exception as e:
                print(f"Failed to load model: {e}")
                return None

    def _load_feature_extractor(
        self, experiment_path: str, dataset: str
    ) -> Optional[Any]:
        """Load a feature extractor from disk with caching."""
        cache_key = f"{experiment_path}/{dataset}"

        if cache_key in self._feature_extractors_cache:
            return self._feature_extractors_cache[cache_key]

        fe_path = Path(experiment_path) / dataset / "feature_extractor.pkl"

        if not fe_path.exists():
            print(f"Feature extractor not found: {fe_path}")
            return None

        try:
            from src.preprocessing.feature_extractor import FeatureExtractor
            fe = FeatureExtractor.load(str(fe_path))
            self._feature_extractors_cache[cache_key] = fe
            return fe
        except Exception as e:
            print(f"Failed to load feature extractor: {e}")
            return None

    def _get_preprocessor(self, language: str) -> Any:
        """Get or create a text preprocessor for the specified language."""
        if language not in self._preprocessors_cache:
            from src.preprocessing.text_preprocessor import TextPreprocessor
            self._preprocessors_cache[language] = TextPreprocessor(language=language)

        return self._preprocessors_cache[language]

    def clear_cache(self) -> None:
        """Clear all cached models and feature extractors."""
        self._models_cache.clear()
        self._feature_extractors_cache.clear()
        self._preprocessors_cache.clear()
