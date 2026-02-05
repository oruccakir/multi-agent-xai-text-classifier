"""XAI Agent - Explains classification decisions using LIME, SHAP, TF-IDF and Google Gemini LLM."""

import os
import numpy as np
from typing import Any, Dict, List, Optional, Callable

from dotenv import load_dotenv

from .base_agent import BaseAgent

# Load environment variables
load_dotenv()


class XAIAgent(BaseAgent):
    """
    Agent responsible for explaining classification decisions.

    Multi-method analysis:
    1. LIME: Local Interpretable Model-agnostic Explanations
    2. SHAP: SHapley Additive exPlanations
    3. TF-IDF: Extract important words from the text
    4. Gemini LLM: Generate human-friendly natural language explanations
    """

    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize XAI Agent.

        Args:
            api_key: Google Gemini API key. If not provided, will look for GEMINI_API_KEY env var.
        """
        super().__init__(name="XAIAgent")
        self.api_key = api_key or os.getenv("GEMINI_API_KEY")
        self.client = None
        self._initialized = False
        self._lime_available = None
        self._shap_available = None

    def _initialize_gemini(self):
        """Initialize Gemini client lazily."""
        if self._initialized:
            return True

        if not self.api_key:
            return False

        try:
            from google import genai

            self.client = genai.Client(api_key=self.api_key)
            self._initialized = True
            return True
        except Exception as e:
            print(f"Failed to initialize Gemini: {e}")
            return False

    def _check_lime_available(self) -> bool:
        """Check if LIME library is available."""
        if self._lime_available is None:
            try:
                from lime.lime_text import LimeTextExplainer
                self._lime_available = True
            except ImportError:
                self._lime_available = False
                print("LIME not available. Install with: pip install lime")
        return self._lime_available

    def _check_shap_available(self) -> bool:
        """Check if SHAP library is available."""
        if self._shap_available is None:
            try:
                import shap
                self._shap_available = True
            except ImportError:
                self._shap_available = False
                print("SHAP not available. Install with: pip install shap")
        return self._shap_available

    def process(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate explanation for a classification decision.

        Args:
            input_data: dict with:
                - 'text': Original input text
                - 'processed_text': Preprocessed text
                - 'prediction': Model prediction
                - 'confidence': Prediction confidence
                - 'probabilities': Class probabilities dict
                - 'word_impacts': TF-IDF word importance scores (optional)
                - 'dataset': Dataset name
                - 'model_name': Model name used
                - 'lime_explanation': LIME explanation data (optional)
                - 'shap_explanation': SHAP explanation data (optional)

        Returns:
            dict with:
                - 'word_impacts': TF-IDF word importance scores
                - 'lime_explanation': LIME word contributions
                - 'shap_explanation': SHAP values
                - 'natural_explanation': LLM-generated explanation
                - 'technical_explanation': Technical details
                - 'gemini_available': Whether Gemini was used
                - 'lime_available': Whether LIME was used
                - 'shap_available': Whether SHAP was used
        """
        text = input_data.get("text", "")
        processed_text = input_data.get("processed_text", "")
        prediction = input_data.get("prediction", "")
        confidence = input_data.get("confidence", 0.0)
        probabilities = input_data.get("probabilities", {})
        word_impacts = input_data.get("word_impacts", {})
        lime_explanation = input_data.get("lime_explanation", {})
        shap_explanation = input_data.get("shap_explanation", {})
        dataset = input_data.get("dataset", "")
        model_name = input_data.get("model_name", "")

        # Generate natural explanation with Gemini
        gemini_available = self._initialize_gemini()

        if gemini_available:
            natural_explanation = self._generate_gemini_explanation(
                text=text,
                prediction=prediction,
                confidence=confidence,
                probabilities=probabilities,
                word_impacts=word_impacts,
                lime_explanation=lime_explanation,
                shap_explanation=shap_explanation,
                dataset=dataset,
                model_name=model_name,
            )
        else:
            natural_explanation = self._generate_fallback_explanation(
                prediction=prediction,
                confidence=confidence,
                word_impacts=word_impacts,
                lime_explanation=lime_explanation,
            )

        # Generate technical explanation
        technical_explanation = self._generate_technical_explanation(
            prediction=prediction,
            confidence=confidence,
            probabilities=probabilities,
            word_impacts=word_impacts,
            lime_explanation=lime_explanation,
            shap_explanation=shap_explanation,
            model_name=model_name,
        )

        return {
            "word_impacts": word_impacts,
            "lime_explanation": lime_explanation,
            "shap_explanation": shap_explanation,
            "natural_explanation": natural_explanation,
            "technical_explanation": technical_explanation,
            "gemini_available": gemini_available,
            "lime_available": bool(lime_explanation),
            "shap_available": bool(shap_explanation),
        }

    def explain_with_lime(
        self,
        text: str,
        predict_proba_fn: Callable,
        class_names: List[str],
        num_features: int = 10,
        num_samples: int = 500,
    ) -> Dict[str, Any]:
        """
        Generate LIME explanation for text classification.

        LIME (Local Interpretable Model-agnostic Explanations) works by:
        1. Perturbing the input text (removing words)
        2. Getting model predictions for perturbed samples
        3. Fitting a local linear model to understand feature importance

        Args:
            text: Original text to explain
            predict_proba_fn: Function that takes list of texts and returns probabilities
            class_names: List of class names
            num_features: Number of top features to return
            num_samples: Number of perturbed samples to generate

        Returns:
            Dictionary with LIME explanation data
        """
        if not self._check_lime_available():
            return {}

        try:
            from lime.lime_text import LimeTextExplainer

            # Create LIME explainer
            explainer = LimeTextExplainer(
                class_names=class_names,
                split_expression=r'\W+',
                bow=True,
            )

            # Generate explanation
            # Note: top_labels must be specified to populate exp.top_labels
            exp = explainer.explain_instance(
                text,
                predict_proba_fn,
                num_features=num_features,
                num_samples=num_samples,
                top_labels=len(class_names),
            )

            # Get prediction class
            pred_class = exp.top_labels[0] if exp.top_labels else 0

            # Extract word contributions for predicted class
            word_contributions = {}
            for word, weight in exp.as_list(label=pred_class):
                word_contributions[word] = float(weight)

            # Separate positive and negative contributions
            positive_words = {w: s for w, s in word_contributions.items() if s > 0}
            negative_words = {w: s for w, s in word_contributions.items() if s < 0}

            # Sort by absolute value
            positive_words = dict(sorted(positive_words.items(), key=lambda x: -x[1]))
            negative_words = dict(sorted(negative_words.items(), key=lambda x: x[1]))

            return {
                "word_contributions": word_contributions,
                "positive_words": positive_words,
                "negative_words": negative_words,
                "predicted_class": class_names[pred_class] if pred_class < len(class_names) else str(pred_class),
                "prediction_proba": float(exp.predict_proba[pred_class]) if hasattr(exp, 'predict_proba') else None,
                "intercept": float(exp.intercept[pred_class]) if hasattr(exp, 'intercept') else None,
                "score": float(exp.score) if hasattr(exp, 'score') else None,
            }

        except Exception as e:
            print(f"LIME explanation error: {e}")
            return {}

    def explain_with_shap(
        self,
        text: str,
        predict_proba_fn: Callable,
        class_names: List[str],
        background_texts: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """
        Generate SHAP explanation for text classification.

        SHAP (SHapley Additive exPlanations) uses game theory to compute
        the contribution of each feature (word) to the prediction.

        Args:
            text: Original text to explain
            predict_proba_fn: Function that takes list of texts and returns probabilities
            class_names: List of class names
            background_texts: Background texts for SHAP (if None, uses masking)

        Returns:
            Dictionary with SHAP explanation data
        """
        if not self._check_shap_available():
            return {}

        try:
            import shap

            # Create a masker for text
            masker = shap.maskers.Text(tokenizer=r"\W+")

            # Create SHAP explainer
            explainer = shap.Explainer(
                predict_proba_fn,
                masker,
                output_names=class_names,
            )

            # Compute SHAP values
            shap_values = explainer([text])

            # Get the prediction
            probs = predict_proba_fn([text])[0]
            pred_class_idx = int(np.argmax(probs))
            pred_class = class_names[pred_class_idx] if pred_class_idx < len(class_names) else str(pred_class_idx)

            # Extract word-level SHAP values for predicted class
            # shap_values.values shape: (num_samples, num_tokens, num_classes)
            values = shap_values.values[0]  # First (only) sample
            data = shap_values.data[0]  # Token strings

            # Get SHAP values for predicted class
            if len(values.shape) > 1:
                class_shap_values = values[:, pred_class_idx]
            else:
                class_shap_values = values

            # Create word -> SHAP value mapping
            word_shap_values = {}
            for i, (token, shap_val) in enumerate(zip(data, class_shap_values)):
                token_str = str(token).strip()
                if token_str and len(token_str) > 1:  # Skip empty and single char tokens
                    word_shap_values[token_str] = float(shap_val)

            # Separate positive and negative
            positive_words = {w: s for w, s in word_shap_values.items() if s > 0}
            negative_words = {w: s for w, s in word_shap_values.items() if s < 0}

            positive_words = dict(sorted(positive_words.items(), key=lambda x: -x[1]))
            negative_words = dict(sorted(negative_words.items(), key=lambda x: x[1]))

            return {
                "word_shap_values": word_shap_values,
                "positive_words": positive_words,
                "negative_words": negative_words,
                "predicted_class": pred_class,
                "base_value": float(shap_values.base_values[0][pred_class_idx]) if hasattr(shap_values, 'base_values') else None,
            }

        except Exception as e:
            print(f"SHAP explanation error: {e}")
            return {}

    def _generate_gemini_explanation(
        self,
        text: str,
        prediction: str,
        confidence: float,
        probabilities: Dict[str, float],
        word_impacts: Dict[str, float],
        lime_explanation: Dict[str, Any],
        shap_explanation: Dict[str, Any],
        dataset: str,
        model_name: str,
    ) -> str:
        """Generate natural language explanation using Gemini."""
        if not self.client:
            return self._generate_fallback_explanation(prediction, confidence, word_impacts, lime_explanation)

        # Prepare word impacts summary
        top_tfidf_words = list(word_impacts.keys())[:5] if word_impacts else []

        # Get LIME positive/negative words
        lime_positive = list(lime_explanation.get("positive_words", {}).keys())[:3]
        lime_negative = list(lime_explanation.get("negative_words", {}).keys())[:3]

        # Get SHAP positive/negative words
        shap_positive = list(shap_explanation.get("positive_words", {}).keys())[:3]
        shap_negative = list(shap_explanation.get("negative_words", {}).keys())[:3]

        # Prepare probability info
        prob_info = ", ".join([f"{cls}: {prob:.1%}" for cls, prob in probabilities.items()])

        # Build XAI summary
        xai_summary = ""
        if lime_positive or lime_negative:
            xai_summary += f"\n**LIME Analysis:**\n"
            if lime_positive:
                xai_summary += f"- Words supporting '{prediction}': {', '.join(lime_positive)}\n"
            if lime_negative:
                xai_summary += f"- Words against '{prediction}': {', '.join(lime_negative)}\n"

        if shap_positive or shap_negative:
            xai_summary += f"\n**SHAP Analysis:**\n"
            if shap_positive:
                xai_summary += f"- Positive contributions: {', '.join(shap_positive)}\n"
            if shap_negative:
                xai_summary += f"- Negative contributions: {', '.join(shap_negative)}\n"

        # Create prompt for Gemini
        prompt = f"""You are an AI explainability expert. Analyze this text classification result with LIME and SHAP analysis and provide a clear, educational explanation.

**Input Text:** "{text[:500]}{'...' if len(text) > 500 else ''}"

**Classification Result:**
- Prediction: {prediction}
- Confidence: {confidence:.1%}
- All class probabilities: {prob_info}
- Dataset: {dataset}
- Model: {model_name}

**Explainability Analysis:**
- TF-IDF Important Words: {', '.join(top_tfidf_words) if top_tfidf_words else 'N/A'}
{xai_summary}

Please provide a concise explanation (3-4 sentences) that:
1. Explains WHY the text was classified as "{prediction}"
2. Highlights which specific words SUPPORT the prediction (from LIME/SHAP positive contributions)
3. Mentions any words that work AGAINST this prediction (if any)
4. Mentions the confidence level

Keep the explanation simple and understandable for non-technical users. Do not use markdown formatting."""

        try:
            response = self.client.models.generate_content(
                model="gemini-2.0-flash",
                contents=prompt,
            )
            return response.text.strip()
        except Exception as e:
            print(f"Gemini API error: {e}")
            return self._generate_fallback_explanation(prediction, confidence, word_impacts, lime_explanation)

    def _generate_fallback_explanation(
        self,
        prediction: str,
        confidence: float,
        word_impacts: Dict[str, float],
        lime_explanation: Dict[str, Any] = None,
    ) -> str:
        """Generate a simple fallback explanation without LLM."""
        lime_explanation = lime_explanation or {}

        # Prefer LIME words over TF-IDF
        positive_words = list(lime_explanation.get("positive_words", {}).keys())[:3]
        negative_words = list(lime_explanation.get("negative_words", {}).keys())[:3]

        if positive_words:
            pos_str = ", ".join(f"'{w}'" for w in positive_words)
            explanation = (
                f"The text is classified as '{prediction}' with {confidence:.1%} confidence. "
                f"Words supporting this decision: {pos_str}. "
            )
            if negative_words:
                neg_str = ", ".join(f"'{w}'" for w in negative_words)
                explanation += f"Words against this classification: {neg_str}."
            return explanation

        # Fallback to TF-IDF
        top_words = list(word_impacts.keys())[:3] if word_impacts else []
        if top_words:
            words_str = ", ".join(f"'{w}'" for w in top_words)
            return (
                f"The text is classified as '{prediction}' with {confidence:.1%} confidence. "
                f"Key contributing words include: {words_str}. "
                f"These words have high TF-IDF scores, meaning they are distinctive for this classification."
            )

        return (
            f"The text is classified as '{prediction}' with {confidence:.1%} confidence "
            f"based on overall text patterns recognized by the model."
        )

    def _generate_technical_explanation(
        self,
        prediction: str,
        confidence: float,
        probabilities: Dict[str, float],
        word_impacts: Dict[str, float],
        lime_explanation: Dict[str, Any],
        shap_explanation: Dict[str, Any],
        model_name: str,
    ) -> str:
        """Generate technical explanation with model details."""
        lines = [
            f"**Model:** {model_name}",
            f"**Prediction:** {prediction}",
            f"**Confidence:** {confidence:.2%}",
            "",
            "**Class Probabilities:**",
        ]

        for cls, prob in sorted(probabilities.items(), key=lambda x: -x[1]):
            lines.append(f"  - {cls}: {prob:.2%}")

        if word_impacts:
            lines.append("")
            lines.append("**Top TF-IDF Features:**")
            for word, score in list(word_impacts.items())[:5]:
                lines.append(f"  - '{word}': {score:.4f}")

        if lime_explanation:
            lines.append("")
            lines.append("**LIME Analysis:**")
            if lime_explanation.get("score"):
                lines.append(f"  - Local Model R²: {lime_explanation['score']:.4f}")
            pos_words = lime_explanation.get("positive_words", {})
            neg_words = lime_explanation.get("negative_words", {})
            if pos_words:
                lines.append("  - Positive contributions:")
                for word, score in list(pos_words.items())[:5]:
                    lines.append(f"    - '{word}': +{score:.4f}")
            if neg_words:
                lines.append("  - Negative contributions:")
                for word, score in list(neg_words.items())[:5]:
                    lines.append(f"    - '{word}': {score:.4f}")

        if shap_explanation:
            lines.append("")
            lines.append("**SHAP Analysis:**")
            if shap_explanation.get("base_value") is not None:
                lines.append(f"  - Base value: {shap_explanation['base_value']:.4f}")
            pos_words = shap_explanation.get("positive_words", {})
            neg_words = shap_explanation.get("negative_words", {})
            if pos_words:
                lines.append("  - Positive SHAP values:")
                for word, score in list(pos_words.items())[:5]:
                    lines.append(f"    - '{word}': +{score:.4f}")
            if neg_words:
                lines.append("  - Negative SHAP values:")
                for word, score in list(neg_words.items())[:5]:
                    lines.append(f"    - '{word}': {score:.4f}")

        return "\n".join(lines)

    def explain_with_tfidf(
        self, text: str, feature_extractor, top_n: int = 10
    ) -> Dict[str, float]:
        """
        Extract TF-IDF based word importance.

        Args:
            text: Preprocessed text to analyze
            feature_extractor: Fitted TF-IDF feature extractor
            top_n: Number of top words to return

        Returns:
            Dictionary of word -> importance score
        """
        # Get feature names
        if hasattr(feature_extractor, "vectorizer") and hasattr(
            feature_extractor.vectorizer, "get_feature_names_out"
        ):
            feature_names = feature_extractor.vectorizer.get_feature_names_out()
        else:
            return {}

        # Transform text
        features = feature_extractor.transform([text])

        # Get feature values
        if hasattr(features, "toarray"):
            feature_values = features.toarray()[0]
        else:
            feature_values = features[0]

        # Find non-zero features
        word_impacts = {}
        for i, val in enumerate(feature_values):
            if val > 0 and i < len(feature_names):
                word_impacts[feature_names[i]] = float(val)

        # Sort by importance and return top N
        sorted_impacts = dict(
            sorted(word_impacts.items(), key=lambda x: x[1], reverse=True)[:top_n]
        )

        return sorted_impacts

    def generate_explanation(
        self,
        text: str,
        processed_text: str,
        prediction: str,
        confidence: float,
        probabilities: Dict[str, float],
        feature_extractor,
        dataset: str,
        model_name: str,
        predict_proba_fn: Optional[Callable] = None,
        class_names: Optional[List[str]] = None,
        use_lime: bool = True,
        use_shap: bool = True,
    ) -> Dict[str, Any]:
        """
        Generate full explanation including LIME and SHAP.

        Args:
            text: Original input text
            processed_text: Preprocessed text
            prediction: Model prediction
            confidence: Prediction confidence
            probabilities: Class probabilities
            feature_extractor: TF-IDF feature extractor
            dataset: Dataset name
            model_name: Model name
            predict_proba_fn: Function for LIME/SHAP (takes texts, returns probabilities)
            class_names: List of class names for LIME/SHAP
            use_lime: Whether to generate LIME explanation
            use_shap: Whether to generate SHAP explanation

        Returns:
            Full explanation dictionary
        """
        # Get TF-IDF word impacts
        word_impacts = self.explain_with_tfidf(processed_text, feature_extractor)

        # Generate LIME explanation
        lime_explanation = {}
        if use_lime and predict_proba_fn and class_names:
            lime_explanation = self.explain_with_lime(
                text=text,
                predict_proba_fn=predict_proba_fn,
                class_names=class_names,
                num_features=10,
                num_samples=500,
            )

        # Generate SHAP explanation
        shap_explanation = {}
        if use_shap and predict_proba_fn and class_names:
            shap_explanation = self.explain_with_shap(
                text=text,
                predict_proba_fn=predict_proba_fn,
                class_names=class_names,
            )

        # Generate full explanation
        return self.process(
            {
                "text": text,
                "processed_text": processed_text,
                "prediction": prediction,
                "confidence": confidence,
                "probabilities": probabilities,
                "word_impacts": word_impacts,
                "lime_explanation": lime_explanation,
                "shap_explanation": shap_explanation,
                "dataset": dataset,
                "model_name": model_name,
            }
        )
