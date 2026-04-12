"""Main pipeline orchestrating the multi-agent system."""

from typing import Any, Dict

from .agents import IntentClassifierAgent, ClassificationAgent, XAIAgent
from .utils.config import Config


class TextClassificationPipeline:
    """
    Main pipeline that orchestrates the three agents:
    1. Intent Classifier - routes to appropriate model
    2. Classification Agent - performs classification
    3. XAI Agent - explains the decision
    """

    def __init__(self, config: Config = None):
        self.config = config or Config()
        self.intent_agent = IntentClassifierAgent()
        self.classification_agent = ClassificationAgent()
        self.xai_agent = XAIAgent()

    def process(self, text: str, explain: bool = True) -> Dict[str, Any]:
        """
        Process input text through the full pipeline.

        Args:
            text: Input text to classify
            explain: Whether to generate XAI explanation

        Returns:
            dict with 'intent', 'prediction', 'confidence', 'explanation'
        """
        # Step 1: Intent classification
        intent_result = self.intent_agent.process(text)

        # Step 2: Classification
        classification_input = {
            "text": text,
            "model_name": intent_result.get("recommended_model"),
            "language": intent_result.get("language"),
        }
        classification_result = self.classification_agent.process(classification_input)

        # Step 3: XAI explanation (optional)
        explanation = None
        if explain:
            xai_input = {
                "text": text,
                "prediction": classification_result.get("prediction"),
                "confidence": classification_result.get("confidence"),
                "model": classification_result.get("model"),
                "features": classification_result.get("features"),
            }
            explanation = self.xai_agent.process(xai_input)

        return {
            "intent": intent_result,
            "prediction": classification_result.get("prediction"),
            "confidence": classification_result.get("confidence"),
            "explanation": explanation,
        }

