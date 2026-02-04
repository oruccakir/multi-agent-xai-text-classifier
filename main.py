"""Main entry point for the Multi-Agent XAI Text Classification System."""

import argparse
from pathlib import Path

from src.pipeline import TextClassificationPipeline
from src.utils.config import Config


def main():
    parser = argparse.ArgumentParser(
        description="Multi-Agent Explainable AI Text Classification System"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/default.yaml",
        help="Path to configuration file",
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["train", "evaluate", "predict"],
        default="predict",
        help="Operation mode",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        choices=["imdb", "turkish_sentiment", "ag_news", "turkish_news"],
        help="Dataset to use for training/evaluation",
    )
    parser.add_argument(
        "--model",
        type=str,
        choices=[
            "naive_bayes",
            "svm",
            "random_forest",
            "knn",
            "logistic_regression",
            "transformer",
        ],
        help="Specific model to use (default: all)",
    )
    parser.add_argument(
        "--text",
        type=str,
        help="Text to classify (for predict mode)",
    )
    parser.add_argument(
        "--no-explain",
        action="store_true",
        help="Skip XAI explanation generation",
    )

    args = parser.parse_args()

    # Load configuration
    config_path = Path(args.config)
    if config_path.exists():
        config = Config.from_yaml(str(config_path))
    else:
        print(f"Config file not found: {config_path}, using defaults")
        config = Config()

    # Initialize pipeline
    pipeline = TextClassificationPipeline(config)

    if args.mode == "train":
        if not args.dataset:
            parser.error("--dataset is required for training mode")
        print(f"Training on {args.dataset}...")
        results = pipeline.train(args.dataset, args.model)
        print(f"Training results: {results}")

    elif args.mode == "evaluate":
        if not args.dataset:
            parser.error("--dataset is required for evaluation mode")
        print(f"Evaluating on {args.dataset}...")
        results = pipeline.evaluate(args.dataset, args.model)
        print(f"Evaluation results: {results}")

    elif args.mode == "predict":
        if not args.text:
            parser.error("--text is required for prediction mode")
        print(f"Classifying: {args.text}")
        result = pipeline.process(args.text, explain=not args.no_explain)
        print(f"\nPrediction: {result['prediction']}")
        print(f"Confidence: {result['confidence']:.2%}")
        if result.get("explanation"):
            print(f"Explanation: {result['explanation']}")


if __name__ == "__main__":
    main()
