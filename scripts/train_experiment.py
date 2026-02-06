#!/usr/bin/env python
"""
Config-based training script for experiments.

Usage:
    python scripts/train_experiment.py --config configs/<experiment_name>/imdb_baseline.yaml
    python scripts/train_experiment.py --config configs/<experiment_name>/turkish_sentiment_baseline.yaml

    # Train multiple experiments
    python scripts/train_experiment.py --config configs/<experiment_name>/*.yaml
"""

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List

import numpy as np
import pandas as pd
import yaml

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.preprocessing.feature_extractor import FeatureExtractor
from src.models.naive_bayes import NaiveBayesClassifier
from src.models.svm import SVMClassifier
from src.models.random_forest import RandomForestClassifier
from src.models.knn import KNNClassifier
from src.models.logistic_regression import LogisticRegressionClassifier
from src.models.transformer import TransformerClassifier


def load_config(config_path: str) -> Dict[str, Any]:
    """Load experiment configuration from YAML file."""
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config


def create_model(model_name: str, model_config: Dict[str, Any]):
    """Create a model instance from configuration."""
    if model_name == "naive_bayes":
        return NaiveBayesClassifier(
            alpha=model_config.get("alpha", 1.0)
        )
    elif model_name == "svm":
        return SVMClassifier(
            C=model_config.get("C", 1.0),
            max_iter=model_config.get("max_iter", 1000),
            random_state=42,
        )
    elif model_name == "random_forest":
        return RandomForestClassifier(
            n_estimators=model_config.get("n_estimators", 100),
            max_depth=model_config.get("max_depth"),
            n_jobs=model_config.get("n_jobs", -1),
            random_state=42,
        )
    elif model_name == "knn":
        return KNNClassifier(
            n_neighbors=model_config.get("n_neighbors", 5),
            metric=model_config.get("metric", "cosine"),
            weights=model_config.get("weights", "distance"),
            n_jobs=model_config.get("n_jobs", 1),
            algorithm=model_config.get("algorithm", "brute"),
        )
    elif model_name == "logistic_regression":
        return LogisticRegressionClassifier(
            C=model_config.get("C", 1.0),
            max_iter=model_config.get("max_iter", 1000),
            solver=model_config.get("solver", "lbfgs"),
            random_state=42,
        )
    elif model_name == "transformer":
        return TransformerClassifier(
            model_name=model_config.get("model_name", "distilbert-base-uncased"),
            num_labels=model_config.get("num_labels", 2),
            fine_tune_mode=model_config.get("fine_tune_mode", "head_only"),
            learning_rate=model_config.get("learning_rate", 2e-5),
            batch_size=model_config.get("batch_size", 16),
            epochs=model_config.get("epochs", 3),
            max_length=model_config.get("max_length", 256),
            lora_r=model_config.get("lora_r", 8),
            lora_alpha=model_config.get("lora_alpha", 16),
            lora_dropout=model_config.get("lora_dropout", 0.1),
            warmup_ratio=model_config.get("warmup_ratio", 0.1),
            weight_decay=model_config.get("weight_decay", 0.01),
        )
    else:
        raise ValueError(f"Unknown model: {model_name}")


def load_data(config: Dict[str, Any]) -> tuple:
    """Load training and test data based on configuration."""
    dataset_config = config["dataset"]

    train_path = project_root / dataset_config["train_path"]
    test_path = project_root / dataset_config["test_path"]

    print(f"  Loading training data from {train_path}...")
    train_df = pd.read_csv(train_path)

    print(f"  Loading test data from {test_path}...")
    test_df = pd.read_csv(test_path)

    # Sample if specified
    sample_size = dataset_config.get("sample_size")
    if sample_size:
        print(f"  Sampling {sample_size} training examples...")
        train_df = train_df.sample(n=min(sample_size, len(train_df)), random_state=42)

        test_sample = min(sample_size // 5, len(test_df))
        print(f"  Sampling {test_sample} test examples...")
        test_df = test_df.sample(n=test_sample, random_state=42)

    return train_df, test_df


def run_experiment(config_path: str) -> Dict[str, Any]:
    """Run a single experiment based on configuration."""
    config = load_config(config_path)

    experiment_name = config["experiment"]["name"]
    print(f"\n{'=' * 70}")
    print(f"EXPERIMENT: {experiment_name}")
    print(f"Description: {config['experiment']['description']}")
    print(f"{'=' * 70}")

    # Create output directory
    output_dir = project_root / "data" / "models" / experiment_name / config["dataset"]["name"]
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {output_dir}")

    # Load data
    train_df, test_df = load_data(config)
    print(f"  Training samples: {len(train_df)}")
    print(f"  Test samples: {len(test_df)}")
    print(f"  Classes: {train_df['label'].unique().tolist()}")

    # Get texts and labels
    X_train_text = train_df["text"].tolist()
    y_train = train_df["label"].values
    X_test_text = test_df["text"].tolist()
    y_test = test_df["label"].values

    # Check which models need TF-IDF features
    models_config = config["models"]
    sklearn_models = {
        k: v for k, v in models_config.items()
        if k != "transformer" and v.get("enabled", True)
    }
    has_transformer = "transformer" in models_config and models_config["transformer"].get("enabled", True)

    # Feature extraction (only for sklearn models)
    X_train = None
    X_test = None

    if sklearn_models:
        fe_config = config["feature_extraction"]
        print(f"\n  Extracting features (method: {fe_config['method']})...")

        start_time = time.time()
        use_sparse = fe_config.get("sparse", True)

        feature_extractor = FeatureExtractor(
            method=fe_config["method"],
            max_features=fe_config.get("max_features", 10000),
            ngram_range=tuple(fe_config.get("ngram_range", [1, 2])),
            min_df=fe_config.get("min_df", 2),
            max_df=fe_config.get("max_df", 0.95),
            sublinear_tf=fe_config.get("sublinear_tf", True),
        )

        X_train = feature_extractor.fit_transform(X_train_text, sparse=use_sparse)
        X_test = feature_extractor.transform(X_test_text, sparse=use_sparse)
        feature_time = time.time() - start_time

        print(f"  Feature extraction completed in {feature_time:.2f}s")
        print(f"  Feature matrix shape: {X_train.shape}")

        # Save feature extractor
        if config["output"].get("save_feature_extractor", True):
            fe_path = output_dir / "feature_extractor.pkl"
            feature_extractor.save(str(fe_path))

    # Train models
    results = {}
    num_classes = len(train_df["label"].unique())

    for model_name, model_config in models_config.items():
        if not model_config.get("enabled", True):
            print(f"\n  Skipping {model_name} (disabled)")
            continue

        print(f"\n  {'─' * 50}")
        print(f"  Training {model_name}...")

        try:
            if model_name == "transformer":
                model_config_copy = dict(model_config)
                model_config_copy["num_labels"] = num_classes
                model = create_model(model_name, model_config_copy)

                # Train on raw texts
                start_time = time.time()
                model.fit(X_train_text, y_train)
                train_time = time.time() - start_time

                # Evaluate on raw texts
                start_time = time.time()
                eval_results = model.evaluate(X_test_text, y_test)
                eval_time = time.time() - start_time
            else:
                model = create_model(model_name, model_config)

                # Train with TF-IDF features
                start_time = time.time()
                model.fit(X_train, y_train)
                train_time = time.time() - start_time

                # Evaluate
                start_time = time.time()
                eval_results = model.evaluate(X_test, y_test)
                eval_time = time.time() - start_time

            # Add timing
            eval_results["train_time"] = train_time
            eval_results["eval_time"] = eval_time

            # Print results
            print(f"    Accuracy:     {eval_results['accuracy']:.4f}")
            print(f"    F1 (macro):   {eval_results['f1_macro']:.4f}")
            print(f"    F1 (weighted): {eval_results['f1_weighted']:.4f}")
            print(f"    Precision:    {eval_results['precision']:.4f}")
            print(f"    Recall:       {eval_results['recall']:.4f}")
            print(f"    Train time:   {train_time:.2f}s")

            # Save model
            if config["output"].get("save_models", True):
                if model_name == "transformer":
                    model_path = output_dir / "transformer.dir"
                    model.save(str(model_path))
                else:
                    model_path = output_dir / f"{model_name}.pkl"
                    model.save(str(model_path))

            results[model_name] = eval_results

        except Exception as e:
            print(f"    ERROR: {str(e)}")
            results[model_name] = {"error": str(e)}

    # Save experiment results
    save_experiment_results(config, results, output_dir)

    return {
        "experiment_name": experiment_name,
        "output_dir": str(output_dir),
        "results": results,
    }


def save_experiment_results(config: Dict, results: Dict, output_dir: Path):
    """Save experiment results and metadata."""
    # Create summary
    summary = {
        "experiment": config["experiment"],
        "dataset": config["dataset"],
        "feature_extraction": config["feature_extraction"],
        "timestamp": datetime.now().isoformat(),
        "results": {},
    }

    # Add results (convert numpy arrays to lists)
    for model_name, metrics in results.items():
        if "error" in metrics:
            summary["results"][model_name] = metrics
        else:
            summary["results"][model_name] = {
                k: v.tolist() if isinstance(v, np.ndarray) else v
                for k, v in metrics.items()
            }

    # Save as JSON
    results_path = output_dir / "experiment_results.json"
    with open(results_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n  Results saved to {results_path}")

    # Save as CSV summary
    csv_rows = []
    for model_name, metrics in results.items():
        if "error" not in metrics:
            csv_rows.append({
                "model": model_name,
                "accuracy": metrics["accuracy"],
                "f1_macro": metrics["f1_macro"],
                "f1_weighted": metrics["f1_weighted"],
                "precision": metrics["precision"],
                "recall": metrics["recall"],
                "train_time": metrics["train_time"],
            })

    if csv_rows:
        csv_path = output_dir / "metrics_summary.csv"
        pd.DataFrame(csv_rows).to_csv(csv_path, index=False)
        print(f"  Metrics saved to {csv_path}")

    # Save config copy
    config_copy_path = output_dir / "config.yaml"
    with open(config_copy_path, "w") as f:
        yaml.dump(config, f, default_flow_style=False)
    print(f"  Config saved to {config_copy_path}")


def print_summary(all_results: List[Dict]):
    """Print summary of all <experiment_name>."""
    print(f"\n{'=' * 70}")
    print("EXPERIMENT SUMMARY")
    print(f"{'=' * 70}")

    for exp in all_results:
        print(f"\n{exp['experiment_name']}:")
        print(f"  Output: {exp['output_dir']}")

        if exp['results']:
            # Find best model
            best_model = None
            best_f1 = 0
            for model_name, metrics in exp['results'].items():
                if "error" not in metrics and metrics.get("f1_macro", 0) > best_f1:
                    best_f1 = metrics["f1_macro"]
                    best_model = model_name

            if best_model:
                print(f"  Best model: {best_model} (F1={best_f1:.4f})")


def main():
    parser = argparse.ArgumentParser(
        description="Run <experiment_name> based on YAML configuration files"
    )
    parser.add_argument(
        "--config",
        type=str,
        nargs="+",
        required=True,
        help="Path(s) to experiment config file(s)",
    )
    args = parser.parse_args()

    # Handle glob patterns
    config_paths = []
    for pattern in args.config:
        if "*" in pattern:
            config_paths.extend(sorted(Path().glob(pattern)))
        else:
            config_paths.append(Path(pattern))

    print("=" * 70)
    print("CONFIG-BASED EXPERIMENT TRAINING")
    print("=" * 70)
    print(f"Config files: {len(config_paths)}")
    for p in config_paths:
        print(f"  - {p}")

    # Run <experiment_name>
    all_results = []
    total_start = time.time()

    for config_path in config_paths:
        if not config_path.exists():
            print(f"\nWARNING: Config file not found: {config_path}")
            continue

        result = run_experiment(str(config_path))
        all_results.append(result)

    total_time = time.time() - total_start

    # Print summary
    print_summary(all_results)

    print(f"\n{'=' * 70}")
    print(f"ALL <experiment_name> COMPLETE!")
    print(f"Total time: {total_time:.2f}s ({total_time / 60:.1f} minutes)")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()
