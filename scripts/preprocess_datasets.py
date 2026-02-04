#!/usr/bin/env python
"""Script to preprocess all raw datasets and save to processed directory."""

import sys
from pathlib import Path

import pandas as pd

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.preprocessing.text_preprocessor import TextPreprocessor
from src.data.data_loader import DataLoader


def preprocess_dataset(
    input_path: Path,
    output_path: Path,
    language: str,
    dataset_name: str,
) -> dict:
    """Preprocess a single dataset file."""
    print(f"  Loading {input_path.name}...")
    df = pd.read_csv(input_path)

    original_count = len(df)
    original_avg_len = df["text"].apply(len).mean()

    # Initialize preprocessor for the appropriate language
    preprocessor = TextPreprocessor(
        language=language,
        remove_stopwords=True,
        remove_punctuation=True,
        remove_numbers=False,  # Keep numbers for news classification
        remove_urls=True,
        remove_html=True,
        lowercase=True,
        min_word_length=2,
    )

    print(f"  Preprocessing {len(df)} samples ({language})...")

    # Preprocess texts
    df["text"] = [preprocessor.preprocess(text) for text in df["text"]]

    # Remove empty texts after preprocessing
    df = df[df["text"].str.strip().str.len() > 0]

    processed_count = len(df)
    processed_avg_len = df["text"].apply(len).mean()

    # Save processed data
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)

    return {
        "original_count": original_count,
        "processed_count": processed_count,
        "removed_count": original_count - processed_count,
        "original_avg_len": original_avg_len,
        "processed_avg_len": processed_avg_len,
        "reduction_pct": (1 - processed_avg_len / original_avg_len) * 100,
    }


def main():
    print("=" * 70)
    print("Preprocessing datasets for Multi-Agent XAI Text Classifier")
    print("=" * 70)

    raw_dir = Path("data/raw")
    processed_dir = Path("data/processed")

    # Dataset configurations
    datasets = {
        "imdb": {"language": "english", "splits": ["train", "test"]},
        "turkish_sentiment": {"language": "turkish", "splits": ["train", "test"]},
        "ag_news": {"language": "english", "splits": ["train", "test"]},
        "turkish_news": {"language": "turkish", "splits": ["train", "test"]},
    }

    all_stats = {}

    for dataset_name, config in datasets.items():
        print(f"\n{'─' * 70}")
        print(f"Processing: {dataset_name.upper()} ({config['language']})")
        print(f"{'─' * 70}")

        dataset_stats = {}

        for split in config["splits"]:
            input_path = raw_dir / f"{dataset_name}_{split}.csv"
            output_path = processed_dir / f"{dataset_name}_{split}.csv"

            if not input_path.exists():
                print(f"  WARNING: {input_path} not found, skipping...")
                continue

            stats = preprocess_dataset(
                input_path=input_path,
                output_path=output_path,
                language=config["language"],
                dataset_name=dataset_name,
            )

            dataset_stats[split] = stats
            print(f"  Saved to {output_path}")
            print(f"    Samples: {stats['original_count']} → {stats['processed_count']} ({stats['removed_count']} removed)")
            print(f"    Avg length: {stats['original_avg_len']:.0f} → {stats['processed_avg_len']:.0f} chars ({stats['reduction_pct']:.1f}% reduction)")

        all_stats[dataset_name] = dataset_stats

    # Summary
    print(f"\n{'=' * 70}")
    print("PREPROCESSING COMPLETE - SUMMARY")
    print(f"{'=' * 70}")

    for dataset_name, dataset_stats in all_stats.items():
        print(f"\n{dataset_name.upper()}:")
        for split, stats in dataset_stats.items():
            print(f"  {split}: {stats['processed_count']} samples, avg {stats['processed_avg_len']:.0f} chars")

    # List processed files
    print(f"\n{'─' * 70}")
    print("Processed files:")
    print(f"{'─' * 70}")
    for f in sorted(processed_dir.glob("*.csv")):
        size_mb = f.stat().st_size / (1024 * 1024)
        print(f"  {f.name}: {size_mb:.1f} MB")


if __name__ == "__main__":
    main()
