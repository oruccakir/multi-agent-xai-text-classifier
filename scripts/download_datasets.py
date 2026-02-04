#!/usr/bin/env python
"""Script to download all datasets for the project."""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.data.data_loader import DataLoader


def main():
    print("=" * 60)
    print("Downloading datasets for Multi-Agent XAI Text Classifier")
    print("=" * 60)

    loader = DataLoader()

    # Download all datasets
    datasets_info = {
        "imdb": "IMDB Movie Reviews (English Sentiment)",
        "turkish_sentiment": "Turkish Sentiment Analysis",
        "ag_news": "AG News (English News Classification)",
        "turkish_news": "Turkish News Classification",
    }

    for dataset_name, description in datasets_info.items():
        print(f"\n{'─' * 60}")
        print(f"Downloading: {description}")
        print(f"{'─' * 60}")

        try:
            loader.download_dataset(dataset_name)

            # Get and display info
            info = loader.get_dataset_info(dataset_name)
            print(f"  Language: {info['language']}")
            print(f"  Task: {info['task']}")
            print(f"  Classes: {info['classes']}")
            print(f"  Source: {info['source']}")

        except Exception as e:
            print(f"  ERROR: {e}")

    print(f"\n{'=' * 60}")
    print("Dataset download complete!")
    print("=" * 60)

    # Show statistics for all datasets
    print("\nDataset Statistics:")
    print("-" * 60)

    for dataset_name in loader.SUPPORTED_DATASETS:
        try:
            print(f"\n{dataset_name.upper()}:")
            stats = loader.get_dataset_stats(dataset_name)
            for split, split_stats in stats.items():
                if "error" not in split_stats:
                    print(f"  {split}: {split_stats['num_samples']} samples")
                    print(f"    Avg text length: {split_stats['avg_text_length']:.0f} chars")
                    print(f"    Labels: {split_stats['label_distribution']}")
        except Exception as e:
            print(f"  Could not get stats: {e}")


if __name__ == "__main__":
    main()
