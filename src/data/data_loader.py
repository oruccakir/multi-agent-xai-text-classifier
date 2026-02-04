"""Data loader for various text classification datasets."""

from pathlib import Path
from typing import Dict, List, Tuple


class DataLoader:
    """
    Data loader supporting multiple datasets:
    - IMDB (English sentiment)
    - Turkish Sentiment
    - AG News (English news classification)
    - Turkish News (TTC4900)
    """

    SUPPORTED_DATASETS = ["imdb", "turkish_sentiment", "ag_news", "turkish_news"]

    # Label mappings for each dataset
    LABEL_MAPS = {
        "imdb": {0: "negative", 1: "positive"},
        "turkish_sentiment": {"Negative": "negatif", "Positive": "pozitif", "Notr": "notr"},
        "ag_news": {0: "World", 1: "Sports", 2: "Business", 3: "Sci/Tech"},
        "turkish_news": {
            0: "siyaset",
            1: "dunya",
            2: "ekonomi",
            3: "kultur",
            4: "saglik",
            5: "spor",
            6: "teknoloji",
        },
    }

    def __init__(self, data_dir: str = "data"):
        self.data_dir = Path(data_dir)
        self.raw_dir = self.data_dir / "raw"
        self.processed_dir = self.data_dir / "processed"
        self._datasets_cache = {}

    def load_dataset(
        self,
        dataset_name: str,
        split: str = "train",
    ) -> Tuple[List[str], List[str]]:
        """
        Load a dataset.

        Args:
            dataset_name: Name of the dataset
            split: 'train', 'test', or 'validation'

        Returns:
            Tuple of (texts, labels)
        """
        if dataset_name not in self.SUPPORTED_DATASETS:
            raise ValueError(
                f"Unknown dataset: {dataset_name}. "
                f"Supported: {self.SUPPORTED_DATASETS}"
            )

        loader_method = getattr(self, f"_load_{dataset_name}", None)
        if loader_method is None:
            raise NotImplementedError(f"Loader for {dataset_name} not implemented")

        return loader_method(split)

    def _load_imdb(self, split: str) -> Tuple[List[str], List[str]]:
        """Load IMDB sentiment dataset from Hugging Face."""
        from datasets import load_dataset

        # IMDB only has train and test splits
        hf_split = "train" if split == "train" else "test"

        if "imdb" not in self._datasets_cache:
            self._datasets_cache["imdb"] = load_dataset("imdb")

        dataset = self._datasets_cache["imdb"][hf_split]

        texts = dataset["text"]
        labels = [self.LABEL_MAPS["imdb"][label] for label in dataset["label"]]

        return texts, labels

    def _load_turkish_sentiment(self, split: str) -> Tuple[List[str], List[str]]:
        """Load Turkish sentiment dataset from Hugging Face."""
        from datasets import load_dataset

        if "turkish_sentiment" not in self._datasets_cache:
            self._datasets_cache["turkish_sentiment"] = load_dataset(
                "winvoker/turkish-sentiment-analysis-dataset"
            )

        dataset = self._datasets_cache["turkish_sentiment"]

        if split == "train":
            data = dataset["train"]
        else:
            data = dataset["test"]

        texts = list(data["text"])
        # Labels are strings in this dataset ("Positive", "Negative")
        labels = [self.LABEL_MAPS["turkish_sentiment"].get(label, label) for label in data["label"]]

        return texts, labels

    def _load_ag_news(self, split: str) -> Tuple[List[str], List[str]]:
        """Load AG News dataset from Hugging Face."""
        from datasets import load_dataset

        # AG News only has train and test splits
        hf_split = "train" if split == "train" else "test"

        if "ag_news" not in self._datasets_cache:
            self._datasets_cache["ag_news"] = load_dataset("ag_news")

        dataset = self._datasets_cache["ag_news"][hf_split]

        texts = dataset["text"]
        labels = [self.LABEL_MAPS["ag_news"][label] for label in dataset["label"]]

        return texts, labels

    def _load_turkish_news(self, split: str) -> Tuple[List[str], List[str]]:
        """Load Turkish news dataset (TTC4900) from Hugging Face."""
        from datasets import load_dataset
        from sklearn.model_selection import train_test_split

        if "turkish_news" not in self._datasets_cache:
            # TTC4900 only has train split, so we'll create our own splits
            dataset = load_dataset("savasy/ttc4900")
            full_data = dataset["train"]

            # Create train/test split (80/20)
            texts = list(full_data["text"])
            labels = list(full_data["category"])

            train_texts, test_texts, train_labels, test_labels = train_test_split(
                texts, labels, test_size=0.2, random_state=42, stratify=labels
            )

            self._datasets_cache["turkish_news"] = {
                "train": {"text": train_texts, "category": train_labels},
                "test": {"text": test_texts, "category": test_labels},
            }

        data = self._datasets_cache["turkish_news"]

        if split == "train":
            texts = data["train"]["text"]
            category_labels = data["train"]["category"]
        else:
            texts = data["test"]["text"]
            category_labels = data["test"]["category"]

        labels = [self.LABEL_MAPS["turkish_news"][label] for label in category_labels]

        return texts, labels

    def get_dataset_info(self, dataset_name: str) -> Dict:
        """Get information about a dataset."""
        info = {
            "imdb": {
                "language": "english",
                "task": "binary_sentiment",
                "classes": list(self.LABEL_MAPS["imdb"].values()),
                "num_classes": 2,
                "description": "IMDB movie review sentiment classification",
                "source": "huggingface:imdb",
            },
            "turkish_sentiment": {
                "language": "turkish",
                "task": "multiclass_sentiment",
                "classes": list(self.LABEL_MAPS["turkish_sentiment"].values()),
                "num_classes": 3,
                "description": "Turkish product review sentiment classification (3-class)",
                "source": "huggingface:winvoker/turkish-sentiment-analysis-dataset",
            },
            "ag_news": {
                "language": "english",
                "task": "multiclass_news",
                "classes": list(self.LABEL_MAPS["ag_news"].values()),
                "num_classes": 4,
                "description": "AG News topic classification",
                "source": "huggingface:ag_news",
            },
            "turkish_news": {
                "language": "turkish",
                "task": "multiclass_news",
                "classes": list(self.LABEL_MAPS["turkish_news"].values()),
                "num_classes": 7,
                "description": "Turkish news topic classification (TTC4900)",
                "source": "huggingface:savasy/ttc4900",
            },
        }
        return info.get(dataset_name, {})

    def download_dataset(self, dataset_name: str) -> None:
        """Download a dataset if not already present."""
        from datasets import load_dataset

        print(f"Downloading {dataset_name}...")

        if dataset_name == "imdb":
            load_dataset("imdb")
        elif dataset_name == "turkish_sentiment":
            load_dataset("winvoker/turkish-sentiment-analysis-dataset")
        elif dataset_name == "ag_news":
            load_dataset("ag_news")
        elif dataset_name == "turkish_news":
            load_dataset("savasy/ttc4900")
        else:
            raise ValueError(f"Unknown dataset: {dataset_name}")

        print(f"Successfully downloaded {dataset_name}")

    def download_all_datasets(self) -> None:
        """Download all supported datasets."""
        for dataset_name in self.SUPPORTED_DATASETS:
            try:
                self.download_dataset(dataset_name)
            except Exception as e:
                print(f"Error downloading {dataset_name}: {e}")

    def get_dataset_stats(self, dataset_name: str) -> Dict:
        """Get statistics for a dataset."""
        stats = {}

        for split in ["train", "test"]:
            try:
                texts, labels = self.load_dataset(dataset_name, split)
                label_counts = {}
                for label in labels:
                    label_counts[label] = label_counts.get(label, 0) + 1

                stats[split] = {
                    "num_samples": len(texts),
                    "label_distribution": label_counts,
                    "avg_text_length": sum(len(t) for t in texts) / len(texts) if texts else 0,
                }
            except Exception as e:
                stats[split] = {"error": str(e)}

        return stats

    def save_to_csv(self, dataset_name: str, split: str, output_path: str = None) -> str:
        """Save a dataset split to CSV file."""
        import pandas as pd

        texts, labels = self.load_dataset(dataset_name, split)

        df = pd.DataFrame({"text": texts, "label": labels})

        if output_path is None:
            self.processed_dir.mkdir(parents=True, exist_ok=True)
            output_path = self.processed_dir / f"{dataset_name}_{split}.csv"

        df.to_csv(output_path, index=False)
        print(f"Saved {len(df)} samples to {output_path}")

        return str(output_path)
