"""
Create intent classifier dataset from existing datasets.

Her datasetten sample çekip dataset adını label olarak kullanır.
Intent classifier, gelen textin hangi datasete ait olduğunu classify eder.

Labels: imdb, turkish_sentiment, ag_news, turkish_news
"""

import pandas as pd
import os
from pathlib import Path

# Config
DATA_DIR = Path(__file__).parent.parent / "data" / "processed"
OUTPUT_DIR = Path(__file__).parent.parent / "data" / "intent_classifier"
OUTPUT_FILE = OUTPUT_DIR / "intent_dataset.csv"

SAMPLES_PER_DATASET = 2000  # her datasetten kaç sample alınacak (train)
TEST_SAMPLES_PER_DATASET = 500  # test için

DATASETS = {
    "imdb": {
        "train": "imdb_train.csv",
        "test": "imdb_test.csv",
    },
    "turkish_sentiment": {
        "train": "turkish_sentiment_train.csv",
        "test": "turkish_sentiment_test.csv",
    },
    "ag_news": {
        "train": "ag_news_train.csv",
        "test": "ag_news_test.csv",
    },
    "turkish_news": {
        "train": "turkish_news_train.csv",
        "test": "turkish_news_test.csv",
    },
}


def load_and_sample(filepath: Path, n: int, dataset_label: str, seed: int = 42) -> pd.DataFrame:
    df = pd.read_csv(filepath)
    # Sadece text kolonunu al, label'ı dataset adıyla değiştir
    df = df[["text"]].dropna()
    df["text"] = df["text"].astype(str).str.strip()
    df = df[df["text"].str.len() > 10]  # çok kısa metinleri filtrele

    n = min(n, len(df))
    sampled = df.sample(n=n, random_state=seed)
    sampled["label"] = dataset_label
    return sampled.reset_index(drop=True)


def create_intent_dataset(split: str, samples_per_dataset: int) -> pd.DataFrame:
    frames = []
    for dataset_name, files in DATASETS.items():
        filepath = DATA_DIR / files[split]
        if not filepath.exists():
            print(f"  [SKIP] {filepath} bulunamadı.")
            continue

        df = load_and_sample(filepath, samples_per_dataset, dataset_label=dataset_name)
        print(f"  {dataset_name:25s} → {len(df):5d} sample ({split})")
        frames.append(df)

    combined = pd.concat(frames, ignore_index=True)
    combined = combined.sample(frac=1, random_state=42).reset_index(drop=True)  # karıştır
    return combined


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"\n=== Intent Dataset Oluşturuluyor ===")
    print(f"Train: {SAMPLES_PER_DATASET} sample/dataset")
    print(f"Test : {TEST_SAMPLES_PER_DATASET} sample/dataset\n")

    print("[TRAIN]")
    train_df = create_intent_dataset("train", SAMPLES_PER_DATASET)
    train_path = OUTPUT_DIR / "intent_train.csv"
    train_df.to_csv(train_path, index=False)
    print(f"\n  Kaydedildi: {train_path}  ({len(train_df)} toplam)")

    print("\n[TEST]")
    test_df = create_intent_dataset("test", TEST_SAMPLES_PER_DATASET)
    test_path = OUTPUT_DIR / "intent_test.csv"
    test_df.to_csv(test_path, index=False)
    print(f"\n  Kaydedildi: {test_path}  ({len(test_df)} toplam)")

    # Birleşik tek dosya da kaydet
    all_df = pd.concat(
        [train_df.assign(split="train"), test_df.assign(split="test")],
        ignore_index=True
    )
    all_df.to_csv(OUTPUT_FILE, index=False)
    print(f"\n[ALL]  Kaydedildi: {OUTPUT_FILE}  ({len(all_df)} toplam)\n")

    # Label dağılımını göster
    print("=== Label Dağılımı (train) ===")
    print(train_df["label"].value_counts().to_string())
    print("\n=== Label Dağılımı (test) ===")
    print(test_df["label"].value_counts().to_string())
    print()


if __name__ == "__main__":
    main()
