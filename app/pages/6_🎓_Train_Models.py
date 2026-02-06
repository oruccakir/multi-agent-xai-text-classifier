"""
Model Training Page
Train classification models directly from the Streamlit UI.
Supports dataset selection, preprocessing configuration, and model training.
"""

import streamlit as st
import sys
import os
import time
import json
from pathlib import Path
from datetime import datetime
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from dotenv import load_dotenv

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

# Load environment variables
load_dotenv(project_root / ".env.example")
load_dotenv(project_root / ".env", override=True)

from src.preprocessing.text_preprocessor import TextPreprocessor
from src.preprocessing.feature_extractor import FeatureExtractor
from src.models.naive_bayes import NaiveBayesClassifier
from src.models.svm import SVMClassifier
from src.models.random_forest import RandomForestClassifier
from src.models.knn import KNNClassifier
from src.models.logistic_regression import LogisticRegressionClassifier
from src.models.transformer import TransformerClassifier

# Page configuration
st.set_page_config(
    page_title="Train Models - XAI Classifier",
    page_icon="🎓",
    layout="wide",
)

# Directories
DATA_DIR = project_root / "data" / "processed"
MODELS_DIR = project_root / "data" / "models"

# Model info with all configurable parameters and detailed tooltips
MODELS_INFO = {
    "naive_bayes": {
        "name": "Naive Bayes",
        "description": "Fast, probabilistic classifier. Good baseline.",
        "icon": "📊",
        "params": {
            "alpha": {
                "type": "float",
                "default": 1.0,
                "min": 0.01,
                "max": 10.0,
                "step": 0.1,
                "help": "🎯 **Laplace Smoothing Parameter**\n\n"
                        "• **What it does:** Prevents zero probabilities for unseen words\n"
                        "• **Low values (0.01-0.5):** Less smoothing, model trusts training data more. Better for large datasets.\n"
                        "• **High values (1.0-10.0):** More smoothing, reduces overfitting. Better for small datasets.\n"
                        "• **Default (1.0):** Standard Laplace smoothing, good starting point."
            },
        }
    },
    "svm": {
        "name": "SVM",
        "description": "Support Vector Machine. High accuracy, slower training.",
        "icon": "📐",
        "params": {
            "C": {
                "type": "float",
                "default": 1.0,
                "min": 0.01,
                "max": 100.0,
                "step": 0.1,
                "help": "🎯 **Regularization Parameter (C)**\n\n"
                        "• **What it does:** Controls trade-off between smooth decision boundary and classifying training points correctly\n"
                        "• **Low values (0.01-0.5):** Stronger regularization, simpler model. Prevents overfitting but may underfit.\n"
                        "• **High values (10-100):** Weaker regularization, complex model. May overfit on small datasets.\n"
                        "• **Default (1.0):** Balanced regularization, good starting point."
            },
            "max_iter": {
                "type": "int",
                "default": 1000,
                "min": 100,
                "max": 10000,
                "step": 100,
                "help": "🎯 **Maximum Iterations**\n\n"
                        "• **What it does:** Maximum number of iterations for the solver to converge\n"
                        "• **Low values (100-500):** Faster training but may not converge on complex data\n"
                        "• **High values (5000-10000):** More time to converge, needed for large/complex datasets\n"
                        "• **Default (1000):** Usually sufficient for most datasets"
            },
        }
    },
    "random_forest": {
        "name": "Random Forest",
        "description": "Ensemble method. Robust and interpretable.",
        "icon": "🌲",
        "params": {
            "n_estimators": {
                "type": "int",
                "default": 100,
                "min": 10,
                "max": 500,
                "step": 10,
                "help": "🎯 **Number of Trees**\n\n"
                        "• **What it does:** Number of decision trees in the forest. More trees = more stable predictions\n"
                        "• **Low values (10-50):** Faster training, less stable, may underfit\n"
                        "• **High values (200-500):** Better accuracy, slower training, diminishing returns after ~300\n"
                        "• **Default (100):** Good balance of speed and accuracy"
            },
            "max_depth": {
                "type": "int",
                "default": 50,
                "min": 5,
                "max": 100,
                "step": 5,
                "help": "🎯 **Maximum Tree Depth**\n\n"
                        "• **What it does:** Maximum depth of each decision tree. Deeper = more complex patterns\n"
                        "• **Low values (5-20):** Simpler trees, faster, prevents overfitting\n"
                        "• **High values (50-100):** Can capture complex patterns but may overfit\n"
                        "• **Default (50):** Allows complex patterns while limiting overfitting"
            },
            "n_jobs": {
                "type": "int",
                "default": 4,
                "min": 1,
                "max": 16,
                "step": 1,
                "help": "🎯 **Parallel Jobs**\n\n"
                        "• **What it does:** Number of CPU cores to use for training\n"
                        "• **1:** Single core, slower but uses less memory\n"
                        "• **4-8:** Good balance for most systems\n"
                        "• **Higher values:** Faster but uses more RAM. Don't exceed your CPU cores."
            },
        }
    },
    "knn": {
        "name": "KNN",
        "description": "Instance-based learning. Simple but memory intensive.",
        "icon": "🎯",
        "params": {
            "n_neighbors": {
                "type": "int",
                "default": 5,
                "min": 1,
                "max": 50,
                "step": 1,
                "help": "🎯 **Number of Neighbors (K)**\n\n"
                        "• **What it does:** Number of nearest neighbors to consider for classification\n"
                        "• **Low values (1-3):** More sensitive to noise, can overfit\n"
                        "• **High values (15-50):** Smoother boundaries, may underfit\n"
                        "• **Default (5):** Good balance. Use odd numbers to avoid ties."
            },
            "metric": {
                "type": "select",
                "default": "euclidean",
                "options": ["euclidean", "manhattan"],
                "help": "🎯 **Distance Metric**\n\n"
                        "• **Euclidean:** Straight-line distance. Good for continuous features.\n"
                        "• **Manhattan:** Sum of absolute differences. More robust to outliers.\n"
                        "• **Note:** 'cosine' requires brute algorithm which is memory-intensive."
            },
            "weights": {
                "type": "select",
                "default": "distance",
                "options": ["uniform", "distance"],
                "help": "🎯 **Weight Function**\n\n"
                        "• **Uniform:** All neighbors have equal vote\n"
                        "• **Distance:** Closer neighbors have more influence (1/distance weighting)\n"
                        "• **Recommendation:** 'distance' usually performs better"
            },
            "algorithm": {
                "type": "select",
                "default": "ball_tree",
                "options": ["ball_tree", "kd_tree", "brute", "auto"],
                "help": "🎯 **Nearest Neighbor Algorithm**\n\n"
                        "• **ball_tree:** Memory efficient, good for high dimensions\n"
                        "• **kd_tree:** Fast for low dimensions (<20), memory efficient\n"
                        "• **brute:** Computes all distances. Required for 'cosine' metric. Memory intensive!\n"
                        "• **auto:** Automatically selects best algorithm"
            },
            "n_jobs": {
                "type": "int",
                "default": 1,
                "min": 1,
                "max": 8,
                "step": 1,
                "help": "🎯 **Parallel Jobs**\n\n"
                        "• **What it does:** Number of CPU cores for distance computation\n"
                        "• **1:** Single core, safest option for memory\n"
                        "• **2-4:** Moderate parallelization\n"
                        "• **Warning:** Higher values increase memory usage significantly for large datasets!"
            },
        },
        "warning": "Not recommended for datasets > 50K samples"
    },
    "logistic_regression": {
        "name": "Logistic Regression",
        "description": "Linear model. Fast and interpretable.",
        "icon": "📈",
        "params": {
            "C": {
                "type": "float",
                "default": 1.0,
                "min": 0.01,
                "max": 100.0,
                "step": 0.1,
                "help": "🎯 **Inverse Regularization Strength**\n\n"
                        "• **What it does:** Controls model complexity. Smaller C = stronger regularization\n"
                        "• **Low values (0.01-0.5):** Strong regularization, simpler model. Good for noisy data.\n"
                        "• **High values (10-100):** Weak regularization, may overfit\n"
                        "• **Default (1.0):** Balanced regularization"
            },
            "max_iter": {
                "type": "int",
                "default": 1000,
                "min": 100,
                "max": 10000,
                "step": 100,
                "help": "🎯 **Maximum Iterations**\n\n"
                        "• **What it does:** Maximum iterations for solver convergence\n"
                        "• **If you see 'ConvergenceWarning':** Increase this value\n"
                        "• **Large datasets:** May need 2000-5000 iterations"
            },
            "solver": {
                "type": "select",
                "default": "lbfgs",
                "options": ["lbfgs", "liblinear", "newton-cg", "sag", "saga"],
                "help": "🎯 **Optimization Algorithm**\n\n"
                        "• **lbfgs:** Good default, handles multiclass natively. Best for small-medium datasets.\n"
                        "• **liblinear:** Good for small datasets, supports L1 penalty\n"
                        "• **newton-cg:** Good for multiclass, no L1 support\n"
                        "• **sag/saga:** Fast for large datasets (>10K samples). saga supports L1."
            },
        }
    },
    "transformer": {
        "name": "Transformer",
        "description": "HuggingFace transformer. State-of-the-art accuracy, GPU recommended.",
        "icon": "🤖",
        "params": {
            "model_name": {
                "type": "text",
                "default": "distilbert-base-uncased",
                "help": "🎯 **HuggingFace Model ID**\n\n"
                        "• Any model from huggingface.co/models\n"
                        "• **distilbert-base-uncased:** Fast English model\n"
                        "• **bert-base-uncased:** Standard English BERT\n"
                        "• **dbmdz/bert-base-turkish-cased:** Turkish BERT\n"
                        "• **xlm-roberta-base:** Multilingual model"
            },
            "fine_tune_mode": {
                "type": "select",
                "default": "head_only",
                "options": ["head_only", "full", "lora"],
                "help": "🎯 **Fine-tuning Strategy**\n\n"
                        "• **head_only:** Freeze base, train classifier head only. Fastest, least memory.\n"
                        "• **full:** Train all parameters. Best accuracy, most memory.\n"
                        "• **lora:** Low-Rank Adaptation. Good accuracy with low memory overhead."
            },
            "epochs": {
                "type": "int",
                "default": 3,
                "min": 1,
                "max": 20,
                "step": 1,
                "help": "🎯 **Training Epochs**\n\n"
                        "• **1-2:** Quick training, may underfit\n"
                        "• **3-5:** Good balance for most tasks\n"
                        "• **10+:** Risk of overfitting, use with care"
            },
            "learning_rate": {
                "type": "float_scientific",
                "default": 2e-5,
                "options": [1e-5, 2e-5, 3e-5, 5e-5, 1e-4, 2e-4],
                "help": "🎯 **Learning Rate**\n\n"
                        "• **1e-5 - 2e-5:** Conservative, good for full fine-tuning\n"
                        "• **3e-5 - 5e-5:** Standard range for most tasks\n"
                        "• **1e-4 - 2e-4:** Aggressive, good for head_only or LoRA"
            },
            "batch_size": {
                "type": "select_int",
                "default": 16,
                "options": [4, 8, 16, 32, 64],
                "help": "🎯 **Batch Size**\n\n"
                        "• **4-8:** Low memory, slower training\n"
                        "• **16:** Good default\n"
                        "• **32-64:** Faster but requires more GPU memory"
            },
            "max_length": {
                "type": "select_int",
                "default": 256,
                "options": [64, 128, 256, 512],
                "help": "🎯 **Max Sequence Length**\n\n"
                        "• **64-128:** Short texts (tweets, titles)\n"
                        "• **256:** Medium texts (reviews, paragraphs)\n"
                        "• **512:** Long texts (articles). Uses more memory."
            },
        },
        "conditional_params": {
            "lora": {
                "lora_r": {
                    "type": "select_int",
                    "default": 8,
                    "options": [4, 8, 16, 32],
                    "help": "🎯 **LoRA Rank**\n\n"
                            "• **4:** Minimal parameters, fastest\n"
                            "• **8:** Good default balance\n"
                            "• **16-32:** More expressive, closer to full fine-tuning"
                },
                "lora_alpha": {
                    "type": "select_int",
                    "default": 16,
                    "options": [8, 16, 32, 64],
                    "help": "🎯 **LoRA Alpha (Scaling)**\n\n"
                            "• Usually set to 2x lora_r\n"
                            "• Higher values = stronger adaptation"
                },
                "lora_dropout": {
                    "type": "float",
                    "default": 0.1,
                    "min": 0.0,
                    "max": 0.5,
                    "step": 0.05,
                    "help": "🎯 **LoRA Dropout**\n\n"
                            "• **0.0:** No dropout\n"
                            "• **0.1:** Light regularization (default)\n"
                            "• **0.3-0.5:** Strong regularization for small datasets"
                },
            },
            "full": {
                "weight_decay": {
                    "type": "float",
                    "default": 0.01,
                    "min": 0.0,
                    "max": 0.3,
                    "step": 0.01,
                    "help": "🎯 **Weight Decay**\n\n"
                            "• **0.0:** No regularization\n"
                            "• **0.01:** Standard for transformers\n"
                            "• **0.1-0.3:** Strong regularization for small datasets"
                },
                "warmup_ratio": {
                    "type": "float",
                    "default": 0.1,
                    "min": 0.0,
                    "max": 0.3,
                    "step": 0.05,
                    "help": "🎯 **Warmup Ratio**\n\n"
                            "• Fraction of training steps for learning rate warmup\n"
                            "• **0.0:** No warmup\n"
                            "• **0.1:** 10% warmup (recommended)\n"
                            "• **0.2-0.3:** Longer warmup for stability"
                },
            },
        },
    },
}

# Preprocessing options with detailed tooltips
PREPROCESSING_OPTIONS = {
    "remove_stopwords": {
        "default": True,
        "help": "🎯 **Remove Stopwords**\n\n"
                "• **What it does:** Removes common words like 'the', 'is', 'and' that don't carry meaning\n"
                "• **Enable:** Reduces noise, focuses on meaningful words. Recommended for most cases.\n"
                "• **Disable:** Keep all words. Useful if stopwords carry meaning in your domain."
    },
    "remove_punctuation": {
        "default": True,
        "help": "🎯 **Remove Punctuation**\n\n"
                "• **What it does:** Removes punctuation marks (.,!?;: etc.)\n"
                "• **Enable:** Cleaner text, reduces vocabulary size. Recommended.\n"
                "• **Disable:** Keep punctuation. Useful for sentiment (e.g., '!!!' may indicate strong emotion)."
    },
    "remove_numbers": {
        "default": False,
        "help": "🎯 **Remove Numbers**\n\n"
                "• **What it does:** Removes all numeric characters\n"
                "• **Enable:** If numbers are noise (e.g., IDs, random numbers)\n"
                "• **Disable:** Keep numbers if they carry meaning (prices, dates, ratings)"
    },
    "remove_urls": {
        "default": True,
        "help": "🎯 **Remove URLs**\n\n"
                "• **What it does:** Removes web links (http://, www., etc.)\n"
                "• **Enable:** URLs are usually noise in text classification\n"
                "• **Disable:** Only if URLs contain relevant information"
    },
    "remove_html": {
        "default": True,
        "help": "🎯 **Remove HTML Tags**\n\n"
                "• **What it does:** Strips HTML tags like <p>, <div>, &nbsp;\n"
                "• **Enable:** If your data comes from web scraping\n"
                "• **Disable:** If your data is already clean text"
    },
    "lowercase": {
        "default": True,
        "help": "🎯 **Convert to Lowercase**\n\n"
                "• **What it does:** Converts all text to lowercase\n"
                "• **Enable:** Treats 'Good' and 'good' as same word. Recommended.\n"
                "• **Disable:** If case carries meaning (e.g., proper nouns, acronyms)"
    },
    "min_word_length": {
        "default": 2,
        "min": 1,
        "max": 5,
        "help": "🎯 **Minimum Word Length**\n\n"
                "• **What it does:** Removes words shorter than this length\n"
                "• **1:** Keep all words including single letters\n"
                "• **2:** Remove single letters (a, I). Good default.\n"
                "• **3+:** Also removes short words like 'is', 'it', 'to'"
    },
}

# Feature extraction tooltips
FEATURE_EXTRACTION_HELP = {
    "max_features": "🎯 **Maximum Features (Vocabulary Size)**\n\n"
                    "• **What it does:** Limits the number of unique words/n-grams in vocabulary\n"
                    "• **Low (1K-5K):** Faster, less memory. May miss important rare words.\n"
                    "• **Medium (10K-15K):** Good balance. Recommended starting point.\n"
                    "• **High (20K-30K):** Captures more vocabulary. Better for morphologically rich languages (Turkish).",

    "ngram_range": "🎯 **N-gram Range**\n\n"
                   "• **What it does:** Considers word combinations (unigrams, bigrams, trigrams)\n"
                   "• **(1,1):** Only single words. Fastest, smallest vocabulary.\n"
                   "• **(1,2):** Words + word pairs ('not good'). Captures negation. Recommended.\n"
                   "• **(1,3):** Also trigrams. Larger vocabulary, may overfit.",

    "min_df": "🎯 **Minimum Document Frequency**\n\n"
              "• **What it does:** Ignores words appearing in fewer than N documents\n"
              "• **1:** Keep all words including rare ones\n"
              "• **2-5:** Remove very rare words (typos, names). Recommended.\n"
              "• **10+:** Only frequent words. May lose important domain terms.",

    "max_df": "🎯 **Maximum Document Frequency**\n\n"
              "• **What it does:** Ignores words appearing in more than X% of documents\n"
              "• **0.5-0.7:** Aggressive filtering of common words\n"
              "• **0.9-0.95:** Remove only very common words. Recommended.\n"
              "• **1.0:** Keep all words regardless of frequency",

    "sublinear_tf": "🎯 **Sublinear TF Scaling**\n\n"
                    "• **What it does:** Applies logarithmic scaling: tf → 1 + log(tf)\n"
                    "• **Enable:** Reduces impact of word frequency. A word appearing 10x isn't 10x more important.\n"
                    "• **Disable:** Linear term frequency. Words appearing more often have proportionally higher weight.\n"
                    "• **Recommendation:** Enable for most text classification tasks.",

    "sparse": "🎯 **Sparse Matrix Format**\n\n"
              "• **What it does:** Store feature matrix in sparse format (only non-zero values)\n"
              "• **Enable (Recommended):** Much less memory usage. Essential for large datasets.\n"
              "• **Disable:** Dense matrix (all values stored). Uses more memory but some algorithms may be faster.\n"
              "• **Warning:** Disabling for large datasets (>50K samples) may cause out-of-memory errors!",
}

# Initialize session state
if "training_results" not in st.session_state:
    st.session_state.training_results = None
if "training_in_progress" not in st.session_state:
    st.session_state.training_in_progress = False


def get_available_datasets():
    """Get list of available datasets from processed data directory."""
    datasets = []
    if DATA_DIR.exists():
        # Find unique dataset names (files ending with _train.csv)
        for f in DATA_DIR.glob("*_train.csv"):
            name = f.stem.replace("_train", "")
            test_file = DATA_DIR / f"{name}_test.csv"
            if test_file.exists():
                train_df = pd.read_csv(f, nrows=5)
                if "text" in train_df.columns and "label" in train_df.columns:
                    # Get dataset info
                    train_size = sum(1 for _ in open(f)) - 1
                    test_size = sum(1 for _ in open(test_file)) - 1
                    datasets.append({
                        "name": name,
                        "train_path": f,
                        "test_path": test_file,
                        "train_size": train_size,
                        "test_size": test_size,
                    })
    return datasets


def detect_language(df: pd.DataFrame) -> str:
    """Detect language from sample texts."""
    sample_text = " ".join(df["text"].head(10).tolist())
    turkish_chars = set("çğıöşüÇĞİÖŞÜ")
    if turkish_chars & set(sample_text):
        return "turkish"
    return "english"


def create_model(model_name: str, params: dict):
    """Create a model instance with given parameters."""
    if model_name == "naive_bayes":
        return NaiveBayesClassifier(
            alpha=params.get("alpha", 1.0)
        )
    elif model_name == "svm":
        return SVMClassifier(
            C=params.get("C", 1.0),
            max_iter=params.get("max_iter", 1000),
            random_state=42,
        )
    elif model_name == "random_forest":
        return RandomForestClassifier(
            n_estimators=params.get("n_estimators", 100),
            max_depth=params.get("max_depth", 50),
            n_jobs=params.get("n_jobs", 4),
            random_state=42,
        )
    elif model_name == "knn":
        return KNNClassifier(
            n_neighbors=params.get("n_neighbors", 5),
            metric=params.get("metric", "euclidean"),
            weights=params.get("weights", "distance"),
            n_jobs=params.get("n_jobs", 1),
            algorithm=params.get("algorithm", "ball_tree"),
        )
    elif model_name == "logistic_regression":
        return LogisticRegressionClassifier(
            C=params.get("C", 1.0),
            max_iter=params.get("max_iter", 1000),
            solver=params.get("solver", "lbfgs"),
            random_state=42,
        )
    elif model_name == "transformer":
        return TransformerClassifier(
            model_name=params.get("model_name", "distilbert-base-uncased"),
            num_labels=params.get("num_labels", 2),
            fine_tune_mode=params.get("fine_tune_mode", "head_only"),
            learning_rate=params.get("learning_rate", 2e-5),
            batch_size=params.get("batch_size", 16),
            epochs=params.get("epochs", 3),
            max_length=params.get("max_length", 256),
            lora_r=params.get("lora_r", 8),
            lora_alpha=params.get("lora_alpha", 16),
            lora_dropout=params.get("lora_dropout", 0.1),
            warmup_ratio=params.get("warmup_ratio", 0.1),
            weight_decay=params.get("weight_decay", 0.01),
        )
    else:
        raise ValueError(f"Unknown model: {model_name}")


def train_models(
    dataset: dict,
    experiment_name: str,
    experiment_description: str,
    selected_models: list,
    model_params: dict,
    preprocessing_config: dict,
    feature_config: dict,
    sample_size: int = None,
    progress_callback=None,
):
    """Train selected models on the dataset."""
    results = {
        "experiment_name": experiment_name,
        "experiment_description": experiment_description,
        "dataset": dataset["name"],
        "timestamp": datetime.now().isoformat(),
        "models": {},
        "feature_extraction_time": 0,
        "total_time": 0,
    }

    total_start = time.time()

    # Load data
    if progress_callback:
        progress_callback(0.05, "Loading dataset...")

    train_df = pd.read_csv(dataset["train_path"])
    test_df = pd.read_csv(dataset["test_path"])

    # Sample if specified
    if sample_size and sample_size < len(train_df):
        train_df = train_df.sample(n=sample_size, random_state=42)
        test_sample = min(sample_size // 5, len(test_df))
        test_df = test_df.sample(n=test_sample, random_state=42)

    results["train_size"] = len(train_df)
    results["test_size"] = len(test_df)
    results["classes"] = train_df["label"].unique().tolist()

    X_train_text = train_df["text"].tolist()
    y_train = train_df["label"].values
    X_test_text = test_df["text"].tolist()
    y_test = test_df["label"].values

    # Separate sklearn and transformer models
    sklearn_models = [m for m in selected_models if m != "transformer"]
    has_transformer = "transformer" in selected_models

    # Preprocessing (for sklearn TF-IDF pipeline)
    if progress_callback:
        progress_callback(0.1, "Preprocessing texts...")

    preprocessor = TextPreprocessor(
        language=preprocessing_config.get("language", "english"),
        remove_stopwords=preprocessing_config.get("remove_stopwords", True),
        remove_punctuation=preprocessing_config.get("remove_punctuation", True),
        remove_numbers=preprocessing_config.get("remove_numbers", False),
        remove_urls=preprocessing_config.get("remove_urls", True),
        remove_html=preprocessing_config.get("remove_html", True),
        lowercase=preprocessing_config.get("lowercase", True),
        min_word_length=preprocessing_config.get("min_word_length", 2),
    )

    X_train_preprocessed = [preprocessor.preprocess(text) for text in X_train_text]
    X_test_preprocessed = [preprocessor.preprocess(text) for text in X_test_text]

    # Create output directory
    output_dir = MODELS_DIR / experiment_name / dataset["name"]
    output_dir.mkdir(parents=True, exist_ok=True)

    # Feature extraction (only needed for sklearn models)
    if sklearn_models:
        if progress_callback:
            progress_callback(0.15, "Extracting features...")

        fe_start = time.time()
        use_sparse = feature_config.get("sparse", True)

        feature_extractor = FeatureExtractor(
            method="tfidf",
            max_features=feature_config.get("max_features", 10000),
            ngram_range=tuple(feature_config.get("ngram_range", [1, 2])),
            min_df=feature_config.get("min_df", 2),
            max_df=feature_config.get("max_df", 0.95),
            sublinear_tf=feature_config.get("sublinear_tf", True),
        )

        X_train = feature_extractor.fit_transform(X_train_preprocessed, sparse=use_sparse)
        X_test = feature_extractor.transform(X_test_preprocessed, sparse=use_sparse)
        results["feature_extraction_time"] = time.time() - fe_start
        results["feature_shape"] = X_train.shape

        # Save feature extractor
        fe_path = output_dir / "feature_extractor.pkl"
        feature_extractor.save(str(fe_path))

    # Train each model
    num_models = len(selected_models)
    for i, model_name in enumerate(selected_models):
        progress_base = 0.2 + (0.7 * i / num_models)

        if progress_callback:
            progress_callback(progress_base, f"Training {MODELS_INFO[model_name]['name']}...")

        try:
            params = model_params.get(model_name, {})

            if model_name == "transformer":
                # Auto-detect num_labels
                num_labels = len(results["classes"])
                params["num_labels"] = num_labels

                model = create_model(model_name, params)

                # Transformer uses raw texts directly
                train_start = time.time()
                model.fit(X_train_text, y_train)
                train_time = time.time() - train_start

                # Evaluate with raw texts
                eval_start = time.time()
                eval_results = model.evaluate(X_test_text, y_test)
                eval_time = time.time() - eval_start

                # Save as directory (not .pkl)
                model_path = output_dir / "transformer.dir"
                model.save(str(model_path))
            else:
                model = create_model(model_name, params)

                # Train with TF-IDF features
                train_start = time.time()
                model.fit(X_train, y_train)
                train_time = time.time() - train_start

                # Evaluate
                eval_start = time.time()
                eval_results = model.evaluate(X_test, y_test)
                eval_time = time.time() - eval_start

                # Save model
                model_path = output_dir / f"{model_name}.pkl"
                model.save(str(model_path))

            results["models"][model_name] = {
                "accuracy": eval_results["accuracy"],
                "f1_macro": eval_results["f1_macro"],
                "f1_weighted": eval_results["f1_weighted"],
                "precision": eval_results["precision"],
                "recall": eval_results["recall"],
                "train_time": train_time,
                "eval_time": eval_time,
                "confusion_matrix": eval_results["confusion_matrix"].tolist(),
                "params": params,
                "status": "success",
            }

        except Exception as e:
            results["models"][model_name] = {
                "status": "error",
                "error": str(e),
            }

    # Save experiment results
    results["total_time"] = time.time() - total_start

    results_path = output_dir / "experiment_results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)

    # Build full config in the standard format
    import yaml
    import getpass

    # Get unique classes from training data
    full_train_df = pd.read_csv(dataset["train_path"])
    classes = full_train_df["label"].unique().tolist()

    # Build models config with enabled flag
    models_config = {}
    all_model_names = ["naive_bayes", "svm", "random_forest", "knn", "logistic_regression", "transformer"]
    for model_name in all_model_names:
        if model_name in selected_models:
            model_cfg = {"enabled": True}
            model_cfg.update(model_params.get(model_name, {}))
            models_config[model_name] = model_cfg
        else:
            models_config[model_name] = {"enabled": False}

    # Determine task type based on classes
    language = preprocessing_config.get("language", "english")
    if len(classes) == 2:
        task = "binary_classification"
    elif language == "turkish":
        task = "multiclass_sentiment" if any(c in str(classes).lower() for c in ["pozitif", "negatif", "notr"]) else "multiclass"
    else:
        task = "multiclass_news" if any(c in str(classes) for c in ["World", "Sports", "Business", "Sci/Tech"]) else "multiclass"

    config = {
        "experiment": {
            "name": experiment_name,
            "description": experiment_description or f"Experiment for {dataset['name']} classification",
            "author": getpass.getuser(),
        },
        "dataset": {
            "name": dataset["name"],
            "train_path": f"data/processed/{dataset['name']}_train.csv",
            "test_path": f"data/processed/{dataset['name']}_test.csv",
            "language": language,
            "task": task,
            "classes": classes,
            "sample_size": sample_size,
        },
        "preprocessing": {
            "remove_stopwords": preprocessing_config.get("remove_stopwords", True),
            "remove_punctuation": preprocessing_config.get("remove_punctuation", True),
            "remove_numbers": preprocessing_config.get("remove_numbers", False),
            "remove_urls": preprocessing_config.get("remove_urls", True),
            "remove_html": preprocessing_config.get("remove_html", True),
            "lowercase": preprocessing_config.get("lowercase", True),
            "min_word_length": preprocessing_config.get("min_word_length", 2),
        },
        "feature_extraction": {
            "method": "tfidf",
            "max_features": feature_config.get("max_features", 10000),
            "ngram_range": feature_config.get("ngram_range", [1, 2]),
            "min_df": feature_config.get("min_df", 2),
            "max_df": feature_config.get("max_df", 0.95),
            "sublinear_tf": feature_config.get("sublinear_tf", True),
            "sparse": feature_config.get("sparse", True),
        },
        "models": models_config,
        "training": {
            "random_state": 42,
        },
        "output": {
            "save_models": True,
            "save_feature_extractor": True,
            "save_metrics": True,
            "save_confusion_matrix": True,
        },
    }

    # Save config to output directory (data/models/...)
    config_path = output_dir / "config.yaml"
    with open(config_path, "w") as f:
        yaml.dump(config, f, default_flow_style=False, allow_unicode=True, sort_keys=False)

    # Also save config to configs/{experiment_name}/{dataset_name}.yaml
    configs_dir = project_root / "configs" / experiment_name
    configs_dir.mkdir(parents=True, exist_ok=True)
    configs_path = configs_dir / f"{dataset['name']}.yaml"

    # Add header comment
    config_content = f"# {dataset['name'].replace('_', ' ').title()} {experiment_name.title()} Experiment Configuration\n"
    config_content += f"# Generated from UI on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
    config_content += yaml.dump(config, default_flow_style=False, allow_unicode=True, sort_keys=False)

    with open(configs_path, "w") as f:
        f.write(config_content)

    if progress_callback:
        progress_callback(1.0, "Training complete!")

    return results


def main():
    st.title("🎓 Train Models")
    st.markdown("Train classification models directly from the UI")

    # Check if training is in progress
    if st.session_state.training_in_progress:
        st.warning("Training is in progress. Please wait...")
        return

    # Get available datasets
    datasets = get_available_datasets()

    if not datasets:
        st.error("No datasets found! Please add CSV files to `data/processed/` directory.")
        st.markdown("""
        ### Expected format:
        - `data/processed/{dataset_name}_train.csv`
        - `data/processed/{dataset_name}_test.csv`

        Each CSV should have `text` and `label` columns.
        """)
        return

    # Sidebar for quick actions
    with st.sidebar:
        st.markdown("## 🚀 Quick Start")
        st.markdown("""
        1. Select a dataset
        2. Choose models to train
        3. Configure parameters
        4. Click "Start Training"
        """)

        st.divider()
        st.markdown("## 📊 Available Datasets")
        for ds in datasets:
            st.markdown(f"**{ds['name']}**")
            st.caption(f"Train: {ds['train_size']:,} | Test: {ds['test_size']:,}")

    # Main content
    tab1, tab2, tab3 = st.tabs(["📁 Dataset", "🔧 Configuration", "🚀 Train"])

    # Tab 1: Dataset Selection
    with tab1:
        st.markdown("### Select Dataset")

        # Option to use existing or upload new
        dataset_source = st.radio(
            "Dataset source:",
            options=["Use existing dataset", "Upload new dataset"],
            horizontal=True,
        )

        if dataset_source == "Use existing dataset":
            col1, col2 = st.columns([2, 1])

            with col1:
                dataset_options = {ds["name"]: ds for ds in datasets}
                selected_dataset_name = st.selectbox(
                    "Choose dataset:",
                    options=list(dataset_options.keys()),
                    format_func=lambda x: f"📁 {x}",
                )
                selected_dataset = dataset_options[selected_dataset_name]

            with col2:
                st.markdown("### Dataset Info")
                st.metric("Training Samples", f"{selected_dataset['train_size']:,}")
                st.metric("Test Samples", f"{selected_dataset['test_size']:,}")

        else:
            # Upload custom dataset
            st.markdown("### Upload Custom Dataset")

            st.info("""
            **CSV Format Requirements:**
            - Must have `text` and `label` columns
            - You can upload a single file (will be split 80/20) or separate train/test files
            """)

            upload_mode = st.radio(
                "Upload mode:",
                options=["Single file (auto-split)", "Separate train/test files"],
                horizontal=True,
            )

            if upload_mode == "Single file (auto-split)":
                uploaded_file = st.file_uploader(
                    "Upload CSV file:",
                    type=["csv"],
                    help="File with 'text' and 'label' columns",
                )

                if uploaded_file:
                    df = pd.read_csv(uploaded_file)

                    if "text" not in df.columns or "label" not in df.columns:
                        st.error("CSV must have 'text' and 'label' columns!")
                    else:
                        # Dataset name
                        custom_name = st.text_input(
                            "Dataset name:",
                            value=uploaded_file.name.replace(".csv", ""),
                        )

                        # Split ratio
                        test_ratio = st.slider(
                            "Test set ratio:",
                            min_value=0.1,
                            max_value=0.3,
                            value=0.2,
                            step=0.05,
                        )

                        if st.button("Save Dataset", type="primary"):
                            # Split data
                            from sklearn.model_selection import train_test_split
                            train_df, test_df = train_test_split(
                                df, test_size=test_ratio, random_state=42, stratify=df["label"]
                            )

                            # Ensure directory exists
                            DATA_DIR.mkdir(parents=True, exist_ok=True)

                            # Save files
                            train_path = DATA_DIR / f"{custom_name}_train.csv"
                            test_path = DATA_DIR / f"{custom_name}_test.csv"

                            train_df.to_csv(train_path, index=False)
                            test_df.to_csv(test_path, index=False)

                            st.success(f"Dataset saved! Train: {len(train_df)}, Test: {len(test_df)}")
                            st.rerun()

                        # Show preview
                        st.dataframe(df.head(5), width="stretch")

                        # Create temporary dataset info
                        selected_dataset = {
                            "name": custom_name,
                            "train_size": int(len(df) * (1 - test_ratio)),
                            "test_size": int(len(df) * test_ratio),
                        }

                else:
                    selected_dataset = None

            else:
                col1, col2 = st.columns(2)

                with col1:
                    train_file = st.file_uploader("Training CSV:", type=["csv"], key="train_upload")
                with col2:
                    test_file = st.file_uploader("Test CSV:", type=["csv"], key="test_upload")

                if train_file and test_file:
                    train_df = pd.read_csv(train_file)
                    test_df = pd.read_csv(test_file)

                    if "text" not in train_df.columns or "label" not in train_df.columns:
                        st.error("Training CSV must have 'text' and 'label' columns!")
                        selected_dataset = None
                    elif "text" not in test_df.columns or "label" not in test_df.columns:
                        st.error("Test CSV must have 'text' and 'label' columns!")
                        selected_dataset = None
                    else:
                        custom_name = st.text_input(
                            "Dataset name:",
                            value=train_file.name.replace("_train.csv", "").replace(".csv", ""),
                        )

                        if st.button("Save Dataset", type="primary"):
                            # Ensure directory exists
                            DATA_DIR.mkdir(parents=True, exist_ok=True)

                            train_path = DATA_DIR / f"{custom_name}_train.csv"
                            test_path = DATA_DIR / f"{custom_name}_test.csv"

                            train_df.to_csv(train_path, index=False)
                            test_df.to_csv(test_path, index=False)

                            st.success(f"Dataset saved! Train: {len(train_df)}, Test: {len(test_df)}")
                            st.rerun()

                        selected_dataset = {
                            "name": custom_name,
                            "train_size": len(train_df),
                            "test_size": len(test_df),
                        }
                else:
                    selected_dataset = None

            if selected_dataset is None:
                st.warning("Please upload dataset files to continue.")
                return

        # After this point, we need train_path for the dataset
        # Check if it's an existing dataset with paths
        if "train_path" not in selected_dataset:
            # Dataset was just configured but not saved yet
            st.info("👆 Click 'Save Dataset' above to save and continue.")
            return

        # Show dataset preview
        st.markdown("### Data Preview")

        preview_df = pd.read_csv(selected_dataset["train_path"], nrows=10)
        st.dataframe(preview_df, width="stretch")

        # Detect language
        language = detect_language(preview_df)
        st.info(f"🌐 Detected Language: **{language.upper()}**")

        # Show class distribution
        st.markdown("### Class Distribution")

        full_train_df = pd.read_csv(selected_dataset["train_path"])
        class_counts = full_train_df["label"].value_counts()

        fig = px.pie(
            values=class_counts.values,
            names=class_counts.index,
            title="Training Set Class Distribution",
            hole=0.4,
        )
        st.plotly_chart(fig, width="stretch")

        # Sampling option
        st.markdown("### Sampling (Optional)")

        use_sampling = st.checkbox(
            "Use data sampling",
            help="Reduce training time by using a subset of data"
        )

        if use_sampling:
            sample_size = st.slider(
                "Sample size:",
                min_value=1000,
                max_value=min(100000, selected_dataset["train_size"]),
                value=min(50000, selected_dataset["train_size"]),
                step=1000,
            )
            st.info(f"Will use {sample_size:,} samples for training")
        else:
            sample_size = None

        # Store in session state
        st.session_state.selected_dataset = selected_dataset
        st.session_state.sample_size = sample_size
        st.session_state.detected_language = language

    # Tab 2: Configuration
    with tab2:
        st.markdown("### Experiment Info")

        col1, col2 = st.columns([1, 2])

        with col1:
            experiment_name = st.text_input(
                "Experiment name:",
                value="baseline",
                help="Models will be saved to data/models/{experiment_name}/{dataset}/",
            )

        with col2:
            experiment_description = st.text_input(
                "Description:",
                value="",
                placeholder="e.g., Baseline experiment with default hyperparameters",
                help="Brief description of this experiment for documentation",
            )

        st.session_state.experiment_name = experiment_name
        st.session_state.experiment_description = experiment_description

        st.divider()

        # Model Selection
        st.markdown("### Select Models to Train")

        selected_models = []
        model_params = {}

        # Get effective training size (sample size if sampling, else full size)
        sample_size = st.session_state.get("sample_size")
        effective_train_size = sample_size if sample_size else selected_dataset["train_size"]

        cols = st.columns(2)
        for i, (model_key, model_info) in enumerate(MODELS_INFO.items()):
            with cols[i % 2]:
                with st.container(border=True):
                    # Check if dataset is too large for KNN (use effective size with sampling)
                    disabled = False
                    knn_warning = None
                    if model_key == "knn":
                        if effective_train_size > 50000:
                            disabled = True
                            knn_warning = f"Dataset too large for KNN ({effective_train_size:,} samples)"
                        elif selected_dataset["train_size"] > 50000:
                            knn_warning = f"KNN enabled due to sampling ({effective_train_size:,} samples)"

                    enabled = st.checkbox(
                        f"{model_info['icon']} {model_info['name']}",
                        value=not disabled,
                        disabled=disabled,
                        key=f"model_{model_key}",
                    )

                    st.caption(model_info["description"])

                    if knn_warning:
                        if disabled:
                            st.warning(knn_warning)
                        else:
                            st.info(knn_warning)

                    if enabled and not disabled:
                        selected_models.append(model_key)

                        # Show parameters
                        with st.expander("Parameters"):
                            params = {}
                            for param_name, param_config in model_info["params"].items():
                                if param_config["type"] == "float":
                                    params[param_name] = st.slider(
                                        param_name,
                                        min_value=float(param_config["min"]),
                                        max_value=float(param_config["max"]),
                                        value=float(param_config["default"]),
                                        step=float(param_config.get("step", 0.1)),
                                        key=f"{model_key}_{param_name}",
                                        help=param_config["help"],
                                    )
                                elif param_config["type"] == "int":
                                    params[param_name] = st.slider(
                                        param_name,
                                        min_value=int(param_config["min"]),
                                        max_value=int(param_config["max"]),
                                        value=int(param_config["default"]),
                                        step=int(param_config.get("step", 1)),
                                        key=f"{model_key}_{param_name}",
                                        help=param_config["help"],
                                    )
                                elif param_config["type"] == "select":
                                    params[param_name] = st.selectbox(
                                        param_name,
                                        options=param_config["options"],
                                        index=param_config["options"].index(param_config["default"]),
                                        key=f"{model_key}_{param_name}",
                                        help=param_config["help"],
                                    )
                                elif param_config["type"] == "text":
                                    params[param_name] = st.text_input(
                                        param_name,
                                        value=param_config["default"],
                                        key=f"{model_key}_{param_name}",
                                        help=param_config["help"],
                                    )
                                elif param_config["type"] == "float_scientific":
                                    options = param_config["options"]
                                    option_labels = [f"{v:.0e}" for v in options]
                                    default_idx = options.index(param_config["default"])
                                    selected_label = st.selectbox(
                                        param_name,
                                        options=option_labels,
                                        index=default_idx,
                                        key=f"{model_key}_{param_name}",
                                        help=param_config["help"],
                                    )
                                    params[param_name] = options[option_labels.index(selected_label)]
                                elif param_config["type"] == "select_int":
                                    options = param_config["options"]
                                    default_idx = options.index(param_config["default"])
                                    params[param_name] = st.selectbox(
                                        param_name,
                                        options=options,
                                        index=default_idx,
                                        key=f"{model_key}_{param_name}",
                                        help=param_config["help"],
                                    )

                            # Conditional parameters (e.g., LoRA or full fine-tune specific)
                            conditional = model_info.get("conditional_params", {})
                            if conditional:
                                mode_key = params.get("fine_tune_mode", "")
                                mode_params = conditional.get(mode_key, {})
                                if mode_params:
                                    st.markdown(f"**{mode_key} settings:**")
                                    for cp_name, cp_config in mode_params.items():
                                        if cp_config["type"] == "float":
                                            params[cp_name] = st.slider(
                                                cp_name,
                                                min_value=float(cp_config["min"]),
                                                max_value=float(cp_config["max"]),
                                                value=float(cp_config["default"]),
                                                step=float(cp_config.get("step", 0.01)),
                                                key=f"{model_key}_{cp_name}",
                                                help=cp_config["help"],
                                            )
                                        elif cp_config["type"] == "select_int":
                                            options = cp_config["options"]
                                            default_idx = options.index(cp_config["default"])
                                            params[cp_name] = st.selectbox(
                                                cp_name,
                                                options=options,
                                                index=default_idx,
                                                key=f"{model_key}_{cp_name}",
                                                help=cp_config["help"],
                                            )

                            model_params[model_key] = params

        st.session_state.selected_models = selected_models
        st.session_state.model_params = model_params

        st.divider()

        # Preprocessing Config
        st.markdown("### Preprocessing Settings")

        with st.expander("Text Preprocessing Options", expanded=True):
            col1, col2 = st.columns(2)

            with col1:
                remove_stopwords = st.checkbox(
                    "Remove stopwords",
                    value=PREPROCESSING_OPTIONS["remove_stopwords"]["default"],
                    help=PREPROCESSING_OPTIONS["remove_stopwords"]["help"],
                )
                remove_punctuation = st.checkbox(
                    "Remove punctuation",
                    value=PREPROCESSING_OPTIONS["remove_punctuation"]["default"],
                    help=PREPROCESSING_OPTIONS["remove_punctuation"]["help"],
                )
                remove_numbers = st.checkbox(
                    "Remove numbers",
                    value=PREPROCESSING_OPTIONS["remove_numbers"]["default"],
                    help=PREPROCESSING_OPTIONS["remove_numbers"]["help"],
                )

            with col2:
                remove_urls = st.checkbox(
                    "Remove URLs",
                    value=PREPROCESSING_OPTIONS["remove_urls"]["default"],
                    help=PREPROCESSING_OPTIONS["remove_urls"]["help"],
                )
                remove_html = st.checkbox(
                    "Remove HTML tags",
                    value=PREPROCESSING_OPTIONS["remove_html"]["default"],
                    help=PREPROCESSING_OPTIONS["remove_html"]["help"],
                )
                lowercase = st.checkbox(
                    "Convert to lowercase",
                    value=PREPROCESSING_OPTIONS["lowercase"]["default"],
                    help=PREPROCESSING_OPTIONS["lowercase"]["help"],
                )

            min_word_length = st.slider(
                "Minimum word length:",
                min_value=PREPROCESSING_OPTIONS["min_word_length"]["min"],
                max_value=PREPROCESSING_OPTIONS["min_word_length"]["max"],
                value=PREPROCESSING_OPTIONS["min_word_length"]["default"],
                help=PREPROCESSING_OPTIONS["min_word_length"]["help"],
            )

        # Store preprocessing config
        st.session_state.preprocessing_config = {
            "language": st.session_state.get("detected_language", "english"),
            "remove_stopwords": remove_stopwords,
            "remove_punctuation": remove_punctuation,
            "remove_numbers": remove_numbers,
            "remove_urls": remove_urls,
            "remove_html": remove_html,
            "lowercase": lowercase,
            "min_word_length": min_word_length,
        }

        st.divider()

        # Feature Extraction Config
        st.markdown("### Feature Extraction (TF-IDF)")

        col1, col2 = st.columns(2)

        with col1:
            max_features = st.slider(
                "Max features:",
                min_value=1000,
                max_value=30000,
                value=10000,
                step=1000,
                help=FEATURE_EXTRACTION_HELP["max_features"],
            )

            min_df = st.slider(
                "Min document frequency:",
                min_value=1,
                max_value=10,
                value=2,
                help=FEATURE_EXTRACTION_HELP["min_df"],
            )

            sublinear_tf = st.checkbox(
                "Sublinear TF scaling",
                value=True,
                help=FEATURE_EXTRACTION_HELP["sublinear_tf"],
            )

            sparse = st.checkbox(
                "Use sparse matrix",
                value=True,
                help=FEATURE_EXTRACTION_HELP["sparse"],
            )

        with col2:
            ngram_min = st.number_input(
                "N-gram min:",
                min_value=1,
                max_value=3,
                value=1,
                help=FEATURE_EXTRACTION_HELP["ngram_range"],
            )
            ngram_max = st.number_input(
                "N-gram max:",
                min_value=1,
                max_value=3,
                value=2,
                help=FEATURE_EXTRACTION_HELP["ngram_range"],
            )

            max_df = st.slider(
                "Max document frequency:",
                min_value=0.5,
                max_value=1.0,
                value=0.95,
                step=0.05,
                help=FEATURE_EXTRACTION_HELP["max_df"],
            )

        st.session_state.feature_config = {
            "max_features": max_features,
            "ngram_range": [ngram_min, ngram_max],
            "min_df": min_df,
            "max_df": max_df,
            "sublinear_tf": sublinear_tf,
            "sparse": sparse,
        }

    # Tab 3: Train
    with tab3:
        st.markdown("### Training Summary")

        # Show experiment description if provided
        exp_desc = st.session_state.get("experiment_description", "")
        if exp_desc:
            st.info(f"**Description:** {exp_desc}")

        # Show summary
        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric("Dataset", st.session_state.get("selected_dataset", {}).get("name", "N/A"))
            sample = st.session_state.get("sample_size")
            if sample:
                st.metric("Sample Size", f"{sample:,}")
            else:
                train_size = st.session_state.get("selected_dataset", {}).get("train_size", 0)
                st.metric("Training Samples", f"{train_size:,}")

        with col2:
            st.metric("Experiment", st.session_state.get("experiment_name", "baseline"))
            st.metric("Models", len(st.session_state.get("selected_models", [])))

        with col3:
            st.metric("Max Features", st.session_state.get("feature_config", {}).get("max_features", 10000))
            ngram = st.session_state.get("feature_config", {}).get("ngram_range", [1, 2])
            st.metric("N-gram Range", f"{ngram[0]}-{ngram[1]}")

        # Models to train
        st.markdown("### Models to Train")

        selected_models = st.session_state.get("selected_models", [])
        if selected_models:
            for model_key in selected_models:
                info = MODELS_INFO[model_key]
                params = st.session_state.get("model_params", {}).get(model_key, {})
                params_str = ", ".join([f"{k}={v}" for k, v in params.items()])
                st.markdown(f"- {info['icon']} **{info['name']}** ({params_str or 'default'})")
        else:
            st.warning("No models selected! Go to Configuration tab to select models.")

        st.divider()

        # Training button
        col1, col2, col3 = st.columns([1, 2, 1])

        with col2:
            train_button = st.button(
                "🚀 Start Training",
                type="primary",
                width="stretch",
                disabled=len(selected_models) == 0,
            )

        if train_button:
            st.session_state.training_in_progress = True

            # Progress tracking
            progress_bar = st.progress(0)
            status_text = st.empty()

            def update_progress(progress, status):
                progress_bar.progress(progress)
                status_text.text(status)

            try:
                results = train_models(
                    dataset=st.session_state.selected_dataset,
                    experiment_name=st.session_state.experiment_name,
                    experiment_description=st.session_state.get("experiment_description", ""),
                    selected_models=st.session_state.selected_models,
                    model_params=st.session_state.model_params,
                    preprocessing_config=st.session_state.preprocessing_config,
                    feature_config=st.session_state.feature_config,
                    sample_size=st.session_state.sample_size,
                    progress_callback=update_progress,
                )

                st.session_state.training_results = results
                st.session_state.training_in_progress = False

                st.success("Training complete!")

            except Exception as e:
                st.session_state.training_in_progress = False
                st.error(f"Training failed: {str(e)}")
                raise e

    # Show results if available
    if st.session_state.training_results:
        st.divider()
        st.markdown("## 📊 Training Results")

        results = st.session_state.training_results

        # Summary metrics
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric("Dataset", results["dataset"])
        with col2:
            st.metric("Training Samples", f"{results['train_size']:,}")
        with col3:
            st.metric("Total Time", f"{results['total_time']:.1f}s")
        with col4:
            successful = sum(1 for m in results["models"].values() if m.get("status") == "success")
            st.metric("Models Trained", f"{successful}/{len(results['models'])}")

        # Model results table
        st.markdown("### Model Performance")

        model_rows = []
        for model_name, metrics in results["models"].items():
            if metrics.get("status") == "success":
                model_rows.append({
                    "Model": MODELS_INFO[model_name]["name"],
                    "Accuracy": metrics["accuracy"],
                    "F1 (Macro)": metrics["f1_macro"],
                    "F1 (Weighted)": metrics["f1_weighted"],
                    "Precision": metrics["precision"],
                    "Recall": metrics["recall"],
                    "Train Time (s)": metrics["train_time"],
                })

        if model_rows:
            df_results = pd.DataFrame(model_rows)
            df_results = df_results.sort_values("F1 (Macro)", ascending=False)

            st.dataframe(
                df_results.style.format({
                    "Accuracy": "{:.4f}",
                    "F1 (Macro)": "{:.4f}",
                    "F1 (Weighted)": "{:.4f}",
                    "Precision": "{:.4f}",
                    "Recall": "{:.4f}",
                    "Train Time (s)": "{:.2f}",
                }).background_gradient(subset=["F1 (Macro)"], cmap="Greens"),
                width="stretch",
                hide_index=True,
            )

            # Visualization
            st.markdown("### Performance Comparison")

            fig = px.bar(
                df_results,
                x="Model",
                y=["Accuracy", "F1 (Macro)", "Precision", "Recall"],
                barmode="group",
                title="Model Performance Metrics",
            )
            fig.update_layout(yaxis_title="Score", xaxis_title="")
            st.plotly_chart(fig, width="stretch")

            # Best model
            best_model = df_results.iloc[0]
            st.success(f"🏆 Best Model: **{best_model['Model']}** with F1 (Macro) = {best_model['F1 (Macro)']:.4f}")

        # Show errors if any
        errors = [(m, r) for m, r in results["models"].items() if r.get("status") == "error"]
        if errors:
            st.markdown("### Errors")
            for model_name, result in errors:
                st.error(f"**{MODELS_INFO[model_name]['name']}**: {result['error']}")

        # Output locations (show relative paths from repo root)
        st.markdown("### Output Locations")
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**Models & Results:**")
            output_path = f"data/models/{results['experiment_name']}/{results['dataset']}"
            st.code(output_path)
        with col2:
            st.markdown("**Config File:**")
            config_path = f"configs/{results['experiment_name']}/{results['dataset']}.yaml"
            st.code(config_path)

        # Clear results button
        if st.button("🗑️ Clear Results"):
            st.session_state.training_results = None
            st.rerun()


if __name__ == "__main__":
    main()
