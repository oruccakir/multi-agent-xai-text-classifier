# Multi-Agent Explainable AI Text Classification System

A multi-agent system for explainable text classification combining traditional ML and transformer models with LIME/SHAP interpretability.

## Project Overview

This project implements a multi-agent architecture for text classification with explainability:

### Three-Agent Architecture

1. **Intent Classifier Agent** - Analyzes input text to determine context (movie review, product review, news) and routes to the appropriate model
2. **Classification Agent** - Performs text classification using one of six methods
3. **XAI Agent** - Explains predictions using LIME/SHAP analysis and LLM-generated natural language explanations

### Supported Classification Methods

- Naive Bayes
- Support Vector Machines (SVM)
- Random Forest
- K-Nearest Neighbors (KNN)
- Logistic Regression
- Transformer Models (BERT, DistilBERT)

### Supported Datasets

| Dataset | Language | Task | Classes |
|---------|----------|------|---------|
| IMDB | English | Binary Sentiment | Positive/Negative |
| Turkish Sentiment | Turkish | Binary Sentiment | Pozitif/Negatif |
| AG News | English | Multi-class News | World, Sports, Business, Sci/Tech |
| Turkish News | Turkish | Multi-class News | Spor, Ekonomi, Siyaset, Magazin, Teknoloji |

## Project Structure

```
multi-agent-xai-text-classifier/
├── src/
│   ├── agents/                 # Multi-agent components
│   │   ├── base_agent.py
│   │   ├── intent_classifier.py
│   │   ├── classification_agent.py
│   │   └── xai_agent.py
│   ├── models/                 # Classification models
│   │   ├── naive_bayes.py
│   │   ├── svm.py
│   │   ├── random_forest.py
│   │   ├── knn.py
│   │   ├── logistic_regression.py
│   │   └── transformer.py
│   ├── preprocessing/          # Text preprocessing
│   │   ├── text_preprocessor.py
│   │   └── feature_extractor.py
│   ├── explainability/         # XAI modules
│   │   ├── lime_explainer.py
│   │   ├── shap_explainer.py
│   │   └── llm_explainer.py
│   ├── data/                   # Data loading
│   │   └── data_loader.py
│   ├── utils/                  # Utilities
│   │   └── config.py
│   └── pipeline.py             # Main pipeline
├── data/
│   ├── raw/                    # Raw datasets
│   ├── processed/              # Processed data
│   └── models/                 # Trained models
├── configs/                    # Configuration files
├── notebooks/                  # Jupyter notebooks
├── tests/                      # Unit tests
├── reports/                    # Project reports
├── main.py                     # Entry point
├── requirements.txt
└── setup.py
```

## Installation

```bash
# Clone the repository
git clone <repository-url>
cd multi-agent-xai-text-classifier

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or
venv\Scripts\activate     # Windows

# Install dependencies
pip install -r requirements.txt

# Install package in development mode
pip install -e .
```

## Usage

### Training

```bash
python main.py --mode train --dataset imdb --model naive_bayes
```

### Evaluation

```bash
python main.py --mode evaluate --dataset imdb
```

### Prediction

```bash
python main.py --mode predict --text "This movie was fantastic!"
```

## Tech Stack

- **Python 3.9+**
- **scikit-learn** - Traditional ML models
- **transformers** - Transformer models
- **sentence-transformers** - Sentence embeddings
- **LIME** - Local interpretable explanations
- **SHAP** - Shapley value explanations
- **LangChain** - LLM integration

## License

MIT License
