# Multi-Agent XAI Text Classification System

## Detailed Project Documentation

**Project:** Multi-Agent Explainable AI-Enhanced Text Classification System
**Author:** Oruç Çakır
**Course:** BİL 443/564 - TOBB ETÜ
**Date:** Spring 2025-26

---

## Table of Contents

1. [Project Goal](#1-project-goal)
2. [System Architecture](#2-system-architecture)
3. [The Three Agents Explained](#3-the-three-agents-explained)
4. [Classification Methods](#4-classification-methods)
5. [Datasets](#5-datasets)
6. [Preprocessing Pipeline](#6-preprocessing-pipeline)
7. [Feature Extraction](#7-feature-extraction)
8. [Explainability (XAI)](#8-explainability-xai)
9. [Project Structure](#9-project-structure)
10. [Training vs Inference Flow](#10-training-vs-inference-flow)
11. [Example End-to-End Flow](#11-example-end-to-end-flow)
12. [What Makes This Project Unique](#12-what-makes-this-project-unique)

---

## 1. Project Goal

The goal is to build a **smart text classification system** that:

1. **Automatically classifies text** into categories (sentiment or news topics)
2. **Explains its decisions** in human-understandable language
3. **Adapts to different contexts** using a multi-agent architecture

### Key Features

- **Bilingual Support:** Works with both English and Turkish text
- **Multiple Models:** Compares 6 different classification algorithms
- **Explainable AI:** Uses LIME, SHAP, and LLM to explain predictions
- **Context-Aware:** Automatically detects text type and routes to appropriate model

---

## 2. System Architecture

The system uses a **three-agent architecture** where each agent has a specific responsibility:

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              USER INPUT                                      │
│                    "Bu ürün gerçekten harika!"                              │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                        AGENT 1: INTENT CLASSIFIER                            │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │ • Detects language: Turkish                                          │    │
│  │ • Detects domain: Product review (sentiment)                         │    │
│  │ • Recommends: turkish_sentiment model                                │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                      AGENT 2: CLASSIFICATION AGENT                           │
│  ┌───────────────┐    ┌───────────────┐    ┌───────────────────────────┐   │
│  │ PREPROCESSING │───▶│   FEATURE     │───▶│     CLASSIFICATION        │   │
│  │               │    │  EXTRACTION   │    │                           │   │
│  │ • Lowercase   │    │               │    │  ┌─────────────────────┐  │   │
│  │ • Remove stop │    │ • TF-IDF      │    │  │ 1. Naive Bayes      │  │   │
│  │ • Remove punct│    │     OR        │    │  │ 2. SVM              │  │   │
│  │ • Clean HTML  │    │ • Transformer │    │  │ 3. Random Forest    │  │   │
│  │               │    │   Embeddings  │    │  │ 4. KNN              │  │   │
│  └───────────────┘    └───────────────┘    │  │ 5. Logistic Reg.    │  │   │
│                                            │  │ 6. Transformer      │  │   │
│                                            │  └─────────────────────┘  │   │
│                                            └───────────────────────────┘   │
│                                                         │                   │
│                                    Output: "pozitif" (91% confidence)       │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                          AGENT 3: XAI AGENT                                  │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │ LIME/SHAP Analysis:                                                  │    │
│  │   "harika" → +0.45 (positive impact)                                │    │
│  │   "gerçekten" → +0.12 (positive impact)                             │    │
│  │   "ürün" → +0.02 (neutral)                                          │    │
│  ├─────────────────────────────────────────────────────────────────────┤    │
│  │ LLM Explanation:                                                     │    │
│  │   "Bu metin POZİTİF olarak sınıflandırıldı çünkü 'harika'           │    │
│  │    kelimesi güçlü bir memnuniyet ifadesi taşımaktadır."             │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                              FINAL OUTPUT                                    │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │ Prediction: POZİTİF                                                  │    │
│  │ Confidence: 91%                                                      │    │
│  │ Explanation: "harika" ve "gerçekten" kelimeleri memnuniyet           │    │
│  │              gösteriyor.                                             │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 3. The Three Agents Explained

### 3.1 Agent 1: Intent Classifier (Niyet Sınıflandırıcı)

**Purpose:** Understand what kind of text the user is providing and route it to the right model.

**Example:**
```
Input: "Apple announces new iPhone with revolutionary AI features"

Agent 1 Analysis:
├── Language Detection: English
├── Domain Detection: Technology/News
├── Context: News article
└── Recommendation: Use ag_news model with English preprocessing
```

**Why is this needed?**
- Different text types need different models
- Turkish sentiment model won't work well on English news
- This agent makes the system "context-aware"

**Implementation Strategy:**
- **Phase 1:** Start with Zero-Shot LLM (no training data needed)
- **Phase 2:** Optimize with a trained classifier for speed/cost reduction

**Key Functions:**
- `detect_language()`: Identifies if text is Turkish or English
- `detect_domain()`: Determines if text is sentiment, news, etc.
- `recommend_model()`: Suggests the best model for the input

---

### 3.2 Agent 2: Classification Agent (Sınıflandırıcı)

**Purpose:** The core classification engine with 3 sub-steps.

#### Step A: Preprocessing (Veri Ön İşleme)

Transforms raw text into clean, normalized form:

```
Raw Text:    "I LOVED this movie!!! <br><br> It was AMAZING :)"
                            │
                            ▼
After HTML removal:    "I LOVED this movie!!!  It was AMAZING :)"
                            │
                            ▼
After lowercase:       "i loved this movie!!! it was amazing :)"
                            │
                            ▼
After punctuation:     "i loved this movie it was amazing"
                            │
                            ▼
After stopwords:       "loved movie amazing"
                            │
                            ▼
Final:                 "loved movie amazing"
```

#### Step B: Feature Extraction (Öznitelik Çıkarımı)

Converts text to numerical vectors for machine learning.

#### Step C: Classification (Sınıflandırma)

Applies one of 6 algorithms to predict the class.

---

### 3.3 Agent 3: XAI Agent (Açıklanabilir YZ)

**Purpose:** Make the "black box" transparent by explaining WHY a decision was made.

**Components:**
1. **LIME:** Local perturbation-based explanations
2. **SHAP:** Game theory-based feature attribution
3. **LLM:** Natural language explanation generation

---

## 4. Classification Methods

The system implements 6 different classification algorithms:

| # | Model | Type | How It Works | Strengths | Weaknesses |
|---|-------|------|--------------|-----------|------------|
| 1 | **Naive Bayes** | Probabilistic | Uses Bayes theorem to calculate P(class\|features) | Fast training, works well with text, handles high dimensions | Assumes feature independence |
| 2 | **SVM** | Geometric | Finds optimal hyperplane to separate classes | Excellent with high-dimensional data, memory efficient | Slow on large datasets, sensitive to feature scaling |
| 3 | **Random Forest** | Ensemble | Combines multiple decision trees | Robust to noise, handles non-linear relationships | Memory intensive, slower prediction |
| 4 | **KNN** | Instance-based | Classifies based on k nearest neighbors | Simple, no training phase, adapts easily | Slow at prediction, sensitive to irrelevant features |
| 5 | **Logistic Regression** | Linear | Models probability using logistic function | Fast, interpretable coefficients, probabilistic output | Limited to linear decision boundaries |
| 6 | **Transformer** | Deep Learning | Uses attention mechanism (BERT architecture) | State-of-the-art accuracy, understands context | Requires GPU, slow training, needs lots of data |

### Model Comparison Matrix

```
                    Speed       Accuracy    Interpretability    Memory
                    ─────       ────────    ────────────────    ──────
Naive Bayes         █████       ███         ████                █
SVM                 ███         ████        ██                  ██
Random Forest       ██          ████        ███                 ████
KNN                 █           ███         █████               █████
Logistic Reg.       █████       ███         █████               █
Transformer         █           █████       █                   █████
```

---

## 5. Datasets

### 5.1 Overview

| Dataset | Language | Task | Classes | Train Size | Test Size |
|---------|----------|------|---------|------------|-----------|
| **IMDB** | English | Binary Sentiment | 2 | 25,000 | 25,000 |
| **Turkish Sentiment** | Turkish | Multi-class Sentiment | 3 | 440,679 | 48,965 |
| **AG News** | English | News Classification | 4 | 120,000 | 7,600 |
| **Turkish News** | Turkish | News Classification | 7 | 3,920 | 980 |

### 5.2 IMDB Dataset (English Sentiment)

**Task:** Binary classification of movie reviews

**Classes:**
- `negative`: Negative movie reviews
- `positive`: Positive movie reviews

**Example:**
```
Text:  "This movie was amazing, I loved every minute!"
Label: positive
```

### 5.3 Turkish Sentiment Dataset

**Task:** 3-class sentiment classification of product reviews

**Classes:**
- `negatif`: Negative reviews
- `pozitif`: Positive reviews
- `notr`: Neutral reviews

**Example:**
```
Text:  "Ürün fena değil ama beklediğim gibi değildi"
Label: notr
```

### 5.4 AG News Dataset (English News)

**Task:** 4-class news topic classification

**Classes:**
- `World`: International news
- `Sports`: Sports news
- `Business`: Business/finance news
- `Sci/Tech`: Science and technology news

**Example:**
```
Text:  "Apple stock rises after strong iPhone sales report"
Label: Business
```

### 5.5 Turkish News Dataset (TTC4900)

**Task:** 7-class news topic classification

**Classes:**
- `siyaset`: Politics
- `dünya`: World news
- `ekonomi`: Economy
- `kültür`: Culture
- `sağlık`: Health
- `spor`: Sports
- `teknoloji`: Technology

**Example:**
```
Text:  "Galatasaray Fenerbahçe derbisini 2-1 kazandı"
Label: spor
```

### 5.6 Class Distribution Visualization

```
IMDB (Balanced):
negative  ████████████████████  50%
positive  ████████████████████  50%

Turkish Sentiment (Imbalanced):
pozitif   ████████████████████████████  54%
notr      █████████████████             35%
negatif   █████                         11%

AG News (Balanced):
World     █████████████████████  25%
Sports    █████████████████████  25%
Business  █████████████████████  25%
Sci/Tech  █████████████████████  25%

Turkish News (Balanced):
siyaset   ██████████████  14.3%
dünya     ██████████████  14.3%
ekonomi   ██████████████  14.3%
kültür    ██████████████  14.3%
sağlık    ██████████████  14.3%
spor      ██████████████  14.3%
teknoloji ██████████████  14.3%
```

---

## 6. Preprocessing Pipeline

### 6.1 Overview

The preprocessing pipeline cleans and normalizes text data before feature extraction.

```
┌─────────────┐    ┌─────────────┐    ┌─────────────┐    ┌─────────────┐
│  Raw Text   │───▶│ HTML/URL    │───▶│  Lowercase  │───▶│ Punctuation │
│             │    │  Removal    │    │             │    │  Removal    │
└─────────────┘    └─────────────┘    └─────────────┘    └─────────────┘
                                                                │
                                                                ▼
┌─────────────┐    ┌─────────────┐    ┌─────────────┐    ┌─────────────┐
│   Clean     │◀───│ Min Length  │◀───│  Stopword   │◀───│  Tokenize   │
│   Text      │    │   Filter    │    │  Removal    │    │             │
└─────────────┘    └─────────────┘    └─────────────┘    └─────────────┘
```

### 6.2 Preprocessing Steps

| Step | Description | Example |
|------|-------------|---------|
| **HTML Removal** | Remove HTML tags and entities | `<br>hello</br>` → `hello` |
| **URL Removal** | Remove web links | `check https://example.com` → `check` |
| **Lowercase** | Convert to lowercase | `HELLO World` → `hello world` |
| **Punctuation** | Remove punctuation marks | `hello, world!` → `hello world` |
| **Stopwords** | Remove common words | `this is a movie` → `movie` |
| **Min Length** | Filter short words (< 2 chars) | `a I am here` → `am here` |

### 6.3 Stopwords

**English Stopwords (179 words):**
```
a, about, above, after, again, against, all, am, an, and, any, are,
aren't, as, at, be, because, been, before, being, below, between,
both, but, by, can't, cannot, could, couldn't, did, didn't, do, does,
doesn't, doing, don't, down, during, each, few, for, from, further,
had, hadn't, has, hasn't, have, haven't, having, he, he'd, he'll,
he's, her, here, here's, hers, herself, him, himself, his, how, ...
```

**Turkish Stopwords (128 words):**
```
acaba, ama, aslında, az, bazı, belki, beri, bile, bir, birçok, biri,
birkaç, birkez, birşey, birşeyi, biz, bize, bizden, bizi, bizim, bu,
buna, bunda, bundan, bunlar, bunları, bunların, bunu, bunun, burada,
çok, çünkü, da, daha, dahi, de, defa, değil, diğer, diye, dolayı, ...
```

### 6.4 Before/After Examples

**English (IMDB):**
```
BEFORE: "I LOVED this movie!!! <br><br> It was AMAZING :)"
AFTER:  "loved movie amazing"
```

**Turkish (Sentiment):**
```
BEFORE: "Bu ürün gerçekten çok harika, kesinlikle tavsiye ederim!"
AFTER:  "ürün gerçekten harika kesinlikle tavsiye ederim"
```

### 6.5 Preprocessing Statistics

| Dataset | Original Avg Length | Processed Avg Length | Reduction |
|---------|---------------------|----------------------|-----------|
| IMDB | 1,325 chars | 847 chars | 36.1% |
| Turkish Sentiment | 140 chars | 122 chars | 12.9% |
| AG News | 236 chars | 183 chars | 22.4% |
| Turkish News | 1,981 chars | 1,800 chars | 9.1% |

---

## 7. Feature Extraction

### 7.1 Overview

Feature extraction converts text into numerical vectors that machine learning models can process.

```
Text: "loved movie amazing"
              │
              ▼
    ┌─────────────────┐
    │ Feature         │
    │ Extraction      │
    │                 │
    │ ┌─────────────┐ │
    │ │   TF-IDF    │ │
    │ └─────────────┘ │
    │       OR        │
    │ ┌─────────────┐ │
    │ │ Transformer │ │
    │ │ Embeddings  │ │
    │ └─────────────┘ │
    └─────────────────┘
              │
              ▼
    [0.12, 0.45, 0.23, ...]
    (Numerical Vector)
```

### 7.2 Method 1: TF-IDF

**TF-IDF = Term Frequency × Inverse Document Frequency**

**How it works:**
1. **TF (Term Frequency):** How often a word appears in a document
2. **IDF (Inverse Document Frequency):** How rare a word is across all documents
3. **TF-IDF:** Words that are frequent in a document but rare overall get high scores

**Example:**
```
Document: "loved movie amazing"
Vocabulary: ["loved", "movie", "amazing", "bad", "terrible", ...]

TF-IDF Vector (sparse, 10,000 dimensions):
┌─────────────────────────────────────────────────────┐
│ [0, 0, 0.45, 0, 0, 0.32, 0, ..., 0.28, 0, 0]       │
│        ↑              ↑              ↑              │
│     "loved"       "movie"       "amazing"           │
└─────────────────────────────────────────────────────┘
```

**Characteristics:**
- Sparse vector (mostly zeros)
- Based on word frequency statistics
- Fast to compute
- Ignores word order and context
- "good" and "great" have completely different vectors

### 7.3 Method 2: Transformer Embeddings (Sentence-BERT)

**How it works:**
1. Pass text through pre-trained transformer model (BERT)
2. Get contextualized embeddings for each token
3. Pool into single sentence vector

**Example:**
```
Text: "loved movie amazing"
              │
              ▼
    ┌─────────────────┐
    │  Sentence-BERT  │
    │    Model        │
    └─────────────────┘
              │
              ▼
Dense Vector (384 or 768 dimensions):
┌─────────────────────────────────────────────────────┐
│ [0.12, -0.45, 0.78, 0.23, -0.11, ..., 0.34]        │
│                                                     │
│ Captures semantic meaning and context               │
└─────────────────────────────────────────────────────┘
```

**Characteristics:**
- Dense vector (no zeros)
- Captures semantic meaning
- "good" and "great" have similar vectors
- Understands context: "bank" (river) vs "bank" (financial)
- Slower to compute, requires more memory

### 7.4 Comparison

| Aspect | TF-IDF | Transformer Embeddings |
|--------|--------|------------------------|
| **Vector Type** | Sparse | Dense |
| **Dimensions** | 10,000+ | 384-768 |
| **Semantics** | No | Yes |
| **Context** | No | Yes |
| **Speed** | Fast | Slow |
| **Memory** | Low | High |
| **Best For** | Traditional ML | Deep Learning |

---

## 8. Explainability (XAI)

### 8.1 Why Explainability?

Traditional ML models are "black boxes" - they give predictions but don't explain why.

```
┌─────────────────────────────────────────────────────────┐
│                    BLACK BOX MODEL                       │
│                                                         │
│  Input: "This movie was terrible"                       │
│                    │                                    │
│                    ▼                                    │
│              ┌───────────┐                              │
│              │    ???    │                              │
│              └───────────┘                              │
│                    │                                    │
│                    ▼                                    │
│  Output: NEGATIVE (89%)                                 │
│                                                         │
│  WHY? We don't know!                                    │
└─────────────────────────────────────────────────────────┘
```

XAI makes models transparent:

```
┌─────────────────────────────────────────────────────────┐
│                  EXPLAINABLE MODEL                       │
│                                                         │
│  Input: "This movie was terrible"                       │
│                    │                                    │
│                    ▼                                    │
│              ┌───────────┐                              │
│              │  Classify │                              │
│              └───────────┘                              │
│                    │                                    │
│                    ▼                                    │
│  Output: NEGATIVE (89%)                                 │
│                                                         │
│  WHY?                                                   │
│  • "terrible" has strong negative impact (+0.52)       │
│  • "movie" is neutral (+0.02)                          │
│  • "this" and "was" are stopwords (ignored)            │
└─────────────────────────────────────────────────────────┘
```

### 8.2 LIME (Local Interpretable Model-agnostic Explanations)

**How LIME Works:**

1. Take the original text and prediction
2. Create many variations by removing/replacing words
3. See how each variation affects the prediction
4. Words that cause biggest changes are most important

**Example:**
```
Original: "This movie was absolutely fantastic and wonderful"
Prediction: POSITIVE (94%)

LIME Process:

Step 1: Create variations
┌────────────────────────────────────────────────────────────────┐
│ Variation                                        │ Prediction  │
├────────────────────────────────────────────────────────────────┤
│ "movie was absolutely fantastic and wonderful"   │ POSITIVE 89%│
│ "This was absolutely fantastic and wonderful"    │ POSITIVE 92%│
│ "This movie was fantastic and wonderful"         │ POSITIVE 91%│
│ "This movie was absolutely and wonderful"        │ POSITIVE 85%│
│ "This movie was absolutely fantastic and"        │ POSITIVE 88%│
│ "This movie was absolutely fantastic wonderful"  │ POSITIVE 90%│
└────────────────────────────────────────────────────────────────┘

Step 2: Calculate impact of each word removal
┌────────────────────────────────────────────────────────────────┐
│ Removing "This"       → 94% - 89% = 5% drop                    │
│ Removing "movie"      → 94% - 92% = 2% drop                    │
│ Removing "absolutely" → 94% - 91% = 3% drop                    │
│ Removing "fantastic"  → 94% - 85% = 9% drop  ← Most important! │
│ Removing "wonderful"  → 94% - 88% = 6% drop                    │
└────────────────────────────────────────────────────────────────┘

Step 3: Generate explanation
┌────────────────────────────────────────┐
│ Word Impact on POSITIVE prediction:    │
│   fantastic  ████████████  +0.42       │
│   wonderful  ████████      +0.31       │
│   absolutely ████          +0.15       │
│   movie      ██            +0.08       │
│   this       █             +0.02       │
└────────────────────────────────────────┘
```

### 8.3 SHAP (SHapley Additive exPlanations)

**Based on Game Theory:** Each word is a "player" and we calculate its fair contribution to the final prediction.

**How SHAP Works:**

1. Start with base prediction (average across all samples)
2. Calculate how each word shifts the prediction
3. Sum of all contributions = final prediction

**Example:**
```
Base prediction (average): 50% positive
Final prediction: 94% positive
Gap to explain: +44%

SHAP Values (how each word contributes to the +44%):
┌─────────────────────────────────────────────────────────────────┐
│                                                                 │
│  Base    fantastic   wonderful   absolutely   movie    Final    │
│  50%  ──── +18% ───── +12% ────── +8% ─────── +6% ──── 94%     │
│                                                                 │
│  Waterfall visualization:                                       │
│                                                                 │
│  50% ├────────────────────────────────────────────────┤        │
│       │████████████████████│ fantastic (+18%)                   │
│       │████████████│ wonderful (+12%)                           │
│       │████████│ absolutely (+8%)                               │
│       │██████│ movie (+6%)                                      │
│  94% ├────────────────────────────────────────────────┤        │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### 8.4 LLM Natural Language Explanation

Converts technical LIME/SHAP values into human-readable text.

**Input to LLM:**
```json
{
  "text": "This movie was absolutely fantastic and wonderful",
  "prediction": "POSITIVE",
  "confidence": 0.94,
  "top_features": [
    {"word": "fantastic", "impact": 0.42},
    {"word": "wonderful", "impact": 0.31},
    {"word": "absolutely", "impact": 0.15}
  ],
  "language": "english"
}
```

**LLM Prompt:**
```
Based on the following classification result, generate a human-friendly
explanation in 1-2 sentences:

Text: "This movie was absolutely fantastic and wonderful"
Prediction: POSITIVE (94% confidence)
Key words: "fantastic" (very positive), "wonderful" (positive)

Explain why this text was classified as positive.
```

**LLM Output:**
```
"The text is classified as POSITIVE with 94% confidence because the words
'fantastic' and 'wonderful' are strong positive sentiment indicators that
express enthusiastic approval of the movie."
```

### 8.5 XAI Comparison

| Aspect | LIME | SHAP | LLM |
|--------|------|------|-----|
| **Type** | Perturbation-based | Game theory | Generative |
| **Output** | Word weights | Feature attributions | Natural text |
| **Speed** | Slow (many predictions) | Medium | Fast |
| **Accuracy** | Good local | Mathematically sound | Depends on LLM |
| **Interpretability** | Technical | Technical | Human-friendly |

---

## 9. Project Structure

```
multi-agent-xai-text-classifier/
│
├── src/                              # Source code
│   │
│   ├── agents/                       # The 3 agents
│   │   ├── __init__.py
│   │   ├── base_agent.py             # Abstract base class
│   │   ├── intent_classifier.py      # Agent 1: Routes to correct model
│   │   ├── classification_agent.py   # Agent 2: Classifies text
│   │   └── xai_agent.py              # Agent 3: Explains decisions
│   │
│   ├── models/                       # 6 classification algorithms
│   │   ├── __init__.py
│   │   ├── base_model.py             # Abstract base class
│   │   ├── naive_bayes.py            # P(class|features) using Bayes theorem
│   │   ├── svm.py                    # Find optimal hyperplane separator
│   │   ├── random_forest.py          # Ensemble of decision trees
│   │   ├── knn.py                    # Classify by nearest neighbors
│   │   ├── logistic_regression.py    # Linear probability model
│   │   └── transformer.py            # BERT-based deep learning
│   │
│   ├── preprocessing/                # Text cleaning
│   │   ├── __init__.py
│   │   ├── text_preprocessor.py      # Clean, normalize text
│   │   └── feature_extractor.py      # TF-IDF or Transformer embeddings
│   │
│   ├── explainability/               # XAI modules
│   │   ├── __init__.py
│   │   ├── lime_explainer.py         # Local perturbation-based
│   │   ├── shap_explainer.py         # Game theory-based
│   │   └── llm_explainer.py          # Natural language generation
│   │
│   ├── data/
│   │   ├── __init__.py
│   │   └── data_loader.py            # Load datasets from HuggingFace
│   │
│   ├── utils/
│   │   ├── __init__.py
│   │   └── config.py                 # Configuration management
│   │
│   ├── __init__.py
│   └── pipeline.py                   # Orchestrates all 3 agents
│
├── data/
│   ├── raw/                          # Original unprocessed data
│   │   ├── imdb_train.csv
│   │   ├── imdb_test.csv
│   │   ├── turkish_sentiment_train.csv
│   │   ├── turkish_sentiment_test.csv
│   │   ├── ag_news_train.csv
│   │   ├── ag_news_test.csv
│   │   ├── turkish_news_train.csv
│   │   └── turkish_news_test.csv
│   │
│   ├── processed/                    # Cleaned, preprocessed data
│   │   ├── imdb_train.csv
│   │   ├── imdb_test.csv
│   │   ├── turkish_sentiment_train.csv
│   │   ├── turkish_sentiment_test.csv
│   │   ├── ag_news_train.csv
│   │   ├── ag_news_test.csv
│   │   ├── turkish_news_train.csv
│   │   └── turkish_news_test.csv
│   │
│   └── models/                       # Saved trained models (.pkl)
│
├── configs/
│   └── default.yaml                  # Configuration parameters
│
├── scripts/
│   ├── download_datasets.py          # Download from HuggingFace
│   └── preprocess_datasets.py        # Clean all datasets
│
├── notebooks/                        # Jupyter notebooks for experiments
│
├── tests/                            # Unit tests
│
├── reports/                          # Project reports
│   └── project_proposal.pdf
│
├── main.py                           # Entry point
├── setup.py                          # Package setup
├── requirements.txt                  # Dependencies
├── README.md                         # Quick start guide
├── ABOUT.md                          # This file
└── .gitignore
```

---

## 10. Training vs Inference Flow

### 10.1 Training Phase (Offline)

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           TRAINING PIPELINE                                  │
└─────────────────────────────────────────────────────────────────────────────┘

┌─────────────┐    ┌──────────────┐    ┌─────────────┐    ┌────────────────┐
│ Raw Dataset │───▶│ Preprocessor │───▶│  Feature    │───▶│ Train Model    │
│  (CSV)      │    │              │    │  Extractor  │    │                │
│             │    │ • lowercase  │    │             │    │ • fit(X, y)    │
│ • text      │    │ • stopwords  │    │ • TF-IDF    │    │ • tune params  │
│ • label     │    │ • punctuation│    │   OR        │    │ • validate     │
│             │    │              │    │ • BERT      │    │                │
└─────────────┘    └──────────────┘    └─────────────┘    └────────────────┘
                                                                   │
                                                                   ▼
                                                          ┌────────────────┐
                                                          │  Save Model    │
                                                          │  (.pkl file)   │
                                                          │                │
                                                          │ • model weights│
                                                          │ • vectorizer   │
                                                          │ • config       │
                                                          └────────────────┘
```

**Steps:**
1. Load raw data from CSV
2. Apply preprocessing (clean text)
3. Extract features (TF-IDF or embeddings)
4. Train model on features + labels
5. Evaluate using cross-validation
6. Save best model to disk

### 10.2 Inference Phase (Online)

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                          INFERENCE PIPELINE                                  │
└─────────────────────────────────────────────────────────────────────────────┘

┌─────────────┐    ┌─────────────┐    ┌─────────────┐    ┌─────────────────┐
│ User Input  │───▶│   Agent 1   │───▶│   Agent 2   │───▶│    Agent 3      │
│  (text)     │    │  (Intent)   │    │ (Classify)  │    │    (XAI)        │
│             │    │             │    │             │    │                 │
│ "Great      │    │ • detect    │    │ • preprocess│    │ • LIME analysis │
│  movie!"    │    │   language  │    │ • extract   │    │ • SHAP values   │
│             │    │ • detect    │    │   features  │    │ • LLM explain   │
│             │    │   domain    │    │ • predict   │    │                 │
│             │    │ • route     │    │             │    │                 │
└─────────────┘    └─────────────┘    └─────────────┘    └─────────────────┘
                                                                   │
                                                                   ▼
                                                    ┌───────────────────────┐
                                                    │     FINAL OUTPUT      │
                                                    │                       │
                                                    │ Prediction: POSITIVE  │
                                                    │ Confidence: 91%       │
                                                    │ Explanation: "Great"  │
                                                    │   indicates positive  │
                                                    │   sentiment...        │
                                                    └───────────────────────┘
```

**Steps:**
1. Receive user input text
2. Agent 1 detects language and domain
3. Agent 2 preprocesses, extracts features, predicts
4. Agent 3 generates explanation
5. Return prediction + confidence + explanation

---

## 11. Example End-to-End Flow

### Example 1: Turkish Product Review

```
┌─────────────────────────────────────────────────────────────────────────────┐
│ USER INPUT                                                                   │
│ "iPhone 15 Pro'nun kamerası harika ama pil ömrü beni hayal kırıklığına      │
│  uğrattı"                                                                    │
└─────────────────────────────────────────────────────────────────────────────┘
                                        │
                                        ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│ AGENT 1: INTENT CLASSIFIER                                                   │
│                                                                             │
│ Analysis:                                                                   │
│ ├── Language Detection: Turkish (detected "harika", "ama", "beni")         │
│ ├── Domain Detection: Product Review (detected "iPhone", "kamera", "pil")  │
│ └── Routing Decision: turkish_sentiment model                               │
└─────────────────────────────────────────────────────────────────────────────┘
                                        │
                                        ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│ AGENT 2: CLASSIFICATION                                                      │
│                                                                             │
│ Step A - Preprocessing:                                                     │
│ Input:  "iPhone 15 Pro'nun kamerası harika ama pil ömrü beni hayal         │
│          kırıklığına uğrattı"                                               │
│ Output: "iphone pro kamerası harika pil ömrü hayal kırıklığına uğrattı"    │
│                                                                             │
│ Step B - Feature Extraction:                                                │
│ Method: TF-IDF                                                              │
│ Vector: [0.12, 0.0, 0.34, 0.0, 0.28, ...]                                  │
│                                                                             │
│ Step C - Classification:                                                    │
│ Model: Logistic Regression (best performer on Turkish sentiment)            │
│                                                                             │
│ Results:                                                                    │
│ ┌─────────────────────────────────────────────┐                            │
│ │  pozitif:  31%                              │                            │
│ │  notr:     52%  ← WINNER                    │                            │
│ │  negatif:  17%                              │                            │
│ └─────────────────────────────────────────────┘                            │
└─────────────────────────────────────────────────────────────────────────────┘
                                        │
                                        ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│ AGENT 3: XAI                                                                 │
│                                                                             │
│ LIME Analysis:                                                              │
│ ┌───────────────────────────────────────────────────────────────┐          │
│ │ Word               │ Impact    │ Direction                    │          │
│ ├───────────────────────────────────────────────────────────────┤          │
│ │ harika             │ +0.28     │ → pushes toward pozitif     │          │
│ │ hayal kırıklığı    │ -0.31     │ → pushes toward negatif     │          │
│ │ iphone             │ +0.05     │ → slightly positive         │          │
│ │ pil                │ -0.08     │ → slightly negative         │          │
│ └───────────────────────────────────────────────────────────────┘          │
│                                                                             │
│ Observation: Positive and negative words cancel out → NOTR                  │
│                                                                             │
│ LLM Explanation (Turkish):                                                  │
│ "Bu yorum NÖTR olarak sınıflandırıldı. 'Harika' kelimesi olumlu duygu      │
│  taşırken, 'hayal kırıklığı' ifadesi olumsuz duygu içeriyor. Bu karşıt     │
│  duygular birbirini dengeleyerek nötr bir sonuç oluşturdu."                │
└─────────────────────────────────────────────────────────────────────────────┘
                                        │
                                        ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│ FINAL OUTPUT                                                                 │
│                                                                             │
│ ┌─────────────────────────────────────────────────────────────────────┐    │
│ │ Tahmin (Prediction): NÖTR                                           │    │
│ │ Güven (Confidence):  52%                                            │    │
│ │                                                                     │    │
│ │ Açıklama (Explanation):                                             │    │
│ │ Metin hem olumlu ("harika") hem olumsuz ("hayal kırıklığı")        │    │
│ │ ifadeler içerdiğinden nötr olarak değerlendirildi.                 │    │
│ │                                                                     │    │
│ │ Kelime Etkileri:                                                    │    │
│ │   harika           ████████████  +0.28 (pozitif)                   │    │
│ │   hayal kırıklığı  ████████████  -0.31 (negatif)                   │    │
│ └─────────────────────────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Example 2: English News Article

```
┌─────────────────────────────────────────────────────────────────────────────┐
│ USER INPUT                                                                   │
│ "Tesla stock surges 15% after record quarterly earnings report"             │
└─────────────────────────────────────────────────────────────────────────────┘
                                        │
                                        ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│ AGENT 1: INTENT CLASSIFIER                                                   │
│                                                                             │
│ ├── Language: English                                                       │
│ ├── Domain: News (Business/Finance)                                         │
│ └── Route to: ag_news model                                                 │
└─────────────────────────────────────────────────────────────────────────────┘
                                        │
                                        ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│ AGENT 2: CLASSIFICATION                                                      │
│                                                                             │
│ Preprocessing: "tesla stock surges record quarterly earnings report"        │
│                                                                             │
│ Classification Results:                                                     │
│ ┌─────────────────────────────────────────────┐                            │
│ │  World:     3%                              │                            │
│ │  Sports:    1%                              │                            │
│ │  Business: 89%  ← WINNER                    │                            │
│ │  Sci/Tech:  7%                              │                            │
│ └─────────────────────────────────────────────┘                            │
└─────────────────────────────────────────────────────────────────────────────┘
                                        │
                                        ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│ AGENT 3: XAI                                                                 │
│                                                                             │
│ Key Features:                                                               │
│   stock     ████████████████  +0.45 (strong business indicator)            │
│   earnings  ████████████      +0.32 (financial term)                       │
│   quarterly ████████          +0.18 (business reporting term)              │
│   Tesla     ██████            +0.12 (company name)                         │
│                                                                             │
│ LLM Explanation:                                                            │
│ "This article is classified as BUSINESS news because it discusses stock    │
│  performance ('surges 15%'), earnings reports, and quarterly results -     │
│  all key indicators of financial/business content."                        │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 12. What Makes This Project Unique

### Comparison with Traditional Systems

| Feature | Traditional System | This System |
|---------|-------------------|-------------|
| **Model Selection** | Manual - user must choose | Automatic - Agent 1 routes intelligently |
| **Language Support** | Usually single language | Bilingual (English + Turkish) |
| **Explainability** | None (black box) | Full (LIME + SHAP + LLM) |
| **Model Variety** | Single model | 6 models compared and benchmarked |
| **User Output** | Just class label | Label + Confidence + Human-readable explanation |
| **Adaptability** | Static | Context-aware, adapts to input type |

### Key Innovations

1. **Multi-Agent Architecture:** Separates concerns into specialized agents
2. **Automatic Routing:** No need for users to specify model or language
3. **Triple Explainability:** LIME for local explanations, SHAP for global, LLM for natural language
4. **Bilingual Support:** Native handling of both Turkish and English
5. **Model Benchmarking:** Compare 6 different algorithms on same data
6. **Human-Centric Output:** Explanations anyone can understand

### Research Questions Addressed

1. Which classification method performs best on Turkish vs English text?
2. How does binary classification compare to multi-class classification?
3. Can explainable AI make text classification more trustworthy?
4. Can a multi-agent system improve classification accuracy through intelligent routing?

---

## References

- **LIME:** Ribeiro, M.T., Singh, S., & Guestrin, C. (2016). "Why Should I Trust You?": Explaining the Predictions of Any Classifier
- **SHAP:** Lundberg, S.M., & Lee, S.I. (2017). A Unified Approach to Interpreting Model Predictions
- **BERT:** Devlin, J., et al. (2019). BERT: Pre-training of Deep Bidirectional Transformers
- **Sentence-BERT:** Reimers, N., & Gurevych, I. (2019). Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks

---

*Last Updated: February 2026*
