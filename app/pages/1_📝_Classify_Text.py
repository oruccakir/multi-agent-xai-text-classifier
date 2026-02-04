"""
Single Text Classification Page
Allows users to classify individual texts with full XAI explanation.
"""

import streamlit as st
import sys
from pathlib import Path
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.preprocessing.text_preprocessor import TextPreprocessor

# Page configuration
st.set_page_config(
    page_title="Classify Text - XAI Classifier",
    page_icon="📝",
    layout="wide",
)

# Initialize session state
if "classification_result" not in st.session_state:
    st.session_state.classification_result = None


def detect_language(text: str) -> str:
    """Simple language detection based on character analysis."""
    turkish_chars = set("çğıöşüÇĞİÖŞÜ")
    text_chars = set(text)
    if turkish_chars & text_chars:
        return "turkish"
    # Check for common Turkish words
    turkish_words = {"ve", "bir", "bu", "için", "ile", "çok", "ama", "değil", "var", "olan"}
    words = set(text.lower().split())
    if len(turkish_words & words) >= 2:
        return "turkish"
    return "english"


def detect_domain(text: str, language: str) -> tuple:
    """Simple domain detection based on keywords."""
    text_lower = text.lower()

    if language == "english":
        # Check for movie review indicators
        movie_words = {"movie", "film", "actor", "actress", "director", "scene", "plot", "character"}
        if any(word in text_lower for word in movie_words):
            return "sentiment", "imdb"

        # Check for news indicators
        news_words = {"reuters", "ap", "announced", "reported", "official", "government", "company"}
        if any(word in text_lower for word in news_words):
            return "news", "ag_news"

        # Default to sentiment for English
        return "sentiment", "imdb"

    else:  # Turkish
        # Check for product review indicators
        product_words = {"ürün", "kargo", "teslimat", "sipariş", "satıcı", "fiyat", "kalite"}
        if any(word in text_lower for word in product_words):
            return "sentiment", "turkish_sentiment"

        # Check for news indicators
        news_words = {"haber", "açıkladı", "dedi", "bildirdi", "başbakan", "cumhurbaşkanı", "meclis"}
        if any(word in text_lower for word in news_words):
            return "news", "turkish_news"

        # Default to sentiment for Turkish
        return "sentiment", "turkish_sentiment"


def simulate_classification(text: str, dataset: str, model: str) -> dict:
    """
    Simulate classification results.
    In production, this would use actual trained models.
    """
    import random
    random.seed(hash(text) % 2**32)

    # Define classes based on dataset
    if dataset == "imdb":
        classes = ["negative", "positive"]
    elif dataset == "turkish_sentiment":
        classes = ["negatif", "notr", "pozitif"]
    elif dataset == "ag_news":
        classes = ["World", "Sports", "Business", "Sci/Tech"]
    else:  # turkish_news
        classes = ["siyaset", "dünya", "ekonomi", "kültür", "sağlık", "spor", "teknoloji"]

    # Generate random probabilities
    probs = [random.random() for _ in classes]
    total = sum(probs)
    probs = [p / total for p in probs]

    # Get prediction
    max_idx = probs.index(max(probs))
    prediction = classes[max_idx]
    confidence = probs[max_idx]

    # Create probability dict
    probabilities = {cls: prob for cls, prob in zip(classes, probs)}

    return {
        "prediction": prediction,
        "confidence": confidence,
        "probabilities": probabilities,
    }


def simulate_xai(text: str, prediction: str) -> dict:
    """
    Simulate XAI explanation.
    In production, this would use LIME/SHAP.
    """
    import random
    random.seed(hash(text + prediction) % 2**32)

    # Preprocess to get words
    words = text.lower().split()[:10]  # Take first 10 words
    words = [w.strip(".,!?") for w in words if len(w) > 2]

    # Generate random impacts
    impacts = {}
    for word in words:
        impact = random.uniform(-0.3, 0.5)
        impacts[word] = round(impact, 3)

    # Sort by absolute impact
    impacts = dict(sorted(impacts.items(), key=lambda x: abs(x[1]), reverse=True))

    # Generate explanation
    top_words = list(impacts.keys())[:3]
    explanation = f"The text is classified as '{prediction}' primarily due to the words: {', '.join(top_words)}."

    return {
        "word_impacts": impacts,
        "explanation": explanation,
    }


def main():
    st.title("📝 Single Text Classification")
    st.markdown("Classify a single text with full XAI explanation")

    st.divider()

    # Input section
    col1, col2 = st.columns([2, 1])

    with col1:
        st.markdown("### Enter Text")
        text_input = st.text_area(
            "Type or paste your text here:",
            height=150,
            placeholder="Example: Bu ürün gerçekten harika, kesinlikle tavsiye ederim!",
            key="text_input",
        )

        # Example texts
        st.markdown("**Quick Examples:**")
        example_col1, example_col2 = st.columns(2)

        with example_col1:
            if st.button("🎬 IMDB Example", use_container_width=True):
                st.session_state.text_input = "This movie was absolutely fantastic! The acting was superb and the plot kept me engaged throughout."
                st.rerun()

            if st.button("🇹🇷 Turkish Sentiment", use_container_width=True):
                st.session_state.text_input = "Bu ürün gerçekten harika, kesinlikle tavsiye ederim! Kargo çok hızlı geldi."
                st.rerun()

        with example_col2:
            if st.button("📰 AG News Example", use_container_width=True):
                st.session_state.text_input = "Apple Inc. announced record quarterly earnings, with iPhone sales exceeding analyst expectations by 15%."
                st.rerun()

            if st.button("🇹🇷 Turkish News", use_container_width=True):
                st.session_state.text_input = "Galatasaray, Süper Lig'in 25. haftasında Fenerbahçe'yi 2-1 mağlup ederek liderliğini sürdürdü."
                st.rerun()

    with col2:
        st.markdown("### Settings")

        # Auto-detect or manual selection
        detection_mode = st.radio(
            "Detection Mode:",
            ["🔍 Auto-detect", "⚙️ Manual"],
            index=0,
        )

        if detection_mode == "⚙️ Manual":
            dataset = st.selectbox(
                "Dataset:",
                ["imdb", "turkish_sentiment", "ag_news", "turkish_news"],
                format_func=lambda x: {
                    "imdb": "🎬 IMDB (English Sentiment)",
                    "turkish_sentiment": "🇹🇷 Turkish Sentiment",
                    "ag_news": "📰 AG News (English)",
                    "turkish_news": "🇹🇷 Turkish News",
                }[x]
            )

            model = st.selectbox(
                "Model:",
                ["naive_bayes", "svm", "random_forest", "knn", "logistic_regression", "transformer"],
                format_func=lambda x: {
                    "naive_bayes": "Naive Bayes",
                    "svm": "SVM",
                    "random_forest": "Random Forest",
                    "knn": "KNN",
                    "logistic_regression": "Logistic Regression",
                    "transformer": "Transformer",
                }[x]
            )
        else:
            dataset = None
            model = "auto"

        # Classify button
        st.markdown("---")
        classify_button = st.button(
            "🚀 Classify Text",
            type="primary",
            use_container_width=True,
            disabled=not text_input,
        )

    # Process classification
    if classify_button and text_input:
        with st.spinner("Processing..."):
            # Agent 1: Intent Detection
            language = detect_language(text_input)
            domain, detected_dataset = detect_domain(text_input, language)

            if detection_mode == "🔍 Auto-detect":
                dataset = detected_dataset
                model = "logistic_regression"  # Default model

            # Preprocess text
            preprocessor = TextPreprocessor(language=language)
            processed_text = preprocessor.preprocess(text_input)

            # Agent 2: Classification
            classification_result = simulate_classification(text_input, dataset, model)

            # Agent 3: XAI
            xai_result = simulate_xai(text_input, classification_result["prediction"])

            # Store results
            st.session_state.classification_result = {
                "text": text_input,
                "processed_text": processed_text,
                "language": language,
                "domain": domain,
                "dataset": dataset,
                "model": model,
                "classification": classification_result,
                "xai": xai_result,
            }

    # Display results
    if st.session_state.classification_result:
        result = st.session_state.classification_result

        st.divider()
        st.markdown("## 📊 Results")

        # Agent 1: Intent Detection
        st.markdown("### 🎯 Agent 1: Intent Detection")
        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric("Language", result["language"].upper())
        with col2:
            st.metric("Domain", result["domain"].capitalize())
        with col3:
            st.metric("Dataset", result["dataset"])

        # Show preprocessing
        with st.expander("🔧 Preprocessing Details"):
            st.markdown("**Original Text:**")
            st.info(result["text"])
            st.markdown("**Processed Text:**")
            st.success(result["processed_text"])

        st.divider()

        # Agent 2: Classification
        st.markdown("### 🔮 Agent 2: Classification")

        col1, col2 = st.columns([1, 2])

        with col1:
            # Prediction box
            st.markdown(f"""
            <div style="background-color: #e8f5e9; border-radius: 15px; padding: 2rem; text-align: center; margin: 1rem 0;">
                <p style="font-size: 1rem; color: #666; margin-bottom: 0.5rem;">Prediction</p>
                <p style="font-size: 2.5rem; font-weight: bold; color: #2e7d32; margin: 0;">{result["classification"]["prediction"].upper()}</p>
                <p style="font-size: 1.5rem; color: #666; margin-top: 0.5rem;">{result["classification"]["confidence"]:.1%} confidence</p>
            </div>
            """, unsafe_allow_html=True)

            st.markdown(f"**Model:** {result['model']}")

        with col2:
            # Probability chart
            probs = result["classification"]["probabilities"]
            df_probs = pd.DataFrame({
                "Class": list(probs.keys()),
                "Probability": list(probs.values()),
            })

            fig = px.bar(
                df_probs,
                x="Probability",
                y="Class",
                orientation="h",
                title="Class Probabilities",
                color="Probability",
                color_continuous_scale="Greens",
            )
            fig.update_layout(
                showlegend=False,
                xaxis_title="Probability",
                yaxis_title="",
                height=300,
            )
            st.plotly_chart(fig, use_container_width=True)

        st.divider()

        # Agent 3: XAI
        st.markdown("### 💡 Agent 3: Explainability (XAI)")

        col1, col2 = st.columns([1, 1])

        with col1:
            st.markdown("#### Word Impact Analysis")

            impacts = result["xai"]["word_impacts"]
            df_impacts = pd.DataFrame({
                "Word": list(impacts.keys()),
                "Impact": list(impacts.values()),
            })

            # Color based on positive/negative
            colors = ["#4caf50" if v > 0 else "#f44336" for v in impacts.values()]

            fig = go.Figure(go.Bar(
                x=list(impacts.values()),
                y=list(impacts.keys()),
                orientation="h",
                marker_color=colors,
            ))
            fig.update_layout(
                title="Word Contributions",
                xaxis_title="Impact Score",
                yaxis_title="",
                height=400,
            )
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            st.markdown("#### Natural Language Explanation")

            st.markdown(f"""
            <div style="background-color: #fff3e0; border-radius: 10px; padding: 1.5rem; border-left: 4px solid #ff9800; margin: 1rem 0;">
                <p style="font-size: 1.1rem; line-height: 1.6;">
                    💬 {result["xai"]["explanation"]}
                </p>
            </div>
            """, unsafe_allow_html=True)

            st.markdown("#### Impact Legend")
            st.markdown("""
            - 🟢 **Positive values**: Push toward predicted class
            - 🔴 **Negative values**: Push away from predicted class
            - Higher absolute values = stronger influence
            """)

        # Clear results button
        st.divider()
        if st.button("🗑️ Clear Results"):
            st.session_state.classification_result = None
            st.rerun()


if __name__ == "__main__":
    main()
