"""
Model Comparison Page
Compare all 6 classification models on the same input.
"""

import streamlit as st
import sys
from pathlib import Path
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import time

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

# Page configuration
st.set_page_config(
    page_title="Model Comparison - XAI Classifier",
    page_icon="⚖️",
    layout="wide",
)

# Model information
MODELS = {
    "naive_bayes": {
        "name": "Naive Bayes",
        "type": "Probabilistic",
        "description": "Uses Bayes theorem with feature independence assumption",
        "icon": "📊",
    },
    "svm": {
        "name": "SVM",
        "type": "Geometric",
        "description": "Finds optimal hyperplane to separate classes",
        "icon": "📐",
    },
    "random_forest": {
        "name": "Random Forest",
        "type": "Ensemble",
        "description": "Combines multiple decision trees for robust predictions",
        "icon": "🌲",
    },
    "knn": {
        "name": "KNN",
        "type": "Instance-based",
        "description": "Classifies based on nearest neighbors in feature space",
        "icon": "🎯",
    },
    "logistic_regression": {
        "name": "Logistic Regression",
        "type": "Linear",
        "description": "Linear model with sigmoid activation for probabilities",
        "icon": "📈",
    },
    "transformer": {
        "name": "Transformer",
        "type": "Deep Learning",
        "description": "BERT-based model with attention mechanism",
        "icon": "🤖",
    },
}


def simulate_model_comparison(text: str, dataset: str) -> dict:
    """Simulate predictions from all models."""
    import random

    # Define classes based on dataset
    if dataset == "imdb":
        classes = ["negative", "positive"]
    elif dataset == "turkish_sentiment":
        classes = ["negatif", "notr", "pozitif"]
    elif dataset == "ag_news":
        classes = ["World", "Sports", "Business", "Sci/Tech"]
    else:
        classes = ["siyaset", "dünya", "ekonomi", "kültür", "sağlık", "spor", "teknoloji"]

    results = {}

    for model_key in MODELS.keys():
        # Use different seeds for different models
        random.seed(hash(text + model_key) % 2**32)

        # Generate probabilities
        probs = [random.random() for _ in classes]
        total = sum(probs)
        probs = [p / total for p in probs]

        max_idx = probs.index(max(probs))
        prediction = classes[max_idx]
        confidence = probs[max_idx]

        # Simulate processing time
        if model_key == "transformer":
            proc_time = random.uniform(0.8, 1.5)
        elif model_key in ["svm", "random_forest"]:
            proc_time = random.uniform(0.1, 0.3)
        else:
            proc_time = random.uniform(0.01, 0.1)

        results[model_key] = {
            "prediction": prediction,
            "confidence": confidence,
            "probabilities": dict(zip(classes, probs)),
            "processing_time": proc_time,
        }

    return results


def main():
    st.title("⚖️ Model Comparison")
    st.markdown("Compare all 6 classification models on the same input")

    st.divider()

    # Model overview
    st.markdown("### 🔧 Available Models")

    cols = st.columns(3)
    for i, (key, info) in enumerate(MODELS.items()):
        with cols[i % 3]:
            st.markdown(f"""
            <div style="background-color: #f0f2f6; border-radius: 10px; padding: 1rem; margin: 0.5rem 0; height: 120px;">
                <h4>{info['icon']} {info['name']}</h4>
                <p style="color: #1E88E5; font-size: 0.9rem; margin: 0;">{info['type']}</p>
                <p style="font-size: 0.8rem; color: #666;">{info['description']}</p>
            </div>
            """, unsafe_allow_html=True)

    st.divider()

    # Input section
    st.markdown("### 📝 Input Text")

    col1, col2 = st.columns([3, 1])

    with col1:
        text_input = st.text_area(
            "Enter text to compare across all models:",
            height=100,
            placeholder="Enter your text here...",
        )

    with col2:
        dataset = st.selectbox(
            "Dataset:",
            ["imdb", "turkish_sentiment", "ag_news", "turkish_news"],
            format_func=lambda x: {
                "imdb": "🎬 IMDB",
                "turkish_sentiment": "🇹🇷 TR Sentiment",
                "ag_news": "📰 AG News",
                "turkish_news": "🇹🇷 TR News",
            }[x]
        )

        compare_button = st.button(
            "🔍 Compare Models",
            type="primary",
            use_container_width=True,
            disabled=not text_input,
        )

    # Run comparison
    if compare_button and text_input:
        st.divider()
        st.markdown("### 📊 Comparison Results")

        with st.spinner("Running all models..."):
            # Progress display
            progress_bar = st.progress(0)
            status_text = st.empty()

            results = {}
            for i, model_key in enumerate(MODELS.keys()):
                status_text.text(f"Running {MODELS[model_key]['name']}...")
                time.sleep(0.2)  # Simulate processing

                # Get simulated results
                all_results = simulate_model_comparison(text_input, dataset)
                results[model_key] = all_results[model_key]

                progress_bar.progress((i + 1) / len(MODELS))

            status_text.text("✅ All models completed!")
            progress_bar.progress(1.0)

        # Display results
        st.markdown("#### 🏆 Model Predictions")

        # Create comparison dataframe
        comparison_data = []
        for model_key, result in results.items():
            comparison_data.append({
                "Model": f"{MODELS[model_key]['icon']} {MODELS[model_key]['name']}",
                "Type": MODELS[model_key]["type"],
                "Prediction": result["prediction"],
                "Confidence": result["confidence"],
                "Time (s)": result["processing_time"],
            })

        df_comparison = pd.DataFrame(comparison_data)

        # Sort by confidence
        df_comparison = df_comparison.sort_values("Confidence", ascending=False)

        # Display table with highlighting
        st.dataframe(
            df_comparison.style.format({
                "Confidence": "{:.1%}",
                "Time (s)": "{:.3f}",
            }).background_gradient(subset=["Confidence"], cmap="Greens"),
            use_container_width=True,
            hide_index=True,
        )

        # Visualizations
        col1, col2 = st.columns(2)

        with col1:
            st.markdown("#### Confidence Comparison")

            fig = px.bar(
                df_comparison,
                x="Model",
                y="Confidence",
                color="Confidence",
                color_continuous_scale="Greens",
                title="Model Confidence Scores",
            )
            fig.update_layout(
                xaxis_title="",
                yaxis_title="Confidence",
                showlegend=False,
            )
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            st.markdown("#### Processing Time")

            fig = px.bar(
                df_comparison,
                x="Model",
                y="Time (s)",
                color="Time (s)",
                color_continuous_scale="Reds_r",
                title="Processing Time (seconds)",
            )
            fig.update_layout(
                xaxis_title="",
                yaxis_title="Time (s)",
                showlegend=False,
            )
            st.plotly_chart(fig, use_container_width=True)

        # Consensus analysis
        st.markdown("#### 🤝 Model Consensus")

        predictions = [r["prediction"] for r in results.values()]
        unique_preds = set(predictions)
        most_common = max(set(predictions), key=predictions.count)
        agreement = predictions.count(most_common) / len(predictions)

        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric("Models Agreeing", f"{predictions.count(most_common)}/{len(predictions)}")

        with col2:
            st.metric("Consensus Prediction", most_common.upper())

        with col3:
            st.metric("Agreement Rate", f"{agreement:.0%}")

        if agreement == 1.0:
            st.success("✅ All models agree on the prediction!")
        elif agreement >= 0.67:
            st.info(f"ℹ️ Majority ({predictions.count(most_common)}/6) of models predict: **{most_common}**")
        else:
            st.warning("⚠️ Models have significant disagreement. Consider reviewing the input.")

        # Detailed probability comparison
        st.divider()
        st.markdown("#### 📈 Detailed Probability Distribution")

        # Create heatmap of probabilities
        prob_data = []
        classes = list(results[list(results.keys())[0]]["probabilities"].keys())

        for model_key, result in results.items():
            for cls, prob in result["probabilities"].items():
                prob_data.append({
                    "Model": MODELS[model_key]["name"],
                    "Class": cls,
                    "Probability": prob,
                })

        df_probs = pd.DataFrame(prob_data)
        df_pivot = df_probs.pivot(index="Model", columns="Class", values="Probability")

        fig = px.imshow(
            df_pivot,
            labels=dict(x="Class", y="Model", color="Probability"),
            color_continuous_scale="Greens",
            title="Probability Heatmap (Model × Class)",
        )
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)


if __name__ == "__main__":
    main()
