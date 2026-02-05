"""
Single Text Classification Page
Allows users to classify individual texts with full XAI explanation.
Supports experiment selection with automatic dataset/model detection.
Uses a 3-agent architecture: Intent Classifier, Classification Agent, XAI Agent.
Includes LIME and SHAP analysis for mathematical explainability.
"""

import streamlit as st
import sys
import os
from pathlib import Path
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

from src.agents.intent_classifier import IntentClassifierAgent
from src.agents.classification_agent import ClassificationAgent
from src.agents.xai_agent import XAIAgent
from src.preprocessing.text_preprocessor import TextPreprocessor
from src.preprocessing.feature_extractor import FeatureExtractor
from src.models.base_model import BaseModel

# Page configuration
st.set_page_config(
    page_title="Classify Text - XAI Classifier",
    page_icon="📝",
    layout="wide",
)

# Model directory
MODELS_DIR = project_root / "data" / "models"

# Available models
AVAILABLE_MODELS = ["naive_bayes", "svm", "random_forest", "knn", "logistic_regression"]

MODEL_DISPLAY_NAMES = {
    "naive_bayes": "Naive Bayes",
    "svm": "SVM",
    "random_forest": "Random Forest",
    "knn": "KNN",
    "logistic_regression": "Logistic Regression",
}

# Initialize session state
if "classification_result" not in st.session_state:
    st.session_state.classification_result = None
if "gemini_api_key" not in st.session_state:
    st.session_state.gemini_api_key = os.getenv("GEMINI_API_KEY", "")


def get_available_experiments():
    """Get list of available experiments (top-level directories in models folder)."""
    experiments = []
    if MODELS_DIR.exists():
        for exp_dir in sorted(MODELS_DIR.iterdir()):
            if exp_dir.is_dir() and not exp_dir.name.startswith('.'):
                has_models = False
                for dataset_dir in exp_dir.iterdir():
                    if dataset_dir.is_dir() and (dataset_dir / "feature_extractor.pkl").exists():
                        has_models = True
                        break
                if has_models:
                    experiments.append({
                        "name": exp_dir.name,
                        "path": exp_dir,
                    })
    return experiments


def get_available_datasets_for_experiment(exp_path: Path):
    """Get list of datasets available in an experiment."""
    datasets = []
    for dataset_dir in sorted(exp_path.iterdir()):
        if dataset_dir.is_dir() and (dataset_dir / "feature_extractor.pkl").exists():
            datasets.append(dataset_dir.name)
    return datasets


def get_available_models_for_dataset(exp_path: Path, dataset: str):
    """Get list of trained models for a dataset within an experiment."""
    available = []
    dataset_path = exp_path / dataset
    if dataset_path.exists():
        for model_name in AVAILABLE_MODELS:
            if (dataset_path / f"{model_name}.pkl").exists():
                available.append(model_name)
    return available


# Initialize agents as singletons
@st.cache_resource
def get_intent_classifier_agent(api_key: str = None):
    """Get or create Intent Classifier agent."""
    return IntentClassifierAgent(api_key=api_key if api_key else None)


@st.cache_resource
def get_classification_agent():
    """Get or create Classification agent."""
    return ClassificationAgent()


@st.cache_resource
def get_xai_agent(api_key: str = None):
    """Get or create XAI agent."""
    return XAIAgent(api_key=api_key if api_key else None)


@st.cache_resource
def load_model_and_extractor(exp_path: str, dataset: str, model_name: str):
    """Load model and feature extractor."""
    fe_path = Path(exp_path) / dataset / "feature_extractor.pkl"
    model_path = Path(exp_path) / dataset / f"{model_name}.pkl"

    feature_extractor = FeatureExtractor.load(str(fe_path))
    model = BaseModel.load(str(model_path))

    return model, feature_extractor


def create_predict_proba_fn(model, feature_extractor, preprocessor):
    """Create a predict_proba function for LIME/SHAP."""
    def predict_proba(texts):
        # Preprocess texts
        processed = [preprocessor.preprocess(t) for t in texts]
        # Extract features
        features = feature_extractor.transform(processed)
        # Get probabilities
        return model.predict_proba(features)
    return predict_proba


def plot_lime_explanation(lime_data: dict, prediction: str):
    """Create LIME visualization."""
    if not lime_data:
        return None

    word_contributions = lime_data.get("word_contributions", {})
    if not word_contributions:
        return None

    # Prepare data
    words = list(word_contributions.keys())
    values = list(word_contributions.values())

    # Sort by absolute value
    sorted_indices = np.argsort(np.abs(values))[::-1][:10]
    words = [words[i] for i in sorted_indices]
    values = [values[i] for i in sorted_indices]

    # Create colors based on positive/negative
    colors = ['#2ecc71' if v > 0 else '#e74c3c' for v in values]

    fig = go.Figure()
    fig.add_trace(go.Bar(
        y=words,
        x=values,
        orientation='h',
        marker_color=colors,
        text=[f"{v:+.4f}" for v in values],
        textposition='outside',
    ))

    fig.update_layout(
        title=f"LIME: Word Contributions to '{prediction}'",
        xaxis_title="Contribution Score",
        yaxis_title="",
        height=400,
        yaxis=dict(autorange="reversed"),
        showlegend=False,
    )

    # Add a vertical line at 0
    fig.add_vline(x=0, line_dash="dash", line_color="gray")

    return fig


def plot_shap_explanation(shap_data: dict, prediction: str):
    """Create SHAP visualization."""
    if not shap_data:
        return None

    word_shap_values = shap_data.get("word_shap_values", {})
    if not word_shap_values:
        return None

    # Prepare data
    words = list(word_shap_values.keys())
    values = list(word_shap_values.values())

    # Sort by absolute value
    sorted_indices = np.argsort(np.abs(values))[::-1][:10]
    words = [words[i] for i in sorted_indices]
    values = [values[i] for i in sorted_indices]

    # Create colors based on positive/negative
    colors = ['#3498db' if v > 0 else '#9b59b6' for v in values]

    fig = go.Figure()
    fig.add_trace(go.Bar(
        y=words,
        x=values,
        orientation='h',
        marker_color=colors,
        text=[f"{v:+.4f}" for v in values],
        textposition='outside',
    ))

    fig.update_layout(
        title=f"SHAP: Feature Importance for '{prediction}'",
        xaxis_title="SHAP Value",
        yaxis_title="",
        height=400,
        yaxis=dict(autorange="reversed"),
        showlegend=False,
    )

    # Add a vertical line at 0
    fig.add_vline(x=0, line_dash="dash", line_color="gray")

    return fig


def main():
    st.title("📝 Single Text Classification")
    st.markdown("Classify text with LIME & SHAP explainability analysis")

    # Get available experiments
    experiments = get_available_experiments()

    if not experiments:
        st.error("No trained experiments found! Please run training first:")
        st.code("python scripts/train_experiment.py --config configs/baseline/imdb.yaml", language="bash")
        return

    # Sidebar for experiment selection and settings
    with st.sidebar:
        st.markdown("## 🔬 Experiment Selection")

        exp_options = {exp["name"]: exp for exp in experiments}
        selected_exp_name = st.selectbox(
            "Select Experiment:",
            options=list(exp_options.keys()),
            format_func=lambda x: f"📁 {x}",
        )
        selected_exp = exp_options[selected_exp_name]

        # Show available datasets in this experiment
        available_datasets = get_available_datasets_for_experiment(selected_exp["path"])

        st.divider()
        st.markdown("## 📊 Available Datasets")
        for ds in available_datasets:
            models = get_available_models_for_dataset(selected_exp["path"], ds)
            st.markdown(f"**{ds}**: {len(models)} models")

        st.divider()
        st.markdown("## 🤖 Gemini API")

        # Check if API key is set
        has_api_key = bool(st.session_state.gemini_api_key)

        if has_api_key:
            st.success("API Key configured")
            if st.button("Clear API Key"):
                st.session_state.gemini_api_key = ""
                st.rerun()
        else:
            st.warning("No API key found")
            api_key_input = st.text_input(
                "Enter Gemini API Key:",
                type="password",
                help="Get your API key from https://aistudio.google.com/app/apikey",
            )
            if api_key_input:
                st.session_state.gemini_api_key = api_key_input
                st.rerun()

        st.divider()
        st.markdown("## 🔍 XAI Settings")
        use_lime = st.checkbox("Enable LIME Analysis", value=True, help="Local Interpretable Model-agnostic Explanations")
        use_shap = st.checkbox("Enable SHAP Analysis", value=True, help="SHapley Additive exPlanations")

        st.divider()
        st.markdown("## 🏗️ Agent Architecture")
        st.markdown("""
        **Agent 1: Intent Classifier**
        - Detects language & domain
        - Powered by Gemini

        **Agent 2: Classification**
        - ML model inference

        **Agent 3: XAI**
        - LIME analysis
        - SHAP analysis
        - TF-IDF analysis
        - Gemini explanation
        """)

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
        st.markdown("### Classification Info")
        gemini_status = "Enabled" if st.session_state.gemini_api_key else "Disabled"
        xai_methods = []
        if use_lime:
            xai_methods.append("LIME")
        if use_shap:
            xai_methods.append("SHAP")
        xai_methods.append("TF-IDF")

        st.info(f"""
        **Experiment:** {selected_exp_name}
        **Mode:** Auto-detect
        **Gemini:** {gemini_status}
        **XAI Methods:** {', '.join(xai_methods)}
        """)

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
        with st.spinner("Processing with 3-agent pipeline..."):
            api_key = st.session_state.gemini_api_key or None

            # ==========================================
            # AGENT 1: Intent Classification
            # ==========================================
            intent_agent = get_intent_classifier_agent(api_key)
            intent_result = intent_agent.process(text_input)

            language = intent_result["language"]
            domain = intent_result["domain"]
            detected_dataset = intent_result["dataset"]
            intent_confidence = intent_result["confidence"]
            intent_reasoning = intent_result["reasoning"]
            intent_gemini_available = intent_result["gemini_available"]

            # Check if dataset is available in experiment
            available_datasets = get_available_datasets_for_experiment(selected_exp["path"])

            if detected_dataset not in available_datasets:
                st.error(f"Dataset '{detected_dataset}' not found in experiment '{selected_exp_name}'.")
                st.info(f"Available datasets: {', '.join(available_datasets)}")
                return

            # Get available models for this dataset
            available_models = get_available_models_for_dataset(selected_exp["path"], detected_dataset)

            if not available_models:
                st.error(f"No trained models found for dataset '{detected_dataset}'.")
                return

            # Use logistic_regression as default, or first available
            if "logistic_regression" in available_models:
                selected_model = "logistic_regression"
            else:
                selected_model = available_models[0]

            # ==========================================
            # AGENT 2: Classification
            # ==========================================
            classification_agent = get_classification_agent()
            classification_result = classification_agent.classify_text(
                text=text_input,
                experiment_path=str(selected_exp["path"]),
                dataset=detected_dataset,
                model_name=selected_model,
                language=language,
            )

            if "error" in classification_result and classification_result.get("prediction") is None:
                st.error(f"Classification failed: {classification_result['error']}")
                return

            prediction = classification_result["prediction"]
            confidence = classification_result["confidence"]
            prob_dict = classification_result["probabilities"]
            processed_text = classification_result["processed_text"]
            class_names = classification_result.get("classes", list(prob_dict.keys()))

            # ==========================================
            # AGENT 3: XAI Explanation with LIME & SHAP
            # ==========================================
            xai_agent = get_xai_agent(api_key)

            # Load model and feature extractor for LIME/SHAP
            model, feature_extractor = load_model_and_extractor(
                str(selected_exp["path"]), detected_dataset, selected_model
            )

            # Create predict_proba function for LIME/SHAP
            preprocessor = TextPreprocessor(language=language)
            predict_proba_fn = create_predict_proba_fn(model, feature_extractor, preprocessor)

            # Generate explanation with LIME and SHAP
            xai_result = xai_agent.generate_explanation(
                text=text_input,
                processed_text=processed_text,
                prediction=prediction,
                confidence=confidence,
                probabilities=prob_dict,
                feature_extractor=feature_extractor,
                dataset=detected_dataset,
                model_name=MODEL_DISPLAY_NAMES.get(selected_model, selected_model),
                predict_proba_fn=predict_proba_fn,
                class_names=class_names,
                use_lime=use_lime,
                use_shap=use_shap,
            )

            # Store results
            st.session_state.classification_result = {
                "text": text_input,
                "processed_text": processed_text,
                "language": language,
                "domain": domain,
                "dataset": detected_dataset,
                "model": selected_model,
                "experiment": selected_exp_name,
                "intent": {
                    "confidence": intent_confidence,
                    "reasoning": intent_reasoning,
                    "gemini_available": intent_gemini_available,
                },
                "classification": {
                    "prediction": prediction,
                    "confidence": confidence,
                    "probabilities": prob_dict,
                    "class_names": class_names,
                },
                "xai": xai_result,
            }

    # Display results
    if st.session_state.classification_result:
        result = st.session_state.classification_result

        st.divider()
        st.markdown("## 📊 Results")

        # Agent 1: Intent Detection
        st.markdown("### 🎯 Agent 1: Intent Detection")

        # Show Gemini status for intent detection
        if result.get("intent", {}).get("gemini_available"):
            st.success("🤖 Powered by Google Gemini")
        else:
            st.warning("⚠️ Using fallback keyword-based detection")

        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric("Language", result["language"].upper())
        with col2:
            st.metric("Domain", result["domain"].capitalize())
        with col3:
            st.metric("Dataset", result["dataset"])
        with col4:
            intent_conf = result.get("intent", {}).get("confidence", 0)
            st.metric("Confidence", f"{intent_conf:.0%}")

        # Show reasoning if available
        intent_reasoning = result.get("intent", {}).get("reasoning", "")
        if intent_reasoning:
            st.info(f"💭 **Reasoning:** {intent_reasoning}")

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
            # Prediction box using native Streamlit components
            st.metric(
                label="Prediction",
                value=result["classification"]["prediction"].upper(),
                delta=f"{result['classification']['confidence']:.1%} confidence",
            )

            st.markdown(f"**Model:** {MODEL_DISPLAY_NAMES.get(result['model'], result['model'])}")

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

        # Show available XAI methods
        xai_methods_used = []
        if result["xai"].get("lime_available"):
            xai_methods_used.append("LIME")
        if result["xai"].get("shap_available"):
            xai_methods_used.append("SHAP")
        xai_methods_used.append("TF-IDF")
        if result["xai"].get("gemini_available"):
            xai_methods_used.append("Gemini")

        st.success(f"🔍 XAI Methods Used: {', '.join(xai_methods_used)}")

        # Natural Language Explanation
        st.markdown("#### 🤖 Natural Language Explanation")
        st.info(result['xai']['natural_explanation'])

        # LIME and SHAP visualizations
        col1, col2 = st.columns(2)

        with col1:
            st.markdown("#### 📊 LIME Analysis")
            lime_data = result["xai"].get("lime_explanation", {})
            if lime_data:
                lime_fig = plot_lime_explanation(lime_data, result["classification"]["prediction"])
                if lime_fig:
                    st.plotly_chart(lime_fig, use_container_width=True)

                # Show LIME details
                with st.expander("LIME Details"):
                    if lime_data.get("score"):
                        st.metric("Local Model R²", f"{lime_data['score']:.4f}")
                    st.markdown("**Positive Words (support prediction):**")
                    for word, score in list(lime_data.get("positive_words", {}).items())[:5]:
                        st.markdown(f"- `{word}`: +{score:.4f}")
                    st.markdown("**Negative Words (against prediction):**")
                    for word, score in list(lime_data.get("negative_words", {}).items())[:5]:
                        st.markdown(f"- `{word}`: {score:.4f}")
            else:
                st.warning("LIME analysis not available. Enable it in settings.")

        with col2:
            st.markdown("#### 📈 SHAP Analysis")
            shap_data = result["xai"].get("shap_explanation", {})
            if shap_data:
                shap_fig = plot_shap_explanation(shap_data, result["classification"]["prediction"])
                if shap_fig:
                    st.plotly_chart(shap_fig, use_container_width=True)

                # Show SHAP details
                with st.expander("SHAP Details"):
                    if shap_data.get("base_value") is not None:
                        st.metric("Base Value", f"{shap_data['base_value']:.4f}")
                    st.markdown("**Positive SHAP Values:**")
                    for word, score in list(shap_data.get("positive_words", {}).items())[:5]:
                        st.markdown(f"- `{word}`: +{score:.4f}")
                    st.markdown("**Negative SHAP Values:**")
                    for word, score in list(shap_data.get("negative_words", {}).items())[:5]:
                        st.markdown(f"- `{word}`: {score:.4f}")
            else:
                st.warning("SHAP analysis not available. Enable it in settings.")

        # TF-IDF Analysis
        st.markdown("#### 📚 TF-IDF Word Importance")
        col1, col2 = st.columns([1, 1])

        with col1:
            impacts = result["xai"]["word_impacts"]
            if impacts:
                df_impacts = pd.DataFrame({
                    "Word": list(impacts.keys()),
                    "Impact": list(impacts.values()),
                })

                fig = px.bar(
                    df_impacts,
                    x="Impact",
                    y="Word",
                    orientation="h",
                    title="TF-IDF Word Importance",
                    color="Impact",
                    color_continuous_scale="Greens",
                )
                fig.update_layout(
                    showlegend=False,
                    xaxis_title="TF-IDF Score",
                    yaxis_title="",
                    height=400,
                )
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning("No word impacts available for this text.")

        with col2:
            # Technical details
            with st.expander("📊 Technical Details", expanded=True):
                st.markdown(result["xai"]["technical_explanation"])

        # Clear results button
        st.divider()
        if st.button("🗑️ Clear Results"):
            st.session_state.classification_result = None
            st.rerun()


if __name__ == "__main__":
    main()
