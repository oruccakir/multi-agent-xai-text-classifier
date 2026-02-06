"""
Model Comparison Page
Compare all classification models on the same input using the Classification Agent.
"""

import streamlit as st
import sys
import os
from pathlib import Path
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import time
from dotenv import load_dotenv

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

# Load environment variables
load_dotenv(project_root / ".env.example")
load_dotenv(project_root / ".env", override=True)

from src.agents.intent_classifier import IntentClassifierAgent
from src.agents.classification_agent import ClassificationAgent
from src.utils.hardware import get_hardware_summary, get_device_display_name

# Page configuration
st.set_page_config(
    page_title="Model Comparison - XAI Classifier",
    page_icon="⚖️",
    layout="wide",
)

# Model directory
MODELS_DIR = project_root / "data" / "models"

# Available models
AVAILABLE_MODELS = ["naive_bayes", "svm", "random_forest", "knn", "logistic_regression", "transformer"]

# Example texts for quick testing
COMPARISON_EXAMPLES = {
    "en_positive": "This movie was absolutely fantastic! The acting was superb and the plot kept me engaged throughout. Highly recommended!",
    "tr_positive": "Bu ürün gerçekten harika, kesinlikle tavsiye ederim! Kargo çok hızlı geldi ve kalitesi mükemmel.",
    "en_negative": "Terrible movie. Waste of time and money. The acting was awful and the story made no sense at all.",
    "tr_negative": "Bu ürün berbat, kesinlikle almayın. Kalitesiz ve pahalı. Çok hayal kırıklığına uğradım.",
}


def set_comparison_example(example_key: str):
    """Callback to set example text in session state."""
    # Set the widget's key directly - this works because callback runs BEFORE widget is rendered on rerun
    st.session_state.comparison_text = COMPARISON_EXAMPLES[example_key]

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
        "description": "HuggingFace transformer for state-of-the-art accuracy",
        "icon": "🤖",
    },
}

# Initialize session state
if "gemini_api_key" not in st.session_state:
    st.session_state.gemini_api_key = os.getenv("GEMINI_API_KEY", "")
if "selected_models_to_compare" not in st.session_state:
    st.session_state.selected_models_to_compare = list(AVAILABLE_MODELS)
if "selected_device" not in st.session_state:
    hw_summary = get_hardware_summary()
    st.session_state.selected_device = hw_summary["default"]


def get_available_experiments():
    """Get list of available experiments (top-level directories in models folder)."""
    experiments = []
    if MODELS_DIR.exists():
        for exp_dir in sorted(MODELS_DIR.iterdir()):
            if exp_dir.is_dir() and not exp_dir.name.startswith('.'):
                # Check if it has any dataset subdirectories with models
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
            if model_name == "transformer":
                # Transformer saved as directory
                if (dataset_path / "transformer.dir").exists():
                    available.append(model_name)
            else:
                # sklearn models saved as .pkl
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


def compare_models(text: str, exp_path: Path, dataset: str, language: str, selected_models: list = None, device: str = None) -> dict:
    """Run selected models on the text and return results.
    
    Args:
        selected_models: List of model names to compare. If None, uses all available.
        device: Device for transformer model inference.
    """
    classification_agent = get_classification_agent()

    # Get available models
    available_models = get_available_models_for_dataset(exp_path, dataset)
    
    # Filter by selected models if provided
    if selected_models:
        available_models = [m for m in available_models if m in selected_models]

    results = {}
    for model_name in available_models:
        # Time the prediction
        start_time = time.time()

        result = classification_agent.classify_text(
            text=text,
            experiment_path=str(exp_path),
            dataset=dataset,
            model_name=model_name,
            language=language,
            device=device,
        )

        pred_time = time.time() - start_time

        if "error" not in result or result.get("prediction") is not None:
            results[model_name] = {
                "prediction": result["prediction"],
                "confidence": result["confidence"],
                "probabilities": result["probabilities"],
                "processing_time": pred_time,
            }

    return results


def main():
    st.title("⚖️ Model Comparison")
    st.markdown("Compare all trained models using the Classification Agent")

    # Get available experiments
    experiments = get_available_experiments()

    if not experiments:
        st.error("No trained experiments found! Please run training first:")
        st.code("python scripts/train_experiment.py --config configs/baseline/imdb.yaml", language="bash")
        return

    # Sidebar for experiment selection
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
        st.markdown("## 🎯 Model Selection")
        st.caption("Choose which models to compare")
        
        # Model multiselect with icons
        model_options = {k: f"{v['icon']} {v['name']}" for k, v in MODELS.items()}
        selected_models = st.multiselect(
            "Select Models:",
            options=list(MODELS.keys()),
            default=st.session_state.selected_models_to_compare,
            format_func=lambda x: model_options[x],
            help="Select which models to include in comparison"
        )
        st.session_state.selected_models_to_compare = selected_models
        
        if not selected_models:
            st.warning("Select at least one model")
        else:
            st.info(f"{len(selected_models)} model(s) selected")

        st.divider()
        st.markdown("## 🖥️ Hardware")
        hw_summary = get_hardware_summary()
        available_devices = hw_summary["devices"]
        if hw_summary["cuda_available"]:
            st.success(f"🎮 CUDA {hw_summary.get('cuda_version', '')} available")
            if hw_summary.get("vram_gb"):
                st.caption(f"VRAM: {hw_summary['vram_gb']} GB")
        else:
            st.info("💻 CPU-only mode")
        if hw_summary.get("ram_gb"):
            st.caption(f"System RAM: {hw_summary['ram_gb']} GB")
        device_options = {dev: get_device_display_name(dev) for dev in available_devices}
        selected_device = st.selectbox(
            "Select Device:",
            options=list(device_options.keys()),
            format_func=lambda x: device_options[x],
            index=available_devices.index(st.session_state.selected_device) if st.session_state.selected_device in available_devices else 0,
            help="Device for transformer model inference"
        )
        st.session_state.selected_device = selected_device

        st.divider()
        st.markdown("## 🤖 Gemini API")

        has_api_key = bool(st.session_state.gemini_api_key)

        if has_api_key:
            st.success("API Key configured")
        else:
            st.warning("No API key (fallback mode)")
            api_key_input = st.text_input(
                "Enter Gemini API Key:",
                type="password",
                help="For better intent detection",
            )
            if api_key_input:
                st.session_state.gemini_api_key = api_key_input
                st.rerun()

        st.divider()
        st.markdown("## 🏗️ Agent Architecture")
        st.markdown("""
        **Agent 1: Intent Classifier**
        - Detects language & domain
        - Selects dataset

        **Agent 2: Classification**
        - Runs all models
        - Returns predictions
        """)

    st.divider()

    # Model overview
    st.markdown("### 🔧 Available Model Types")

    cols = st.columns(3)
    for i, (key, info) in enumerate(MODELS.items()):
        with cols[i % 3]:
            with st.container(border=True):
                st.markdown(f"**{info['icon']} {info['name']}**")
                st.caption(f"Type: {info['type']}")
                st.caption(info['description'])

    st.divider()

    # Input section
    st.markdown("### 📝 Input Text")

    col1, col2 = st.columns([3, 1])

    with col1:
        text_input = st.text_area(
            "Enter text to compare across all models:",
            height=100,
            placeholder="Enter your text here...",
            key="comparison_text",
        )

        # Example buttons with callbacks
        st.markdown("**Quick Examples:**")
        example_col1, example_col2 = st.columns(2)

        with example_col1:
            st.button(
                "🎬 English Positive",
                width="stretch",
                on_click=set_comparison_example,
                args=("en_positive",),
            )
            st.button(
                "🇹🇷 Turkish Positive",
                width="stretch",
                on_click=set_comparison_example,
                args=("tr_positive",),
            )

        with example_col2:
            st.button(
                "🎬 English Negative",
                width="stretch",
                on_click=set_comparison_example,
                args=("en_negative",),
            )
            st.button(
                "🇹🇷 Turkish Negative",
                width="stretch",
                on_click=set_comparison_example,
                args=("tr_negative",),
            )

    with col2:
        st.markdown("### Current Settings")
        gemini_status = "Enabled" if st.session_state.gemini_api_key else "Fallback"
        st.info(f"""
        **Experiment:** {selected_exp_name}
        **Mode:** Auto-detect
        **Intent:** {gemini_status}

        Dataset will be automatically
        selected based on your text.
        """)

        compare_button = st.button(
            "🔍 Compare Models",
            type="primary",
            width="stretch",
            disabled=not text_input,
        )

    # Run comparison
    if compare_button and text_input:
        # Use Intent Classifier Agent to detect language and domain
        api_key = st.session_state.gemini_api_key or None
        intent_agent = get_intent_classifier_agent(api_key)
        intent_result = intent_agent.process(text_input)

        language = intent_result["language"]
        domain = intent_result["domain"]
        detected_dataset = intent_result["dataset"]
        intent_gemini = intent_result["gemini_available"]

        # Check if dataset is available
        available_datasets = get_available_datasets_for_experiment(selected_exp["path"])

        if detected_dataset not in available_datasets:
            st.error(f"Dataset '{detected_dataset}' not found in experiment '{selected_exp_name}'.")
            st.info(f"Available datasets: {', '.join(available_datasets)}")
            return

        st.divider()
        st.markdown("### 🎯 Intent Detection")

        if intent_gemini:
            st.success("🤖 Powered by Google Gemini")
        else:
            st.warning("⚠️ Using fallback keyword-based detection")

        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Language", language.upper())
        with col2:
            st.metric("Domain", domain.capitalize())
        with col3:
            st.metric("Dataset", detected_dataset)

        if intent_result.get("reasoning"):
            st.info(f"💭 **Reasoning:** {intent_result['reasoning']}")

        st.divider()
        st.markdown("### 📊 Comparison Results")

        with st.spinner("Running selected models with Classification Agent..."):
            results = compare_models(
                text_input, 
                selected_exp["path"], 
                detected_dataset, 
                language,
                selected_models=st.session_state.selected_models_to_compare,
                device=st.session_state.selected_device
            )

        if not results:
            st.error("No models could be loaded for this dataset.")
            return

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
                "Time (ms)": result["processing_time"] * 1000,
            })

        df_comparison = pd.DataFrame(comparison_data)

        # Sort by confidence
        df_comparison = df_comparison.sort_values("Confidence", ascending=False)

        # Display table with highlighting
        st.dataframe(
            df_comparison.style.format({
                "Confidence": "{:.1%}",
                "Time (ms)": "{:.2f}",
            }).background_gradient(subset=["Confidence"], cmap="Greens"),
            width="stretch",
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
                yaxis_tickformat=".0%",
            )
            st.plotly_chart(fig, width="stretch")

        with col2:
            st.markdown("#### Processing Time")

            fig = px.bar(
                df_comparison,
                x="Model",
                y="Time (ms)",
                color="Time (ms)",
                color_continuous_scale="Reds_r",
                title="Prediction Time (milliseconds)",
            )
            fig.update_layout(
                xaxis_title="",
                yaxis_title="Time (ms)",
                showlegend=False,
            )
            st.plotly_chart(fig, width="stretch")

        # Consensus analysis
        st.markdown("#### 🤝 Model Consensus")

        predictions = [r["prediction"] for r in results.values()]
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
            st.success("All models agree on the prediction!")
        elif agreement >= 0.6:
            st.info(f"Majority ({predictions.count(most_common)}/{len(predictions)}) of models predict: **{most_common}**")
        else:
            st.warning("Models have significant disagreement. Consider reviewing the input.")

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
            text_auto=".2f",
        )
        fig.update_layout(height=400)
        st.plotly_chart(fig, width="stretch")

        # Individual model details
        st.divider()
        st.markdown("#### 📋 Individual Model Details")

        for model_key, result in results.items():
            with st.expander(f"{MODELS[model_key]['icon']} {MODELS[model_key]['name']}"):
                col1, col2 = st.columns([1, 2])

                with col1:
                    st.metric("Prediction", result["prediction"].upper())
                    st.metric("Confidence", f"{result['confidence']:.1%}")
                    st.metric("Time", f"{result['processing_time']*1000:.2f} ms")

                with col2:
                    probs = result["probabilities"]
                    df_prob = pd.DataFrame({
                        "Class": list(probs.keys()),
                        "Probability": list(probs.values()),
                    }).sort_values("Probability", ascending=True)

                    fig = px.bar(
                        df_prob,
                        x="Probability",
                        y="Class",
                        orientation="h",
                        color="Probability",
                        color_continuous_scale="Greens",
                    )
                    fig.update_layout(
                        showlegend=False,
                        height=200,
                        margin=dict(l=0, r=0, t=0, b=0),
                    )
                    st.plotly_chart(fig, width="stretch")


if __name__ == "__main__":
    main()
