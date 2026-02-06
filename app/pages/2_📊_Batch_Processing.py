"""
Batch Processing Page
Process multiple texts from CSV file using the 3-agent architecture.
Uses Intent Classifier for auto-detection and Classification Agent for batch processing.
"""

import streamlit as st
import sys
import os
from pathlib import Path
import pandas as pd
import plotly.express as px
from dotenv import load_dotenv

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

# Load environment variables
load_dotenv(project_root / ".env.example")
load_dotenv(project_root / ".env", override=True)

from src.agents.intent_classifier import IntentClassifierAgent
from src.agents.classification_agent import ClassificationAgent

# Page configuration
st.set_page_config(
    page_title="Batch Processing - XAI Classifier",
    page_icon="📊",
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
if "gemini_api_key" not in st.session_state:
    st.session_state.gemini_api_key = os.getenv("GEMINI_API_KEY", "")


def get_available_experiments():
    """Get list of available experiments."""
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


def main():
    st.title("📊 Batch Processing")
    st.markdown("Process multiple texts using the Classification Agent")

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

        # Show available datasets
        available_datasets = get_available_datasets_for_experiment(selected_exp["path"])

        st.divider()
        st.markdown("## 📊 Available Datasets")
        for ds in available_datasets:
            models = get_available_models_for_dataset(selected_exp["path"], ds)
            st.markdown(f"**{ds}**: {len(models)} models")

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
        - Auto-detects language
        - Selects dataset

        **Agent 2: Classification**
        - Batch processing
        - Efficient inference
        """)

    st.divider()

    # Instructions
    with st.expander("📋 Instructions", expanded=False):
        st.markdown("""
        ### How to use:
        1. **Prepare your CSV file** with a column named `text` containing the texts to classify
        2. **Upload the CSV** using the file uploader below
        3. Language and dataset are **auto-detected** using the Intent Classifier Agent
        4. **Click Process** to classify all texts with the Classification Agent
        5. **Download results** as a new CSV file

        ### CSV Format Example:
        ```csv
        text
        "This movie was great!"
        "The product quality is poor."
        "Average experience, nothing special."
        ```
        """)

    st.divider()

    # File upload section
    col1, col2 = st.columns([2, 1])

    with col1:
        st.markdown("### 📁 Upload CSV File")
        uploaded_file = st.file_uploader(
            "Choose a CSV file",
            type=["csv"],
            help="Upload a CSV file with a 'text' column",
        )

    with col2:
        st.markdown("### ⚙️ Settings")
        gemini_status = "Enabled" if st.session_state.gemini_api_key else "Fallback"
        st.info(f"""
        **Experiment:** {selected_exp_name}
        **Mode:** Auto-detect
        **Intent Detection:** {gemini_status}

        Language and dataset will be
        detected from text content.
        """)

    # Process uploaded file
    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)

            # Check for text column
            if "text" not in df.columns:
                st.error("CSV must have a 'text' column!")
                st.markdown("**Columns found:** " + ", ".join(df.columns))
                return

            st.divider()

            # Preview data
            st.markdown("### 👀 Data Preview")
            st.dataframe(df.head(10), width="stretch")
            st.info(f"📊 Total rows: **{len(df)}**")

            # Use Intent Classifier Agent to detect language and dataset
            api_key = st.session_state.gemini_api_key or None
            intent_agent = get_intent_classifier_agent(api_key)

            # Detect from first few texts
            sample_text = " ".join(df["text"].head(5).tolist())
            intent_result = intent_agent.process(sample_text)

            detected_language = intent_result["language"]
            detected_dataset = intent_result["dataset"]
            intent_gemini = intent_result["gemini_available"]

            # Check if dataset is available
            if detected_dataset not in available_datasets:
                st.warning(f"Detected dataset '{detected_dataset}' not found. Using first available dataset.")
                detected_dataset = available_datasets[0] if available_datasets else None

            if not detected_dataset:
                st.error("No datasets available for this experiment.")
                return

            # Get available models
            available_models = get_available_models_for_dataset(selected_exp["path"], detected_dataset)

            if not available_models:
                st.error(f"No models found for dataset '{detected_dataset}'.")
                return

            # Use logistic_regression as default
            selected_model = "logistic_regression" if "logistic_regression" in available_models else available_models[0]

            # Show detection info
            st.markdown("### 🎯 Intent Detection Result")

            if intent_gemini:
                st.success("🤖 Powered by Google Gemini")
            else:
                st.warning("⚠️ Using fallback keyword-based detection")

            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Detected Language", detected_language.upper())
            with col2:
                st.metric("Selected Dataset", detected_dataset)
            with col3:
                st.metric("Model", MODEL_DISPLAY_NAMES.get(selected_model, selected_model))

            if intent_result.get("reasoning"):
                st.info(f"💭 **Reasoning:** {intent_result['reasoning']}")

            # Process button
            if st.button("🚀 Process All Texts", type="primary", width="stretch"):
                with st.spinner(f"Processing {len(df)} texts with Classification Agent..."):
                    # Get Classification Agent
                    classification_agent = get_classification_agent()

                    # Get texts
                    texts = df["text"].tolist()

                    # Progress bar
                    progress_bar = st.progress(0)
                    status_text = st.empty()

                    # Process in batches for progress tracking
                    batch_size = max(1, min(100, len(texts) // 10))
                    all_results = []

                    for i in range(0, len(texts), batch_size):
                        batch = texts[i:i + batch_size]

                        # Use Classification Agent for batch processing
                        batch_results = classification_agent.classify_batch(
                            texts=batch,
                            experiment_path=str(selected_exp["path"]),
                            dataset=detected_dataset,
                            model_name=selected_model,
                            language=detected_language,
                        )

                        # Format results
                        for j, result in enumerate(batch_results):
                            text = batch[j]
                            all_results.append({
                                "text": text[:100] + "..." if len(text) > 100 else text,
                                "prediction": result.get("prediction", "error"),
                                "confidence": result.get("confidence", 0.0),
                            })

                        progress = min(1.0, (i + batch_size) / len(texts))
                        progress_bar.progress(progress)
                        status_text.text(f"Processed {min(i + batch_size, len(texts))}/{len(texts)} texts")

                    progress_bar.progress(1.0)
                    status_text.text("✅ Processing complete!")

                if not all_results:
                    st.error("Failed to process texts. Check if models are properly trained.")
                    return

                # Create results dataframe
                results_df = pd.DataFrame(all_results)

                st.divider()

                # Show results
                st.markdown("### 📈 Results")

                # Summary statistics
                col1, col2, col3 = st.columns(3)

                with col1:
                    st.metric("Total Processed", len(results_df))

                with col2:
                    avg_confidence = results_df["confidence"].mean()
                    st.metric("Avg Confidence", f"{avg_confidence:.1%}")

                with col3:
                    unique_preds = results_df["prediction"].nunique()
                    st.metric("Unique Predictions", unique_preds)

                # Prediction distribution
                st.markdown("#### Prediction Distribution")

                pred_counts = results_df["prediction"].value_counts()
                fig = px.pie(
                    values=pred_counts.values,
                    names=pred_counts.index,
                    title="Distribution of Predictions",
                    hole=0.4,
                )
                st.plotly_chart(fig, width="stretch")

                # Confidence distribution
                st.markdown("#### Confidence Distribution")

                fig = px.histogram(
                    results_df,
                    x="confidence",
                    nbins=20,
                    title="Confidence Score Distribution",
                    color_discrete_sequence=["#1E88E5"],
                )
                fig.update_layout(
                    xaxis_title="Confidence",
                    yaxis_title="Count",
                )
                st.plotly_chart(fig, width="stretch")

                # Results table
                st.markdown("#### Detailed Results")
                st.dataframe(results_df, width="stretch")

                # Download button
                csv = results_df.to_csv(index=False)
                st.download_button(
                    label="📥 Download Results as CSV",
                    data=csv,
                    file_name="classification_results.csv",
                    mime="text/csv",
                    type="primary",
                )

        except Exception as e:
            st.error(f"Error processing file: {str(e)}")

    else:
        # Show sample data option
        st.divider()
        st.markdown("### 🧪 Try with Sample Data")

        if st.button("Load Sample Data", width="stretch"):
            # Create sample data
            sample_data = pd.DataFrame({
                "text": [
                    "This movie was absolutely amazing! Best film I've seen all year.",
                    "Terrible waste of time. Poor acting and boring plot.",
                    "It was okay, nothing special but not bad either.",
                    "Loved every minute of it! Highly recommended!",
                    "Complete disappointment. Do not watch.",
                    "A masterpiece of modern cinema. Outstanding performances.",
                    "Average movie with some good moments.",
                    "Worst movie I have ever seen in my entire life.",
                    "Pretty good overall, would watch again.",
                    "Boring and predictable. Save your money.",
                ]
            })

            st.session_state.sample_df = sample_data
            st.info("✅ Sample data loaded! Upload it as a CSV or copy the format.")
            st.dataframe(sample_data, width="stretch")

            # Provide download of sample
            csv = sample_data.to_csv(index=False)
            st.download_button(
                label="📥 Download Sample CSV",
                data=csv,
                file_name="sample_texts.csv",
                mime="text/csv",
            )


if __name__ == "__main__":
    main()
