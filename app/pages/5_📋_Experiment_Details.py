"""
Experiment Details Page
View training configurations, model hyperparameters, and performance metrics.
"""

import streamlit as st
import sys
from pathlib import Path
import pandas as pd
import yaml
import json
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

# Page configuration
st.set_page_config(
    page_title="Experiment Details - XAI Classifier",
    page_icon="📋",
    layout="wide",
)

# Model directory
MODELS_DIR = project_root / "data" / "models"

# Model display names
MODEL_DISPLAY_NAMES = {
    "naive_bayes": "Naive Bayes",
    "svm": "SVM",
    "random_forest": "Random Forest",
    "knn": "KNN",
    "logistic_regression": "Logistic Regression",
}

MODEL_ICONS = {
    "naive_bayes": "📊",
    "svm": "📐",
    "random_forest": "🌲",
    "knn": "🎯",
    "logistic_regression": "📈",
}


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
            datasets.append({
                "name": dataset_dir.name,
                "path": dataset_dir,
            })
    return datasets


def load_config(dataset_path: Path):
    """Load experiment config from dataset directory."""
    config_path = dataset_path / "config.yaml"
    if config_path.exists():
        with open(config_path) as f:
            return yaml.safe_load(f)
    return None


def load_results(dataset_path: Path):
    """Load experiment results from dataset directory."""
    results_path = dataset_path / "experiment_results.json"
    if results_path.exists():
        with open(results_path) as f:
            return json.load(f)
    return None


def load_metrics(dataset_path: Path):
    """Load metrics summary from dataset directory."""
    metrics_path = dataset_path / "metrics_summary.csv"
    if metrics_path.exists():
        return pd.read_csv(metrics_path)
    return None


def display_config_section(title: str, config_dict: dict, icon: str = "⚙️"):
    """Display a configuration section as a formatted table."""
    st.markdown(f"#### {icon} {title}")

    # Convert to displayable format
    rows = []
    for key, value in config_dict.items():
        if isinstance(value, list):
            value = str(value)
        elif isinstance(value, bool):
            value = "✅ Yes" if value else "❌ No"
        elif value is None:
            value = "None (full data)"
        rows.append({"Parameter": key.replace("_", " ").title(), "Value": value})

    df = pd.DataFrame(rows)
    st.dataframe(df, use_container_width=True, hide_index=True)


def display_model_config(model_name: str, model_config: dict):
    """Display model-specific configuration."""
    icon = MODEL_ICONS.get(model_name, "📦")
    display_name = MODEL_DISPLAY_NAMES.get(model_name, model_name)

    with st.expander(f"{icon} {display_name} Configuration"):
        # Filter out 'enabled' from display
        config_to_show = {k: v for k, v in model_config.items() if k != "enabled"}

        if config_to_show:
            rows = []
            for key, value in config_to_show.items():
                rows.append({
                    "Hyperparameter": key.replace("_", " ").title(),
                    "Value": value
                })
            df = pd.DataFrame(rows)
            st.dataframe(df, use_container_width=True, hide_index=True)
        else:
            st.info("Using default parameters")


def display_metrics_comparison(metrics_df: pd.DataFrame):
    """Display model performance comparison charts."""
    st.markdown("### 📊 Model Performance Comparison")

    # Add display names
    metrics_df["Model"] = metrics_df["model"].map(
        lambda x: f"{MODEL_ICONS.get(x, '📦')} {MODEL_DISPLAY_NAMES.get(x, x)}"
    )

    # Sort by F1 score
    metrics_df = metrics_df.sort_values("f1_macro", ascending=False)

    col1, col2 = st.columns(2)

    with col1:
        # Accuracy and F1 comparison
        fig = go.Figure()

        fig.add_trace(go.Bar(
            name="Accuracy",
            x=metrics_df["Model"],
            y=metrics_df["accuracy"],
            marker_color="#2196F3",
        ))

        fig.add_trace(go.Bar(
            name="F1 (Macro)",
            x=metrics_df["Model"],
            y=metrics_df["f1_macro"],
            marker_color="#4CAF50",
        ))

        fig.update_layout(
            title="Accuracy vs F1 Score",
            barmode="group",
            yaxis_title="Score",
            yaxis_tickformat=".0%",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            height=400,
        )
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        # Training time comparison
        fig = px.bar(
            metrics_df,
            x="Model",
            y="train_time",
            color="train_time",
            color_continuous_scale="Reds",
            title="Training Time (seconds)",
        )
        fig.update_layout(
            yaxis_title="Time (s)",
            showlegend=False,
            height=400,
        )
        st.plotly_chart(fig, use_container_width=True)

    # Detailed metrics table
    st.markdown("#### Detailed Metrics")

    display_df = metrics_df[["Model", "accuracy", "f1_macro", "f1_weighted", "precision", "recall", "train_time"]].copy()
    display_df.columns = ["Model", "Accuracy", "F1 (Macro)", "F1 (Weighted)", "Precision", "Recall", "Train Time (s)"]

    st.dataframe(
        display_df.style.format({
            "Accuracy": "{:.2%}",
            "F1 (Macro)": "{:.2%}",
            "F1 (Weighted)": "{:.2%}",
            "Precision": "{:.2%}",
            "Recall": "{:.2%}",
            "Train Time (s)": "{:.2f}",
        }).background_gradient(subset=["Accuracy", "F1 (Macro)"], cmap="Greens"),
        use_container_width=True,
        hide_index=True,
    )


def display_confusion_matrices(results: dict):
    """Display confusion matrices for all models."""
    st.markdown("### 🔢 Confusion Matrices")

    model_results = results.get("results", {})
    num_models = len(model_results)

    if num_models == 0:
        st.warning("No model results found.")
        return

    # Create grid of confusion matrices
    cols = st.columns(min(3, num_models))

    for i, (model_name, model_data) in enumerate(model_results.items()):
        if "confusion_matrix" not in model_data:
            continue

        cm = np.array(model_data["confusion_matrix"])
        classes = results.get("dataset", {}).get("classes", ["Class 0", "Class 1"])

        icon = MODEL_ICONS.get(model_name, "📦")
        display_name = MODEL_DISPLAY_NAMES.get(model_name, model_name)

        with cols[i % len(cols)]:
            fig = px.imshow(
                cm,
                labels=dict(x="Predicted", y="Actual", color="Count"),
                x=classes,
                y=classes,
                color_continuous_scale="Blues",
                text_auto=True,
            )
            fig.update_layout(
                title=f"{icon} {display_name}",
                height=300,
                margin=dict(l=10, r=10, t=40, b=10),
            )
            st.plotly_chart(fig, use_container_width=True)


def display_classification_reports(results: dict):
    """Display per-class metrics for each model."""
    st.markdown("### 📈 Per-Class Performance")

    model_results = results.get("results", {})
    classes = results.get("dataset", {}).get("classes", [])

    if not classes:
        return

    # Prepare data for comparison
    data = []
    for model_name, model_data in model_results.items():
        report = model_data.get("classification_report", {})
        display_name = MODEL_DISPLAY_NAMES.get(model_name, model_name)

        for cls in classes:
            if cls in report:
                data.append({
                    "Model": display_name,
                    "Class": cls,
                    "Precision": report[cls]["precision"],
                    "Recall": report[cls]["recall"],
                    "F1-Score": report[cls]["f1-score"],
                })

    if data:
        df = pd.DataFrame(data)

        # Heatmap of F1 scores per class
        df_pivot = df.pivot(index="Model", columns="Class", values="F1-Score")

        fig = px.imshow(
            df_pivot,
            labels=dict(x="Class", y="Model", color="F1-Score"),
            color_continuous_scale="Greens",
            text_auto=".2%",
        )
        fig.update_layout(
            title="F1-Score by Model and Class",
            height=400,
        )
        st.plotly_chart(fig, use_container_width=True)


def main():
    st.title("📋 Experiment Details")
    st.markdown("View training configurations, hyperparameters, and performance metrics")

    # Get available experiments
    experiments = get_available_experiments()

    if not experiments:
        st.error("No trained experiments found! Please run training first:")
        st.code("python scripts/train_experiment.py --config configs/baseline/imdb.yaml", language="bash")
        return

    # Sidebar for navigation
    with st.sidebar:
        st.markdown("## 🔬 Select Experiment")

        exp_options = {exp["name"]: exp for exp in experiments}
        selected_exp_name = st.selectbox(
            "Experiment:",
            options=list(exp_options.keys()),
            format_func=lambda x: f"📁 {x}",
        )
        selected_exp = exp_options[selected_exp_name]

        # Get datasets for this experiment
        datasets = get_available_datasets_for_experiment(selected_exp["path"])

        st.divider()
        st.markdown("## 📊 Select Dataset")

        dataset_options = {ds["name"]: ds for ds in datasets}
        selected_dataset_name = st.selectbox(
            "Dataset:",
            options=list(dataset_options.keys()),
            format_func=lambda x: f"📂 {x}",
        )
        selected_dataset = dataset_options[selected_dataset_name]

        st.divider()
        st.markdown("## 📑 Sections")
        st.markdown("""
        - [Overview](#overview)
        - [Configuration](#configuration)
        - [Performance](#model-performance-comparison)
        - [Confusion Matrices](#confusion-matrices)
        """)

    # Load data
    config = load_config(selected_dataset["path"])
    results = load_results(selected_dataset["path"])
    metrics = load_metrics(selected_dataset["path"])

    if not config:
        st.error(f"No configuration found for {selected_dataset_name}")
        return

    # Overview Section
    st.markdown("## Overview")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Experiment", config.get("experiment", {}).get("name", "Unknown"))
    with col2:
        st.metric("Dataset", config.get("dataset", {}).get("name", "Unknown"))
    with col3:
        st.metric("Language", config.get("dataset", {}).get("language", "Unknown").upper())
    with col4:
        st.metric("Task", config.get("dataset", {}).get("task", "Unknown").replace("_", " ").title())

    # Experiment info
    exp_info = config.get("experiment", {})
    if exp_info.get("description"):
        st.info(f"**Description:** {exp_info['description']}")
    if exp_info.get("author"):
        st.caption(f"Author: {exp_info['author']}")
    if results and results.get("timestamp"):
        st.caption(f"Trained: {results['timestamp'][:19].replace('T', ' ')}")

    st.divider()

    # Configuration Section
    st.markdown("## Configuration")

    tab1, tab2, tab3, tab4 = st.tabs(["📊 Dataset", "🔤 Feature Extraction", "🔧 Preprocessing", "🤖 Models"])

    with tab1:
        dataset_config = config.get("dataset", {})
        display_config_section("Dataset Configuration", dataset_config, "📊")

    with tab2:
        fe_config = config.get("feature_extraction", {})
        display_config_section("Feature Extraction", fe_config, "🔤")

    with tab3:
        prep_config = config.get("preprocessing", {})
        display_config_section("Preprocessing", prep_config, "🔧")

    with tab4:
        st.markdown("#### 🤖 Model Hyperparameters")
        models_config = config.get("models", {})

        # Show enabled models count
        enabled_models = [m for m, c in models_config.items() if c.get("enabled", True)]
        st.success(f"**{len(enabled_models)} models trained:** {', '.join(enabled_models)}")

        # Show each model's config
        for model_name, model_config in models_config.items():
            if model_config.get("enabled", True):
                display_model_config(model_name, model_config)

    st.divider()

    # Performance Section
    if metrics is not None:
        display_metrics_comparison(metrics)

        # Best model highlight
        best_idx = metrics["f1_macro"].idxmax()
        best_model = metrics.loc[best_idx]

        st.success(f"""
        🏆 **Best Model:** {MODEL_DISPLAY_NAMES.get(best_model['model'], best_model['model'])}
        - Accuracy: {best_model['accuracy']:.2%}
        - F1 (Macro): {best_model['f1_macro']:.2%}
        - Training Time: {best_model['train_time']:.2f}s
        """)

    st.divider()

    # Confusion Matrices Section
    if results:
        display_confusion_matrices(results)

        st.divider()

        # Per-class performance
        display_classification_reports(results)

    st.divider()

    # Raw Config Section
    with st.expander("📄 View Raw Configuration (YAML)"):
        st.code(yaml.dump(config, default_flow_style=False, allow_unicode=True), language="yaml")

    with st.expander("📄 View Raw Results (JSON)"):
        if results:
            st.json(results)
        else:
            st.warning("No results file found.")


if __name__ == "__main__":
    main()
