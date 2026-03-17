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
    "xgboost": "XGBoost",
    "decision_tree": "Decision Tree",
    "transformer": "Transformer",
}

MODEL_ICONS = {
    "naive_bayes": "📊",
    "svm": "📐",
    "random_forest": "🌲",
    "knn": "🎯",
    "logistic_regression": "📈",
    "xgboost": "⚡",
    "decision_tree": "🌿",
    "transformer": "🤖",
}


def get_model_results(results: dict) -> dict:
    """Extract model results from experiment results, handling both formats.

    - Script format: results["results"]
    - UI format: results["models"]
    """
    # Try "results" first (script format), then "models" (UI format)
    return results.get("results", results.get("models", {}))


def get_classes(results: dict) -> list:
    """Extract class names from experiment results, handling both formats.

    - Script format: results["dataset"]["classes"]
    - UI format: results["classes"]
    """
    dataset_info = results.get("dataset", {})
    if isinstance(dataset_info, dict):
        return dataset_info.get("classes", [])
    return results.get("classes", [])


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
        else:
            value = str(value)
        rows.append({"Parameter": key.replace("_", " ").title(), "Value": value})

    df = pd.DataFrame(rows)
    st.dataframe(df, width="stretch", hide_index=True)


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
                    "Value": str(value),
                })
            df = pd.DataFrame(rows)
            st.dataframe(df, width="stretch", hide_index=True)
        else:
            st.info("Using default parameters")


def display_metrics_comparison(metrics_df: pd.DataFrame):
    """Display model performance comparison charts."""
    st.markdown("### 📊 Model Performance Comparison")

    # Add display names and sort
    metrics_df = metrics_df.copy()
    metrics_df["Model"] = metrics_df["model"].map(
        lambda x: MODEL_DISPLAY_NAMES.get(x, x)
    )
    metrics_df = metrics_df.sort_values("f1_macro", ascending=False)

    models = metrics_df["Model"].tolist()

    # shared axis label style
    _axis_font = dict(color="black", size=13, family="Arial")
    _tick_font = dict(color="black", size=11)

    # ── Grouped bar: Accuracy, F1 Macro, Precision, Recall ──────────────────
    perf_cols = ["accuracy", "f1_macro", "precision", "recall"]
    perf_vals = metrics_df[perf_cols].values.flatten()
    y_min = max(0.0, float(perf_vals.min()) - 0.06)
    y_max = min(1.0, float(perf_vals.max()) + 0.06)

    fig = go.Figure()
    for col, label, color in [
        ("accuracy",  "Accuracy",    "#2196F3"),
        ("f1_macro",  "F1 (Macro)",  "#4CAF50"),
        ("precision", "Precision",   "#FF9800"),
        ("recall",    "Recall",      "#E91E63"),
    ]:
        vals = metrics_df[col].tolist()
        fig.add_trace(go.Bar(
            name=label,
            x=models,
            y=vals,
            marker_color=color,
            text=[f"{v:.2f}" for v in vals],
            textposition="outside",
            textfont=dict(color="black", size=9, family="Arial"),
        ))

    fig.update_layout(
        title="Model Performance Comparison",
        barmode="group",
        yaxis=dict(
            title="Score",
            title_font=_axis_font,
            tickfont=_tick_font,
            tickformat=".0%",
            range=[y_min, y_max],
            gridcolor="#e0e0e0",
        ),
        xaxis=dict(
            title="Model",
            title_font=_axis_font,
            tickfont=_tick_font,
        ),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        height=440,
        plot_bgcolor="white",
        paper_bgcolor="white",
    )
    st.plotly_chart(fig, use_container_width=True)

    # ── Second row: F1 Macro vs Weighted  |  Training Time ──────────────────
    col1, col2 = st.columns(2)

    with col1:
        f1_vals = metrics_df[["f1_macro", "f1_weighted"]].values.flatten()
        f1_min = max(0.0, float(f1_vals.min()) - 0.06)
        f1_max = min(1.0, float(f1_vals.max()) + 0.06)

        fig2 = go.Figure()
        f1_macro_vals = metrics_df["f1_macro"].tolist()
        f1_weighted_vals = metrics_df["f1_weighted"].tolist()
        fig2.add_trace(go.Bar(
            name="F1 (Macro)",
            x=models,
            y=f1_macro_vals,
            marker_color="#4CAF50",
            text=[f"{v:.2f}" for v in f1_macro_vals],
            textposition="outside",
            textfont=dict(color="black", size=9, family="Arial"),
        ))
        fig2.add_trace(go.Bar(
            name="F1 (Weighted)",
            x=models,
            y=f1_weighted_vals,
            marker_color="#8BC34A",
            text=[f"{v:.2f}" for v in f1_weighted_vals],
            textposition="outside",
            textfont=dict(color="black", size=9, family="Arial"),
        ))
        fig2.update_layout(
            title="F1 Macro vs F1 Weighted",
            barmode="group",
            yaxis=dict(
                title="Score",
                title_font=_axis_font,
                tickfont=_tick_font,
                tickformat=".0%",
                range=[f1_min, f1_max],
                gridcolor="#e0e0e0",
            ),
            xaxis=dict(
                title="Model",
                title_font=_axis_font,
                tickfont=_tick_font,
            ),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            height=400,
            plot_bgcolor="white",
            paper_bgcolor="white",
        )
        st.plotly_chart(fig2, use_container_width=True)

    with col2:
        times = metrics_df["train_time"].tolist()
        use_log = max(times) / (min(t for t in times if t > 0) or 1) > 20

        # On log scale "outside" labels clip; use "inside" so transformer label stays visible
        text_pos = "inside" if use_log else "outside"
        text_color = "white" if use_log else "black"

        fig3 = go.Figure()
        fig3.add_trace(go.Bar(
            x=models,
            y=times,
            marker_color="#F44336",
            showlegend=False,
            text=[f"{t:.1f}s" for t in times],
            textposition=text_pos,
            textfont=dict(color=text_color, size=10, family="Arial"),
        ))
        fig3.update_layout(
            title="Training Time (seconds)" + (" — log scale" if use_log else ""),
            yaxis=dict(
                title="Time (s)",
                title_font=_axis_font,
                tickfont=_tick_font,
                type="log" if use_log else "linear",
                gridcolor="#e0e0e0",
            ),
            xaxis=dict(
                title="Model",
                title_font=_axis_font,
                tickfont=_tick_font,
            ),
            height=400,
            plot_bgcolor="white",
            paper_bgcolor="white",
        )
        st.plotly_chart(fig3, use_container_width=True)

    # ── Detailed metrics table ───────────────────────────────────────────────
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
        }),
        width="stretch",
        hide_index=True,
    )


def build_metrics_df_from_results(results: dict) -> pd.DataFrame:
    """Build a metrics DataFrame from experiment_results.json when metrics_summary.csv is absent."""
    model_results = get_model_results(results)
    rows = []
    for model_name, data in model_results.items():
        if data.get("status") == "error" or "accuracy" not in data:
            continue
        rows.append({
            "model":       model_name,
            "accuracy":    data["accuracy"],
            "f1_macro":    data["f1_macro"],
            "f1_weighted": data["f1_weighted"],
            "precision":   data["precision"],
            "recall":      data["recall"],
            "train_time":  data.get("train_time", 0),
        })
    return pd.DataFrame(rows) if rows else None


def display_roc_curves(results: dict):
    """Display ROC/AUC curves for all models."""
    st.markdown("### 📈 ROC / AUC Curves")

    model_results = get_model_results(results)

    # Filter models that have roc_curves data
    models_with_roc = {
        name: data for name, data in model_results.items()
        if data.get("roc_curves")
    }

    if not models_with_roc:
        st.info("ROC curve data not available. Re-train your models to generate ROC data.")
        return

    classes = list(next(iter(models_with_roc.values()))["roc_curves"].keys())
    is_binary = len(classes) == 1

    # Color palette for classes (multiclass)
    PALETTE = [
        "#2196F3", "#4CAF50", "#FF5722", "#9C27B0",
        "#FF9800", "#00BCD4", "#E91E63", "#8BC34A",
    ]

    if is_binary:
        # Binary: one ROC curve per model, all on a single plot
        fig = go.Figure()

        for model_name, model_data in models_with_roc.items():
            roc = model_data["roc_curves"]
            cls = classes[0]
            fpr = roc[cls]["fpr"]
            tpr = roc[cls]["tpr"]
            roc_auc = roc[cls]["auc"]
            display_name = MODEL_DISPLAY_NAMES.get(model_name, model_name)
            icon = MODEL_ICONS.get(model_name, "📦")

            fig.add_trace(go.Scatter(
                x=fpr,
                y=tpr,
                mode="lines",
                name=f"{display_name} (AUC={roc_auc:.3f})",
                line=dict(width=2),
            ))

        # Diagonal reference line
        fig.add_trace(go.Scatter(
            x=[0, 1], y=[0, 1],
            mode="lines",
            name="Random Classifier",
            line=dict(color="gray", dash="dash", width=1),
            showlegend=True,
        ))

        _axis_font = dict(color="black", size=13, family="Arial")
        _tick_font = dict(color="black", size=11)
        fig.update_layout(
            title=f"ROC Curves — All Models (class: {cls})",
            xaxis_title="False Positive Rate",
            yaxis_title="True Positive Rate",
            xaxis=dict(
                range=[0, 1],
                title_font=_axis_font,
                tickfont=_tick_font,
                linecolor="black",
            ),
            yaxis=dict(
                range=[0, 1.02],
                title_font=_axis_font,
                tickfont=_tick_font,
                linecolor="black",
            ),
            legend=dict(orientation="v", x=0.6, y=0.05),
            height=500,
        )
        st.plotly_chart(fig, width="stretch")

    else:
        # Multiclass OvR: grid of subplots, one per model
        num_models = len(models_with_roc)
        ncols = min(2, num_models)
        nrows = (num_models + ncols - 1) // ncols

        fig = make_subplots(
            rows=nrows,
            cols=ncols,
            subplot_titles=[
                MODEL_DISPLAY_NAMES.get(m, m)
                for m in models_with_roc
            ],
        )

        for idx, (model_name, model_data) in enumerate(models_with_roc.items()):
            row = idx // ncols + 1
            col = idx % ncols + 1
            roc = model_data["roc_curves"]

            for cls_idx, cls in enumerate(classes):
                if cls not in roc:
                    continue
                fpr = roc[cls]["fpr"]
                tpr = roc[cls]["tpr"]
                roc_auc = roc[cls]["auc"]
                color = PALETTE[cls_idx % len(PALETTE)]

                fig.add_trace(
                    go.Scatter(
                        x=fpr,
                        y=tpr,
                        mode="lines",
                        name=f"{cls} (AUC={roc_auc:.3f})",
                        line=dict(color=color, width=2),
                        legendgroup=cls,
                        showlegend=(idx == 0),
                    ),
                    row=row,
                    col=col,
                )

            # Diagonal
            fig.add_trace(
                go.Scatter(
                    x=[0, 1], y=[0, 1],
                    mode="lines",
                    line=dict(color="gray", dash="dash", width=1),
                    showlegend=False,
                ),
                row=row,
                col=col,
            )

        fig.update_layout(
            title="ROC Curves — One-vs-Rest (per model)",
            height=350 * nrows,
            legend=dict(orientation="v"),
        )
        for ann in fig.layout.annotations:
            ann.font = dict(color="black", size=13, family="Arial")
        _axis_font = dict(color="black", size=12, family="Arial")
        _tick_font = dict(color="black", size=10)
        fig.update_xaxes(
            title_text="FPR",
            range=[0, 1],
            title_font=_axis_font,
            tickfont=_tick_font,
            linecolor="black",
        )
        fig.update_yaxes(
            title_text="TPR",
            range=[0, 1.02],
            title_font=_axis_font,
            tickfont=_tick_font,
            linecolor="black",
        )
        st.plotly_chart(fig, width="stretch")

    # AUC summary table
    st.markdown("#### AUC Summary")
    auc_rows = []
    for model_name, model_data in models_with_roc.items():
        roc = model_data["roc_curves"]
        row = {"Model": MODEL_DISPLAY_NAMES.get(model_name, model_name)}
        for cls, vals in roc.items():
            row[f"AUC ({cls})"] = vals["auc"]
        if len(classes) > 1:
            row["Macro AUC"] = float(np.mean([v["auc"] for v in roc.values()]))
        auc_rows.append(row)

    auc_df = pd.DataFrame(auc_rows)
    auc_cols = [c for c in auc_df.columns if c != "Model"]
    st.dataframe(
        auc_df.style.format({c: "{:.4f}" for c in auc_cols})
        ,
        width="stretch",
        hide_index=True,
    )


def display_confusion_matrices(results: dict):
    """Display confusion matrices for all models."""
    st.markdown("### 🔢 Confusion Matrices")

    model_results = get_model_results(results)
    num_models = len(model_results)

    if num_models == 0:
        st.warning("No model results found.")
        return

    # Create grid of confusion matrices
    cols = st.columns(min(3, num_models))
    global_classes = get_classes(results)

    for i, (model_name, model_data) in enumerate(model_results.items()):
        if "confusion_matrix" not in model_data:
            continue

        cm = np.array(model_data["confusion_matrix"])
        n = cm.shape[0]
        classes = global_classes if len(global_classes) == n else [f"Class {j}" for j in range(n)]

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
            _axis_font = dict(color="black", size=12, family="Arial")
            _tick_font = dict(color="black", size=10)
            fig.update_layout(
                title=display_name,
                height=300,
                margin=dict(l=10, r=10, t=40, b=10),
                xaxis=dict(
                    title_font=_axis_font,
                    tickfont=_tick_font,
                    linecolor="black",
                ),
                yaxis=dict(
                    title_font=_axis_font,
                    tickfont=_tick_font,
                    linecolor="black",
                ),
                coloraxis_colorbar=dict(
                    tickfont=dict(color="black", size=10),
                    title_font=dict(color="black", size=11),
                ),
            )
            st.plotly_chart(fig, width="stretch")


def display_classification_reports(results: dict):
    """Display per-class metrics for each model."""
    st.markdown("### 📈 Per-Class Performance")

    model_results = get_model_results(results)
    classes = get_classes(results)

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
        st.plotly_chart(fig, width="stretch")


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
        - [ROC / AUC Curves](#roc-auc-curves)
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
        # Handle both formats: dataset as dict with name, or just name string
        dataset_info = config.get("dataset", {})
        dataset_name = dataset_info.get("name", dataset_info) if isinstance(dataset_info, dict) else dataset_info
        st.metric("Dataset", dataset_name or "Unknown")
    with col3:
        # Language can be in dataset (script) or preprocessing (UI)
        language = (
            config.get("dataset", {}).get("language") or
            config.get("preprocessing", {}).get("language") or
            "Unknown"
        )
        st.metric("Language", language.upper())
    with col4:
        # Task is only in script format
        task = config.get("dataset", {}).get("task", "classification")
        st.metric("Task", task.replace("_", " ").title())

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

    # Performance Section — use CSV if available, fall back to results JSON
    metrics_df = metrics if metrics is not None else build_metrics_df_from_results(results)

    if metrics_df is not None:
        display_metrics_comparison(metrics_df)

        # Best model highlight
        best_idx = metrics_df["f1_macro"].idxmax()
        best_model = metrics_df.loc[best_idx]

        st.success(f"""
        🏆 **Best Model:** {MODEL_DISPLAY_NAMES.get(best_model['model'], best_model['model'])}
        - Accuracy: {best_model['accuracy']:.2%}
        - F1 (Macro): {best_model['f1_macro']:.2%}
        - Training Time: {best_model['train_time']:.2f}s
        """)

    st.divider()

    # ROC/AUC Section
    if results:
        display_roc_curves(results)

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
