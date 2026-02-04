"""
Dataset Explorer Page
Explore and visualize the available datasets.
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

# Page configuration
st.set_page_config(
    page_title="Dataset Explorer - XAI Classifier",
    page_icon="📚",
    layout="wide",
)

# Dataset information
DATASETS = {
    "imdb": {
        "name": "IMDB Movie Reviews",
        "language": "English",
        "task": "Binary Sentiment",
        "classes": ["negative", "positive"],
        "icon": "🎬",
        "description": "Movie reviews from IMDB for sentiment classification",
    },
    "turkish_sentiment": {
        "name": "Turkish Sentiment",
        "language": "Turkish",
        "task": "3-Class Sentiment",
        "classes": ["negatif", "notr", "pozitif"],
        "icon": "🇹🇷",
        "description": "Turkish product reviews for sentiment analysis",
    },
    "ag_news": {
        "name": "AG News",
        "language": "English",
        "task": "4-Class News",
        "classes": ["World", "Sports", "Business", "Sci/Tech"],
        "icon": "📰",
        "description": "News articles classified into 4 categories",
    },
    "turkish_news": {
        "name": "Turkish News (TTC4900)",
        "language": "Turkish",
        "task": "7-Class News",
        "classes": ["siyaset", "dünya", "ekonomi", "kültür", "sağlık", "spor", "teknoloji"],
        "icon": "🗞️",
        "description": "Turkish news articles from 7 different categories",
    },
}


@st.cache_data
def load_dataset(dataset_name: str, split: str) -> pd.DataFrame:
    """Load dataset from processed CSV files."""
    file_path = project_root / "data" / "processed" / f"{dataset_name}_{split}.csv"
    if file_path.exists():
        return pd.read_csv(file_path)
    return None


def main():
    st.title("📚 Dataset Explorer")
    st.markdown("Explore the datasets used for text classification")

    st.divider()

    # Dataset overview
    st.markdown("### 📊 Dataset Overview")

    cols = st.columns(4)
    for i, (key, info) in enumerate(DATASETS.items()):
        with cols[i]:
            st.markdown(f"""
            <div style="background-color: #f0f2f6; border-radius: 10px; padding: 1rem; text-align: center; height: 180px;">
                <h2>{info['icon']}</h2>
                <h4>{info['name']}</h4>
                <p style="color: #1E88E5;">{info['language']} | {info['task']}</p>
                <p style="font-size: 0.8rem; color: #666;">{len(info['classes'])} classes</p>
            </div>
            """, unsafe_allow_html=True)

    st.divider()

    # Dataset selector
    st.markdown("### 🔍 Explore Dataset")

    col1, col2 = st.columns([1, 3])

    with col1:
        selected_dataset = st.selectbox(
            "Select Dataset:",
            list(DATASETS.keys()),
            format_func=lambda x: f"{DATASETS[x]['icon']} {DATASETS[x]['name']}",
        )

        split = st.radio(
            "Split:",
            ["train", "test"],
            format_func=lambda x: "🏋️ Training" if x == "train" else "🧪 Test",
        )

    # Load dataset
    df = load_dataset(selected_dataset, split)

    with col2:
        info = DATASETS[selected_dataset]
        st.markdown(f"""
        ### {info['icon']} {info['name']}

        | Property | Value |
        |----------|-------|
        | **Language** | {info['language']} |
        | **Task** | {info['task']} |
        | **Classes** | {', '.join(info['classes'])} |
        | **Description** | {info['description']} |
        """)

    if df is not None:
        st.divider()

        # Statistics
        st.markdown("### 📈 Dataset Statistics")

        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric("Total Samples", f"{len(df):,}")

        with col2:
            avg_len = df["text"].str.len().mean()
            st.metric("Avg Text Length", f"{avg_len:,.0f} chars")

        with col3:
            num_classes = df["label"].nunique()
            st.metric("Number of Classes", num_classes)

        with col4:
            min_class = df["label"].value_counts().min()
            max_class = df["label"].value_counts().max()
            balance = min_class / max_class
            st.metric("Class Balance", f"{balance:.1%}")

        st.divider()

        # Visualizations
        col1, col2 = st.columns(2)

        with col1:
            st.markdown("#### Class Distribution")

            class_counts = df["label"].value_counts()
            fig = px.pie(
                values=class_counts.values,
                names=class_counts.index,
                title=f"Label Distribution ({split})",
                hole=0.4,
                color_discrete_sequence=px.colors.qualitative.Set2,
            )
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            st.markdown("#### Text Length Distribution")

            df["text_length"] = df["text"].str.len()
            fig = px.histogram(
                df,
                x="text_length",
                nbins=50,
                title="Distribution of Text Lengths",
                color_discrete_sequence=["#1E88E5"],
            )
            fig.update_layout(
                xaxis_title="Text Length (characters)",
                yaxis_title="Count",
            )
            st.plotly_chart(fig, use_container_width=True)

        # Text length by class
        st.markdown("#### Text Length by Class")

        fig = px.box(
            df,
            x="label",
            y="text_length",
            title="Text Length Distribution per Class",
            color="label",
            color_discrete_sequence=px.colors.qualitative.Set2,
        )
        fig.update_layout(
            xaxis_title="Class",
            yaxis_title="Text Length",
            showlegend=False,
        )
        st.plotly_chart(fig, use_container_width=True)

        st.divider()

        # Sample data
        st.markdown("### 📝 Sample Data")

        # Class filter
        selected_class = st.selectbox(
            "Filter by class:",
            ["All"] + list(df["label"].unique()),
        )

        if selected_class != "All":
            filtered_df = df[df["label"] == selected_class]
        else:
            filtered_df = df

        # Number of samples to show
        num_samples = st.slider("Number of samples:", 5, 50, 10)

        # Show samples
        sample_df = filtered_df.sample(min(num_samples, len(filtered_df)))

        for _, row in sample_df.iterrows():
            with st.expander(f"**{row['label'].upper()}** - {row['text'][:80]}..."):
                st.markdown(f"**Label:** `{row['label']}`")
                st.markdown("**Full Text:**")
                st.info(row["text"])

        st.divider()

        # Word frequency analysis
        st.markdown("### 📊 Word Frequency Analysis")

        if st.button("🔍 Analyze Word Frequencies"):
            with st.spinner("Analyzing..."):
                from collections import Counter

                # Get all words
                all_text = " ".join(df["text"].tolist())
                words = all_text.lower().split()

                # Filter short words
                words = [w.strip(".,!?()[]\"'") for w in words if len(w) > 3]

                # Count
                word_counts = Counter(words).most_common(30)

                # Create dataframe
                df_words = pd.DataFrame(word_counts, columns=["Word", "Count"])

                # Plot
                fig = px.bar(
                    df_words,
                    x="Count",
                    y="Word",
                    orientation="h",
                    title="Top 30 Most Common Words",
                    color="Count",
                    color_continuous_scale="Blues",
                )
                fig.update_layout(
                    yaxis=dict(autorange="reversed"),
                    height=600,
                )
                st.plotly_chart(fig, use_container_width=True)

        # Download option
        st.divider()
        st.markdown("### 📥 Download Data")

        col1, col2 = st.columns(2)

        with col1:
            csv = df.to_csv(index=False)
            st.download_button(
                label=f"📥 Download {selected_dataset}_{split}.csv",
                data=csv,
                file_name=f"{selected_dataset}_{split}.csv",
                mime="text/csv",
            )

        with col2:
            # Stats summary
            stats = {
                "dataset": selected_dataset,
                "split": split,
                "total_samples": len(df),
                "num_classes": df["label"].nunique(),
                "avg_text_length": df["text"].str.len().mean(),
                "class_distribution": df["label"].value_counts().to_dict(),
            }

            import json
            stats_json = json.dumps(stats, indent=2)
            st.download_button(
                label="📥 Download Statistics (JSON)",
                data=stats_json,
                file_name=f"{selected_dataset}_{split}_stats.json",
                mime="application/json",
            )

    else:
        st.error(f"❌ Dataset file not found: data/processed/{selected_dataset}_{split}.csv")
        st.info("Please run the preprocessing script first to generate the processed datasets.")


if __name__ == "__main__":
    main()
