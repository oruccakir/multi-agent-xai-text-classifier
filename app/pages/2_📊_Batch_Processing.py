"""
Batch Processing Page
Process multiple texts from CSV file.
"""

import streamlit as st
import sys
from pathlib import Path
import pandas as pd
import plotly.express as px

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.preprocessing.text_preprocessor import TextPreprocessor

# Page configuration
st.set_page_config(
    page_title="Batch Processing - XAI Classifier",
    page_icon="📊",
    layout="wide",
)


def simulate_batch_classification(texts: list, dataset: str) -> list:
    """Simulate classification for batch of texts."""
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

    results = []
    for i, text in enumerate(texts):
        random.seed(hash(text) % 2**32)
        probs = [random.random() for _ in classes]
        total = sum(probs)
        probs = [p / total for p in probs]
        max_idx = probs.index(max(probs))

        results.append({
            "text": text[:100] + "..." if len(text) > 100 else text,
            "prediction": classes[max_idx],
            "confidence": probs[max_idx],
        })

    return results


def main():
    st.title("📊 Batch Processing")
    st.markdown("Process multiple texts at once from a CSV file")

    st.divider()

    # Instructions
    with st.expander("📋 Instructions", expanded=True):
        st.markdown("""
        ### How to use:
        1. **Prepare your CSV file** with a column named `text` containing the texts to classify
        2. **Upload the CSV** using the file uploader below
        3. **Select the dataset** that matches your text type
        4. **Click Process** to classify all texts
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
        dataset = st.selectbox(
            "Dataset/Model:",
            ["imdb", "turkish_sentiment", "ag_news", "turkish_news"],
            format_func=lambda x: {
                "imdb": "🎬 IMDB (English Sentiment)",
                "turkish_sentiment": "🇹🇷 Turkish Sentiment",
                "ag_news": "📰 AG News (English)",
                "turkish_news": "🇹🇷 Turkish News",
            }[x]
        )

    # Process uploaded file
    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)

            # Check for text column
            if "text" not in df.columns:
                st.error("❌ CSV must have a 'text' column!")
                st.markdown("**Columns found:** " + ", ".join(df.columns))
                return

            st.divider()

            # Preview data
            st.markdown("### 👀 Data Preview")
            st.dataframe(df.head(10), use_container_width=True)
            st.info(f"📊 Total rows: **{len(df)}**")

            # Process button
            if st.button("🚀 Process All Texts", type="primary", use_container_width=True):
                with st.spinner(f"Processing {len(df)} texts..."):
                    # Get texts
                    texts = df["text"].tolist()

                    # Progress bar
                    progress_bar = st.progress(0)
                    status_text = st.empty()

                    # Process in batches for progress tracking
                    batch_size = max(1, len(texts) // 20)
                    results = []

                    for i in range(0, len(texts), batch_size):
                        batch = texts[i:i + batch_size]
                        batch_results = simulate_batch_classification(batch, dataset)
                        results.extend(batch_results)

                        progress = min(1.0, (i + batch_size) / len(texts))
                        progress_bar.progress(progress)
                        status_text.text(f"Processed {min(i + batch_size, len(texts))}/{len(texts)} texts")

                    progress_bar.progress(1.0)
                    status_text.text("✅ Processing complete!")

                # Create results dataframe
                results_df = pd.DataFrame(results)

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
                st.plotly_chart(fig, use_container_width=True)

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
                st.plotly_chart(fig, use_container_width=True)

                # Results table
                st.markdown("#### Detailed Results")
                st.dataframe(results_df, use_container_width=True)

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
            st.error(f"❌ Error processing file: {str(e)}")

    else:
        # Show sample data option
        st.divider()
        st.markdown("### 🧪 Try with Sample Data")

        if st.button("Load Sample Data", use_container_width=True):
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

            # Save to session state and trigger reload
            st.session_state.sample_df = sample_data
            st.info("✅ Sample data loaded! Click 'Process All Texts' to classify.")
            st.dataframe(sample_data, use_container_width=True)


if __name__ == "__main__":
    main()
