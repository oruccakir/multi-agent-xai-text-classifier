"""
Multi-Agent XAI Text Classification System - Streamlit UI
Main entry point for the application.
"""

import streamlit as st
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Page configuration
st.set_page_config(
    page_title="Multi-Agent XAI Text Classifier",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1E88E5;
        text-align: center;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #666;
        text-align: center;
        margin-bottom: 2rem;
    }
    .agent-box {
        background-color: #f0f2f6;
        border-radius: 10px;
        padding: 1rem;
        margin: 0.5rem 0;
    }
    .prediction-box {
        background-color: #e8f5e9;
        border-radius: 10px;
        padding: 1.5rem;
        text-align: center;
        margin: 1rem 0;
    }
    .prediction-label {
        font-size: 2rem;
        font-weight: bold;
        color: #2e7d32;
    }
    .confidence-label {
        font-size: 1.5rem;
        color: #666;
    }
    .explanation-box {
        background-color: #fff3e0;
        border-radius: 10px;
        padding: 1rem;
        margin: 1rem 0;
        border-left: 4px solid #ff9800;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 2rem;
    }
    .stTabs [data-baseweb="tab"] {
        padding: 1rem 2rem;
    }
</style>
""", unsafe_allow_html=True)


def main():
    # Header
    st.markdown('<p class="main-header">🤖 Multi-Agent XAI Text Classifier</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Intelligent Text Classification with Explainable AI</p>', unsafe_allow_html=True)

    # Divider
    st.divider()

    # Welcome section
    col1, col2, col3 = st.columns([1, 2, 1])

    with col2:
        st.markdown("""
        ### Welcome! 👋

        This system uses a **multi-agent architecture** to classify text and explain its decisions:

        | Agent | Role |
        |-------|------|
        | 🎯 **Agent 1** | Intent Classifier - Detects language & domain |
        | 🔮 **Agent 2** | Classification - Predicts category |
        | 💡 **Agent 3** | XAI - Explains the decision |
        """)

    st.divider()

    # Features overview
    st.markdown("### 🚀 Features")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.markdown("""
        <div style="background-color: #e3f2fd; padding: 1rem; border-radius: 10px; text-align: center;">
            <h4>📝 Single Text</h4>
            <p>Classify one text with full explanation</p>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("""
        <div style="background-color: #f3e5f5; padding: 1rem; border-radius: 10px; text-align: center;">
            <h4>📊 Batch Processing</h4>
            <p>Process multiple texts from CSV</p>
        </div>
        """, unsafe_allow_html=True)

    with col3:
        st.markdown("""
        <div style="background-color: #e8f5e9; padding: 1rem; border-radius: 10px; text-align: center;">
            <h4>⚖️ Model Comparison</h4>
            <p>Compare all 6 models</p>
        </div>
        """, unsafe_allow_html=True)

    with col4:
        st.markdown("""
        <div style="background-color: #fff3e0; padding: 1rem; border-radius: 10px; text-align: center;">
            <h4>📚 Dataset Explorer</h4>
            <p>Explore the datasets</p>
        </div>
        """, unsafe_allow_html=True)

    st.divider()

    # Supported datasets and models
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### 📁 Supported Datasets")
        st.markdown("""
        | Dataset | Language | Classes |
        |---------|----------|---------|
        | IMDB | English | 2 (positive/negative) |
        | Turkish Sentiment | Turkish | 3 (pozitif/negatif/notr) |
        | AG News | English | 4 (World/Sports/Business/Tech) |
        | Turkish News | Turkish | 7 categories |
        """)

    with col2:
        st.markdown("### 🔧 Classification Models")
        st.markdown("""
        | Model | Type |
        |-------|------|
        | Naive Bayes | Probabilistic |
        | SVM | Geometric |
        | Random Forest | Ensemble |
        | KNN | Instance-based |
        | Logistic Regression | Linear |
        | Transformer | Deep Learning |
        """)

    st.divider()

    # Quick start
    st.markdown("### 🎯 Quick Start")
    st.info("👈 Use the **sidebar** to navigate to different pages, or click the buttons below:")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        if st.button("📝 Classify Text", width="stretch"):
            st.switch_page("pages/1_📝_Classify_Text.py")

    with col2:
        if st.button("📊 Batch Process", width="stretch"):
            st.switch_page("pages/2_📊_Batch_Processing.py")

    with col3:
        if st.button("⚖️ Compare Models", width="stretch"):
            st.switch_page("pages/3_⚖️_Model_Comparison.py")

    with col4:
        if st.button("📚 Explore Data", width="stretch"):
            st.switch_page("pages/4_📚_Dataset_Explorer.py")

    # Footer
    st.divider()
    st.markdown("""
    <div style="text-align: center; color: #888; font-size: 0.9rem;">
        <p>Built by Oruç Çakır | TOBB ETÜ - BİL 443/564 | Spring 2025-26</p>
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
