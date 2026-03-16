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

# Custom CSS for better styling (dark theme compatible)
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        background: linear-gradient(135deg, #60a5fa, #a78bfa);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #94a3b8;
        text-align: center;
        margin-bottom: 2rem;
    }
    .agent-box {
        background-color: rgba(30, 41, 59, 0.7);
        border: 1px solid rgba(100, 116, 139, 0.3);
        border-radius: 10px;
        padding: 1rem;
        margin: 0.5rem 0;
    }
    .prediction-box {
        background-color: rgba(16, 185, 129, 0.1);
        border: 1px solid rgba(16, 185, 129, 0.3);
        border-radius: 10px;
        padding: 1.5rem;
        text-align: center;
        margin: 1rem 0;
    }
    .prediction-label {
        font-size: 2rem;
        font-weight: bold;
        color: #34d399;
    }
    .confidence-label {
        font-size: 1.5rem;
        color: #94a3b8;
    }
    .explanation-box {
        background-color: rgba(245, 158, 11, 0.1);
        border-radius: 10px;
        padding: 1rem;
        margin: 1rem 0;
        border-left: 4px solid #f59e0b;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 2rem;
    }
    .stTabs [data-baseweb="tab"] {
        padding: 1rem 2rem;
    }
    /* Feature card styling */
    .feature-card {
        border-radius: 12px;
        padding: 1.5rem 1rem;
        text-align: center;
        min-height: 120px;
        display: flex;
        flex-direction: column;
        align-items: center;
        justify-content: center;
        transition: transform 0.2s ease, box-shadow 0.2s ease;
    }
    .feature-card:hover {
        transform: translateY(-3px);
        box-shadow: 0 8px 25px rgba(0,0,0,0.3);
    }
    .feature-card h4 {
        margin: 0 0 0.5rem 0;
        font-size: 1.1rem;
    }
    .feature-card p {
        margin: 0;
        font-size: 0.9rem;
        opacity: 0.85;
    }
    .card-blue {
        background: linear-gradient(135deg, rgba(59, 130, 246, 0.15), rgba(37, 99, 235, 0.25));
        border: 1px solid rgba(59, 130, 246, 0.35);
        color: #93c5fd;
    }
    .card-blue h4 { color: #93c5fd; }
    .card-blue p { color: #bfdbfe; }
    .card-purple {
        background: linear-gradient(135deg, rgba(168, 85, 247, 0.15), rgba(139, 92, 246, 0.25));
        border: 1px solid rgba(168, 85, 247, 0.35);
        color: #c4b5fd;
    }
    .card-purple h4 { color: #c4b5fd; }
    .card-purple p { color: #ddd6fe; }
    .card-green {
        background: linear-gradient(135deg, rgba(16, 185, 129, 0.15), rgba(5, 150, 105, 0.25));
        border: 1px solid rgba(16, 185, 129, 0.35);
        color: #6ee7b7;
    }
    .card-green h4 { color: #6ee7b7; }
    .card-green p { color: #a7f3d0; }
    .card-amber {
        background: linear-gradient(135deg, rgba(245, 158, 11, 0.15), rgba(217, 119, 6, 0.25));
        border: 1px solid rgba(245, 158, 11, 0.35);
        color: #fcd34d;
    }
    .card-amber h4 { color: #fcd34d; }
    .card-amber p { color: #fde68a; }
    /* Footer styling */
    .footer-text {
        text-align: center;
        color: #64748b;
        font-size: 0.9rem;
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
        <div class="feature-card card-blue">
            <h4>📝 Single Text</h4>
            <p>Classify one text with full explanation</p>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("""
        <div class="feature-card card-purple">
            <h4>📊 Batch Processing</h4>
            <p>Process multiple texts from CSV</p>
        </div>
        """, unsafe_allow_html=True)

    with col3:
        st.markdown("""
        <div class="feature-card card-green">
            <h4>⚖️ Model Comparison</h4>
            <p>Compare all 6 models</p>
        </div>
        """, unsafe_allow_html=True)

    with col4:
        st.markdown("""
        <div class="feature-card card-amber">
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
    <div class="footer-text">
        <p>Built by Oruç Çakır | TOBB ETÜ - BİL 443/564 | Spring 2025-26</p>
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
