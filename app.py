"""
NeuroLens: End-to-End LLM & Multi-Agent Explainability Workbench
================================================================
Main Streamlit application entry point.

Tabs
----
1. 📝 Text Interpretability  — DistilBERT + Captum LayerIntegratedGradients
2. 🖼️ Image Interpretability  — CLIP ViT-B/32 patch cosine-similarity grounding
3. 📊 Tabular Guardrails     — RandomForest + SHAP TreeExplainer (prompt risk)

Run with:
    streamlit run app.py
"""

import os
import sys
import streamlit as st

# Ensure the project root is on the path so `modules` is importable
ROOT = os.path.dirname(os.path.abspath(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from modules import text_saliency, vision_grounding, tabular_guardrails

# ---------------------------------------------------------------------------
# Page configuration
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="NeuroLens — XAI Workbench",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Global CSS
# ---------------------------------------------------------------------------
st.markdown(
    """
    <style>
    /* ---- Base ---- */
    html, body, [data-testid="stAppViewContainer"] {
        background-color: #0e1117;
        color: #e0e0e0;
    }

    /* ---- Sidebar ---- */
    [data-testid="stSidebar"] {
        background: #13151f;
        border-right: 1px solid #222;
    }

    /* ---- Metric cards ---- */
    [data-testid="stMetric"] {
        background: #1a1d27;
        border-radius: 10px;
        padding: 14px 18px;
        border: 1px solid #2a2d3a;
    }
    [data-testid="stMetricLabel"] { color: #aaa !important; font-size: 13px !important; }
    [data-testid="stMetricValue"] { color: #fff !important; font-size: 24px !important; }

    /* ---- Tabs ---- */
    [data-testid="stTabs"] button {
        font-size: 15px;
        font-weight: 600;
        color: #aaa;
    }
    [data-testid="stTabs"] button[aria-selected="true"] {
        color: #7c3aed !important;
        border-bottom: 3px solid #7c3aed;
    }

    /* ---- Buttons ---- */
    .stButton > button {
        background: #7c3aed;
        color: white;
        border-radius: 8px;
        border: none;
        font-weight: 600;
        transition: background 0.2s ease;
    }
    .stButton > button:hover {
        background: #6d28d9;
        color: white;
    }
    .stButton > button[kind="primary"] {
        background: #7c3aed;
        font-size: 15px;
        padding: 10px;
    }

    /* ---- Expanders ---- */
    [data-testid="stExpander"] {
        background: #1a1d27;
        border: 1px solid #2a2d3a;
        border-radius: 10px;
    }

    /* ---- Info / Warning boxes ---- */
    .stAlert {
        border-radius: 10px;
    }

    /* ---- Header banner ---- */
    .nl-header {
        background: linear-gradient(135deg, #1a1d27 0%, #0f1020 60%, #12162b 100%);
        border-radius: 14px;
        padding: 28px 32px 22px;
        border: 1px solid #2a2d3a;
        margin-bottom: 24px;
    }
    .nl-title {
        font-size: 36px;
        font-weight: 900;
        background: linear-gradient(90deg, #7c3aed, #06b6d4, #10b981);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        margin-bottom: 6px;
    }
    .nl-subtitle {
        font-size: 15px;
        color: #9ca3af;
        line-height: 1.6;
    }

    /* ---- KPI cards ---- */
    .kpi-row { display: flex; gap: 16px; margin: 18px 0 24px; }
    .kpi-card {
        flex: 1;
        background: #1a1d27;
        border: 1px solid #2a2d3a;
        border-radius: 12px;
        padding: 18px 20px;
        text-align: center;
    }
    .kpi-number { font-size: 34px; font-weight: 900; }
    .kpi-label  { font-size: 13px; color: #9ca3af; margin-top: 4px; }

    /* ---- Divider ---- */
    hr { border-color: #2a2d3a !important; }

    /* ---- File uploader ---- */
    [data-testid="stFileUploader"] {
        background: #1a1d27;
        border-radius: 10px;
        border: 1px dashed #3a3d4a;
    }

    /* ---- Text area ---- */
    textarea {
        background: #1a1d27 !important;
        border-color: #2a2d3a !important;
        color: #e0e0e0 !important;
    }

    /* Selectbox / slider labels */
    label { color: #ccc !important; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
with st.sidebar:
    st.markdown("## 🧠 NeuroLens")
    st.markdown("**Explainable AI Workbench**")
    st.markdown("---")
    st.markdown(
        """
        ### About
        NeuroLens is a portfolio-grade XAI project demonstrating
        interpretability across **three modalities**:

        | Tab | Modality | Technique |
        |-----|----------|-----------|
        | 📝 Text | NLP | Dual Integrated Gradients (QA) |
        | 🖼️ Image | Vision | Patch Cosine Similarity |
        | 📊 Tabular | Security | SHAP TreeExplainer |

        ---
        ### Quick-Start Guide
        1. Select any **preset** demo or enter custom input
        2. Press the **Analyze / Run** button
        3. Inspect the real-time attribution highlights & charts

        ---
        ### Models Used
        - `distilbert-base-cased-distilled-squad`
        - `openai/clip-vit-base-patch32`
        - `RandomForestClassifier` (trained in-session)

        ---
        ### Capabilities
        ✅ Interactive Multi-Tab Workbench  
        ✅ Tabular Explainability (SHAP)  
        ✅ Computer Vision Grounding (CLIP ViT)  
        ✅ Text Transformer Attribution (Captum LIG)  
        ✅ Enterprise Guardrail Risk Auditing  
        ✅ 100% Local CPU Execution (No API Keys)
        """
    )
    st.markdown("---")
    st.markdown(
        "<small style='color:#555'>NeuroLens XAI Workbench · 100% CPU Inference</small>",
        unsafe_allow_html=True,
    )

# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------
st.markdown(
    """
    <div class="nl-header">
        <div class="nl-title">🧠 NeuroLens</div>
        <div class="nl-subtitle">
            <b>End-to-End LLM & Multi-Agent Explainability Workbench</b><br>
            Interpretable AI across Text, Image, and Tabular modalities —
            powered by Captum, CLIP, and SHAP.
            All models run entirely on CPU. No API keys required.
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# KPI cards
# ---------------------------------------------------------------------------
st.markdown(
    """
    <div class="kpi-row">
        <div class="kpi-card">
            <div class="kpi-number" style="color:#7c3aed;">3</div>
            <div class="kpi-label">AI Models Deployed</div>
        </div>
        <div class="kpi-card">
            <div class="kpi-number" style="color:#06b6d4;">3</div>
            <div class="kpi-label">Data Modalities</div>
        </div>
        <div class="kpi-card">
            <div class="kpi-number" style="color:#10b981;">3</div>
            <div class="kpi-label">XAI Techniques</div>
        </div>
        <div class="kpi-card">
            <div class="kpi-number" style="color:#f59e0b;">100%</div>
            <div class="kpi-label">CPU Inference</div>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Tabs
# ---------------------------------------------------------------------------
tab1, tab2, tab3 = st.tabs([
    "📝  Text Interpretability",
    "🖼️  Image Interpretability",
    "📊  Tabular Guardrails",
])

SAMPLE_IMAGES_DIR = os.path.join(ROOT, "sample_data", "sample_images")

with tab1:
    try:
        text_saliency.render_tab()
    except Exception as exc:
        st.error(f"❌ Text tab error: {exc}")
        st.exception(exc)

with tab2:
    try:
        vision_grounding.render_tab(sample_images_dir=SAMPLE_IMAGES_DIR)
    except Exception as exc:
        st.error(f"❌ Vision tab error: {exc}")
        st.exception(exc)

with tab3:
    try:
        tabular_guardrails.render_tab()
    except Exception as exc:
        st.error(f"❌ Tabular tab error: {exc}")
        st.exception(exc)

# ---------------------------------------------------------------------------
# Footer
# ---------------------------------------------------------------------------
st.markdown("---")
st.markdown(
    "<div style='text-align:center; color:#555; font-size:13px;'>"
    "NeuroLens XAI Workbench · "
    "Captum &nbsp;|&nbsp; HuggingFace Transformers &nbsp;|&nbsp; SHAP &nbsp;|&nbsp; Streamlit"
    "</div>",
    unsafe_allow_html=True,
)
