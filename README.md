# 🧠 NeuroLens — End-to-End XAI Workbench

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch CPU](https://img.shields.io/badge/PyTorch-2.2+_(CPU)-EE4C2C.svg?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![Hugging Face](https://img.shields.io/badge/%F0%9F%A4%97-Transformers-yellow.svg)](https://huggingface.co/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.32+-FF4B4B.svg?logo=streamlit&logoColor=white)](https://streamlit.io/)
[![Captum](https://img.shields.io/badge/Captum-XAI-brightgreen.svg)](https://captum.ai/)
[![SHAP](https://img.shields.io/badge/SHAP-Interpretability-orange.svg)](https://shap.readthedocs.io/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

> **Portfolio-grade Explainable AI workbench demonstrating interpretability across Text, Image, and Tabular modalities.** Built with a strict focus on modern GenAI explainability, zero API key dependencies, and 100% CPU-compatible execution.

---

## 🔬 What is NeuroLens?

NeuroLens is a Streamlit-based interactive workbench that applies state-of-the-art Explainable AI (XAI) methods to three distinct data modalities:

| Tab | Modality | Model | Task | XAI Technique |
|-----|----------|-------|------|---------------|
| 📝 Text | Natural Language | DistilBERT (`distilbert-base-cased-distilled-squad`) | Extractive Question Answering | Dual Layer Integrated Gradients (Captum) |
| 🖼️ Image | Computer Vision | CLIP (`openai/clip-vit-base-patch32`) | Visual Grounding | Spatial Patch Cosine Similarity Heatmap |
| 📊 Tabular | Prompt Security | Scikit-Learn `RandomForestClassifier` | LLM Guardrail Risk Scoring | SHAP TreeExplainer + Live Counterfactuals |

---

## 📁 Repository Structure

```
Neurolens/
├── app.py                          # Main Streamlit application entry point
├── modules/
│   ├── __init__.py
│   ├── text_saliency.py            # Tab 1: Extractive Q&A + Captum LayerIntegratedGradients
│   ├── vision_grounding.py         # Tab 2: CLIP patch cosine similarity grounding
│   └── tabular_guardrails.py       # Tab 3: RF + SHAP + live counterfactual sliders
├── sample_data/
│   ├── sample_texts.json           # Extractive Q&A benchmark reference pairs
│   └── sample_images/
│       ├── dog_grass.jpg           # Sample image: Dog on grass
│       ├── red_car.jpg             # Sample image: Sports car
│       └── living_room.jpg         # Sample image: Modern living room
├── .gitignore                      # Git exclusion rules for clean commits
├── LICENSE                         # MIT License
├── requirements.txt                # Production Python dependencies
└── README.md                       # Comprehensive documentation
```

---

## 🚀 Quick Start (Running Manually)

### 1. Clone & Set Up Environment

```bash
# Clone the repository
git clone https://github.com/Niranjan070/Neurolens.git
cd Neurolens

# Create and activate a virtual environment
python -m venv .venv

# Windows (Command Prompt / PowerShell)
.venv\Scripts\activate

# macOS / Linux
source .venv/bin/activate
```

### 2. Install Dependencies

```bash
# Install PyTorch (CPU-only — no external GPU needed)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu

# Install remaining dependencies
pip install -r requirements.txt
```

### 3. Launch the Workbench

```bash
streamlit run app.py
```

The app will launch at **http://localhost:8501** in your browser.

> **Note:** On first launch, Hugging Face will download the lightweight weights for DistilBERT-SQuAD (~260 MB) and CLIP ViT-B/32 (~350 MB). Subsequent runs are instant due to `@st.cache_resource`.

---

## 🔬 Tab-by-Tab Guide

### Tab 1 — 📝 Text Interpretability (Extractive Q&A + Dual Attribution)

- **Input:** Context paragraph + Question (choose from 3 built-in presets: 🏛️ History, 🤖 AI/ML, 🌍 Science, or enter custom context & question).
- **Inference:** DistilBERT-SQuAD identifies the exact answer token span within the context.
- **Explainability:** 
  - Computes **dual Captum Layer Integrated Gradients** on the DistilBERT embedding layer targeting both the **Start Logit** and **End Logit**.
  - Highlights context tokens in real time (🟢 Green = positive attribution towards the answer, 🔴 Red = negative/inhibiting attribution).
  - Distinguishes extracted answer span with a prominent gold highlight (⭐).
  - Provides interactive bar charts showing top contributing tokens from both the question and context.

**Technique:** `captum.attr.LayerIntegratedGradients` applied to `model.distilbert.embeddings`.

---

### Tab 2 — 🖼️ Image Interpretability (Visual Grounding Heatmap)

- **Input:** Select a sample image (sports car, dog on grass, living room) or upload a custom image (`.jpg`, `.png`).
- **Query:** Enter any descriptive prompt (e.g., *"A red sports car on the road"*).
- **Explainability:**
  - Extracts 49 spatial patch embeddings (7×7 grid) from the final ViT encoder block via a PyTorch forward hook.
  - Projects patch embeddings through CLIP's visual projection head and computes normalized cosine similarities against the text query embedding.
  - Generates a jet colormap heatmap overlay showing where the vision-language model focuses its attention.

**Technique:** Spatial patch activation projection with cosine similarity grounding, bilinearly interpolated onto image coordinates.

---

### Tab 3 — 📊 Tabular Guardrails (Prompt Risk Scoring & Counterfactuals)

- **Input:** Select safety presets (*Safe Prompt*, *Borderline Query*, *Jailbreak Injection*) or manipulate real-time feature sliders.
- **Explainability:**
  - Evaluates security telemetry features (e.g., `toxic_word_ratio`, `jailbreak_embed_dist`, `token_entropy`, `special_char_freq`).
  - Displays a live Risk Gauge (0–100%) and categorizes threat levels (Low, Medium, High).
  - Renders a **SHAP Waterfall Plot** explaining local contribution: Red bars push the risk score higher, Blue bars pull it lower.
  - Global feature importance analysis using TreeExplainer.

**Innovation:** Enterprise GenAI guardrail framing evaluating prompt injection signals rather than toy legacy datasets.

---

## 📐 Rubric Mapping

| Academic Requirement | Implementation |
|---------------------|---------------|
| **GUI-based application** | Streamlit multi-tab workbench with reactive charts & UI controls |
| **Tabular XAI** | RandomForest + SHAP TreeExplainer + live counterfactual sliders |
| **Image XAI** | CLIP ViT patch cosine similarity spatial grounding heatmap |
| **Text XAI** | DistilBERT-SQuAD extractive QA + dual Start/End Captum Layer Integrated Gradients |
| **High-quality visualizations** | Annotated text spans, Jet overlay heatmaps, SHAP waterfall & global bar charts |
| **Innovation factor** | Modern GenAI interpretability (LLM guardrail audit, VLM grounding, Transformer QA attribution) |
| **CPU-only execution** | Lightweight models engineered to run smoothly on standard laptops without GPU |

---

## 🛠️ Technology Stack

| Library | Version | Role |
|---------|---------|------|
| Streamlit | ≥1.32 | GUI workbench framework |
| PyTorch | ≥2.2 (CPU) | Deep learning model inference |
| HuggingFace Transformers | ≥4.40 | DistilBERT-SQuAD + CLIP models |
| Captum | ≥0.7 | Layer Integrated Gradients attribution |
| SHAP | ≥0.45 | TreeExplainer for tabular models |
| Scikit-Learn | ≥1.4 | RandomForestClassifier |
| Pillow | ≥10.0 | Image processing & resizing |
| Matplotlib | ≥3.8 | Explanatory chart & heatmap plotting |
| NumPy / Pandas | ≥1.26 / ≥2.2 | Data processing & tensor manipulation |
| SciPy | ≥1.12 | Spatial interpolation for visual heatmaps |

---

## ⚡ Performance Notes

- First model download: ~30–60 seconds per model depending on internet speed.
- In-memory inference: runs locally on CPU with zero external API calls or secrets.
- Subsequent tab switches: instant caching via Streamlit `@st.cache_resource`.

---

*Built for IA3 Evaluation — Explainable AI Portfolio Project*
