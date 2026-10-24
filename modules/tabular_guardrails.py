"""
Tab 3: Tabular Interpretability — LLM Guardrail Audit / Prompt Risk Scoring
Model  : RandomForestClassifier trained on synthetic prompt-security features
Method : SHAP TreeExplainer → Waterfall plot + live counterfactual sliders
Output : Risk gauge · SHAP waterfall · What-If slider panel
"""

import warnings
warnings.filterwarnings("ignore")

import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import shap
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# ---------------------------------------------------------------------------
# Feature schema
# ---------------------------------------------------------------------------
FEATURE_NAMES = [
    "prompt_length_tokens",
    "toxic_word_ratio",
    "entropy_score",
    "special_char_ratio",
    "semantic_distance_to_jailbreak",
    "readability_fk_grade",
]

# (description, min, max, safe_default)
FEATURE_META = {
    "prompt_length_tokens":            ("Token count of the prompt",                      10,   512, 80.0),
    "toxic_word_ratio":                ("Fraction of toxic / harmful words  [0–1]",        0.0,  1.0, 0.05),
    "entropy_score":                   ("Shannon entropy of token distribution  [0–8]",    0.0,  8.0, 4.5),
    "special_char_ratio":              ("Ratio of special chars (@#$% etc.)  [0–1]",       0.0,  1.0, 0.02),
    "semantic_distance_to_jailbreak":  ("Cosine distance from known jailbreak embeds [0–1]", 0.0, 1.0, 0.80),
    "readability_fk_grade":            ("Flesch-Kincaid grade level  [0–20]",              0.0, 20.0, 9.0),
}

# Preset scenarios
PRESETS = {
    "✅ Safe Query": {
        "prompt_length_tokens": 45.0,
        "toxic_word_ratio": 0.01,
        "entropy_score": 4.2,
        "special_char_ratio": 0.01,
        "semantic_distance_to_jailbreak": 0.88,
        "readability_fk_grade": 9.5,
    },
    "⚠️ Borderline Prompt": {
        "prompt_length_tokens": 180.0,
        "toxic_word_ratio": 0.15,
        "entropy_score": 5.8,
        "special_char_ratio": 0.18,
        "semantic_distance_to_jailbreak": 0.35,
        "readability_fk_grade": 5.2,
    },
    "🚨 Jailbreak Attempt": {
        "prompt_length_tokens": 420.0,
        "toxic_word_ratio": 0.45,
        "entropy_score": 7.2,
        "special_char_ratio": 0.52,
        "semantic_distance_to_jailbreak": 0.08,
        "readability_fk_grade": 3.1,
    },
}


# ---------------------------------------------------------------------------
# Data synthesis
# ---------------------------------------------------------------------------
def _synthesize_dataset(n: int = 700, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    n_safe = n // 2
    n_risk = n - n_safe

    safe = {
        "prompt_length_tokens":           rng.integers(10, 150, n_safe).astype(float),
        "toxic_word_ratio":               rng.beta(1, 20, n_safe),
        "entropy_score":                  rng.normal(4.5, 0.8, n_safe).clip(1.0, 7.0),
        "special_char_ratio":             rng.beta(1, 30, n_safe),
        "semantic_distance_to_jailbreak": rng.beta(8, 2, n_safe),
        "readability_fk_grade":           rng.normal(9.0, 2.5, n_safe).clip(2.0, 18.0),
        "label": np.zeros(n_safe, dtype=int),
    }

    risky = {
        "prompt_length_tokens":           rng.integers(80, 512, n_risk).astype(float),
        "toxic_word_ratio":               rng.beta(4, 4, n_risk),
        "entropy_score":                  rng.normal(6.2, 1.0, n_risk).clip(3.0, 8.0),
        "special_char_ratio":             rng.beta(5, 3, n_risk),
        "semantic_distance_to_jailbreak": rng.beta(2, 6, n_risk),
        "readability_fk_grade":           rng.normal(5.0, 2.0, n_risk).clip(1.0, 15.0),
        "label": np.ones(n_risk, dtype=int),
    }

    df = pd.concat(
        [pd.DataFrame(safe), pd.DataFrame(risky)],
        ignore_index=True,
    ).sample(frac=1, random_state=seed).reset_index(drop=True)
    return df


# ---------------------------------------------------------------------------
# Model training — cached per session
# ---------------------------------------------------------------------------
@st.cache_resource(show_spinner=False)
def train_model():
    """Synthesize data, train RF, build SHAP explainer. Cached after first call."""
    df = _synthesize_dataset()
    X = df[FEATURE_NAMES].values
    y = df["label"].values

    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    clf = RandomForestClassifier(n_estimators=150, max_depth=8, random_state=42, n_jobs=-1)
    clf.fit(X_tr, y_tr)

    explainer = shap.TreeExplainer(clf)

    report = classification_report(y_te, clf.predict(X_te), output_dict=True)
    return clf, explainer, df, report


# ---------------------------------------------------------------------------
# Prediction helper
# ---------------------------------------------------------------------------
def predict_single(clf: RandomForestClassifier, feat_dict: dict):
    X = np.array([[feat_dict[f] for f in FEATURE_NAMES]])
    proba = clf.predict_proba(X)[0]
    pred = int(clf.predict(X)[0])
    return pred, proba


# ---------------------------------------------------------------------------
# Visualization helpers
# ---------------------------------------------------------------------------
def render_risk_gauge(risk_prob: float):
    """Styled risk indicator card."""
    if risk_prob > 0.65:
        color, label = "#e84040", "🔴 HIGH RISK — Likely Injection / Jailbreak"
    elif risk_prob > 0.35:
        color, label = "#f0a500", "🟡 MEDIUM RISK — Review Required"
    else:
        color, label = "#00b450", "🟢 SAFE — Normal Query"

    st.markdown(
        f"""
        <div style="
            background:#1a1d27;
            border-radius:14px;
            padding:24px 16px;
            text-align:center;
            border:2px solid {color};
        ">
            <div style="font-size:42px; font-weight:900; color:{color};">{risk_prob:.1%}</div>
            <div style="font-size:17px; color:white; margin-top:10px;">{label}</div>
            <div style="font-size:12px; color:#888; margin-top:6px;">
                Predicted probability of being a jailbreak / injection attempt
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def plot_shap_waterfall(explainer: shap.TreeExplainer, feat_dict: dict) -> plt.Figure:
    """Custom SHAP waterfall bar chart (avoids JS rendering issues in Streamlit)."""
    X = np.array([[feat_dict[f] for f in FEATURE_NAMES]])
    shap_values = explainer(X)

    # For multi-output (binary RF), pick class 1 (risk)
    if len(shap_values.shape) == 3:
        sv = shap_values[:, :, 1]
    else:
        sv = shap_values

    vals = sv.values[0]
    base_val = float(sv.base_values[0])
    feat_values = X[0]

    # Sort by absolute magnitude (ascending for bottom-up barh)
    order = np.argsort(np.abs(vals))
    sorted_vals = vals[order]
    feat_names_sorted = FEATURE_NAMES
    sorted_labels = [
        f"{feat_names_sorted[i]}\n({feat_values[i]:.3f})"
        for i in order
    ]
    colors = ["#e84040" if v > 0 else "#00b4d8" for v in sorted_vals]

    fig, ax = plt.subplots(figsize=(9, 5))
    fig.patch.set_facecolor("#0e1117")
    ax.set_facecolor("#1a1d27")

    bars = ax.barh(sorted_labels, sorted_vals, color=colors, height=0.55, edgecolor="none")
    ax.axvline(0, color="#666", linewidth=1.2)

    # Annotate bars
    for bar, val in zip(bars, sorted_vals):
        x = bar.get_width()
        ax.text(
            x + (0.003 if x >= 0 else -0.003),
            bar.get_y() + bar.get_height() / 2,
            f"{val:+.3f}",
            va="center",
            ha="left" if x >= 0 else "right",
            fontsize=8,
            color="white",
        )

    ax.set_xlabel("SHAP Value  (↑ increases risk  |  ↓ reduces risk)", color="#aaa", fontsize=10)
    ax.set_title(
        f"SHAP Waterfall  —  base risk score: {base_val:.3f}",
        color="white", fontsize=12, pad=12,
    )
    ax.tick_params(axis="both", colors="white", labelsize=9)
    ax.spines[["top", "right", "bottom", "left"]].set_visible(False)
    for lbl in ax.get_yticklabels():
        lbl.set_color("white")
    for lbl in ax.get_xticklabels():
        lbl.set_color("#aaaaaa")

    plt.tight_layout()
    return fig


def plot_global_importance(clf: RandomForestClassifier) -> plt.Figure:
    """Horizontal bar chart of Random Forest feature importances."""
    importances = clf.feature_importances_
    order = np.argsort(importances)

    fig, ax = plt.subplots(figsize=(8, 3.5))
    fig.patch.set_facecolor("#0e1117")
    ax.set_facecolor("#1a1d27")

    ax.barh(
        [FEATURE_NAMES[i] for i in order],
        importances[order],
        color="#7c3aed",
        edgecolor="none",
        height=0.55,
    )
    ax.set_title("Global Feature Importance (Mean Decrease Impurity)", color="white", fontsize=11, pad=10)
    ax.set_xlabel("Importance", color="#aaa", fontsize=10)
    ax.tick_params(colors="white", labelsize=9)
    ax.spines[["top", "right", "bottom", "left"]].set_visible(False)
    for lbl in ax.get_yticklabels():
        lbl.set_color("white")
    for lbl in ax.get_xticklabels():
        lbl.set_color("#aaaaaa")

    plt.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Streamlit tab renderer
# ---------------------------------------------------------------------------
def render_tab():
    st.markdown("### 📊 LLM Guardrail Audit — Prompt Risk Scoring")
    st.markdown(
        "A **RandomForestClassifier** trained on 700 synthetic prompt-security samples. "
        "**SHAP TreeExplainer** explains *why* each prompt is labeled **Safe** or **High-Risk Injection**. "
        "Use the sliders below for live **What-If counterfactual testing**."
    )

    # ---- Train / load model ----
    with st.spinner("⏳ Training classifier & initializing SHAP explainer…"):
        clf, explainer, df, report = train_model()

    # ---- Model summary ----
    with st.expander("📈 Model Performance on Hold-out Test Set"):
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Accuracy", f"{report['accuracy']:.1%}")
        m2.metric("Precision (Risk)", f"{report['1']['precision']:.1%}")
        m3.metric("Recall (Risk)", f"{report['1']['recall']:.1%}")
        m4.metric("F1 (Risk)", f"{report['1']['f1-score']:.1%}")

    st.markdown("---")

    # ---- Preset scenarios ----
    st.markdown("#### 🎭 Quick Scenario Presets")
    preset_cols = st.columns(len(PRESETS))
    for col, (preset_name, preset_vals) in zip(preset_cols, PRESETS.items()):
        with col:
            if st.button(preset_name, use_container_width=True, key=f"preset_{preset_name}"):
                for feat, val in preset_vals.items():
                    st.session_state[f"tab3_{feat}"] = float(val)

    st.markdown("---")
    st.markdown("#### 🎛️ Feature Sliders — What-If Counterfactual Testing")

    # ---- Feature sliders ----
    feature_values: dict[str, float] = {}
    left_col, right_col = st.columns(2)

    for i, feat in enumerate(FEATURE_NAMES):
        desc, mn, mx, default = FEATURE_META[feat]
        col = left_col if i % 2 == 0 else right_col
        step = (mx - mn) / 200.0
        with col:
            val = st.slider(
                f"**{feat}**",
                min_value=float(mn),
                max_value=float(mx),
                value=float(st.session_state.get(f"tab3_{feat}", default)),
                step=step,
                help=desc,
                key=f"tab3_{feat}",
            )
            feature_values[feat] = val

    st.markdown("---")

    # ---- Live prediction ----
    pred, proba = predict_single(clf, feature_values)
    risk_prob = proba[1]

    gauge_col, shap_col = st.columns([1, 2])

    with gauge_col:
        st.markdown("#### Risk Score")
        render_risk_gauge(risk_prob)
        st.markdown("<br>", unsafe_allow_html=True)
        st.metric("Safe Probability", f"{proba[0]:.1%}")

    with shap_col:
        st.markdown("#### SHAP Explanation")
        st.caption(
            "🔴 Red bars = features that **increase** the risk score "
            "| 🔵 Blue bars = features that **reduce** it."
        )
        try:
            fig = plot_shap_waterfall(explainer, feature_values)
            st.pyplot(fig)
            plt.close(fig)
        except Exception as exc:
            st.error(f"SHAP plot error: {exc}")

    # ---- Global importance ----
    st.markdown("---")
    with st.expander("🌍 Global Feature Importance (across all prompts)"):
        fig_imp = plot_global_importance(clf)
        st.pyplot(fig_imp)
        plt.close(fig_imp)
        st.caption(
            "This chart shows which features the Random Forest relies on most "
            "across **all** training examples (global view). "
            "The SHAP waterfall above shows the **local** explanation for your specific slider values."
        )

    # ---- Interpretation note ----
    verdict = "🔴 HIGH-RISK INJECTION" if risk_prob > 0.65 else ("🟡 BORDERLINE" if risk_prob > 0.35 else "🟢 SAFE")
    st.info(
        f"**Current verdict:** {verdict} ({risk_prob:.1%} injection risk). "
        "Try moving the **Semantic Distance** slider down and the **Toxic Word Ratio** up "
        "to watch the score flip to HIGH RISK in real time."
    )
