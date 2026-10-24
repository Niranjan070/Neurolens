"""
Tab 1: Text Interpretability — Extractive Q&A with Attribution
================================================================
Model  : distilbert-base-cased-distilled-squad  (CPU-friendly)
Task   : Extractive Question Answering
         - User provides a context paragraph + a question
         - Model extracts the exact answer span from the context
Method : Captum LayerIntegratedGradients on the embedding layer,
         targeted at the START and END token logits of the answer span
Output :
  - The extracted answer, highlighted in the context
  - Dual attribution maps (start-token & end-token attributions)
  - Bar chart of top contributing tokens from both context & question
"""

import streamlit as st
import torch
import numpy as np
import matplotlib.pyplot as plt
from transformers import AutoTokenizer, AutoModelForQuestionAnswering
from captum.attr import LayerIntegratedGradients

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
MODEL_NAME = "distilbert-base-cased-distilled-squad"

# Preloaded context + question pairs for 1-click demonstration
PRESETS = [
    {
        "label": "🏛️ History",
        "context": (
            "The Eiffel Tower is a wrought-iron lattice tower located on the Champ de Mars "
            "in Paris, France. It was constructed from 1887 to 1889 as the centerpiece of "
            "the 1889 World's Fair. The tower is named after Gustave Eiffel, whose company "
            "designed and built it. It stands 330 metres tall and is the most-visited paid "
            "monument in the world."
        ),
        "question": "When was the Eiffel Tower constructed?",
    },
    {
        "label": "🤖 AI / ML",
        "context": (
            "Transformers are a type of deep learning architecture introduced in the paper "
            "'Attention Is All You Need' by Vaswani et al. in 2017. They rely entirely on "
            "a self-attention mechanism to compute representations of their input and output "
            "without using sequence-aligned recurrent neural networks. BERT, GPT, and T5 "
            "are all based on the Transformer architecture and have revolutionised natural "
            "language processing."
        ),
        "question": "What mechanism do Transformers rely on?",
    },
    {
        "label": "🌍 Science",
        "context": (
            "Photosynthesis is the process by which green plants, algae, and some bacteria "
            "convert light energy into chemical energy stored in glucose. It takes place "
            "primarily in the chloroplasts of plant cells, using carbon dioxide from the air "
            "and water from the soil. Oxygen is released as a by-product of this reaction. "
            "Photosynthesis is the foundation of almost all food chains on Earth."
        ),
        "question": "Where does photosynthesis take place in plant cells?",
    },
]

SPECIAL_TOKENS = {"[CLS]", "[SEP]", "[PAD]"}


# ---------------------------------------------------------------------------
# Model loading — cached per session
# ---------------------------------------------------------------------------
@st.cache_resource(show_spinner=False)
def load_qa_model():
    """Load DistilBERT-SQuAD tokenizer and QA model from HuggingFace Hub."""
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForQuestionAnswering.from_pretrained(MODEL_NAME)
    model.eval()
    return tokenizer, model


# ---------------------------------------------------------------------------
# Inference + attribution
# ---------------------------------------------------------------------------
def run_qa_with_attribution(question: str, context: str, tokenizer, model):
    """
    Run extractive QA and compute Captum attributions for both
    the start-token and end-token logits.

    Returns
    -------
    answer          : str   — the extracted answer span
    tokens          : list[str]
    start_attrs     : np.ndarray  — normalized attributions for start logit
    end_attrs       : np.ndarray  — normalized attributions for end logit
    answer_start    : int   — token index of answer start
    answer_end      : int   — token index of answer end (inclusive)
    question_len    : int   — number of question tokens (incl. [CLS])
    confidence      : float — max start logit probability
    """
    # Tokenize — keep offset mapping to reconstruct span
    encoding = tokenizer(
        question,
        context,
        return_tensors="pt",
        truncation=True,
        max_length=384,
        return_offsets_mapping=True,
        padding=False,
    )

    input_ids = encoding["input_ids"]
    attention_mask = encoding["attention_mask"]
    token_type_ids = encoding.get("token_type_ids", None)

    # Forward pass to get predicted span
    with torch.no_grad():
        if token_type_ids is not None:
            outputs = model(input_ids=input_ids, attention_mask=attention_mask,
                            token_type_ids=token_type_ids)
        else:
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)

    start_logits = outputs.start_logits[0]
    end_logits = outputs.end_logits[0]

    # Mask question tokens (type_id == 0) from being the answer
    seq_len = input_ids.shape[1]
    if token_type_ids is not None:
        context_mask = token_type_ids[0].bool()
    else:
        # DistilBERT has no token_type_ids — find [SEP] position
        sep_positions = (input_ids[0] == tokenizer.sep_token_id).nonzero(as_tuple=True)[0]
        first_sep = sep_positions[0].item()
        context_mask = torch.zeros(seq_len, dtype=torch.bool)
        context_mask[first_sep + 1:] = True

    # Best valid (start <= end) answer
    start_logits_masked = start_logits.clone()
    start_logits_masked[~context_mask] = -1e9
    end_logits_masked = end_logits.clone()
    end_logits_masked[~context_mask] = -1e9

    # Greedy best span
    best_start = int(torch.argmax(start_logits_masked))
    best_end = int(torch.argmax(end_logits_masked))
    if best_end < best_start:
        best_end = best_start

    # Decode answer
    answer_ids = input_ids[0][best_start: best_end + 1]
    answer = tokenizer.decode(answer_ids, skip_special_tokens=True).strip()

    # Confidence
    start_probs = torch.softmax(start_logits, dim=0)
    confidence = float(start_probs[best_start].item())

    tokens = tokenizer.convert_ids_to_tokens(input_ids[0])

    # Question length = index of first [SEP]
    question_len = int((input_ids[0] == tokenizer.sep_token_id).nonzero()[0].item()) + 1

    # ---- Captum: attributions for START logit ----
    def forward_start(ids, mask):
        out = model(input_ids=ids, attention_mask=mask)
        return out.start_logits

    def forward_end(ids, mask):
        out = model(input_ids=ids, attention_mask=mask)
        return out.end_logits

    lig = LayerIntegratedGradients(forward_start, model.distilbert.embeddings)
    baseline_ids = torch.zeros_like(input_ids)

    start_attr, _ = lig.attribute(
        inputs=input_ids,
        baselines=baseline_ids,
        additional_forward_args=(attention_mask,),
        target=best_start,
        return_convergence_delta=True,
        n_steps=50,
    )
    start_attrs = start_attr.sum(dim=-1).squeeze(0).detach().numpy()

    lig_end = LayerIntegratedGradients(forward_end, model.distilbert.embeddings)
    end_attr, _ = lig_end.attribute(
        inputs=input_ids,
        baselines=baseline_ids,
        additional_forward_args=(attention_mask,),
        target=best_end,
        return_convergence_delta=True,
        n_steps=50,
    )
    end_attrs = end_attr.sum(dim=-1).squeeze(0).detach().numpy()

    # Normalize each independently to [-1, 1]
    def _norm(arr):
        m = np.max(np.abs(arr)) + 1e-8
        return arr / m

    return (
        answer, tokens, _norm(start_attrs), _norm(end_attrs),
        best_start, best_end, question_len, confidence
    )


# ---------------------------------------------------------------------------
# Visualization helpers
# ---------------------------------------------------------------------------
def _clean(tok: str) -> str:
    return tok.replace("##", "").replace("Ġ", "").strip()


def render_attributed_context(tokens, attributions, answer_start, answer_end,
                               question_len, label="") -> str:
    """
    HTML span visualization.
    - Question tokens: shown in a dimmed header row
    - Context tokens: color-coded by attribution
    - Answer span: bold gold border
    """
    # ---- Question section ----
    q_html = '<div style="font-size:13px; color:#888; margin-bottom:6px;">Question tokens:</div>'
    q_html += '<div style="font-size:14px; line-height:2.4; font-family:monospace; margin-bottom:12px;">'
    for i, (tok, attr) in enumerate(zip(tokens[:question_len], attributions[:question_len])):
        if tok in SPECIAL_TOKENS:
            continue
        clean = _clean(tok)
        if not clean:
            continue
        alpha = min(abs(attr) * 0.7, 0.7)
        bg = f"rgba(6,182,212,{alpha:.2f})" if attr > 0 else f"rgba(249,115,22,{alpha:.2f})"
        q_html += (
            f'<span style="background:{bg}; padding:3px 5px; border-radius:4px; '
            f'margin:2px; color:white; font-weight:600;">{clean}</span> '
        )
    q_html += '</div>'

    # ---- Context section ----
    c_html = f'<div style="font-size:13px; color:#888; margin-bottom:6px;">Context — {label}:</div>'
    c_html += (
        '<div style="font-size:15px; line-height:2.8; font-family:monospace; '
        'background:#1a1d27; border-radius:10px; padding:14px;">'
    )
    for i, (tok, attr) in enumerate(zip(tokens[question_len:], attributions[question_len:]),
                                     start=question_len):
        if tok in SPECIAL_TOKENS:
            continue
        clean = _clean(tok)
        if not clean:
            continue

        is_answer = (answer_start <= i <= answer_end)
        alpha = min(abs(attr) * 0.8, 0.8)
        bg = f"rgba(0,180,80,{alpha:.2f})" if attr > 0 else f"rgba(220,50,50,{alpha:.2f})"

        border = "3px solid #f59e0b" if is_answer else "none"
        font_w = "900" if is_answer else "600"
        extra = "⭐" if is_answer else ""

        c_html += (
            f'<span style="background:{bg}; padding:4px 6px; border-radius:5px; '
            f'margin:2px; color:white; font-weight:{font_w}; '
            f'border:{border}; outline:{border};">'
            f"{extra}{clean}{extra}</span> "
        )
    c_html += '</div>'

    return q_html + c_html


def plot_top_tokens(tokens, attributions, question_len, title="") -> plt.Figure:
    """Bar chart of top contributing tokens from question + context."""
    pairs = []
    for i, (tok, attr) in enumerate(zip(tokens, attributions)):
        if tok in SPECIAL_TOKENS or not _clean(tok):
            continue
        section = "Q" if i < question_len else "C"
        pairs.append((f"[{section}] {_clean(tok)}", float(attr)))

    pairs.sort(key=lambda x: abs(x[1]), reverse=True)
    pairs = pairs[:14]
    pairs.reverse()

    labels = [p[0] for p in pairs]
    values = [p[1] for p in pairs]
    colors = ["#00b450" if v > 0 else "#e84040" for v in values]

    fig, ax = plt.subplots(figsize=(9, 5))
    fig.patch.set_facecolor("#0e1117")
    ax.set_facecolor("#1a1d27")

    ax.barh(labels, values, color=colors, height=0.6, edgecolor="none")
    ax.axvline(0, color="#555", linewidth=1.2)
    ax.set_xlabel("Attribution Score", color="#aaa", fontsize=11)
    ax.set_title(title, color="white", fontsize=12, pad=12)
    ax.tick_params(axis="both", colors="white", labelsize=9)
    ax.spines[["top", "right", "bottom", "left"]].set_visible(False)
    for lbl in ax.get_yticklabels():
        lbl.set_color("white")
        lbl.set_fontsize(9)
    for lbl in ax.get_xticklabels():
        lbl.set_color("#aaa")

    # Legend note
    ax.text(
        0.99, 0.02,
        "[Q] = Question token  [C] = Context token",
        transform=ax.transAxes,
        ha="right", va="bottom",
        fontsize=8, color="#888",
    )
    plt.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Streamlit tab renderer
# ---------------------------------------------------------------------------
def render_tab():
    st.markdown("### 📝 Extractive Q&A Interpretability")
    st.markdown(
        "Ask **any question** about a passage of text. "
        "**DistilBERT-SQuAD** extracts the exact answer from the context. "
        "**Captum Layer Integrated Gradients** then explains *why* — showing which "
        "question tokens and context tokens most influenced where the answer starts and ends."
    )

    st.markdown("---")

    # ---- Preset buttons ----
    st.markdown("**⚡ Quick Presets** — load a sample context + question:")
    preset_cols = st.columns(len(PRESETS))
    for col, preset in zip(preset_cols, PRESETS):
        with col:
            if st.button(preset["label"], key=f"qa_preset_{preset['label']}", use_container_width=True):
                st.session_state["qa_context"] = preset["context"]
                st.session_state["qa_question"] = preset["question"]

    st.markdown("---")

    # ---- Inputs ----
    col_ctx, col_q = st.columns([3, 2])

    with col_ctx:
        context = st.text_area(
            "📄 Context Paragraph",
            value=st.session_state.get("qa_context", PRESETS[0]["context"]),
            height=180,
            key="qa_context_input",
            help="Paste any paragraph of text. The model will extract the answer FROM this context.",
            placeholder="Paste a paragraph of text here...",
        )

    with col_q:
        question = st.text_input(
            "❓ Your Question",
            value=st.session_state.get("qa_question", PRESETS[0]["question"]),
            key="qa_question_input",
            help="Ask anything that can be answered from the context above.",
            placeholder="e.g. When was it built?",
        )
        st.markdown("<br>", unsafe_allow_html=True)
        st.info(
            "💡 **Tip:** The model only answers from the context. "
            "If your question can't be answered from the paragraph, the answer may be inaccurate."
        )

    analyze_btn = st.button(
        "🔍 Get Answer + Explain",
        type="primary",
        use_container_width=True,
        key="qa_analyze_btn",
    )

    if analyze_btn:
        if not context.strip():
            st.warning("⚠️ Please enter a context paragraph.")
            return
        if not question.strip():
            st.warning("⚠️ Please enter a question.")
            return

        with st.spinner("⏳ Loading model & computing answer + attributions — first run ~30s…"):
            try:
                tokenizer, model = load_qa_model()
                (
                    answer, tokens, start_attrs, end_attrs,
                    ans_start, ans_end, question_len, confidence
                ) = run_qa_with_attribution(question, context, tokenizer, model)
            except Exception as exc:
                st.error(f"❌ Error: {exc}")
                st.exception(exc)
                return

        # ---- Answer card ----
        st.markdown("---")
        ans_color = "#10b981" if confidence > 0.5 else "#f59e0b"
        st.markdown(
            f"""
            <div style="background:#1a1d27; border-radius:12px; padding:20px 24px;
                        border-left:5px solid {ans_color}; margin-bottom:18px;">
                <div style="font-size:13px; color:#888; margin-bottom:6px;">
                    Extracted Answer
                </div>
                <div style="font-size:28px; font-weight:900; color:{ans_color};">
                    "{answer}"
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # ---- Metric cards ----
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Answer Confidence", f"{confidence:.1%}")
        m2.metric("Answer Length", f"{len(answer.split())} words")
        m3.metric("Context Tokens", f"{len(tokens) - question_len}")
        m4.metric("Question Tokens", f"{question_len}")

        st.markdown("---")

        # ---- Dual attribution tabs ----
        attr_tab1, attr_tab2 = st.tabs([
            "🎯 Answer Start Attribution",
            "🏁 Answer End Attribution",
        ])

        with attr_tab1:
            st.markdown(
                "Which tokens influenced **where the answer begins**? "
                "Tokens with high positive attribution (🟢) pushed the model to start the answer here."
            )
            st.markdown(
                render_attributed_context(
                    tokens, start_attrs, ans_start, ans_end, question_len,
                    label="start token attribution"
                ),
                unsafe_allow_html=True,
            )
            st.markdown("<br>", unsafe_allow_html=True)
            fig1 = plot_top_tokens(tokens, start_attrs, question_len, "Top Tokens → Answer START")
            st.pyplot(fig1)
            plt.close(fig1)

        with attr_tab2:
            st.markdown(
                "Which tokens influenced **where the answer ends**? "
                "Tokens with high positive attribution (🟢) pushed the model to end the answer here."
            )
            st.markdown(
                render_attributed_context(
                    tokens, end_attrs, ans_start, ans_end, question_len,
                    label="end token attribution"
                ),
                unsafe_allow_html=True,
            )
            st.markdown("<br>", unsafe_allow_html=True)
            fig2 = plot_top_tokens(tokens, end_attrs, question_len, "Top Tokens → Answer END")
            st.pyplot(fig2)
            plt.close(fig2)

        # ---- Interpretation summary ----
        st.markdown("---")
        st.info(
            f"**How to read this:** The model read your question *\"{question}\"* "
            f"and scanned the context to find the answer span *\"{answer}\"* "
            f"(marked with ⭐). "
            "Green tokens = pushed the model toward this answer location. "
            "Red tokens = pulled the model away from this location. "
            "**[Q]** labels = question tokens, **[C]** labels = context tokens."
        )
