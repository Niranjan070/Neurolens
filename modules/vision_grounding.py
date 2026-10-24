"""
Tab 2: Image Interpretability — Vision-Language Grounding
Model  : openai/clip-vit-base-patch32
Method : Patch-level cosine similarity between visual token embeddings
         and the text query embedding → 7×7 grid → bilinear upsample
Output : Original image + heatmap overlay side-by-side + raw patch grid
"""

import os
import streamlit as st
import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from transformers import CLIPProcessor, CLIPModel
from scipy.ndimage import zoom

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
MODEL_NAME = "openai/clip-vit-base-patch32"

# CLIP ViT-B/32 produces 7×7 = 49 patch tokens + 1 CLS token
PATCH_GRID = 7

SAMPLE_QUERIES = [
    "A dog sitting on grass",
    "A red sports car",
    "A cozy living room with a sofa",
    "The sky and trees in the background",
    "The wheels of a car",
    "Books on a shelf",
]

SAMPLE_FILES = {
    "🐕 Dog on Grass": "dog_grass.jpg",
    "🚗 Red Sports Car": "red_car.jpg",
    "🛋️ Living Room": "living_room.jpg",
}


# ---------------------------------------------------------------------------
# Model loading — cached per session
# ---------------------------------------------------------------------------
@st.cache_resource(show_spinner=False)
def load_clip_model():
    """Load CLIP processor and model from HuggingFace Hub."""
    processor = CLIPProcessor.from_pretrained(MODEL_NAME)
    model = CLIPModel.from_pretrained(MODEL_NAME)
    model.eval()
    return processor, model


# ---------------------------------------------------------------------------
# Core computation
# ---------------------------------------------------------------------------
def get_patch_heatmap(image: Image.Image, text_query: str, processor, model):
    """
    Extract CLIP patch-level cosine similarities with the text embedding.

    CLIP ViT-B/32 splits a 224×224 image into 32×32 patches → 7×7 = 49 tokens.
    We hook the last transformer encoder layer to get the raw patch features,
    project them through the visual projection head, and compute cosine similarity
    against the text embedding.

    Parameters
    ----------
    image      : PIL Image (RGB), any size — resized to 224×224 internally
    text_query : str

    Returns
    -------
    heatmap      : np.ndarray, shape (H, W), values in [0, 1]
    sim_scores   : np.ndarray, shape (49,)  — raw similarity per patch
    image_resized: PIL Image, 224×224
    """
    image_resized = image.convert("RGB").resize((224, 224))
    inputs = processor(
        text=[text_query],
        images=image_resized,
        return_tensors="pt",
        padding=True,
    )

    patch_features = {}

    def _hook(module, inp, out):
        # out: (batch, seq_len, hidden_dim)  — seq_len = 50 (CLS + 49 patches)
        patch_features["last_hidden"] = out.detach()

    hook = model.vision_model.encoder.layers[-1].register_forward_hook(_hook)

    with torch.no_grad():
        outputs = model(**inputs)

    hook.remove()

    # Text embedding — normalized (detach to allow numpy conversion later)
    text_embed = outputs.text_embeds.detach()  # (1, 512)
    text_embed = text_embed / text_embed.norm(dim=-1, keepdim=True)

    # Patch tokens — skip CLS at index 0 → (49, hidden_dim)
    raw_patches = patch_features["last_hidden"][0, 1:, :]

    # Project through visual projection head → (49, 512)
    patches_proj = model.visual_projection(raw_patches).detach()
    patches_proj = patches_proj / patches_proj.norm(dim=-1, keepdim=True)

    # Cosine similarity per patch → (49,)
    sim_scores = (patches_proj @ text_embed.T).squeeze(-1).detach().numpy()

    # Reshape to 7×7 grid
    grid = sim_scores.reshape(PATCH_GRID, PATCH_GRID)

    # Bilinear upsample to original image dimensions
    orig_w, orig_h = image.size
    zoom_factors = (orig_h / PATCH_GRID, orig_w / PATCH_GRID)
    heatmap = zoom(grid, zoom_factors, order=1)

    # Normalize to [0, 1]
    heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min() + 1e-8)

    return heatmap, sim_scores, image_resized


# ---------------------------------------------------------------------------
# Visualization helpers
# ---------------------------------------------------------------------------
def create_overlay_figure(original: Image.Image, heatmap: np.ndarray, query: str) -> plt.Figure:
    """Side-by-side: original image and jet colormap heatmap overlay."""
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.patch.set_facecolor("#0e1117")

    # Left: original
    axes[0].imshow(original)
    axes[0].set_title("Original Image", color="white", fontsize=13, pad=10)
    axes[0].axis("off")

    # Right: overlay
    axes[1].imshow(original, alpha=0.55)
    im = axes[1].imshow(
        heatmap,
        cmap="jet",
        alpha=0.50,
        interpolation="bilinear",
        vmin=0,
        vmax=1,
        extent=[0, original.width, original.height, 0],
    )
    axes[1].set_title(f'Attention: "{query}"', color="white", fontsize=12, pad=10)
    axes[1].axis("off")

    # Colorbar
    cbar = fig.colorbar(im, ax=axes[1], fraction=0.046, pad=0.04)
    cbar.set_label("Patch Similarity", color="white", fontsize=10)
    cbar.ax.yaxis.set_tick_params(color="white")
    plt.setp(cbar.ax.yaxis.get_ticklabels(), color="white")

    plt.tight_layout(pad=1.5)
    return fig


def create_patch_grid_figure(sim_scores: np.ndarray) -> plt.Figure:
    """7×7 raw cosine similarity grid with annotated values."""
    grid = sim_scores.reshape(PATCH_GRID, PATCH_GRID)

    fig, ax = plt.subplots(figsize=(5, 4.5))
    fig.patch.set_facecolor("#0e1117")
    ax.set_facecolor("#1a1d27")

    im = ax.imshow(grid, cmap="RdYlGn", aspect="auto", vmin=grid.min(), vmax=grid.max())
    for i in range(PATCH_GRID):
        for j in range(PATCH_GRID):
            ax.text(
                j, i, f"{grid[i, j]:.2f}",
                ha="center", va="center",
                fontsize=7, color="black", fontweight="bold",
            )

    ax.set_title(f"7×7 Patch Cosine Similarity Grid", color="white", fontsize=11)
    ax.set_xticks(range(PATCH_GRID))
    ax.set_yticks(range(PATCH_GRID))
    ax.tick_params(colors="white", labelsize=8)
    for spine in ax.spines.values():
        spine.set_visible(False)

    plt.colorbar(im, ax=ax, label="Similarity")
    plt.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Streamlit tab renderer
# ---------------------------------------------------------------------------
def render_tab(sample_images_dir: str):
    st.markdown("### 🖼️ Vision-Language Grounding with CLIP")
    st.markdown(
        "**CLIP ViT-B/32** splits the image into a **7×7 grid of patches**. "
        "Each patch embedding is compared against the text query via cosine similarity. "
        "High-similarity patches are highlighted — showing *where* the model looks for the concept."
    )

    st.markdown("---")

    col_img, col_query = st.columns([1, 1])

    # ---- Image selection ----
    with col_img:
        st.markdown("**Step 1 — Choose an Image**")

        # Sample image buttons
        st.markdown("<small style='color:#888'>Use a preloaded sample:</small>", unsafe_allow_html=True)
        sample_cols = st.columns(3)
        for idx, (display_name, filename) in enumerate(SAMPLE_FILES.items()):
            fpath = os.path.join(sample_images_dir, filename)
            with sample_cols[idx]:
                if os.path.exists(fpath):
                    thumb = Image.open(fpath).resize((90, 65))
                    st.image(thumb, use_container_width=True)
                if st.button(display_name.split()[0], key=f"vis_sample_{idx}", use_container_width=True):
                    st.session_state["vis_selected_image"] = fpath
                    # auto-set a sensible query
                    default_queries = [
                        "A dog sitting on grass",
                        "A red sports car on the road",
                        "A cozy living room with a sofa",
                    ]
                    st.session_state["vis_query_preset"] = default_queries[idx]

        st.markdown("---")
        st.markdown("<small style='color:#888'>Or upload your own:</small>", unsafe_allow_html=True)
        uploaded = st.file_uploader("Upload image", type=["jpg", "jpeg", "png"], key="vis_upload")
        if uploaded:
            st.session_state["vis_selected_image"] = uploaded
            st.session_state["vis_uploaded"] = True
        else:
            st.session_state["vis_uploaded"] = False

    # ---- Query input ----
    with col_query:
        st.markdown("**Step 2 — Enter a Text Query**")
        default_q = st.session_state.get("vis_query_preset", SAMPLE_QUERIES[0])
        selected_query = st.selectbox(
            "Select a preset query:",
            SAMPLE_QUERIES,
            index=SAMPLE_QUERIES.index(default_q) if default_q in SAMPLE_QUERIES else 0,
            key="vis_query_select",
        )
        custom_query = st.text_input(
            "Or type your own query:",
            placeholder="e.g., the eyes of the dog",
            key="vis_custom_query",
        )
        final_query = custom_query.strip() if custom_query.strip() else selected_query
        st.info(f"🔍 **Active query:** {final_query}")
        st.markdown(
            "<small style='color:#888'>The model will highlight image regions most similar to this text.</small>",
            unsafe_allow_html=True,
        )

    # ---- Resolve image ----
    image: Image.Image | None = None
    img_source_label = ""

    if st.session_state.get("vis_uploaded") and uploaded:
        try:
            image = Image.open(uploaded).convert("RGB")
            img_source_label = "Uploaded image"
        except Exception:
            st.warning("⚠️ Could not read uploaded image.")
    elif st.session_state.get("vis_selected_image"):
        src = st.session_state["vis_selected_image"]
        if isinstance(src, str) and os.path.exists(src):
            image = Image.open(src).convert("RGB")
            img_source_label = os.path.basename(src)
    else:
        # Auto-load default
        default_path = os.path.join(sample_images_dir, "dog_grass.jpg")
        if os.path.exists(default_path):
            image = Image.open(default_path).convert("RGB")
            img_source_label = "dog_grass.jpg (default)"

    # Show thumbnail of selected image
    if image:
        st.markdown(f"**Selected:** `{img_source_label}`")
        st.image(image.resize((220, 160)), width=220)

    st.markdown("---")
    run_btn = st.button("🔍 Run Visual Grounding", type="primary", use_container_width=True, key="vis_run_btn")

    if run_btn:
        if image is None:
            st.warning("⚠️ Please select or upload an image first.")
            return

        with st.spinner("⏳ Loading CLIP & computing patch similarities — first run may take ~30s…"):
            try:
                processor, model = load_clip_model()
                heatmap, sim_scores, _ = get_patch_heatmap(image, final_query, processor, model)
            except Exception as exc:
                st.error(f"❌ Error during visual grounding: {exc}")
                st.exception(exc)
                return

        # ---- Metric cards ----
        m1, m2, m3 = st.columns(3)
        m1.metric("Max Patch Similarity", f"{sim_scores.max():.3f}")
        m2.metric("Mean Patch Similarity", f"{sim_scores.mean():.3f}")
        high_attn = int((sim_scores > sim_scores.mean()).sum())
        m3.metric("High-Attention Patches", f"{high_attn} / 49")

        # ---- Heatmap overlay ----
        st.markdown("#### Spatial Attention Heatmap")
        fig = create_overlay_figure(image, heatmap, final_query)
        st.pyplot(fig)
        plt.close(fig)

        # ---- Raw grid ----
        st.markdown("---")
        with st.expander("📐 View Raw 7×7 Patch Similarity Grid"):
            fig2 = create_patch_grid_figure(sim_scores)
            st.pyplot(fig2)
            plt.close(fig2)
            st.caption(
                "Each cell shows the cosine similarity between that patch's embedding "
                "and the text query embedding. Warmer colors = higher relevance."
            )

        st.info(
            f"**Interpretation:** The jet-colored heatmap shows which **7×7 image patches** "
            f"CLIP considers most relevant to the query *\"{final_query}\"*. "
            "Red/yellow regions have the highest patch-text cosine similarity."
        )
