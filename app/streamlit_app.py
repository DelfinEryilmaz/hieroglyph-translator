"""Streamlit demo: upload a photo of hieroglyphs, see detected signs + glosses.

Run with: streamlit run app/streamlit_app.py
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import streamlit as st
from PIL import Image

from hieroglyph.lookup.gardiner_lookup import GardinerLookup
from hieroglyph.models.classifier import load_checkpoint
from hieroglyph.pipeline.inference import run_inference
from hieroglyph.segmentation.tiling import load_tiled_or_fallback
from hieroglyph.utils.visualization import draw_annotated_image

CHECKPOINT_PATH = Path(__file__).resolve().parent.parent / "models" / "best_model.pt"
YOLO_CHECKPOINT_PATH = Path(__file__).resolve().parent.parent / "models" / "yolo_seg.pt"

st.set_page_config(page_title="Hieroglyph Translator", page_icon="🏺", layout="wide")

# --- Styling -----------------------------------------------------------
# Streamlit's default component styling is plain; this injects a small
# custom look (hero header, card-based results, confidence pills) on top
# of the .streamlit/config.toml color theme, since the default table/list
# rendering isn't enough on its own for a portfolio-quality result.
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&family=Cormorant+Garamond:wght@600&display=swap');

    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

    .hero { text-align: center; padding: 1.5rem 0 0.5rem; }
    .hero h1 {
        font-family: 'Cormorant Garamond', serif;
        font-size: 3rem;
        font-weight: 600;
        color: #2B2620;
        margin-bottom: 0.2rem;
    }
    .hero p { color: #6b6255; font-size: 1.05rem; max-width: 640px; margin: 0 auto; }

    .gloss-banner {
        background: linear-gradient(135deg, #FCF6E8 0%, #F3E6C4 100%);
        border: 1px solid #E9D9A8;
        border-radius: 16px;
        padding: 1.4rem 1.8rem;
        font-size: 1.25rem;
        color: #4a3c1a;
        text-align: center;
        margin: 1rem 0 1.5rem;
    }

    .sign-card {
        background: #F7F1E3;
        border-radius: 12px;
        padding: 0.85rem 1.1rem;
        margin-bottom: 0.55rem;
        display: flex;
        align-items: center;
        justify-content: space-between;
        box-shadow: 0 1px 2px rgba(0,0,0,0.06);
    }
    .sign-card .left { display: flex; align-items: center; gap: 0.9rem; }
    .sign-index {
        background: #B8860B; color: white; font-weight: 700;
        width: 1.8rem; height: 1.8rem; border-radius: 50%;
        display: flex; align-items: center; justify-content: center;
        font-size: 0.85rem; flex-shrink: 0;
    }
    .sign-code { font-weight: 700; color: #2B2620; font-size: 1.0rem; }
    .sign-translit { color: #8a7f6a; font-style: italic; font-size: 0.9rem; }
    .sign-gloss { color: #4a4335; }

    .confidence-pill {
        padding: 0.2rem 0.65rem; border-radius: 999px;
        font-size: 0.78rem; font-weight: 600; flex-shrink: 0;
    }
    .conf-high { background: #DFF5E1; color: #1B6E2C; }
    .conf-mid  { background: #FFF4D6; color: #8A6D00; }
    .conf-low  { background: #FDE2E2; color: #B3261E; }

    .limitation-note {
        background: #FBF8F1; border-left: 3px solid #B8860B;
        padding: 0.6rem 0.9rem; font-size: 0.85rem; color: #5c5344;
        border-radius: 4px; margin-bottom: 0.6rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource(show_spinner="Loading model...")
def load_model_and_lookup():
    model, class_to_idx = load_checkpoint(CHECKPOINT_PATH)
    lookup = GardinerLookup()
    return model, class_to_idx, lookup


@st.cache_resource(show_spinner="Loading segmenter...")
def load_segmenter():
    return load_tiled_or_fallback(YOLO_CHECKPOINT_PATH)


def confidence_class(confidence: float) -> str:
    if confidence >= 0.85:
        return "conf-high"
    if confidence >= 0.5:
        return "conf-mid"
    return "conf-low"


def pil_to_pipeline_image(pil_image: Image.Image) -> np.ndarray:
    """Convert an uploaded PIL image (RGB) to the BGR array our pipeline
    expects -- see utils/visualization.py's docstring on this convention."""
    return cv2.cvtColor(np.array(pil_image.convert("RGB")), cv2.COLOR_RGB2BGR)


# --- Sidebar -------------------------------------------------------------
with st.sidebar:
    st.markdown("### About")
    st.write(
        "Detects individual Egyptian hieroglyphs in a photo and looks up "
        "each sign's transliteration and English gloss from Gardiner's "
        "sign list."
    )
    st.markdown("### Model")
    st.write("ResNet18 fine-tuned on 171 Gardiner sign classes.")
    st.metric("Test accuracy", "97.3%", help="620 held-out test images; ~51 rare classes have no test coverage.")

    st.markdown("### Known limitations")
    st.markdown(
        '<div class="limitation-note">This is <b>not</b> a sentence translator — '
        "each sign gets its own literal gloss, not fluent English.</div>",
        unsafe_allow_html=True,
    )
    if YOLO_CHECKPOINT_PATH.exists():
        st.markdown(
            '<div class="limitation-note">Segmentation uses a YOLOv8 '
            "detector fine-tuned on synthesized composite images — see "
            "notebooks/05_evaluate_segmenter.ipynb for its held-out test "
            "accuracy.</div>",
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            '<div class="limitation-note">Segmentation uses classical image '
            "processing, not a trained detector — works best on clean, "
            "high-contrast photos. Run notebooks/04_train_segmenter.ipynb "
            "to train a detector and unlock this upgrade.</div>",
            unsafe_allow_html=True,
        )
    st.markdown(
        '<div class="limitation-note">Reading order is a simple top-to-bottom, '
        "left-to-right guess — true Egyptian reading order depends on which "
        "way signs face.</div>",
        unsafe_allow_html=True,
    )
    st.markdown("[View on GitHub](https://github.com/DelfinEryilmaz/hieroglyph-translator)")


# --- Header ----------------------------------------------------------------
st.markdown(
    """
    <div class="hero">
        <h1>𓂀 Hieroglyph Translator</h1>
        <p>Upload a photo of Egyptian hieroglyphs to detect each sign and see
        its transliteration and English gloss.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

if not CHECKPOINT_PATH.exists():
    st.error(
        f"No trained model found at `{CHECKPOINT_PATH}`. Train it first via "
        "`notebooks/02_train_classifier.ipynb` on Colab, then download "
        "`best_model.pt` into the `models/` folder."
    )
    st.stop()

model, class_to_idx, lookup = load_model_and_lookup()
segmenter = load_segmenter()

uploaded_file = st.file_uploader("Upload a photo", type=["png", "jpg", "jpeg"])

if uploaded_file is None:
    st.info("Upload an image to get started — a clear, well-lit photo works best.")
    st.stop()

pil_image = Image.open(uploaded_file)
image_bgr = pil_to_pipeline_image(pil_image)

with st.spinner("Detecting and classifying signs..."):
    result = run_inference(image_bgr, model, class_to_idx, lookup, segmenter=segmenter)

if not result.signs:
    st.warning(
        "No glyphs detected. Try a photo with higher contrast between the "
        "signs and their background, or a closer crop."
    )
    st.image(pil_image, caption="Uploaded photo", use_container_width=True)
    st.stop()

annotated_rgb = draw_annotated_image(image_bgr, result.signs)

col1, col2 = st.columns([3, 2])
with col1:
    st.image(annotated_rgb, caption=f"{len(result.signs)} sign(s) detected, numbered in reading order", use_container_width=True)
with col2:
    st.markdown(f'<div class="gloss-banner">{result.concatenated_gloss}</div>', unsafe_allow_html=True)

    for i, sign in enumerate(result.signs, start=1):
        translit_html = f'<span class="sign-translit">{sign.transliteration}</span>' if sign.transliteration else ""
        gloss_text = sign.gloss or "unknown sign"
        st.markdown(
            f"""
            <div class="sign-card">
                <div class="left">
                    <div class="sign-index">{i}</div>
                    <div>
                        <span class="sign-code">{sign.gardiner_code}</span> {translit_html}<br/>
                        <span class="sign-gloss">{gloss_text}</span>
                    </div>
                </div>
                <div class="confidence-pill {confidence_class(sign.confidence)}">{sign.confidence:.0%}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
