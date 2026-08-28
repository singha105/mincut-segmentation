"""Streamlit front end for interactive min-cut segmentation."""

import numpy as np
import streamlit as st
from PIL import Image
from streamlit_drawable_canvas import st_canvas

FOREGROUND_COLOR = "#ff0000"
BACKGROUND_COLOR = "#0000ff"

# Edmonds-Karp is O(V * E^2), so segmentation runs on a small downscaled copy
# of the image (see graph_builder.py / test_segmentation.py for the same
# constraint). The canvas is shown much larger purely so scribbling is usable.
PROCESSING_MAX_DIM = 50
DISPLAY_MAX_DIM = 400


def _resize_to_max_dim(image: Image.Image, max_dim: int) -> Image.Image:
    """Resize `image` so its longer side is `max_dim`, preserving aspect ratio."""
    width, height = image.size
    scale = max_dim / max(width, height)
    new_size = (max(1, round(width * scale)), max(1, round(height * scale)))
    return image.resize(new_size)


st.title("Mincut Segmentation")
st.write(
    "Upload an image, scribble red for foreground / blue for background, "
    "then click Segment."
)

uploaded_file = st.file_uploader("Upload an image", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    original_image = Image.open(uploaded_file).convert("RGB")
    processing_image = _resize_to_max_dim(original_image, PROCESSING_MAX_DIM)
    display_image = _resize_to_max_dim(processing_image, DISPLAY_MAX_DIM)

    st.caption(
        f"Processing at {processing_image.width}x{processing_image.height} "
        "for fast interactive segmentation."
    )

    scribble_mode = st.radio(
        "Scribble type", ["Foreground (red)", "Background (blue)"], horizontal=True
    )
    brush_size = st.slider("Brush size", min_value=1, max_value=25, value=6)
    stroke_color = FOREGROUND_COLOR if scribble_mode.startswith("Foreground") else BACKGROUND_COLOR

    canvas_result = st_canvas(
        stroke_width=brush_size,
        stroke_color=stroke_color,
        background_image=display_image,
        height=display_image.height,
        width=display_image.width,
        drawing_mode="freedraw",
        key="scribble_canvas",
    )
