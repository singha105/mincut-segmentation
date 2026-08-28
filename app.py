"""Streamlit front end for interactive min-cut segmentation."""

import io
import time

import numpy as np
import streamlit as st
import streamlit.elements.image as st_image
from PIL import Image
from streamlit import runtime
from streamlit_drawable_canvas import st_canvas

from canvas_utils import BACKGROUND_COLOR, FOREGROUND_COLOR, rasterize_scribbles
from segmentation.segmenter import segment

# streamlit-drawable-canvas 0.9.3 calls the private st_image.image_to_url(),
# which newer Streamlit releases renamed/restructured. The canvas frontend
# prepends the app's own origin to whatever URL this returns, so it must be
# a relative media-manager path (like the original) -- not an absolute URL
# or data URI, which would come out corrupted after that concatenation.
# Only applied if the real function is actually missing, so this is a no-op
# on Streamlit versions where streamlit-drawable-canvas still works natively.
if not hasattr(st_image, "image_to_url"):

    def _image_to_url(image, width, clamp, channels, output_format, image_id):
        buffer = io.BytesIO()
        image.save(buffer, format="PNG")
        return runtime.get_instance().media_file_mgr.add(
            buffer.getvalue(), "image/png", image_id
        )

    st_image.image_to_url = _image_to_url

# Edmonds-Karp is O(V * E^2), so segmentation runs on a small downscaled copy
# of the image (see graph_builder.py / test_segmentation.py for the same
# constraint). The canvas is shown much larger purely so scribbling is usable.
PROCESSING_MAX_DIM = 30
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

    has_scribbles = canvas_result.json_data is not None and canvas_result.json_data["objects"]

    if st.button("Segment", disabled=not has_scribbles):
        scale_x = processing_image.width / display_image.width
        scale_y = processing_image.height / display_image.height
        scribble_mask = rasterize_scribbles(
            canvas_result.json_data["objects"],
            mask_shape=(processing_image.height, processing_image.width),
            scale_x=scale_x,
            scale_y=scale_y,
        )
        image_array = np.array(processing_image)

        try:
            start_time = time.perf_counter()
            mask = segment(image_array, scribble_mask)
            elapsed_seconds = time.perf_counter() - start_time
        except ValueError as error:
            st.error(str(error))
        else:
            st.session_state["result"] = {
                "mask": mask,
                "elapsed_seconds": elapsed_seconds,
                "scribble_preview": canvas_result.image_data,
            }

    if "result" in st.session_state:
        mask = st.session_state["result"]["mask"]
        elapsed_seconds = st.session_state["result"]["elapsed_seconds"]
        scribble_preview = st.session_state["result"]["scribble_preview"]

        mask_image = Image.fromarray((mask * 255).astype(np.uint8))
        mask_display = np.array(
            mask_image.resize(display_image.size, Image.Resampling.NEAREST)
        ) > 127

        cutout = np.array(display_image).copy()
        cutout[~mask_display] = cutout[~mask_display] // 4

        result_col1, result_col2, result_col3 = st.columns(3)
        with result_col1:
            st.image(display_image, caption="Original")
        with result_col2:
            st.image(scribble_preview, caption="Scribble mask")
        with result_col3:
            st.image(cutout, caption="Segmented result")

        st.write(f"Segmentation took {elapsed_seconds:.2f} seconds")

        result_buffer = io.BytesIO()
        Image.fromarray(cutout).save(result_buffer, format="PNG")
        st.download_button(
            "Download result",
            data=result_buffer.getvalue(),
            file_name="segmentation_result.png",
            mime="image/png",
        )
