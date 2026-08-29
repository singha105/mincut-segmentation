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
from segmentation.segmenter import DEFAULT_PROCESSING_MAX_DIM, apply_mask, segment_image

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

# The canvas size is a UI choice only -- scribbles are mapped back up to the
# image's own resolution, and segmentation.segmenter handles the downscaling
# the solver needs internally. The uploaded image is never downscaled for
# output, so results keep the resolution you gave them.
DISPLAY_MAX_DIM = 520


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
    display_image = _resize_to_max_dim(original_image, DISPLAY_MAX_DIM)

    st.caption(
        f"Image is {original_image.width}x{original_image.height}. The min-cut is "
        f"solved on a {DEFAULT_PROCESSING_MAX_DIM}px grid for speed, then the mask is "
        "refined back to full resolution — your result is not downscaled."
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

    background_style = st.radio(
        "Background in the result",
        ["Black", "Dimmed", "Transparent (PNG)"],
        horizontal=True,
    )

    if st.button("Segment", disabled=not has_scribbles):
        # Scribbles are drawn on the display canvas but rasterised straight
        # into the image's own resolution, so no precision is lost on the way in.
        scribble_mask = rasterize_scribbles(
            canvas_result.json_data["objects"],
            mask_shape=(original_image.height, original_image.width),
            scale_x=original_image.width / display_image.width,
            scale_y=original_image.height / display_image.height,
        )
        image_array = np.array(original_image)

        try:
            start_time = time.perf_counter()
            mask = segment_image(image_array, scribble_mask)
            elapsed_seconds = time.perf_counter() - start_time
        except ValueError as error:
            st.error(str(error))
        else:
            # The canvas hands back only the strokes on a transparent layer;
            # composite them over the image so the preview shows what was marked.
            strokes = Image.fromarray(
                canvas_result.image_data.astype(np.uint8), mode="RGBA"
            ).resize(original_image.size)
            preview = original_image.convert("RGBA")
            preview.alpha_composite(strokes)

            st.session_state["result"] = {
                "mask": mask,
                "elapsed_seconds": elapsed_seconds,
                "scribble_preview": preview.convert("RGB"),
            }

    if "result" in st.session_state:
        mask = st.session_state["result"]["mask"]
        elapsed_seconds = st.session_state["result"]["elapsed_seconds"]
        scribble_preview = st.session_state["result"]["scribble_preview"]
        image_array = np.array(original_image)

        # Guard against a stale result from a previously uploaded image.
        if mask.shape == image_array.shape[:2]:
            background = {
                "Black": "black",
                "Dimmed": "dim",
                "Transparent (PNG)": "transparent",
            }[background_style]
            result = apply_mask(image_array, mask, background=background)

            result_col1, result_col2, result_col3 = st.columns(3)
            with result_col1:
                st.image(original_image, caption="Original")
            with result_col2:
                st.image(scribble_preview, caption="Scribbles")
            with result_col3:
                st.image(result, caption="Segmented result")

            st.write(
                f"Segmentation took {elapsed_seconds:.2f} seconds — "
                f"result is {result.shape[1]}x{result.shape[0]}, "
                "the same resolution as the upload."
            )

            result_buffer = io.BytesIO()
            Image.fromarray(result).save(result_buffer, format="PNG")
            st.download_button(
                "Download result",
                data=result_buffer.getvalue(),
                file_name="segmentation_result.png",
                mime="image/png",
            )
