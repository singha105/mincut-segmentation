"""Smoke tests for app.py.

The UI has no unit coverage of its own, so a broken import or a Streamlit API
change would only show up by launching the app by hand. These tests import the
module and drive its pure helpers, which is enough to catch the common breakages
(bad import, renamed helper, wrong compositing) without needing a browser.

The end-to-end browser check lives in scripts/capture_screenshots.py.
"""

import io

import numpy as np
import pytest
from PIL import Image


def test_app_imports_without_a_streamlit_runtime():
    """Importing app.py runs its top-level Streamlit calls; they must not raise."""
    import app

    assert app.DISPLAY_MAX_DIM > 0


def test_display_resize_preserves_aspect_ratio():
    import app

    image = Image.new("RGB", (1600, 800))
    resized = app._resize_to_max_dim(image, app.DISPLAY_MAX_DIM)

    assert max(resized.size) == app.DISPLAY_MAX_DIM
    assert resized.width == 2 * resized.height


def test_display_resize_handles_portrait_images():
    import app

    resized = app._resize_to_max_dim(Image.new("RGB", (400, 1200)), 600)

    assert resized.size == (200, 600)


def test_small_images_are_scaled_up_to_the_cap():
    """Small uploads are enlarged, deliberately.

    This sizes the drawing canvas, not the output: a 50px image would be
    unusable to scribble on. Segmentation still runs on the original pixels.
    """
    import app

    resized = app._resize_to_max_dim(Image.new("RGB", (50, 40)), 100)

    assert resized.size == (100, 80)


def test_streamlit_compat_shim_is_installed_when_needed():
    """streamlit-drawable-canvas calls a private Streamlit helper.

    Newer Streamlit versions moved it; app.py shims it back. Whichever the
    installed version, the attribute has to exist by the time the canvas runs.
    """
    import streamlit.elements.image as st_image

    import app  # noqa: F401  (import installs the shim)

    assert hasattr(st_image, "image_to_url")


@pytest.mark.parametrize(
    "background, expected_mode", [("black", "RGB"), ("dim", "RGB"), ("transparent", "RGBA")]
)
def test_apply_mask_backgrounds_used_by_the_app(background, expected_mode):
    """The three options the app's radio offers must all round-trip to a PNG."""
    from segmentation.segmenter import apply_mask

    image = np.full((8, 8, 3), 200, dtype=np.uint8)
    mask = np.zeros((8, 8), dtype=bool)
    mask[2:6, 2:6] = True

    result = apply_mask(image, mask, background=background)

    buffer = io.BytesIO()
    Image.fromarray(result).save(buffer, format="PNG")
    buffer.seek(0)
    assert Image.open(buffer).mode == expected_mode
