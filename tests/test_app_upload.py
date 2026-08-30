"""The uploader must reject unusable files instead of crashing.

Streamlit's file picker filters by extension, which says nothing about the
bytes behind it: a truncated download or a renamed non-image still arrives.
Before this was handled, those surfaced as a raw traceback in the UI.
"""

import io

import pytest
from PIL import Image

from app import load_uploaded_image


def _png_bytes(size=(24, 18), colour=(120, 200, 90)):
    buffer = io.BytesIO()
    Image.new("RGB", size, colour).save(buffer, format="PNG")
    buffer.seek(0)
    return buffer


def test_loads_a_valid_image():
    image, error = load_uploaded_image(_png_bytes())

    assert error is None
    assert image is not None
    assert image.size == (24, 18)
    assert image.mode == "RGB"


def test_converts_non_rgb_images():
    """Palette and grayscale uploads must come back as RGB for the pipeline."""
    buffer = io.BytesIO()
    Image.new("L", (10, 10), 128).save(buffer, format="PNG")
    buffer.seek(0)

    image, error = load_uploaded_image(buffer)

    assert error is None
    assert image.mode == "RGB"


def test_rejects_a_file_that_is_not_an_image():
    image, error = load_uploaded_image(io.BytesIO(b"this is plainly not a PNG"))

    assert image is None
    assert "Couldn't read that file as an image" in error


def test_rejects_a_truncated_image():
    """Pillow decodes lazily, so a truncated file often fails on pixel access."""
    data = _png_bytes(size=(200, 200)).getvalue()
    truncated = io.BytesIO(data[: len(data) // 2])

    image, error = load_uploaded_image(truncated)

    assert image is None
    assert "Couldn't read that file as an image" in error


def test_rejects_an_empty_file():
    image, error = load_uploaded_image(io.BytesIO(b""))

    assert image is None
    assert error


@pytest.mark.parametrize("payload", [b"\x00" * 64, b"GIF89a corrupted tail"])
def test_rejects_assorted_junk_without_raising(payload):
    image, error = load_uploaded_image(io.BytesIO(payload))

    assert image is None
    assert error
