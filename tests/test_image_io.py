"""Loading images, including HEIC/HEIF from Apple devices.

Pillow cannot decode HEIC on its own, so uploads straight off an iPhone used to
be rejected and had to be converted first. image_io registers the pillow-heif
decoder and applies EXIF rotation, which phone cameras rely on.
"""

import io

import numpy as np
import pytest
from PIL import Image

from image_io import UPLOAD_TYPES, heif_available, looks_like_heif, open_image

heif_only = pytest.mark.skipif(
    not heif_available(), reason="pillow-heif not installed"
)


def _png(size=(30, 20), colour=(90, 160, 220)):
    buffer = io.BytesIO()
    Image.new("RGB", size, colour).save(buffer, format="PNG")
    buffer.seek(0)
    return buffer


def test_heic_is_an_accepted_upload_type():
    assert "heic" in UPLOAD_TYPES
    assert "heif" in UPLOAD_TYPES


@pytest.mark.parametrize(
    "name, expected",
    [
        ("IMG_0001.HEIC", True),
        ("photo.heic", True),
        ("photo.heif", True),
        ("photo.hif", True),
        ("photo.jpg", False),
        ("photo.png", False),
        ("", False),
        (None, False),
    ],
)
def test_looks_like_heif(name, expected):
    assert looks_like_heif(name) is expected


def test_opens_an_ordinary_png():
    image = open_image(_png())

    assert image.size == (30, 20)
    assert image.mode == "RGB"


def test_grayscale_is_converted_to_rgb():
    buffer = io.BytesIO()
    Image.new("L", (10, 10), 128).save(buffer, format="PNG")
    buffer.seek(0)

    assert open_image(buffer).mode == "RGB"


def test_unreadable_input_raises_value_error():
    with pytest.raises(ValueError, match="Couldn't read that file as an image"):
        open_image(io.BytesIO(b"definitely not an image"))


@heif_only
def test_opens_a_heic_file(tmp_path):
    """The whole point: a .heic opens directly, with no conversion step."""
    source = Image.new("RGB", (64, 48), (200, 120, 60))
    path = tmp_path / "IMG_0001.heic"
    source.save(path, format="HEIF", quality=80)

    image = open_image(path)

    assert image.size == (64, 48)
    assert image.mode == "RGB"
    # HEIF is lossy, so compare approximately rather than exactly.
    assert abs(np.asarray(image).mean() - np.asarray(source).mean()) < 12


@heif_only
def test_heic_from_a_file_object(tmp_path):
    """Streamlit hands over an in-memory buffer, not a path."""
    path = tmp_path / "photo.heic"
    Image.new("RGB", (40, 30), (30, 200, 90)).save(path, format="HEIF", quality=80)

    buffer = io.BytesIO(path.read_bytes())
    buffer.name = "photo.heic"

    assert open_image(buffer, name="photo.heic").size == (40, 30)


@heif_only
def test_heic_survives_the_segmentation_pipeline(tmp_path):
    """A HEIC upload should segment like any other image."""
    from maxflow.graph_builder import BACKGROUND, FOREGROUND, UNMARKED
    from segmentation.segmenter import segment_image

    array = np.zeros((60, 80, 3), dtype=np.uint8)
    array[:, :40] = 230
    array[:, 40:] = 20
    path = tmp_path / "halves.heic"
    Image.fromarray(array).save(path, format="HEIF", quality=90)

    image = np.asarray(open_image(path))
    scribbles = np.full(image.shape[:2], UNMARKED, dtype=np.uint8)
    scribbles[25:35, 5:15] = FOREGROUND
    scribbles[25:35, 65:75] = BACKGROUND

    mask = segment_image(image, scribbles)

    assert mask.shape == image.shape[:2]
    assert mask[:, :40].mean() > 0.9, "bright half should be foreground"
    assert mask[:, 40:].mean() < 0.1, "dark half should be background"


def test_exif_rotation_is_applied(tmp_path):
    """Phone cameras store rotation in EXIF rather than rotating the pixels.

    Without honouring that, a photo taken sideways would be segmented sideways.
    """
    tall = Image.new("RGB", (40, 20), (10, 10, 10))
    exif = tall.getexif()
    exif[274] = 6  # orientation: rotate 90 degrees
    path = tmp_path / "rotated.jpg"
    tall.save(path, exif=exif)

    # 40x20 stored, tagged to display rotated, so it should come back 20x40.
    assert open_image(path).size == (20, 40)


def test_missing_heif_support_names_the_fix(monkeypatch, tmp_path):
    """Without pillow-heif, the error should say what to install."""
    import image_io

    monkeypatch.setattr(image_io, "register_heif", lambda: False)
    monkeypatch.setattr(image_io, "heif_available", lambda: False)

    path = tmp_path / "photo.heic"
    path.write_bytes(b"not really heic")

    with pytest.raises(ValueError, match="pillow-heif"):
        image_io.open_image(path)
