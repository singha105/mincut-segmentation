"""Tests for the end-to-end segmentation pipeline."""

import numpy as np
from PIL import Image

from maxflow.graph_builder import BACKGROUND, FOREGROUND, UNMARKED
from segmentation.segmenter import apply_mask, cutout, segment, segment_image

SAMPLE_IMAGE_PATH = "sample_images/circle_fg_bg.png"
# Edmonds-Karp is O(V * E^2); a full 200x200 pixel graph is far too large for
# a naive per-pixel graph to solve quickly, so downscale for the test. This
# mirrors the downsampling called out as a future need in graph_builder.py's
# original design notes.
DOWNSCALED_SIZE = 20


def test_segment_separates_two_flat_regions():
    """Left half is dark, right half is bright; one scribble pixel marks each."""
    image = np.zeros((4, 4, 3), dtype=np.uint8)
    image[:, :2] = 20
    image[:, 2:] = 220

    scribble_mask = np.full((4, 4), UNMARKED, dtype=np.uint8)
    scribble_mask[1, 0] = BACKGROUND
    scribble_mask[1, 3] = FOREGROUND

    mask = segment(image, scribble_mask)

    assert mask.shape == (4, 4)
    assert mask[:, 2:].all(), "bright half should be foreground"
    assert not mask[:, :2].any(), "dark half should be background"


def test_segment_sample_image_produces_non_trivial_mask():
    """End-to-end run on a real sample image with a synthetic scribble mask.

    circle_fg_bg.png is a light circle (roughly rows/cols 5-15 after
    downscaling to 20x20) on a dark background. The scribbles mark a small
    patch near the circle's center as foreground and a small patch in the
    far corner (clearly outside the circle) as background.
    """
    image = Image.open(SAMPLE_IMAGE_PATH).convert("RGB")
    image = np.array(image.resize((DOWNSCALED_SIZE, DOWNSCALED_SIZE)))
    height, width = image.shape[:2]

    scribble_mask = np.full((height, width), UNMARKED, dtype=np.uint8)
    scribble_mask[8:12, 8:12] = FOREGROUND
    scribble_mask[0:3, 0:3] = BACKGROUND

    mask = segment(image, scribble_mask)

    assert mask.shape == (height, width)
    assert mask.any(), "segmentation produced an all-background mask"
    assert not mask.all(), "segmentation produced an all-foreground mask"

    # Ground truth we know from how the sample was drawn: the scribbled
    # foreground patch stays foreground, and the far corner stays background.
    assert mask[8:12, 8:12].all()
    assert not mask[0:3, 0:3].any()


def _full_res_circle_case():
    """The 200x200 sample at full size, with scribbles on the circle and a corner."""
    image = np.array(Image.open(SAMPLE_IMAGE_PATH).convert("RGB"))
    height, width = image.shape[:2]

    scribbles = np.full((height, width), UNMARKED, dtype=np.uint8)
    scribbles[95:105, 95:105] = FOREGROUND  # centre of the circle
    scribbles[0:12, 0:12] = BACKGROUND  # far corner
    return image, scribbles


def test_segment_image_returns_a_full_resolution_mask():
    """The mask must match the input image, not the solver's internal grid."""
    image, scribbles = _full_res_circle_case()

    mask = segment_image(image, scribbles)

    assert mask.shape == image.shape[:2] == (200, 200)
    assert mask.dtype == bool
    assert mask.any() and not mask.all()


def test_segment_image_recovers_the_circle():
    """Full-res output should track the drawn circle far more tightly than a coarse grid."""
    image, scribbles = _full_res_circle_case()

    mask = segment_image(image, scribbles)

    # The sample is a light circle of radius ~50 centred at (100, 100).
    yy, xx = np.ogrid[:200, :200]
    distance = np.sqrt((xx - 100) ** 2 + (yy - 100) ** 2)
    inside = distance < 40
    outside = distance > 60

    assert mask[inside].mean() > 0.95, "circle interior should be foreground"
    assert mask[outside].mean() < 0.05, "area well outside the circle should be background"


def test_segment_image_honours_scribbles_at_full_resolution():
    """Hard constraints must survive the upsample-and-refine step."""
    image, scribbles = _full_res_circle_case()

    mask = segment_image(image, scribbles)

    assert mask[scribbles == FOREGROUND].all()
    assert not mask[scribbles == BACKGROUND].any()


def test_apply_mask_preserves_foreground_pixels_exactly():
    """A cutout must not degrade the pixels it keeps."""
    image = np.arange(4 * 4 * 3, dtype=np.uint8).reshape(4, 4, 3)
    mask = np.zeros((4, 4), dtype=bool)
    mask[1:3, 1:3] = True

    blacked = apply_mask(image, mask, background="black")
    assert np.array_equal(blacked[mask], image[mask])
    assert not blacked[~mask].any()

    dimmed = apply_mask(image, mask, background="dim")
    assert np.array_equal(dimmed[mask], image[mask])

    rgba = apply_mask(image, mask, background="transparent")
    assert rgba.shape == (4, 4, 4)
    assert np.array_equal(rgba[..., :3][mask], image[mask])
    assert (rgba[..., 3][mask] == 255).all()
    assert (rgba[..., 3][~mask] == 0).all()


def test_cutout_returns_full_resolution_image_and_mask():
    image, scribbles = _full_res_circle_case()

    result, mask = cutout(image, scribbles)

    assert result.shape == image.shape
    assert mask.shape == image.shape[:2]
    # Kept pixels are untouched originals, not resampled ones.
    assert np.array_equal(result[mask], image[mask])
