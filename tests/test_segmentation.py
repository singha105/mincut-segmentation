"""Tests for the end-to-end segmentation pipeline."""

import numpy as np
from PIL import Image

from maxflow.graph_builder import BACKGROUND, FOREGROUND, UNMARKED
from segmentation.segmenter import segment

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
