"""Tests for the end-to-end segmentation pipeline."""

import numpy as np

from maxflow.graph_builder import BACKGROUND, FOREGROUND, UNMARKED
from segmentation.segmenter import segment


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
