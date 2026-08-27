"""Unit tests for maxflow.graph_builder."""

import numpy as np

from maxflow.graph_builder import _build_n_links, _estimate_beta, _to_grayscale


def test_to_grayscale_averages_equal_channels():
    image = np.full((2, 2, 3), 100, dtype=np.uint8)
    gray = _to_grayscale(image)
    assert np.allclose(gray, 100)


def test_estimate_beta_is_positive_for_varying_image():
    gray = np.array([[0, 255], [255, 0]], dtype=np.float64)
    beta = _estimate_beta(gray)
    assert beta > 0


def test_n_links_favor_similar_neighbors():
    """A flat region should get a much higher N-link weight than a hard edge."""
    gray = np.array(
        [[10, 10, 250],
         [10, 10, 250]],
        dtype=np.float64,
    )
    graph, _ = _build_n_links(gray, n_link_weight=50.0)
    similar_weight = graph[(0, 0)][(0, 1)]
    different_weight = graph[(0, 1)][(0, 2)]
    assert similar_weight > different_weight
