"""Unit tests for maxflow.graph_builder."""

import numpy as np

from maxflow.graph_builder import (
    BACKGROUND,
    FOREGROUND,
    SINK,
    SOURCE,
    UNMARKED,
    _build_n_links,
    _estimate_beta,
    _to_grayscale,
    build_graph,
)


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


def test_build_graph_requires_both_labels():
    image = np.zeros((2, 2, 3), dtype=np.uint8)
    scribble_mask = np.full((2, 2), UNMARKED, dtype=np.uint8)
    try:
        build_graph(image, scribble_mask)
        assert False, "expected ValueError for a scribble mask missing a label"
    except ValueError:
        pass


def test_scribbled_pixels_get_hard_constraints():
    image = np.zeros((2, 2, 3), dtype=np.uint8)
    image[0, 0] = 10
    image[1, 1] = 250
    scribble_mask = np.array(
        [[FOREGROUND, UNMARKED],
         [UNMARKED, BACKGROUND]],
        dtype=np.uint8,
    )
    graph = build_graph(image, scribble_mask)

    max_n_link_weight = max(
        weight
        for node, neighbors in graph.items()
        for target, weight in neighbors.items()
        if node != SOURCE and target != SINK
    )

    # Hard constraints must dominate every N-link weight, or the min-cut
    # could still relabel a pixel the user explicitly scribbled.
    assert graph[SOURCE][(0, 0)] > max_n_link_weight
    assert SINK not in graph.get((0, 0), {})

    assert graph[(1, 1)][SINK] > max_n_link_weight
    assert (1, 1) not in graph[SOURCE]
