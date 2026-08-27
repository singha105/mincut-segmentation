"""
Top-level segmentation orchestration.

Ties maxflow.graph_builder (image + scribbles -> flow network) and
maxflow.edmonds_karp (network -> max-flow / min-cut) together into a single
segment() call that returns a foreground/background pixel mask.
"""

import numpy as np

from maxflow.edmonds_karp import edmonds_karp, find_reachable_from_source
from maxflow.graph_builder import SINK, SOURCE, build_graph


def segment(
    image: np.ndarray,
    scribble_mask: np.ndarray,
    n_link_weight: float = 50.0,
    num_bins: int = 32,
) -> np.ndarray:
    """Segment `image` into foreground/background given `scribble_mask`.

    Returns a boolean (H, W) mask where True marks pixels grouped with the
    source terminal (foreground) by the min-cut.
    """
    height, width = scribble_mask.shape
    graph = build_graph(image, scribble_mask, n_link_weight=n_link_weight, num_bins=num_bins)

    _, residual = edmonds_karp(graph, SOURCE, SINK)
    foreground_nodes = find_reachable_from_source(residual, SOURCE)

    mask = np.zeros((height, width), dtype=bool)
    for node in foreground_nodes:
        if node == SOURCE:
            continue
        row, col = node
        mask[row, col] = True

    return mask
