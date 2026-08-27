"""
Image-to-graph construction for min-cut segmentation (Boykov-Jolly).

Boykov, Y. and Jolly, M-P., "Interactive Graph Cuts for Optimal Boundary &
Region Segmentation of Objects in N-D Images" (ICCV 2001), frames binary
segmentation as an energy minimization solved by a single min-cut:

    E(L) = sum_p D_p(L_p) + sum_{(p,q) in N} V_pq(L_p, L_q)

D_p is a per-pixel "regional" term (how well pixel p's color matches the
foreground/background color models), and V_pq is a "boundary" term that
penalizes cutting between similar-looking neighboring pixels. This module
builds the flow network whose min-cut equals the minimum of E(L):

- One node per pixel, plus two terminals: source S (foreground) and sink T
  (background).
- N-links: an edge between every pair of 4-connected neighboring pixels,
  weighted by color similarity (a Gaussian on intensity difference) so that
  cutting through a strong image edge is cheap and cutting through a flat
  region is expensive.
- T-links: an edge from S to each pixel and from each pixel to T, weighted
  by how well that pixel's color matches simple background/foreground
  histograms built from the user's scribbles.
- Hard constraints: scribbled pixels get one of their two t-links forced to
  a very large capacity K, so the min-cut can never relabel a pixel the
  user has explicitly marked.

The resulting graph is a plain nested dict of capacities, directly
consumable by maxflow.edmonds_karp.
"""

from typing import Dict, Tuple

import numpy as np

Node = Tuple[int, int]
Graph = Dict[object, Dict[object, float]]


def _to_grayscale(image: np.ndarray) -> np.ndarray:
    """Convert an RGB (or already-grayscale) image to a float grayscale array."""
    if image.ndim == 2:
        return image.astype(np.float64)
    weights = np.array([0.299, 0.587, 0.114])
    return image[..., :3].astype(np.float64) @ weights


def _estimate_beta(gray: np.ndarray) -> float:
    """Estimate the smoothness scale beta = 1 / (2 * mean squared neighbor difference).

    This is the standard Boykov-Jolly/GrabCut heuristic for making N-link
    weights adapt to the image's own contrast rather than a fixed constant.
    """
    horizontal_diff = gray[:, 1:] - gray[:, :-1]
    vertical_diff = gray[1:, :] - gray[:-1, :]
    squared_diffs = np.concatenate([horizontal_diff.ravel() ** 2, vertical_diff.ravel() ** 2])
    mean_sq_diff = squared_diffs.mean()
    if mean_sq_diff == 0:
        return 1.0
    return 1.0 / (2.0 * mean_sq_diff)


def _build_n_links(gray: np.ndarray, n_link_weight: float) -> Tuple[Graph, np.ndarray]:
    """Build symmetric pixel-to-pixel edges weighted by color similarity.

    Returns the partial graph plus a per-pixel array of each pixel's total
    outgoing N-link weight, which later sizes the hard-constraint constant K.
    """
    height, width = gray.shape
    beta = _estimate_beta(gray)
    graph: Graph = {}
    neighbor_weight_sum = np.zeros((height, width), dtype=np.float64)

    def add_edge(p: Node, q: Node, weight: float) -> None:
        graph.setdefault(p, {})[q] = weight
        graph.setdefault(q, {})[p] = weight
        neighbor_weight_sum[p] += weight
        neighbor_weight_sum[q] += weight

    for row in range(height):
        for col in range(width):
            if col + 1 < width:
                diff = gray[row, col] - gray[row, col + 1]
                weight = n_link_weight * np.exp(-beta * diff ** 2)
                add_edge((row, col), (row, col + 1), weight)
            if row + 1 < height:
                diff = gray[row, col] - gray[row + 1, col]
                weight = n_link_weight * np.exp(-beta * diff ** 2)
                add_edge((row, col), (row + 1, col), weight)

    return graph, neighbor_weight_sum
