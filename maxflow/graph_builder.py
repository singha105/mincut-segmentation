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

SOURCE = "S"
SINK = "T"

UNMARKED = 0
FOREGROUND = 1
BACKGROUND = 2

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


def _n_link_capacities(gray: np.ndarray, n_link_weight: float) -> Tuple[np.ndarray, np.ndarray]:
    """Return the (east, south) N-link weight arrays for `gray`.

    ``east[y, x]`` is the weight of the undirected link between ``(y, x)`` and
    ``(y, x+1)``; ``south[y, x]`` links ``(y, x)`` to ``(y+1, x)``. The last
    column of ``east`` and last row of ``south`` are zero (no such neighbour).

    This is the single source of truth for N-link weights -- both the dict
    graph and the grid solver's capacity arrays are derived from it, so the
    two representations cannot drift apart.
    """
    beta = _estimate_beta(gray)

    east = np.zeros_like(gray)
    horizontal_diff = gray[:, 1:] - gray[:, :-1]
    east[:, :-1] = n_link_weight * np.exp(-beta * horizontal_diff ** 2)

    south = np.zeros_like(gray)
    vertical_diff = gray[1:, :] - gray[:-1, :]
    south[:-1, :] = n_link_weight * np.exp(-beta * vertical_diff ** 2)

    return east, south


def _neighbor_weight_sum(east: np.ndarray, south: np.ndarray) -> np.ndarray:
    """Total N-link weight incident on each pixel (used to size the K constant)."""
    total = np.zeros_like(east)
    total[:, :-1] += east[:, :-1]  # link to the east
    total[:, 1:] += east[:, :-1]  # same link, seen from the western pixel
    total[:-1, :] += south[:-1, :]
    total[1:, :] += south[:-1, :]
    return total


def _build_n_links(gray: np.ndarray, n_link_weight: float) -> Tuple[Graph, np.ndarray]:
    """Build symmetric pixel-to-pixel edges weighted by color similarity.

    Returns the partial graph plus a per-pixel array of each pixel's total
    outgoing N-link weight, which later sizes the hard-constraint constant K.
    """
    height, width = gray.shape
    east, south = _n_link_capacities(gray, n_link_weight)
    graph: Graph = {}

    for row in range(height):
        for col in range(width):
            if col + 1 < width:
                weight = east[row, col]
                graph.setdefault((row, col), {})[(row, col + 1)] = weight
                graph.setdefault((row, col + 1), {})[(row, col)] = weight
            if row + 1 < height:
                weight = south[row, col]
                graph.setdefault((row, col), {})[(row + 1, col)] = weight
                graph.setdefault((row + 1, col), {})[(row, col)] = weight

    return graph, _neighbor_weight_sum(east, south)


def _color_histogram(gray: np.ndarray, mask: np.ndarray, num_bins: int) -> np.ndarray:
    """Build a normalized intensity histogram over the pixels selected by `mask`."""
    values = gray[mask]
    histogram, _ = np.histogram(values, bins=num_bins, range=(0, 255))
    histogram = histogram.astype(np.float64) + 1.0  # Laplace smoothing avoids log(0).
    return histogram / histogram.sum()


def _bin_indices(gray: np.ndarray, num_bins: int) -> np.ndarray:
    """Map each pixel's intensity to its histogram bin index."""
    bin_width = 256.0 / num_bins
    return np.clip((gray // bin_width).astype(int), 0, num_bins - 1)


def _t_link_capacities(
    gray: np.ndarray,
    scribble_mask: np.ndarray,
    hard_constraint: float,
    num_bins: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """Return the (source, sink) t-link capacity arrays.

    Scribbled pixels get the hard-constraint capacity K on one side and 0 on
    the other, so the min-cut can never relabel them. Everything else is
    scored by how well it matches each scribble's intensity histogram.
    """
    fg_mask = scribble_mask == FOREGROUND
    bg_mask = scribble_mask == BACKGROUND
    if not fg_mask.any() or not bg_mask.any():
        raise ValueError(
            "scribble_mask must contain at least one FOREGROUND and one BACKGROUND pixel"
        )

    fg_histogram = _color_histogram(gray, fg_mask, num_bins)
    bg_histogram = _color_histogram(gray, bg_mask, num_bins)
    bin_indices = _bin_indices(gray, num_bins)

    # Regional cost of a label is low when the pixel resembles the *other*
    # label's model -- see the module docstring.
    source_capacity = -np.log(bg_histogram[bin_indices])
    sink_capacity = -np.log(fg_histogram[bin_indices])

    source_capacity[fg_mask] = hard_constraint
    sink_capacity[fg_mask] = 0.0
    source_capacity[bg_mask] = 0.0
    sink_capacity[bg_mask] = hard_constraint

    return source_capacity, sink_capacity


def build_grid_capacities(
    image: np.ndarray,
    scribble_mask: np.ndarray,
    n_link_weight: float = 50.0,
    num_bins: int = 32,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Build the same Boykov-Jolly graph as `build_graph`, as capacity arrays.

    Returns ``(source, sink, east, south)``, all shaped ``(H, W)`` -- the
    layout ``maxflow.grid_maxflow.solve_grid_min_cut`` expects. This is the
    fast path used for real segmentation; `build_graph` builds the equivalent
    nested-dict graph for the reference solver.
    """
    gray = _to_grayscale(image)
    east, south = _n_link_capacities(gray, n_link_weight)

    # K must exceed any possible sum of N-link weights at a single pixel, so
    # a hard-constrained pixel's t-link is never the cheapest thing to cut.
    hard_constraint = 1.0 + _neighbor_weight_sum(east, south).max()
    source, sink = _t_link_capacities(gray, scribble_mask, hard_constraint, num_bins)
    return source, sink, east, south


def build_graph(
    image: np.ndarray,
    scribble_mask: np.ndarray,
    n_link_weight: float = 50.0,
    num_bins: int = 32,
) -> Graph:
    """Build the Boykov-Jolly pixel graph for `image` given `scribble_mask`.

    `scribble_mask` is an (H, W) array whose entries are one of UNMARKED,
    FOREGROUND, or BACKGROUND. Returns a graph with source node SOURCE and
    sink node SINK, ready to pass to maxflow.edmonds_karp.
    """
    gray = _to_grayscale(image)
    height, width = gray.shape

    graph, neighbor_weight_sum = _build_n_links(gray, n_link_weight)

    # K must exceed any possible sum of N-link weights at a single pixel, so
    # a hard-constrained pixel's t-link is never the cheapest thing to cut.
    hard_constraint = 1.0 + neighbor_weight_sum.max()
    source_capacities, sink_capacities = _t_link_capacities(
        gray, scribble_mask, hard_constraint, num_bins
    )

    graph.setdefault(SOURCE, {})
    for row in range(height):
        for col in range(width):
            pixel: Node = (row, col)
            source_capacity = source_capacities[row, col]
            sink_capacity = sink_capacities[row, col]

            if source_capacity > 0:
                graph[SOURCE][pixel] = source_capacity
            if sink_capacity > 0:
                graph.setdefault(pixel, {})[SINK] = sink_capacity

    return graph
