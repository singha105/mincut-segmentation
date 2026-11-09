"""Use a model's per-pixel probability as the regional term of the graph cut.

The default term compares each pixel's intensity with histograms built from
the scribbles, so it fails when subject and background share a tonal range.
This one asks a trained model instead and leaves the rest of Boykov-Jolly
alone: the N-links, the hard constraints and the solver are all the same.
"""

from typing import Tuple

import numpy as np

from maxflow.graph_builder import (
    BACKGROUND,
    FOREGROUND,
    _n_link_capacities,
    _neighbor_weight_sum,
    _to_grayscale,
)

# Keeps -log finite when the model is completely sure.
_MIN_PROBABILITY = 1e-6


def build_learned_capacities(
    image: np.ndarray,
    scribble_mask: np.ndarray,
    foreground_probability: np.ndarray,
    n_link_weight: float = 5.0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return ``(source, sink, east, south)``, as ``build_grid_capacities`` does.

    The source link is what it costs to call a pixel background, so it is
    ``-log P(background)``, and the sink link is ``-log P(foreground)``.
    """
    gray = _to_grayscale(image)
    east, south = _n_link_capacities(gray, n_link_weight)

    probability = np.clip(foreground_probability, _MIN_PROBABILITY, 1.0 - _MIN_PROBABILITY)
    source = -np.log(1.0 - probability)
    sink = -np.log(probability)

    # A marked pixel keeps its label however confident the model is.
    hard_constraint = 1.0 + _neighbor_weight_sum(east, south).max()
    foreground = scribble_mask == FOREGROUND
    background = scribble_mask == BACKGROUND
    source[foreground] = hard_constraint
    sink[foreground] = 0.0
    source[background] = 0.0
    sink[background] = hard_constraint

    return source, sink, east, south
