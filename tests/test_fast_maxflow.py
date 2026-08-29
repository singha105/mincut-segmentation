"""Fast mode must produce the same min-cut as the from-scratch solver.

`fast=True` swaps in Boykov-Kolmogorov via PyMaxflow. It is a different
algorithm, but it solves the same problem, so on these graphs it should return
the same max-flow value and the same partition.

PyMaxflow is optional, so every test here skips when it is not installed --
CI runs without it and still exercises the default path.
"""

import numpy as np
import pytest

from maxflow.fast_maxflow import is_available
from maxflow.graph_builder import (
    BACKGROUND,
    FOREGROUND,
    UNMARKED,
    build_grid_capacities,
)
from maxflow.grid_maxflow import solve_grid_min_cut
from segmentation.segmenter import segment, segment_image

pytestmark = pytest.mark.skipif(
    not is_available(), reason="PyMaxflow not installed (pip install -r requirements-fast.txt)"
)


def _random_case(size, seed):
    rng = np.random.default_rng(seed)
    image = rng.integers(0, 256, size=(size, size, 3), dtype=np.uint8)

    scribbles = np.full((size, size), UNMARKED, dtype=np.uint8)
    for _ in range(3):
        y, x = rng.integers(0, size, 2)
        scribbles[y, x] = FOREGROUND if rng.random() < 0.5 else BACKGROUND
    # Guarantee both labels are present whatever the random draw did.
    scribbles[0, 0] = FOREGROUND
    scribbles[-1, -1] = BACKGROUND
    return image, scribbles


@pytest.mark.parametrize("seed", range(6))
def test_fast_solver_matches_reference_min_cut(seed):
    from maxflow.fast_maxflow import solve_grid_min_cut_fast

    image, scribbles = _random_case(16, seed)

    expected_flow, expected_mask = solve_grid_min_cut(
        *build_grid_capacities(image, scribbles)
    )
    actual_flow, actual_mask = solve_grid_min_cut_fast(
        *build_grid_capacities(image, scribbles)
    )

    assert actual_flow == pytest.approx(expected_flow)
    assert np.array_equal(actual_mask, expected_mask)


def test_fast_solver_does_not_mutate_its_inputs():
    """Callers may reuse capacities, so the solver must leave them alone."""
    from maxflow.fast_maxflow import solve_grid_min_cut_fast

    image, scribbles = _random_case(12, seed=11)
    capacities = build_grid_capacities(image, scribbles)
    originals = [array.copy() for array in capacities]

    solve_grid_min_cut_fast(*capacities)

    for before, after in zip(originals, capacities):
        assert np.array_equal(before, after)


def test_segment_agrees_between_modes():
    image, scribbles = _random_case(20, seed=3)

    assert np.array_equal(
        segment(image, scribbles, fast=False), segment(image, scribbles, fast=True)
    )


def test_segment_image_agrees_between_modes():
    """The full pipeline, including the full-resolution refinement step."""
    rng = np.random.default_rng(5)
    image = rng.integers(0, 256, size=(180, 200, 3), dtype=np.uint8)
    scribbles = np.full((180, 200), UNMARKED, dtype=np.uint8)
    scribbles[20:40, 20:40] = FOREGROUND
    scribbles[140:170, 150:190] = BACKGROUND

    assert np.array_equal(
        segment_image(image, scribbles, fast=False),
        segment_image(image, scribbles, fast=True),
    )


def test_fast_mode_raises_a_helpful_error_when_unavailable(monkeypatch):
    """The failure mode should tell the user how to fix it."""
    import maxflow.fast_maxflow as fast_maxflow

    monkeypatch.setattr(fast_maxflow, "_load_pymaxflow", lambda: None)

    image, scribbles = _random_case(8, seed=1)
    with pytest.raises(ImportError, match="requirements-fast.txt"):
        fast_maxflow.solve_grid_min_cut_fast(*build_grid_capacities(image, scribbles))
