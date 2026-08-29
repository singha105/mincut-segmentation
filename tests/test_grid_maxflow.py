"""The fast grid solver must agree exactly with the reference implementation.

`maxflow/grid_maxflow.py` exists only as a speed optimisation of
`maxflow/edmonds_karp.py`. It is the one used for real segmentation, so these
tests pin it to the reference solver that `tests/test_maxflow.py` verifies by
hand -- if the two ever diverge, that is a bug in the fast path.
"""

import numpy as np
import pytest

from maxflow.edmonds_karp import edmonds_karp, find_reachable_from_source
from maxflow.graph_builder import (
    BACKGROUND,
    FOREGROUND,
    SINK,
    SOURCE,
    UNMARKED,
    build_graph,
    build_grid_capacities,
)
from maxflow.grid_maxflow import solve_grid_min_cut


def _random_case(size, seed):
    """A random image plus scribbles guaranteed to contain both labels."""
    rng = np.random.default_rng(seed)
    image = rng.integers(0, 256, size=(size, size, 3), dtype=np.uint8)

    scribbles = np.full((size, size), UNMARKED, dtype=np.uint8)
    scribbles[0, 0] = FOREGROUND
    scribbles[-1, -1] = BACKGROUND
    for _ in range(3):
        y, x = rng.integers(0, size, 2)
        scribbles[y, x] = FOREGROUND if rng.random() < 0.5 else BACKGROUND
    scribbles[0, 0] = FOREGROUND
    scribbles[-1, -1] = BACKGROUND
    return image, scribbles


def _reference_solution(image, scribbles):
    """Max-flow value and foreground mask via the hand-verified dict solver."""
    graph = build_graph(image, scribbles)
    flow, residual = edmonds_karp(graph, SOURCE, SINK)
    reachable = find_reachable_from_source(residual, SOURCE)

    mask = np.zeros(scribbles.shape, dtype=bool)
    for node in reachable:
        if node != SOURCE:
            mask[node] = True
    return flow, mask


@pytest.mark.parametrize("seed", range(6))
def test_grid_solver_matches_reference_solver(seed):
    image, scribbles = _random_case(12, seed)

    expected_flow, expected_mask = _reference_solution(image, scribbles)
    actual_flow, actual_mask = solve_grid_min_cut(*build_grid_capacities(image, scribbles))

    assert actual_flow == pytest.approx(expected_flow)
    assert np.array_equal(actual_mask, expected_mask)


def test_grid_capacities_match_dict_graph():
    """Both graph representations must describe the same network."""
    image, scribbles = _random_case(8, seed=99)

    graph = build_graph(image, scribbles)
    source, sink, east, south = build_grid_capacities(image, scribbles)

    height, width = scribbles.shape
    for y in range(height):
        for x in range(width):
            expected_source = graph[SOURCE].get((y, x), 0.0)
            assert source[y, x] == pytest.approx(expected_source)

            expected_sink = graph.get((y, x), {}).get(SINK, 0.0)
            assert sink[y, x] == pytest.approx(expected_sink)

            if x + 1 < width:
                assert east[y, x] == pytest.approx(graph[(y, x)][(y, x + 1)])
            if y + 1 < height:
                assert south[y, x] == pytest.approx(graph[(y, x)][(y + 1, x)])


def test_hard_constraints_survive_the_cut():
    """Scribbled pixels must keep the label the user gave them."""
    image, scribbles = _random_case(10, seed=7)
    _, mask = solve_grid_min_cut(*build_grid_capacities(image, scribbles))

    assert mask[scribbles == FOREGROUND].all()
    assert not mask[scribbles == BACKGROUND].any()
