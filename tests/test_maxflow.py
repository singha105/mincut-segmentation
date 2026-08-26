"""Correctness tests for Edmonds-Karp against small, hand-verifiable graphs."""

from maxflow.edmonds_karp import edmonds_karp


def test_single_path_bottleneck():
    """S -> A -> T: flow is capped by the smaller of the two edges (3)."""
    graph = {
        "S": {"A": 5},
        "A": {"T": 3},
    }
    max_flow, _ = edmonds_karp(graph, "S", "T")
    assert max_flow == 3


def test_two_edge_disjoint_paths_sum():
    """Two independent S->T paths: max flow is just the sum of their bottlenecks."""
    graph = {
        "S": {"A": 3, "B": 2},
        "A": {"T": 3},
        "B": {"T": 2},
    }
    max_flow, _ = edmonds_karp(graph, "S", "T")
    assert max_flow == 5


def test_multiple_augmenting_paths_through_shared_node():
    """Paths share node B, so BFS needs two separate augmenting-path iterations.

    Hand-verified via min-cut: the cut ({S, A, B}, {T}) has capacity
    A->T (4) + B->T (9) = 13, which is achieved by combining the
    S-A-T path (bottleneck 4) with the S-B-T path (bottleneck 9).
    """
    graph = {
        "S": {"A": 10, "B": 10},
        "A": {"B": 2, "T": 4},
        "B": {"T": 9},
    }
    max_flow, _ = edmonds_karp(graph, "S", "T")
    assert max_flow == 13


def test_no_path_gives_zero_flow():
    """T is unreachable from S, so max flow must be 0."""
    graph = {
        "S": {"A": 5},
        "A": {},
        "T": {},
    }
    max_flow, _ = edmonds_karp(graph, "S", "T")
    assert max_flow == 0
