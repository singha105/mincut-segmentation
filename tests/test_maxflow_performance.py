"""Performance check tying Edmonds-Karp to the project's <2s segmentation target."""

import random
import time

from maxflow.edmonds_karp import edmonds_karp


def _random_layered_graph(num_layers, nodes_per_layer, edge_probability=0.3, seed=42):
    """Build a random layered DAG (source -> layers -> sink) with random capacities.

    A layered graph is a reasonable stand-in for a downsampled pixel grid:
    many nodes, bounded fan-out, and a single source/sink pair.
    """
    rng = random.Random(seed)
    source, sink = "S", "T"
    layers = [[f"L{layer}_{i}" for i in range(nodes_per_layer)] for layer in range(num_layers)]

    graph = {source: {node: rng.randint(1, 20) for node in layers[0]}}

    for layer_idx in range(num_layers - 1):
        for u in layers[layer_idx]:
            graph.setdefault(u, {})
            for v in layers[layer_idx + 1]:
                if rng.random() < edge_probability:
                    graph[u][v] = rng.randint(1, 20)

    for u in layers[-1]:
        graph.setdefault(u, {})
        graph[u][sink] = rng.randint(1, 20)

    return graph, source, sink


def test_edmonds_karp_runs_within_time_budget():
    graph, source, sink = _random_layered_graph(num_layers=6, nodes_per_layer=15)

    start = time.perf_counter()
    max_flow, _ = edmonds_karp(graph, source, sink)
    elapsed = time.perf_counter() - start

    assert max_flow > 0
    assert elapsed < 2.0
