"""
Edmonds-Karp max-flow algorithm.

Edmonds-Karp is Ford-Fulkerson with augmenting paths chosen by
breadth-first search (BFS), which guarantees the shortest (fewest-edge)
augmenting path is used at each step. This bounds the algorithm to
O(V * E^2) and avoids the pathological slow cases that arbitrary
augmenting-path selection can hit.

Algorithm:
    1. Build a residual graph from the input capacities. Every edge (u, v)
       with capacity c gets a reverse edge (v, u), initially capacity 0,
       so that flow pushed along (u, v) can later be partially "undone"
       if a better routing is found.
    2. Use BFS to find a path from source to sink using only edges with
       positive residual capacity.
    3. If no such path exists, the current flow is already maximum: stop.
    4. Otherwise, compute the path's bottleneck (its smallest residual
       capacity) and push that much flow along the path: subtract it from
       each forward edge and add it to each corresponding reverse edge.
    5. Repeat from step 2.

Graphs are represented as a nested dict of capacities:
    {u: {v: capacity, ...}, ...}
Nodes may be any hashable value (e.g. pixel coordinates), so this module
can operate directly on graphs produced by graph_builder.py.
"""

from collections import deque
from typing import Dict, Hashable, List, Optional, Tuple

Node = Hashable
Graph = Dict[Node, Dict[Node, float]]


def build_residual_graph(graph: Graph) -> Graph:
    """Return a residual-graph copy of `graph` with reverse edges added.

    Every edge (u, v) with capacity c gets a reverse edge (v, u) with
    capacity 0 if one isn't already present in `graph`, so augmenting
    paths can push flow back through it later.
    """
    residual: Graph = {}
    for u, neighbors in graph.items():
        residual.setdefault(u, {})
        for v, capacity in neighbors.items():
            residual[u][v] = residual[u].get(v, 0) + capacity
            residual.setdefault(v, {})
            residual[v].setdefault(u, 0)
    return residual


def bfs_find_augmenting_path(
    residual: Graph, source: Node, sink: Node
) -> Optional[List[Node]]:
    """Find a shortest source -> sink path using edges with residual capacity > 0.

    Returns the path as a list of nodes [source, ..., sink], or None if
    the sink is unreachable from the source in the current residual graph.
    """
    parent: Dict[Node, Node] = {source: source}
    queue = deque([source])

    while queue:
        u = queue.popleft()
        if u == sink:
            break
        for v, capacity in residual.get(u, {}).items():
            if capacity > 0 and v not in parent:
                parent[v] = u
                queue.append(v)

    if sink not in parent:
        return None

    path = [sink]
    while path[-1] != source:
        path.append(parent[path[-1]])
    path.reverse()
    return path


def path_bottleneck(residual: Graph, path: List[Node]) -> float:
    """Return the smallest residual capacity along the edges of `path`."""
    return min(residual[u][v] for u, v in zip(path, path[1:]))


def augment_along_path(residual: Graph, path: List[Node], bottleneck: float) -> None:
    """Push `bottleneck` units of flow along `path`, updating residual capacities."""
    for u, v in zip(path, path[1:]):
        residual[u][v] -= bottleneck
        residual[v][u] += bottleneck


def edmonds_karp(graph: Graph, source: Node, sink: Node) -> Tuple[float, Graph]:
    """Compute the max-flow from `source` to `sink` in `graph`.

    Returns (max_flow_value, residual_graph). The residual graph is
    returned so callers can later recover the min-cut by finding which
    nodes remain reachable from `source` once no augmenting path remains.
    """
    residual = build_residual_graph(graph)
    max_flow: float = 0

    while True:
        path = bfs_find_augmenting_path(residual, source, sink)
        if path is None:
            break
        bottleneck = path_bottleneck(residual, path)
        augment_along_path(residual, path, bottleneck)
        max_flow += bottleneck

    return max_flow, residual
