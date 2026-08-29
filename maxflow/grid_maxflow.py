"""
Grid-specialised Edmonds-Karp max-flow.

This is the same algorithm as :mod:`maxflow.edmonds_karp` -- BFS-chosen
shortest augmenting paths, repeated until none remain -- but specialised to
the 4-connected pixel grid that :mod:`maxflow.graph_builder` produces.

Why a second implementation exists
----------------------------------
The reference implementation stores the graph as a nested dict and walks it
one edge at a time in Python. That is readable (and is what the hand-verified
tests in ``tests/test_maxflow.py`` exercise), but every edge lookup costs a
dict probe, so an N-pixel image needs roughly 2N augmenting paths x O(N)
probes each -- quadratic with a large constant. Measured on a real photo it
runs ~20s at 80x80, which caps segmentation at a very coarse grid.

Here the grid structure is exploited instead: residual capacities live in
six ``(H, W)`` float arrays and each BFS *level* expands as a handful of
whole-array numpy operations rather than per-edge Python. The asymptotics are
unchanged -- Edmonds-Karp is still O(V * E^2) and still needs ~2N augmenting
paths -- but the constant drops by well over an order of magnitude, which buys
a meaningfully finer mask within the same time budget.

Both implementations must agree exactly; ``tests/test_grid_maxflow.py``
checks the grid solver against the reference solver on the same graphs.

Residual capacity layout
------------------------
For a pixel ``p = (y, x)``:

- ``cap_source[y, x]``   residual  SOURCE -> p
- ``cap_sink[y, x]``     residual  p -> SINK
- ``cap_east[y, x]``     residual  p -> (y, x+1)
- ``cap_west[y, x]``     residual  p -> (y, x-1)
- ``cap_north[y, x]``    residual  p -> (y-1, x)
- ``cap_south[y, x]``    residual  p -> (y+1, x)

Reverse edges are the mirrored entries: the reverse of ``cap_east[y, x]`` is
``cap_west[y, x+1]``, and the reverse of ``cap_south[y, x]`` is
``cap_north[y+1, x]``.
"""

from typing import Tuple

import numpy as np

# How a pixel was reached during BFS, used to walk the augmenting path back.
FROM_SOURCE = 0
FROM_WEST = 1  # reached from (y, x-1) along that pixel's east edge
FROM_EAST = 2  # reached from (y, x+1) along that pixel's west edge
FROM_NORTH = 3  # reached from (y-1, x) along that pixel's south edge
FROM_SOUTH = 4  # reached from (y+1, x) along that pixel's north edge

UNREACHED = -1


class _Residual:
    """Mutable residual capacities for the grid plus its two terminals."""

    __slots__ = ("source", "sink", "east", "west", "north", "south", "height", "width")

    def __init__(self, height: int, width: int):
        shape = (height, width)
        self.height = height
        self.width = width
        self.source = np.zeros(shape, dtype=np.float64)
        self.sink = np.zeros(shape, dtype=np.float64)
        self.east = np.zeros(shape, dtype=np.float64)
        self.west = np.zeros(shape, dtype=np.float64)
        self.north = np.zeros(shape, dtype=np.float64)
        self.south = np.zeros(shape, dtype=np.float64)


def _saturate_direct_paths(residual: _Residual) -> float:
    """Push flow along every ``SOURCE -> pixel -> SINK`` path at once.

    Length-2 paths are the shortest any augmenting path can be, so BFS finds
    them before anything else -- and because each one only touches its own
    pixel's two terminal edges, no two of them share an edge. They can
    therefore all be saturated in a single vectorised step, giving exactly
    the residual graph that running them one BFS at a time would produce.

    This matters a lot in practice: unscribbled pixels get t-links to *both*
    terminals, so without this step the solver would burn one full BFS per
    pixel on paths that never needed a search.
    """
    direct = np.minimum(residual.source, residual.sink)
    residual.source -= direct
    residual.sink -= direct
    return float(direct.sum())


def _bfs_parents(residual: _Residual):
    """Breadth-first search from SOURCE over edges with positive residual.

    Returns ``(parent, sink_pixel)`` where ``parent`` records how each visited
    pixel was reached and ``sink_pixel`` is the ``(y, x)`` of a pixel on a
    shortest source->sink path, or ``None`` when the sink is unreachable.

    Each BFS level is expanded with whole-array operations, so the cost per
    level is a few passes over the grid rather than one dict probe per edge.
    """
    parent = np.full((residual.height, residual.width), UNREACHED, dtype=np.int8)

    # Level 1: every pixel the source can still push flow into.
    frontier = residual.source > 0
    if not frontier.any():
        return parent, None
    parent[frontier] = FROM_SOURCE

    while True:
        # Can any pixel on this level still push into the sink? If several
        # can, they are all on shortest paths, so any of them is a valid
        # Edmonds-Karp choice.
        reaches_sink = frontier & (residual.sink > 0)
        if reaches_sink.any():
            flat = int(np.argmax(reaches_sink))
            return parent, (flat // residual.width, flat % residual.width)

        visited = parent != UNREACHED
        next_frontier = np.zeros_like(frontier)

        # Expand one step in each of the four directions. `visited` is updated
        # after every direction so a pixel first reached going east is not
        # overwritten by a later direction on the same level.
        step = frontier[:, :-1] & (residual.east[:, :-1] > 0) & ~visited[:, 1:]
        if step.any():
            parent[:, 1:][step] = FROM_WEST
            visited[:, 1:] |= step
            next_frontier[:, 1:] |= step

        step = frontier[:, 1:] & (residual.west[:, 1:] > 0) & ~visited[:, :-1]
        if step.any():
            parent[:, :-1][step] = FROM_EAST
            visited[:, :-1] |= step
            next_frontier[:, :-1] |= step

        step = frontier[1:, :] & (residual.north[1:, :] > 0) & ~visited[:-1, :]
        if step.any():
            parent[:-1, :][step] = FROM_SOUTH
            visited[:-1, :] |= step
            next_frontier[:-1, :] |= step

        step = frontier[:-1, :] & (residual.south[:-1, :] > 0) & ~visited[1:, :]
        if step.any():
            parent[1:, :][step] = FROM_NORTH
            visited[1:, :] |= step
            next_frontier[1:, :] |= step

        if not next_frontier.any():
            return parent, None
        frontier = next_frontier


def _walk_back(parent: np.ndarray, sink_pixel: Tuple[int, int]):
    """Return the augmenting path's pixels, ordered source-side first."""
    path = [sink_pixel]
    y, x = sink_pixel
    while parent[y, x] != FROM_SOURCE:
        direction = parent[y, x]
        if direction == FROM_WEST:
            x -= 1
        elif direction == FROM_EAST:
            x += 1
        elif direction == FROM_NORTH:
            y -= 1
        else:  # FROM_SOUTH
            y += 1
        path.append((y, x))
    path.reverse()
    return path


def _augment(residual: _Residual, path) -> float:
    """Push the path's bottleneck flow along it and update residual capacities."""
    first_y, first_x = path[0]
    last_y, last_x = path[-1]

    bottleneck = min(residual.source[first_y, first_x], residual.sink[last_y, last_x])
    for (ay, ax), (by, bx) in zip(path, path[1:]):
        if bx == ax + 1:
            capacity = residual.east[ay, ax]
        elif bx == ax - 1:
            capacity = residual.west[ay, ax]
        elif by == ay + 1:
            capacity = residual.south[ay, ax]
        else:
            capacity = residual.north[ay, ax]
        if capacity < bottleneck:
            bottleneck = capacity

    residual.source[first_y, first_x] -= bottleneck
    residual.sink[last_y, last_x] -= bottleneck
    for (ay, ax), (by, bx) in zip(path, path[1:]):
        if bx == ax + 1:
            residual.east[ay, ax] -= bottleneck
            residual.west[by, bx] += bottleneck
        elif bx == ax - 1:
            residual.west[ay, ax] -= bottleneck
            residual.east[by, bx] += bottleneck
        elif by == ay + 1:
            residual.south[ay, ax] -= bottleneck
            residual.north[by, bx] += bottleneck
        else:
            residual.north[ay, ax] -= bottleneck
            residual.south[by, bx] += bottleneck

    return bottleneck


def _reachable_from_source(residual: _Residual) -> np.ndarray:
    """Pixels still reachable from SOURCE once no augmenting path remains.

    This is the source side of the min-cut, i.e. the foreground mask.
    """
    reached = residual.source > 0
    if not reached.any():
        return reached
    frontier = reached.copy()

    while True:
        next_frontier = np.zeros_like(frontier)

        step = frontier[:, :-1] & (residual.east[:, :-1] > 0) & ~reached[:, 1:]
        next_frontier[:, 1:] |= step
        reached[:, 1:] |= step

        step = frontier[:, 1:] & (residual.west[:, 1:] > 0) & ~reached[:, :-1]
        next_frontier[:, :-1] |= step
        reached[:, :-1] |= step

        step = frontier[1:, :] & (residual.north[1:, :] > 0) & ~reached[:-1, :]
        next_frontier[:-1, :] |= step
        reached[:-1, :] |= step

        step = frontier[:-1, :] & (residual.south[:-1, :] > 0) & ~reached[1:, :]
        next_frontier[1:, :] |= step
        reached[1:, :] |= step

        if not next_frontier.any():
            return reached
        frontier = next_frontier


def solve_grid_min_cut(
    source_capacity: np.ndarray,
    sink_capacity: np.ndarray,
    east_capacity: np.ndarray,
    south_capacity: np.ndarray,
) -> Tuple[float, np.ndarray]:
    """Run Edmonds-Karp on a 4-connected grid and return its min-cut.

    Parameters are the graph's *original* capacities, all shaped ``(H, W)``:

    - ``source_capacity[y, x]``  SOURCE -> pixel
    - ``sink_capacity[y, x]``    pixel -> SINK
    - ``east_capacity[y, x]``    pixel -> (y, x+1); last column must be 0
    - ``south_capacity[y, x]``   pixel -> (y+1, x); last row must be 0

    N-links are undirected, so ``east``/``south`` also seed the matching
    west/north residuals.

    Returns ``(max_flow, foreground_mask)`` where ``foreground_mask`` is a
    boolean ``(H, W)`` array that is True on the source side of the min-cut.
    """
    height, width = source_capacity.shape
    residual = _Residual(height, width)
    residual.source[:] = source_capacity
    residual.sink[:] = sink_capacity

    # An undirected N-link of weight w is a pair of directed edges of weight w.
    residual.east[:] = east_capacity
    residual.west[:, 1:] = east_capacity[:, :-1]
    residual.south[:] = south_capacity
    residual.north[1:, :] = south_capacity[:-1, :]

    max_flow = _saturate_direct_paths(residual)
    while True:
        parent, sink_pixel = _bfs_parents(residual)
        if sink_pixel is None:
            break
        max_flow += _augment(residual, _walk_back(parent, sink_pixel))

    return max_flow, _reachable_from_source(residual)
