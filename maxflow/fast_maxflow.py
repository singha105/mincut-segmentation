"""
Optional Boykov-Kolmogorov solver, backed by the PyMaxflow library.

The from-scratch Edmonds-Karp solvers in this package are the point of the
project. This module is an *opt-in* alternative for when raw speed matters:
Boykov-Kolmogorov is the max-flow variant designed specifically for the
grid graphs that arise in vision, and it solves them far faster than
Edmonds-Karp can, which allows a much finer cut grid within the same time
budget.

It computes the same min-cut, so results are equivalent up to ties -- see
``tests/test_fast_maxflow.py``, which pins it to the reference solver.

Installing
----------
PyMaxflow is deliberately not in ``requirements.txt``; the project works
without it. To enable this path::

    pip install -r requirements-fast.txt

Name collision
--------------
PyMaxflow's own import name is ``maxflow``, which is also this project's
package name. From the repository root our package wins, so a plain
``import maxflow`` would return *this* package rather than the library. The
loader below therefore locates PyMaxflow in site-packages and imports it
explicitly under the name ``pymaxflow``.
"""

import importlib.util
import os
import sys
import sysconfig
from typing import Optional, Tuple

import numpy as np

_PYMAXFLOW_MODULE_NAME = "pymaxflow"


def _load_pymaxflow():
    """Import PyMaxflow from site-packages, bypassing this package's name.

    Returns the module, or None when PyMaxflow is not installed.
    """
    if _PYMAXFLOW_MODULE_NAME in sys.modules:
        return sys.modules[_PYMAXFLOW_MODULE_NAME]

    candidates = {
        sysconfig.get_paths()["purelib"],
        sysconfig.get_paths()["platlib"],
    }
    candidates.update(p for p in sys.path if p.endswith("site-packages"))

    for base in candidates:
        package_dir = os.path.join(base, "maxflow")
        init_file = os.path.join(package_dir, "__init__.py")
        # Guard against finding this project's own package.
        if not os.path.isfile(init_file) or os.path.samefile(
            package_dir, os.path.dirname(os.path.abspath(__file__))
        ):
            continue
        spec = importlib.util.spec_from_file_location(
            _PYMAXFLOW_MODULE_NAME, init_file, submodule_search_locations=[package_dir]
        )
        if spec is None or spec.loader is None:
            continue
        module = importlib.util.module_from_spec(spec)
        sys.modules[_PYMAXFLOW_MODULE_NAME] = module
        try:
            spec.loader.exec_module(module)
        except Exception:
            del sys.modules[_PYMAXFLOW_MODULE_NAME]
            continue
        if hasattr(module, "GraphFloat"):
            return module

    return None


def is_available() -> bool:
    """True when PyMaxflow is installed and importable."""
    return _load_pymaxflow() is not None


def solve_grid_min_cut_fast(
    source_capacity: np.ndarray,
    sink_capacity: np.ndarray,
    east_capacity: np.ndarray,
    south_capacity: np.ndarray,
) -> Tuple[float, np.ndarray]:
    """Boykov-Kolmogorov equivalent of ``grid_maxflow.solve_grid_min_cut``.

    Takes and returns exactly what the from-scratch solver does:
    ``(max_flow, foreground_mask)``, with the mask True on the source side of
    the cut.

    Raises ImportError when PyMaxflow is not installed.
    """
    pymaxflow = _load_pymaxflow()
    if pymaxflow is None:
        raise ImportError(
            "Fast mode needs PyMaxflow. Install it with: pip install -r requirements-fast.txt"
        )

    height, width = source_capacity.shape
    graph = pymaxflow.GraphFloat()
    nodes = graph.add_grid_nodes((height, width))

    # N-links. add_grid_edges with a structure matrix adds the edge and its
    # reverse with the same weight, matching the undirected links the graph
    # builder produces -- but it takes a single scalar weight, so the two
    # directions are added explicitly to preserve per-edge weights.
    east_from = nodes[:, :-1].ravel()
    east_to = nodes[:, 1:].ravel()
    east_weights = east_capacity[:, :-1].ravel()
    graph.add_edges(east_from, east_to, east_weights, east_weights)

    south_from = nodes[:-1, :].ravel()
    south_to = nodes[1:, :].ravel()
    south_weights = south_capacity[:-1, :].ravel()
    graph.add_edges(south_from, south_to, south_weights, south_weights)

    # T-links.
    graph.add_grid_tedges(nodes, source_capacity, sink_capacity)

    max_flow = float(graph.maxflow())

    # PyMaxflow marks the source side as segment 0.
    mask = graph.get_grid_segments(nodes)
    return max_flow, ~mask
