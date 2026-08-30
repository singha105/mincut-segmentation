"""Benchmark the from-scratch Edmonds-Karp solver against PyMaxflow's
Boykov-Kolmogorov, on identical graphs.

Both solvers are given the same capacities from the same photos and scribbles,
so the only variable is the algorithm. The script also checks that the two
agree on the resulting mask -- a speed number is meaningless if the answers
differ.

Writes a markdown table and a CSV next to this file:

    python benchmarks/compare_solvers.py

Requires PyMaxflow (``pip install -r requirements-fast.txt``); without it the
script explains what to install and exits.
"""

import csv
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import cv2
import numpy as np

from maxflow.fast_maxflow import is_available, solve_grid_min_cut_fast
from maxflow.graph_builder import build_grid_capacities
from maxflow.grid_maxflow import solve_grid_min_cut
from measure_timing import SCRIBBLES, load, paint_scribbles
from segmentation.segmenter import (
    DEFAULT_PROCESSING_MAX_DIM,
    _downscale_scribbles,
    _processing_size,
    segment_image,
)

HERE = os.path.dirname(os.path.abspath(__file__))
REPEATS = 3
# Grid sizes to compare at. 56 is the shipped default for the from-scratch
# solver; the larger ones show what fast mode makes reachable.
GRID_SIZES = (56, 96, 160)
# Above this the from-scratch solver takes long enough that timing every image
# is not worth the wall time; it is reported as "not run" instead.
EDMONDS_KARP_LIMIT = 96


def time_best(fn, repeats=REPEATS):
    best = float("inf")
    result = None
    for _ in range(repeats):
        start = time.perf_counter()
        result = fn()
        best = min(best, time.perf_counter() - start)
    return best, result


def build_case(name, grid_size):
    """Capacities for one image at one cut-grid size."""
    image = load(name, 800)
    scribbles = paint_scribbles(image.shape[:2], SCRIBBLES[name])
    height, width = image.shape[:2]

    proc_height, proc_width = _processing_size(height, width, grid_size)
    small_image = cv2.resize(image, (proc_width, proc_height), interpolation=cv2.INTER_AREA)
    small_scribbles = _downscale_scribbles(scribbles, proc_height, proc_width)

    return build_grid_capacities(small_image, small_scribbles), (proc_width, proc_height)


def main():
    if not is_available():
        print("PyMaxflow is not installed, so there is nothing to compare against.")
        print("Install it with:  pip install -r requirements-fast.txt")
        return 1

    rows = []
    for grid_size in GRID_SIZES:
        for name in sorted(SCRIBBLES):
            capacities, (grid_w, grid_h) = build_case(name, grid_size)
            nodes = grid_w * grid_h

            bk_time, (bk_flow, bk_mask) = time_best(
                lambda: solve_grid_min_cut_fast(*capacities)
            )

            if grid_size <= EDMONDS_KARP_LIMIT:
                ek_time, (ek_flow, ek_mask) = time_best(
                    lambda: solve_grid_min_cut(*capacities)
                )
                agree = bool(np.array_equal(ek_mask, bk_mask)) and abs(ek_flow - bk_flow) < 1e-6
                speedup = ek_time / bk_time if bk_time > 0 else float("inf")
            else:
                ek_time, agree, speedup = None, None, None

            rows.append(
                {
                    "image": name,
                    "grid": f"{grid_w}x{grid_h}",
                    "nodes": nodes,
                    "edmonds_karp_s": ek_time,
                    "boykov_kolmogorov_s": bk_time,
                    "speedup": speedup,
                    "same_result": agree,
                }
            )
            print(f"  done: {name} at {grid_w}x{grid_h}")

    csv_path = os.path.join(HERE, "solver_comparison.csv")
    with open(csv_path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    lines = [
        "| Image | Cut grid | Nodes | Edmonds-Karp (from scratch) | Boykov-Kolmogorov (PyMaxflow) | Speed-up | Same result |",
        "| --- | --- | ---: | ---: | ---: | ---: | :---: |",
    ]
    for row in rows:
        ek = f"{row['edmonds_karp_s']:.2f}s" if row["edmonds_karp_s"] is not None else "not run"
        speedup = f"{row['speedup']:.0f}x" if row["speedup"] is not None else "n/a"
        same = "yes" if row["same_result"] else ("n/a" if row["same_result"] is None else "NO")
        lines.append(
            f"| {row['image']} | {row['grid']} | {row['nodes']:,} | {ek} "
            f"| {row['boykov_kolmogorov_s']:.3f}s | {speedup} | {same} |"
        )
    # Second table: what this means end to end, where the cut is only one
    # stage alongside graph construction and full-resolution refinement.
    end_to_end = [
        "",
        "End-to-end `segment_image()` on 800px inputs, worst of the four photos:",
        "",
        "| Mode | Cut grid cap | Slowest photo |",
        "| --- | ---: | ---: |",
    ]
    for label, grid_cap, fast in (
        ("Edmonds-Karp (default)", DEFAULT_PROCESSING_MAX_DIM, False),
        ("Boykov-Kolmogorov", DEFAULT_PROCESSING_MAX_DIM, True),
        ("Boykov-Kolmogorov", 256, True),
        ("Boykov-Kolmogorov", 400, True),
    ):
        slowest = 0.0
        for name in sorted(SCRIBBLES):
            image = load(name, 800)
            scribbles = paint_scribbles(image.shape[:2], SCRIBBLES[name])
            elapsed, _ = time_best(
                lambda: segment_image(
                    image, scribbles, processing_max_dim=grid_cap, fast=fast
                )
            )
            slowest = max(slowest, elapsed)
        end_to_end.append(f"| {label} | {grid_cap}px | {slowest:.2f}s |")
        print(f"  end-to-end: {label} @ {grid_cap}px -> {slowest:.2f}s")

    markdown = "\n".join(lines + end_to_end) + "\n"

    md_path = os.path.join(HERE, "solver_comparison.md")
    with open(md_path, "w") as handle:
        handle.write(markdown)

    print()
    print(markdown)
    print(f"wrote {md_path}")
    print(f"wrote {csv_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
