"""Measure end-to-end segmentation time on realistic photos.

Reports the *input* resolution (what the caller passes in and gets back) and
the internal cut-grid resolution separately, since they are deliberately
different -- see the README's "Output quality" section.

Run from the repo root:

    python benchmarks/measure_timing.py
"""

import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from PIL import Image, ImageDraw

from maxflow.graph_builder import BACKGROUND, FOREGROUND, UNMARKED
from segmentation.segmenter import DEFAULT_PROCESSING_MAX_DIM, segment_image

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REPEATS = 3

# One foreground stroke on the subject, two background strokes off it --
# roughly what a user draws.
SCRIBBLES = {
    "dog.jpg": {
        "fg": [[(0.5, 0.68), (0.62, 0.72)]],
        "bg": [[(0.08, 0.2), (0.08, 0.7)], [(0.3, 0.06), (0.6, 0.06)]],
    },
    "hen.jpg": {
        "fg": [[(0.42, 0.4), (0.52, 0.52)]],
        "bg": [[(0.08, 0.12), (0.35, 0.1)], [(0.2, 0.9), (0.7, 0.92)]],
    },
    "horse.jpg": {
        "fg": [[(0.45, 0.3), (0.72, 0.33)]],
        "bg": [[(0.08, 0.08), (0.4, 0.06)], [(0.1, 0.9), (0.6, 0.92)]],
    },
    "tiger.jpg": {
        "fg": [[(0.2, 0.35), (0.45, 0.5)]],
        "bg": [[(0.75, 0.12), (0.9, 0.2)], [(0.2, 0.9), (0.7, 0.88)]],
    },
}


def paint_scribbles(shape, spec):
    """Rasterise normalised polyline strokes into a full-resolution label mask."""
    height, width = shape
    canvas = Image.new("L", (width, height), UNMARKED)
    draw = ImageDraw.Draw(canvas)
    stroke = max(3, round(max(width, height) * 0.022))
    for label, strokes in ((BACKGROUND, spec["bg"]), (FOREGROUND, spec["fg"])):
        for polyline in strokes:
            draw.line(
                [(x * width, y * height) for x, y in polyline],
                fill=int(label),
                width=stroke,
                joint="curve",
            )
    return np.array(canvas, dtype=np.uint8)


def load(name, longest_side=None):
    image = Image.open(os.path.join(REPO_ROOT, "sample_images", name)).convert("RGB")
    if longest_side is not None:
        scale = longest_side / max(image.size)
        image = image.resize((round(image.width * scale), round(image.height * scale)))
    return np.array(image)


def time_call(fn, repeats=REPEATS):
    """Best-of-N wall time, to reduce noise from other activity on the machine."""
    best = float("inf")
    for _ in range(repeats):
        start = time.perf_counter()
        fn()
        best = min(best, time.perf_counter() - start)
    return best


def main():
    print(f"cut grid capped at {DEFAULT_PROCESSING_MAX_DIM}px on the longest side")
    print(f"best of {REPEATS} runs\n")
    print(f"{'image':<12} {'input resolution':<18} {'megapixels':>10} {'time (s)':>10}")
    print("-" * 54)

    worst = 0.0
    for name in sorted(SCRIBBLES):
        for longest_side in (400, 600, 800):
            image = load(name, longest_side)
            scribbles = paint_scribbles(image.shape[:2], SCRIBBLES[name])
            height, width = image.shape[:2]

            elapsed = time_call(lambda: segment_image(image, scribbles))
            worst = max(worst, elapsed)

            resolution = f"{width}x{height}"
            megapixels = (width * height) / 1e6
            print(f"{name:<12} {resolution:<18} {megapixels:>10.2f} {elapsed:>10.2f}")

    print("-" * 54)
    print(f"slowest case: {worst:.2f}s")
    print("under 2s target:", "PASS" if worst < 2.0 else "FAIL")


if __name__ == "__main__":
    main()
