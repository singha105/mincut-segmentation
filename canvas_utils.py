"""
Convert streamlit-drawable-canvas scribble strokes into a label mask.

streamlit_drawable_canvas returns each freehand stroke as a fabric.js
"path" object: {"stroke": "<hex color>", "strokeWidth": <int>, "path": [...]}.
Each entry in `path` is a drawing command like ["M", x, y] (move) or
["Q", cx, cy, x, y] (quadratic curve); in every case the last two numbers
are that segment's endpoint. This module walks those endpoints, scales them
from canvas display coordinates down to the (smaller) processing-resolution
image, and rasterizes straight-line segments between consecutive endpoints
into a maxflow.graph_builder-compatible label mask.
"""

from typing import Dict, List, Sequence

import numpy as np
from PIL import Image, ImageDraw

from maxflow.graph_builder import BACKGROUND, FOREGROUND, UNMARKED

FOREGROUND_COLOR = "#ff0000"
BACKGROUND_COLOR = "#0000ff"


def _color_to_label(stroke_color: str) -> int:
    """Map a stroke's hex color to a FOREGROUND/BACKGROUND label."""
    color = stroke_color.lower()
    if color == FOREGROUND_COLOR:
        return FOREGROUND
    if color == BACKGROUND_COLOR:
        return BACKGROUND
    raise ValueError(f"Unrecognized stroke color: {stroke_color}")


def _path_endpoints(path_commands: Sequence[Sequence[float]]) -> List[Sequence[float]]:
    """Extract each drawing command's (x, y) endpoint, in canvas coordinates."""
    return [command[-2:] for command in path_commands if len(command) >= 3]


def rasterize_scribbles(
    objects: List[Dict],
    mask_shape: Sequence[int],
    scale_x: float,
    scale_y: float,
) -> np.ndarray:
    """Rasterize canvas stroke `objects` into a label mask of shape `mask_shape`.

    `scale_x`/`scale_y` convert from canvas display coordinates down to the
    mask's (smaller) processing resolution. Later strokes are drawn on top
    of earlier ones, matching what the user sees on screen.
    """
    height, width = mask_shape
    mask_image = Image.new("L", (width, height), UNMARKED)
    draw = ImageDraw.Draw(mask_image)

    for obj in objects:
        if obj.get("type") != "path":
            continue
        label = _color_to_label(obj["stroke"])
        stroke_width = max(1, round(obj.get("strokeWidth", 1) * (scale_x + scale_y) / 2))
        points = [(x * scale_x, y * scale_y) for x, y in _path_endpoints(obj["path"])]

        if len(points) == 1:
            x, y = points[0]
            radius = stroke_width / 2
            draw.ellipse([x - radius, y - radius, x + radius, y + radius], fill=label)
        else:
            draw.line(points, fill=label, width=stroke_width)

    return np.array(mask_image, dtype=np.uint8)
