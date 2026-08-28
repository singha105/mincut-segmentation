"""Unit tests for canvas_utils, using fake fabric.js path objects."""

from canvas_utils import BACKGROUND_COLOR, FOREGROUND_COLOR, rasterize_scribbles
from maxflow.graph_builder import BACKGROUND, FOREGROUND, UNMARKED


def _path_object(color, points, stroke_width=4):
    path = [["M", points[0][0], points[0][1]]]
    for x, y in points[1:]:
        path.append(["L", x, y])
    return {"type": "path", "stroke": color, "strokeWidth": stroke_width, "path": path}


def test_rasterize_marks_foreground_stroke():
    objects = [_path_object(FOREGROUND_COLOR, [(2, 2), (2, 6)], stroke_width=2)]
    mask = rasterize_scribbles(objects, mask_shape=(10, 10), scale_x=1.0, scale_y=1.0)
    assert mask[4, 2] == FOREGROUND


def test_rasterize_marks_background_stroke():
    objects = [_path_object(BACKGROUND_COLOR, [(8, 2), (8, 6)], stroke_width=2)]
    mask = rasterize_scribbles(objects, mask_shape=(10, 10), scale_x=1.0, scale_y=1.0)
    assert mask[4, 8] == BACKGROUND


def test_rasterize_ignores_unmarked_regions():
    objects = [_path_object(FOREGROUND_COLOR, [(0, 0), (0, 2)], stroke_width=1)]
    mask = rasterize_scribbles(objects, mask_shape=(10, 10), scale_x=1.0, scale_y=1.0)
    assert mask[9, 9] == UNMARKED


def test_rasterize_scales_coordinates_down():
    """A stroke drawn at 4x display resolution should land at 1/4 the coordinates."""
    objects = [_path_object(FOREGROUND_COLOR, [(20, 20), (20, 24)], stroke_width=4)]
    mask = rasterize_scribbles(objects, mask_shape=(10, 10), scale_x=0.25, scale_y=0.25)
    assert mask[5, 5] == FOREGROUND
