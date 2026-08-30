"""Tests for the command-line interface."""

import numpy as np
import pytest
from PIL import Image

from maxflow.graph_builder import BACKGROUND, FOREGROUND, UNMARKED
from segmentation.cli import (
    main,
    paint_rectangles,
    rectangle,
    scribbles_from_image,
)


@pytest.fixture
def photo(tmp_path):
    """A 120x90 image: bright block on the left, dark on the right."""
    array = np.zeros((90, 120, 3), dtype=np.uint8)
    array[:, :60] = 225
    array[:, 60:] = 25
    path = tmp_path / "photo.png"
    Image.fromarray(array).save(path)
    return path


def test_rectangle_parsing():
    assert rectangle("10,20,30,40") == (10, 20, 30, 40)


@pytest.mark.parametrize("bad", ["1,2,3", "a,b,c,d", "1,2,0,4", "1,2,3,-4"])
def test_rectangle_rejects_bad_input(bad):
    import argparse

    with pytest.raises(argparse.ArgumentTypeError):
        rectangle(bad)


def test_paint_rectangles_clips_to_the_image():
    """Rectangles hanging off the edge should clip, not wrap or raise."""
    scribbles = np.full((10, 10), UNMARKED, dtype=np.uint8)
    paint_rectangles(scribbles, [(-5, -5, 8, 8), (8, 8, 20, 20)], FOREGROUND)

    assert scribbles[0, 0] == FOREGROUND
    assert scribbles[9, 9] == FOREGROUND
    assert scribbles[5, 5] == UNMARKED


def test_scribbles_from_image_reads_red_and_blue(tmp_path):
    strokes = np.zeros((20, 20, 3), dtype=np.uint8)
    strokes[2:5, 2:5] = (255, 0, 0)
    strokes[15:18, 15:18] = (0, 0, 255)
    path = tmp_path / "strokes.png"
    Image.fromarray(strokes).save(path)

    scribbles = scribbles_from_image(path, (20, 20))

    assert (scribbles[2:5, 2:5] == FOREGROUND).all()
    assert (scribbles[15:18, 15:18] == BACKGROUND).all()
    assert scribbles[10, 10] == UNMARKED


def test_end_to_end_writes_a_result(photo, tmp_path, capsys):
    out = tmp_path / "out.png"
    mask_path = tmp_path / "mask.png"

    code = main([
        str(photo), str(out),
        "--fg", "5,40,20,10",
        "--bg", "95,40,20,10",
        "--mask", str(mask_path),
    ])

    assert code == 0
    assert out.exists() and mask_path.exists()

    result = np.asarray(Image.open(out).convert("RGB"))
    assert result.shape == (90, 120, 3), "result must keep the input's resolution"

    mask = np.asarray(Image.open(mask_path).convert("L")) > 127
    assert mask[:, :60].mean() > 0.9, "bright half should be foreground"
    assert mask[:, 60:].mean() < 0.1, "dark half should be background"

    assert "segmented 120x90" in capsys.readouterr().out


def test_transparent_background_produces_rgba(photo, tmp_path):
    out = tmp_path / "out.png"
    main([
        str(photo), str(out),
        "--fg", "5,40,20,10", "--bg", "95,40,20,10",
        "--background", "transparent",
    ])

    assert Image.open(out).mode == "RGBA"


def test_scribble_image_path(photo, tmp_path):
    strokes = np.zeros((90, 120, 3), dtype=np.uint8)
    strokes[40:50, 5:25] = (255, 0, 0)
    strokes[40:50, 95:115] = (0, 0, 255)
    strokes_path = tmp_path / "strokes.png"
    Image.fromarray(strokes).save(strokes_path)

    out = tmp_path / "out.png"
    assert main([str(photo), str(out), "--scribbles", str(strokes_path)]) == 0
    assert out.exists()


def test_missing_scribbles_is_an_error(photo, tmp_path):
    with pytest.raises(SystemExit) as excinfo:
        main([str(photo), str(tmp_path / "out.png")])
    assert excinfo.value.code == 2


def test_rectangles_entirely_outside_the_image_are_reported(photo, tmp_path, capsys):
    code = main([
        str(photo), str(tmp_path / "out.png"),
        "--fg", "9000,9000,10,10",
        "--bg", "5,40,20,10",
    ])

    assert code == 2
    assert "foreground and one background" in capsys.readouterr().err
