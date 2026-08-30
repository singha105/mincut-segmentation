"""Command-line interface: segment an image without the Streamlit UI.

Scribbles can be given either as rectangles on the command line, or as a
scribble image using the same red/blue convention the app uses.

    python -m segmentation.cli sample_images/horse.jpg out.png \\
        --fg 350,300,250,20 --bg 60,40,340,20 --bg 100,740,500,20

    python -m segmentation.cli photo.jpg out.png --scribbles strokes.png

Run with --help for the full set of options.
"""

import argparse
import sys
import time

import numpy as np
from PIL import Image

from maxflow.graph_builder import BACKGROUND, FOREGROUND, UNMARKED
from segmentation.segmenter import (
    DEFAULT_PROCESSING_MAX_DIM,
    apply_mask,
    segment_image,
)

# A pixel counts as a stroke when one channel clearly dominates, which is how
# the app's red/blue scribbles come out after any resampling.
CHANNEL_DOMINANCE = 40


def rectangle(value):
    """Parse an ``x,y,width,height`` argument."""
    try:
        x, y, width, height = (int(part) for part in value.split(","))
    except ValueError:
        raise argparse.ArgumentTypeError(
            f"expected x,y,width,height (four integers), got {value!r}"
        )
    if width <= 0 or height <= 0:
        raise argparse.ArgumentTypeError(f"width and height must be positive, got {value!r}")
    return x, y, width, height


def paint_rectangles(scribbles, rectangles, label):
    """Mark `rectangles` with `label`, clipped to the image."""
    height, width = scribbles.shape
    for x, y, w, h in rectangles:
        x0, y0 = max(0, x), max(0, y)
        x1, y1 = min(width, x + w), min(height, y + h)
        if x1 > x0 and y1 > y0:
            scribbles[y0:y1, x0:x1] = label


def scribbles_from_image(path, shape):
    """Read a red/blue scribble image into a label mask."""
    strokes = Image.open(path).convert("RGB")
    if strokes.size != (shape[1], shape[0]):
        strokes = strokes.resize((shape[1], shape[0]), Image.Resampling.NEAREST)
    pixels = np.asarray(strokes).astype(int)

    red, green, blue = pixels[..., 0], pixels[..., 1], pixels[..., 2]
    scribbles = np.full(shape, UNMARKED, dtype=np.uint8)
    scribbles[(blue - red > CHANNEL_DOMINANCE) & (blue - green > CHANNEL_DOMINANCE)] = BACKGROUND
    scribbles[(red - blue > CHANNEL_DOMINANCE) & (red - green > CHANNEL_DOMINANCE)] = FOREGROUND
    return scribbles


def build_parser():
    parser = argparse.ArgumentParser(
        prog="python -m segmentation.cli",
        description="Segment an image with graph min-cut, using scribbles as hints.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "examples:\n"
            "  python -m segmentation.cli photo.jpg out.png --fg 350,300,250,20 --bg 60,40,340,20\n"
            "  python -m segmentation.cli photo.jpg out.png --scribbles strokes.png --background transparent\n"
        ),
    )
    parser.add_argument("input", help="image to segment")
    parser.add_argument("output", help="where to write the result")
    parser.add_argument(
        "--fg", type=rectangle, action="append", default=[], metavar="X,Y,W,H",
        help="foreground rectangle; repeatable",
    )
    parser.add_argument(
        "--bg", type=rectangle, action="append", default=[], metavar="X,Y,W,H",
        help="background rectangle; repeatable",
    )
    parser.add_argument(
        "--scribbles", metavar="PATH",
        help="scribble image instead of rectangles (red = foreground, blue = background)",
    )
    parser.add_argument(
        "--background", choices=("black", "dim", "transparent"), default="black",
        help="how to render the background in the result (default: black)",
    )
    parser.add_argument(
        "--grid", type=int, default=DEFAULT_PROCESSING_MAX_DIM, metavar="N",
        help=f"cut-grid cap on the longest side (default: {DEFAULT_PROCESSING_MAX_DIM})",
    )
    parser.add_argument(
        "--fast", action="store_true",
        help="use the Boykov-Kolmogorov solver (needs requirements-fast.txt)",
    )
    parser.add_argument(
        "--mask", metavar="PATH", help="also write the raw binary mask here",
    )
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)

    if not args.scribbles and not (args.fg and args.bg):
        build_parser().error(
            "need scribbles: pass --scribbles, or at least one --fg and one --bg rectangle"
        )

    image = np.asarray(Image.open(args.input).convert("RGB"))
    shape = image.shape[:2]

    if args.scribbles:
        scribbles = scribbles_from_image(args.scribbles, shape)
    else:
        scribbles = np.full(shape, UNMARKED, dtype=np.uint8)
        paint_rectangles(scribbles, args.bg, BACKGROUND)
        paint_rectangles(scribbles, args.fg, FOREGROUND)

    if not (scribbles == FOREGROUND).any() or not (scribbles == BACKGROUND).any():
        print(
            "error: need at least one foreground and one background pixel "
            "(check the rectangles fall inside the image)",
            file=sys.stderr,
        )
        return 2

    start = time.perf_counter()
    mask = segment_image(image, scribbles, processing_max_dim=args.grid, fast=args.fast)
    elapsed = time.perf_counter() - start

    Image.fromarray(apply_mask(image, mask, background=args.background)).save(args.output)
    if args.mask:
        Image.fromarray((mask * 255).astype(np.uint8)).save(args.mask)

    height, width = shape
    print(
        f"segmented {width}x{height} in {elapsed:.2f}s "
        f"({mask.mean() * 100:.1f}% foreground) -> {args.output}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
