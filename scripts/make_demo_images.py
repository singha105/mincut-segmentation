"""Regenerate assets/demo.png and assets/demo_animals.png at full resolution.

    python scripts/make_demo_images.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import time

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from maxflow.graph_builder import BACKGROUND, FOREGROUND, UNMARKED
from segmentation.segmenter import apply_mask, segment_image

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Scribbles as polylines in normalised (x, y) coordinates, plus a stroke width
# as a fraction of the image's longest side -- i.e. what a user would draw.
SHAPES = [
    {
        "file": "circle_fg_bg.png",
        "fg": [[(0.42, 0.5), (0.58, 0.5)]],
        "bg": [[(0.06, 0.06), (0.2, 0.06)]],
    },
    {
        "file": "rectangle_fg_bg.png",
        "fg": [[(0.72, 0.35), (0.72, 0.6)]],
        "bg": [[(0.1, 0.1), (0.3, 0.1)], [(0.1, 0.9), (0.4, 0.9)]],
    },
    {
        "file": "shapes_fg_bg.png",
        "fg": [[(0.28, 0.5), (0.38, 0.56)]],
        "bg": [[(0.06, 0.08), (0.25, 0.08)], [(0.72, 0.5), (0.8, 0.62)]],
    },
]

ANIMALS = [
    {
        "file": "dog.jpg",
        "fg": [[(0.5, 0.68), (0.62, 0.72)]],
        "bg": [
            [(0.08, 0.2), (0.08, 0.7)],
            [(0.62, 0.35), (0.68, 0.55)],
            [(0.3, 0.06), (0.6, 0.06)],
            [(0.12, 0.94), (0.88, 0.94)],
            # the unlit room behind the dog reads as dark as its shadowed fur
            [(0.40, 0.40), (0.52, 0.44)],
            [(0.62, 0.58), (0.66, 0.68)],
        ],
    },
    {
        "file": "hen.jpg",
        "fg": [[(0.42, 0.4), (0.52, 0.52)]],
        "bg": [
            [(0.08, 0.12), (0.35, 0.1)],
            [(0.85, 0.35), (0.9, 0.6)],
            [(0.2, 0.9), (0.7, 0.92)],
        ],
    },
    {
        "file": "horse.jpg",
        "fg": [[(0.45, 0.3), (0.72, 0.33)]],
        "bg": [
            [(0.08, 0.08), (0.4, 0.06)],
            [(0.1, 0.9), (0.6, 0.92)],
        ],
    },
    {
        "file": "tiger.jpg",
        # the face needs its own stroke: its dark stripes sit in the same
        # tonal range as the water behind it
        "fg": [[(0.2, 0.35), (0.45, 0.5)], [(0.52, 0.52), (0.68, 0.6)]],
        "bg": [
            [(0.75, 0.12), (0.9, 0.2)],
            [(0.2, 0.9), (0.7, 0.88)],
        ],
    },
]


def paint_scribbles(shape, spec):
    """Rasterise the polylines into a full-resolution label mask."""
    height, width = shape
    canvas = Image.new("L", (width, height), UNMARKED)
    draw = ImageDraw.Draw(canvas)
    stroke = max(3, round(max(width, height) * 0.022))
    for label, strokes in ((BACKGROUND, spec["bg"]), (FOREGROUND, spec["fg"])):
        for polyline in strokes:
            points = [(x * width, y * height) for x, y in polyline]
            draw.line(points, fill=int(label), width=stroke, joint="curve")
    return np.array(canvas, dtype=np.uint8)


def overlay_scribbles(image, scribbles):
    out = image.copy()
    out[scribbles == FOREGROUND] = [255, 40, 40]
    out[scribbles == BACKGROUND] = [50, 90, 255]
    return out


def build_sheet(specs, out_name, panel_size):
    font = ImageFont.load_default()
    col_titles = ["Original", "Scribbles", "Segmented result"]
    label_w = 108
    header_h = 26
    pad = 12

    rows = []
    for spec in specs:
        image = np.array(Image.open(f"{ROOT}/sample_images/{spec['file']}").convert("RGB"))
        scribbles = paint_scribbles(image.shape[:2], spec)

        start = time.perf_counter()
        mask = segment_image(image, scribbles)
        elapsed = time.perf_counter() - start
        print(f"  {spec['file']:22s} {image.shape[1]}x{image.shape[0]}  {elapsed:5.2f}s  fg={mask.mean():.2f}")

        panels = [image, overlay_scribbles(image, scribbles), apply_mask(image, mask, "black")]
        rendered = [
            Image.fromarray(p).resize((panel_size, panel_size), Image.Resampling.LANCZOS)
            for p in panels
        ]
        rows.append((spec["file"], rendered))

    canvas_w = label_w + len(col_titles) * panel_size + (len(col_titles) + 1) * pad
    canvas_h = header_h + len(rows) * (panel_size + pad) + pad
    sheet = Image.new("RGB", (canvas_w, canvas_h), (24, 24, 27))
    draw = ImageDraw.Draw(sheet)

    for col, title in enumerate(col_titles):
        x = label_w + pad + col * (panel_size + pad)
        draw.text((x, 8), title, fill=(232, 232, 232), font=font)

    for row_idx, (name, panels) in enumerate(rows):
        y = header_h + pad + row_idx * (panel_size + pad)
        draw.text((10, y + panel_size // 2 - 6), name, fill=(232, 232, 232), font=font)
        for col_idx, panel in enumerate(panels):
            sheet.paste(panel, (label_w + pad + col_idx * (panel_size + pad), y))

    path = f"{ROOT}/assets/{out_name}"
    sheet.save(path)
    print(f"saved {path} {sheet.size}\n")


print("synthetic shapes:")
build_sheet(SHAPES, "demo.png", 200)
print("real photos:")
build_sheet(ANIMALS, "demo_animals.png", 220)
