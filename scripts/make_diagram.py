"""Generate assets/how-it-works.png, the graph diagram used in the README.

Draws the Boykov-Jolly construction: a pixel grid wired to two terminals,
with N-links between neighbours, T-links to the terminals, and the min-cut
running through the cheap edges.

    python scripts/make_diagram.py
"""

import os

from PIL import Image, ImageDraw, ImageFont

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_PATH = os.path.join(REPO_ROOT, "assets", "how-it-works.png")

SCALE = 2  # render at 2x then downsample, so the lines come out smooth
W, H = 900 * SCALE, 600 * SCALE

BG = (24, 24, 27)
TEXT = (232, 232, 235)
MUTED = (150, 152, 158)
RED = (239, 68, 68)
BLUE = (59, 130, 246)
NLINK_STRONG = (200, 202, 208)
NLINK_WEAK = (86, 88, 96)
CUT = (250, 204, 21)

GRID_COLS, GRID_ROWS = 5, 3
NODE_R = 15 * SCALE

# Which pixels are "really" foreground -- the left of the image. The N-link
# between columns 2 and 3 is the weak one (an image edge), so the cut goes there.
FOREGROUND_COLS = {0, 1, 2}
WEAK_BOUNDARY_AFTER_COL = 2


def font(size):
    for candidate in (
        "/System/Library/Fonts/Supplemental/Arial.ttf",
        "/System/Library/Fonts/Helvetica.ttc",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    ):
        if os.path.exists(candidate):
            try:
                return ImageFont.truetype(candidate, size * SCALE)
            except OSError:
                continue
    return ImageFont.load_default()


def centered(draw, xy, text, fill, fnt):
    left, top, right, bottom = draw.textbbox((0, 0), text, font=fnt)
    draw.text((xy[0] - (right - left) / 2, xy[1] - (bottom - top) / 2), text, fill=fill, font=fnt)


def main():
    image = Image.new("RGB", (W, H), BG)
    draw = ImageDraw.Draw(image)

    title_font = font(19)
    label_font = font(14)
    small_font = font(12)

    grid_left = 250 * SCALE
    grid_top = 205 * SCALE
    step_x = 88 * SCALE
    step_y = 82 * SCALE

    def node_xy(col, row):
        return grid_left + col * step_x, grid_top + row * step_y

    source_xy = (W // 2, 96 * SCALE)
    sink_xy = (W // 2, H - 86 * SCALE)

    # ---- T-links (drawn first, so they sit behind everything) ----
    for col in range(GRID_COLS):
        for row in range(GRID_ROWS):
            x, y = node_xy(col, row)
            draw.line([source_xy, (x, y)], fill=(70, 42, 46), width=max(1, SCALE))
            draw.line([(x, y), sink_xy], fill=(38, 52, 78), width=max(1, SCALE))

    # ---- N-links ----
    for row in range(GRID_ROWS):
        for col in range(GRID_COLS - 1):
            a, b = node_xy(col, row), node_xy(col + 1, row)
            weak = col == WEAK_BOUNDARY_AFTER_COL
            draw.line(
                [a, b],
                fill=NLINK_WEAK if weak else NLINK_STRONG,
                width=(2 if weak else 6) * SCALE,
            )
    for col in range(GRID_COLS):
        for row in range(GRID_ROWS - 1):
            a, b = node_xy(col, row), node_xy(col, row + 1)
            draw.line([a, b], fill=NLINK_STRONG, width=6 * SCALE)

    # ---- the cut ----
    cut_x = grid_left + (WEAK_BOUNDARY_AFTER_COL + 0.5) * step_x
    dash_top, dash_bottom = 150 * SCALE, H - 132 * SCALE
    y = dash_top
    while y < dash_bottom:
        draw.line([(cut_x, y), (cut_x, min(y + 14 * SCALE, dash_bottom))], fill=CUT, width=3 * SCALE)
        y += 24 * SCALE
    centered(draw, (cut_x + 66 * SCALE, dash_top - 14 * SCALE), "min-cut", CUT, label_font)

    # ---- pixel nodes ----
    for col in range(GRID_COLS):
        for row in range(GRID_ROWS):
            x, y = node_xy(col, row)
            fill = (168, 74, 74) if col in FOREGROUND_COLS else (64, 84, 122)
            draw.ellipse(
                [x - NODE_R, y - NODE_R, x + NODE_R, y + NODE_R],
                fill=fill, outline=(220, 220, 226), width=max(1, SCALE),
            )

    # ---- terminals ----
    term_w, term_h = 62 * SCALE, 40 * SCALE
    for xy, color, letter, caption in (
        (source_xy, RED, "S", "source = foreground"),
        (sink_xy, BLUE, "T", "sink = background"),
    ):
        draw.rounded_rectangle(
            [xy[0] - term_w, xy[1] - term_h, xy[0] + term_w, xy[1] + term_h],
            radius=12 * SCALE, fill=color,
        )
        centered(draw, xy, letter, (255, 255, 255), title_font)
        # Caption sits beside the box: above/below would collide with the
        # title and run off the bottom edge.
        draw.text(
            (xy[0] + term_w + 16 * SCALE, xy[1] - 9 * SCALE), caption, fill=MUTED, font=label_font
        )

    # ---- annotations ----
    draw.text(
        (30 * SCALE, 150 * SCALE),
        "T-links\nhow well each pixel\nmatches the scribbled\ncolour models",
        fill=MUTED, font=small_font, spacing=6 * SCALE,
    )
    draw.text(
        (30 * SCALE, 330 * SCALE),
        "N-links\nthick = similar neighbours\n(expensive to cut)\nthin = an image edge\n(cheap to cut)",
        fill=MUTED, font=small_font, spacing=6 * SCALE,
    )
    draw.text(
        (W - 250 * SCALE, 330 * SCALE),
        "The cheapest cut\nseparating S from T\nfollows the image edge\n-> the object boundary",
        fill=MUTED, font=small_font, spacing=6 * SCALE,
    )

    centered(
        draw, (W // 2, 22 * SCALE),
        "Every pixel is a node between two terminals", TEXT, title_font,
    )

    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    image.resize((W // SCALE, H // SCALE), Image.Resampling.LANCZOS).save(OUT_PATH)
    print("wrote", OUT_PATH)


if __name__ == "__main__":
    main()
