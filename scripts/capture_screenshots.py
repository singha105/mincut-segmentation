"""Drive the Streamlit app headlessly with Playwright and capture screenshots.

Starts the app, uploads a sample photo, draws real foreground/background
strokes on the canvas, runs the segmentation, and saves screenshots to
``screenshots/``. Also stitches the captured frames into a demo GIF.

    python scripts/capture_screenshots.py

Needs the dev extras:

    pip install -r requirements-dev.txt
    python -m playwright install chromium

The strokes are drawn with Playwright's real mouse events rather than
synthetic DOM events, because the drawable canvas is fabric.js-based and
ignores untrusted events.
"""

import argparse
import os
import shutil
import socket
import subprocess
import sys
import time

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCREENSHOT_DIR = os.path.join(REPO_ROOT, "screenshots")
FRAME_DIR = os.path.join(SCREENSHOT_DIR, "_frames")
SAMPLE_IMAGE = os.path.join(REPO_ROOT, "sample_images", "horse.jpg")

VIEWPORT = {"width": 1000, "height": 1000}
PORT = 8899


def wait_for_port(port, timeout=90):
    deadline = time.time() + timeout
    while time.time() < deadline:
        with socket.socket() as sock:
            sock.settimeout(1)
            if sock.connect_ex(("127.0.0.1", port)) == 0:
                return True
        time.sleep(0.5)
    return False


def start_app(port):
    """Launch Streamlit headlessly, with usage stats and the welcome prompt off."""
    env = dict(os.environ, STREAMLIT_BROWSER_GATHER_USAGE_STATS="false")
    process = subprocess.Popen(
        [
            sys.executable, "-m", "streamlit", "run", os.path.join(REPO_ROOT, "app.py"),
            "--server.port", str(port),
            "--server.headless", "true",
            "--browser.gatherUsageStats", "false",
        ],
        cwd=REPO_ROOT,
        env=env,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    if not wait_for_port(port):
        process.terminate()
        raise RuntimeError(f"Streamlit did not start on port {port}")
    return process


def settle(page, seconds=1.2):
    """Let Streamlit finish its rerun and repaint."""
    page.wait_for_timeout(int(seconds * 1000))


def canvas_box(page):
    """Page-relative bounding box of the drawable canvas.

    The canvas lives in a Streamlit component iframe, and fabric.js stacks
    several canvases in the same place -- some of which report as hidden. They
    all share one rectangle, so take the largest box any of them reports rather
    than insisting on a particular element being "visible".
    """
    for frame in page.frames:
        if "st_canvas" not in (frame.url or ""):
            continue
        frame.locator("canvas").first.wait_for(state="attached", timeout=20000)

        best = None
        for index in range(frame.locator("canvas").count()):
            box = frame.locator("canvas").nth(index).bounding_box()
            if box and box["width"] > 50 and box["height"] > 50:
                if best is None or box["width"] * box["height"] > best["width"] * best["height"]:
                    best = box
        if best:
            return best
    raise RuntimeError("drawable canvas not found")


def draw_stroke(page, box, points, frames=None, page_for_frame=None):
    """Draw one freehand stroke through `points` given as (x, y) fractions."""
    x0, y0 = points[0]
    page.mouse.move(box["x"] + box["width"] * x0, box["y"] + box["height"] * y0)
    page.mouse.down()
    for fx, fy in points[1:]:
        page.mouse.move(
            box["x"] + box["width"] * fx, box["y"] + box["height"] * fy, steps=12
        )
        if frames is not None:
            frames.append(capture_frame(page_for_frame or page, len(frames)))
    page.mouse.up()
    page.wait_for_timeout(600)


def capture_frame(page, index):
    os.makedirs(FRAME_DIR, exist_ok=True)
    path = os.path.join(FRAME_DIR, f"frame_{index:03d}.png")
    page.screenshot(path=path)
    return path


def trim(path, margin=24):
    """Crop uniform borders so the screenshot is mostly content.

    A fixed 1000x1000 viewport leaves a lot of empty page below short screens;
    trimming keeps the README images readable at GitHub's display width.
    """
    from PIL import Image, ImageChops

    image = Image.open(path).convert("RGB")
    background = Image.new("RGB", image.size, image.getpixel((image.width - 2, image.height - 2)))
    bbox = ImageChops.difference(image, background).convert("L").point(lambda v: 255 if v > 8 else 0).getbbox()
    if not bbox:
        return

    left, top, right, bottom = bbox
    cropped = image.crop(
        (
            max(0, left - margin),
            max(0, top - margin),
            min(image.width, right + margin),
            min(image.height, bottom + margin),
        )
    )
    cropped.save(path)


def select_radio(page, label):
    page.get_by_text(label, exact=True).first.click()
    settle(page)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=PORT)
    parser.add_argument(
        "--keep-frames", action="store_true", help="keep the intermediate GIF frames"
    )
    args = parser.parse_args()

    from playwright.sync_api import sync_playwright

    os.makedirs(SCREENSHOT_DIR, exist_ok=True)
    shutil.rmtree(FRAME_DIR, ignore_errors=True)

    print("starting Streamlit...")
    app = start_app(args.port)
    frames = []

    try:
        with sync_playwright() as playwright:
            browser = playwright.chromium.launch()
            page = browser.new_page(viewport=VIEWPORT, device_scale_factor=2)
            page.goto(f"http://127.0.0.1:{args.port}", wait_until="networkidle")
            settle(page, 2.0)

            # (1) empty upload screen
            page.screenshot(path=os.path.join(SCREENSHOT_DIR, "01-upload.png"))
            trim(os.path.join(SCREENSHOT_DIR, "01-upload.png"))
            print("captured 01-upload.png")
            frames.append(capture_frame(page, len(frames)))

            # upload a sample photo through the real file input
            page.locator('input[type="file"]').set_input_files(SAMPLE_IMAGE)
            settle(page, 3.0)
            frames.append(capture_frame(page, len(frames)))

            box = canvas_box(page)

            # foreground stroke along the horse's back
            draw_stroke(
                page, box,
                [(0.40, 0.30), (0.55, 0.31), (0.70, 0.34)],
                frames=frames, page_for_frame=page,
            )

            # background strokes: the dark trees, then the grass
            select_radio(page, "Background (blue)")
            draw_stroke(
                page, box,
                [(0.08, 0.07), (0.30, 0.06), (0.50, 0.07)],
                frames=frames, page_for_frame=page,
            )
            draw_stroke(
                page, box,
                [(0.10, 0.93), (0.45, 0.94), (0.75, 0.93)],
                frames=frames, page_for_frame=page,
            )

            # (2) scribbles drawn
            page.screenshot(path=os.path.join(SCREENSHOT_DIR, "02-scribbles.png"))
            trim(os.path.join(SCREENSHOT_DIR, "02-scribbles.png"))
            print("captured 02-scribbles.png")
            frames.append(capture_frame(page, len(frames)))

            # run the segmentation
            page.get_by_role("button", name="Segment").click()
            page.wait_for_selector("text=Segmentation took", timeout=60000)
            settle(page, 1.5)

            # (3) result with timing -- scroll so the three panels are in view
            page.get_by_text("Segmented result").first.scroll_into_view_if_needed()
            settle(page, 0.8)
            page.screenshot(path=os.path.join(SCREENSHOT_DIR, "03-result.png"))
            trim(os.path.join(SCREENSHOT_DIR, "03-result.png"))
            print("captured 03-result.png")
            for _ in range(4):  # hold the final frame in the GIF
                frames.append(capture_frame(page, len(frames)))

            browser.close()
    finally:
        app.terminate()
        app.wait(timeout=15)

    build_gif(frames)
    if not args.keep_frames:
        shutil.rmtree(FRAME_DIR, ignore_errors=True)
    print("done")


def build_gif(frame_paths):
    """Stitch the captured frames into screenshots/demo.gif."""
    import imageio.v2 as imageio
    import numpy as np
    from PIL import Image

    if not frame_paths:
        print("no frames captured; skipping GIF")
        return

    # Full-size retina frames make a huge GIF, so scale down to a sane width.
    target_width = 720
    images = []
    for path in frame_paths:
        image = Image.open(path).convert("RGB")
        height = round(target_width * image.height / image.width)
        images.append(np.asarray(image.resize((target_width, height))))

    gif_path = os.path.join(SCREENSHOT_DIR, "demo.gif")
    imageio.mimsave(gif_path, images, duration=0.7, loop=0)
    size_mb = os.path.getsize(gif_path) / 1e6
    print(f"wrote {gif_path} ({len(images)} frames, {size_mb:.1f} MB)")


if __name__ == "__main__":
    main()
