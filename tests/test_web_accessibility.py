"""The try-it page has to work without a mouse and tell a screen reader what happened.

axe-core catches markup problems, but it cannot tell whether a canvas is usable
from the keyboard, so the keyboard path is driven for real here.

Needs requirements-dev.txt and a local Chrome. The default test jobs don't
install those, so this module skips there and has its own CI job.
"""

import functools
import http.server
import shutil
import threading
from pathlib import Path

import pytest

pytest.importorskip("playwright")
pytest.importorskip("axe_playwright_python")

from axe_playwright_python.sync_playwright import Axe  # noqa: E402
from playwright.sync_api import sync_playwright  # noqa: E402

REPO = Path(__file__).resolve().parent.parent
SAMPLE = REPO / "sample_images" / "circle_fg_bg.png"
AXE_TAGS = ["wcag2a", "wcag2aa", "wcag21a", "wcag21aa", "wcag22aa", "best-practice"]
SEGMENTED = "document.getElementById('tryStatus').textContent.startsWith('Segmented')"


class _QuietHandler(http.server.SimpleHTTPRequestHandler):
    def log_message(self, *args):
        pass


@pytest.fixture(scope="module")
def site_url(tmp_path_factory):
    # Same layout the Pages workflow publishes: web/ with assets/ inside it.
    site = tmp_path_factory.mktemp("site")
    shutil.copytree(REPO / "web", site, dirs_exist_ok=True)
    shutil.copytree(REPO / "assets", site / "assets", dirs_exist_ok=True)
    handler = functools.partial(_QuietHandler, directory=str(site))
    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{server.server_port}/index.html"
    server.shutdown()


@pytest.fixture(scope="module")
def browser():
    with sync_playwright() as playwright:
        chrome = playwright.chromium.launch(channel="chrome", headless=True)
        yield chrome
        chrome.close()


@pytest.fixture
def page(browser, site_url):
    context = browser.new_context(viewport={"width": 1280, "height": 800})
    tab = context.new_page()
    # Web fonts are decoration; don't let a test depend on reaching Google.
    tab.route("**/*", lambda route: route.abort() if "fonts.g" in route.request.url else route.continue_())
    tab.goto(site_url)
    yield tab
    context.close()


def _violations(page):
    result = Axe().run(page, options={"runOnly": {"type": "tag", "values": AXE_TAGS}})
    return [(v["id"], [n["target"] for n in v["nodes"]]) for v in result.response["violations"]]


def _upload(page):
    page.set_input_files("#tryFileInput", str(SAMPLE))
    page.wait_for_selector("#tryCanvas", state="visible")


def _active_id(page):
    return page.evaluate("document.activeElement.id")


def _tab_to(page, element_id, limit=15):
    for _ in range(limit):
        page.keyboard.press("Tab")
        if _active_id(page) == element_id:
            return
    pytest.fail(f"Tab never reached #{element_id}")


def _press(page, key, times=1):
    for _ in range(times):
        page.keyboard.press(key)


def _scribble_colours(page):
    """Count pure red and pure blue pixels in the results' scribble panel."""
    return page.evaluate(
        """() => {
            const c = document.getElementById('scribbleResultCanvas');
            const d = c.getContext('2d').getImageData(0, 0, c.width, c.height).data;
            let red = 0, blue = 0;
            for (let i = 0; i < d.length; i += 4) {
                if (d[i] > 200 && d[i + 1] < 60 && d[i + 2] < 60) red++;
                if (d[i + 2] > 200 && d[i] < 60 && d[i + 1] < 60) blue++;
            }
            return [red, blue];
        }"""
    )


def test_no_axe_violations_on_load(page):
    assert _violations(page) == []


def test_no_axe_violations_with_the_tool_open(page):
    _upload(page)
    assert _violations(page) == []


def test_whole_flow_works_from_the_keyboard_alone(page):
    _upload(page)
    page.focus("#tryFileInput")
    _tab_to(page, "tryCanvas")

    # Foreground stroke through the middle of the image.
    page.keyboard.press("Space")
    _press(page, "Shift+ArrowRight", 2)
    page.keyboard.press("Space")

    # Background stroke along the top-left corner.
    page.keyboard.press("b")
    _press(page, "Shift+ArrowUp", 20)
    _press(page, "Shift+ArrowLeft", 20)
    page.keyboard.press("Space")
    _press(page, "Shift+ArrowRight", 3)
    page.keyboard.press("Space")

    _tab_to(page, "segmentBtn")
    page.keyboard.press("Enter")
    page.wait_for_function(SEGMENTED, timeout=60000)

    assert _active_id(page) == "segmentBtn", "focus must survive the solve"
    assert page.is_visible("#tryResults")
    red, blue = _scribble_colours(page)
    assert red > 0 and blue > 0, "keyboard strokes must reach the solver"
    assert _violations(page) == []


def test_keyboard_cursor_never_becomes_a_scribble(page):
    _upload(page)
    page.focus("#tryFileInput")
    _tab_to(page, "tryCanvas")
    _press(page, "Shift+ArrowRight", 2)
    page.keyboard.press("Tab")
    page.click("#autoMarkBtn")
    page.click("#segmentBtn")
    page.wait_for_function(SEGMENTED, timeout=60000)
    scribbles = page.evaluate(
        """() => {
            const c = document.getElementById('scribbleResultCanvas');
            const o = document.getElementById('originalResultCanvas');
            const a = c.getContext('2d').getImageData(0, 0, c.width, c.height).data;
            const b = o.getContext('2d').getImageData(0, 0, o.width, o.height).data;
            let changedToWhite = 0;
            for (let i = 0; i < a.length; i += 4) {
                const white = a[i] > 250 && a[i + 1] > 250 && a[i + 2] > 250;
                if (white && (a[i] !== b[i] || a[i + 1] !== b[i + 1] || a[i + 2] !== b[i + 2])) changedToWhite++;
            }
            return changedToWhite;
        }"""
    )
    assert scribbles == 0


def test_tab_leaves_the_canvas(page):
    _upload(page)
    page.focus("#tryCanvas")
    page.keyboard.press("Tab")
    assert _active_id(page) != "tryCanvas", "the canvas must not trap focus"


def test_escape_stops_painting(page):
    _upload(page)
    page.focus("#tryFileInput")
    _tab_to(page, "tryCanvas")
    page.keyboard.press("Space")
    page.keyboard.press("Escape")
    assert page.inner_text("#canvasAnnouncer") == "Stopped painting."


def test_keyboard_moves_are_announced(page):
    _upload(page)
    page.focus("#tryFileInput")
    _tab_to(page, "tryCanvas")
    page.keyboard.press("Shift+ArrowRight")
    assert "across" in page.inner_text("#canvasAnnouncer")


def test_drawing_free_option_produces_a_result(page):
    _upload(page)
    page.click("#autoMarkBtn")
    assert "Marked the center" in page.inner_text("#tryStatus")
    page.click("#segmentBtn")
    page.wait_for_function(SEGMENTED, timeout=60000)
    assert "% of the image" in page.inner_text("#tryStatus")
    assert "% of the image" in page.get_attribute("#resultCanvas", "aria-label")


def test_status_is_announced_and_never_inside_hidden_content(page):
    assert page.get_attribute("#tryStatus", "role") == "status"
    assert page.get_attribute("#tryStatus", "aria-live") == "polite"
    assert page.evaluate("document.getElementById('tryStatus').closest('[hidden]') === null")
    assert page.evaluate("document.getElementById('canvasAnnouncer').closest('[hidden]') === null")


def test_sections_stay_hidden_until_they_have_something_to_show(page):
    for selector in ("#scribbleControls", "#canvasWrap", "#tryActions", "#tryResults", "#downloadBtn"):
        assert not page.is_visible(selector), f"{selector} shows before an upload"

    _upload(page)
    assert page.is_visible("#segmentBtn")
    assert not page.is_visible("#downloadBtn"), "nothing to download yet"

    page.click("#autoMarkBtn")
    page.click("#segmentBtn")
    page.wait_for_function(SEGMENTED, timeout=60000)
    assert page.is_visible("#downloadBtn")


def test_a_bad_first_upload_shows_its_error(page, tmp_path):
    bad = tmp_path / "not-an-image.png"
    bad.write_text("this is not an image")
    page.set_input_files("#tryFileInput", str(bad))
    page.wait_for_function("document.getElementById('tryStatus').textContent !== ''")
    assert "Couldn't load" in page.inner_text("#tryStatus")
    assert page.is_visible("#tryStatus")


def test_canvas_explains_itself(page):
    _upload(page)
    assert page.get_attribute("#tryCanvas", "role") == "application"
    assert page.get_attribute("#tryCanvas", "aria-label")
    described_by = page.get_attribute("#tryCanvas", "aria-describedby")
    assert "Space" in page.inner_text(f"#{described_by}")


def test_reduced_motion_is_respected(browser, site_url):
    context = browser.new_context(reduced_motion="reduce")
    tab = context.new_page()
    tab.route("**/*", lambda route: route.abort() if "fonts.g" in route.request.url else route.continue_())
    tab.goto(site_url)
    assert tab.evaluate("getComputedStyle(document.documentElement).scrollBehavior") == "auto"
    context.close()


def test_skip_link_is_first_and_goes_to_the_demo(page):
    page.keyboard.press("Tab")
    assert page.evaluate("document.activeElement.textContent") == "Skip to the demo"
    page.keyboard.press("Enter")
    assert page.evaluate("location.hash") == "#try-it"
