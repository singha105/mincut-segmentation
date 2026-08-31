"""Keep the design documents honest.

design/LLD.md lists module interfaces. Documentation that quietly drifts out of
date is worse than none, so these tests parse the signatures straight out of
the document and check they still exist in the code. Rename or delete a
documented function and this fails.
"""

import importlib
import pathlib
import re

import pytest

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
DESIGN_DIR = REPO_ROOT / "design"
HLD = DESIGN_DIR / "HLD.md"
LLD = DESIGN_DIR / "LLD.md"
GUIDE = REPO_ROOT / "GUIDE.md"

# "### `maxflow/graph_builder.py`" or "### `image_io.py` (reference)"
MODULE_HEADING = re.compile(r"^### `([\w/]+\.py)`")
# "name(args) -> ..." at the start of a line inside a python block
SIGNATURE = re.compile(r"^([a-z_][a-z0-9_]*)\(")


def _documented_functions():
    """Yield (module_path, function_name) for every signature in LLD section 2."""
    lines = LLD.read_text().splitlines()
    module = None
    in_python_block = False
    in_interfaces = False

    for line in lines:
        if line.startswith("## "):
            in_interfaces = line.startswith("## 2. Module interfaces")
            module = None
        if not in_interfaces:
            continue

        heading = MODULE_HEADING.match(line)
        if heading:
            module = heading.group(1)
            continue

        if line.startswith("```"):
            in_python_block = not in_python_block
            continue

        if in_python_block and module:
            match = SIGNATURE.match(line)
            if match:
                yield module, match.group(1)


DOCUMENTED = sorted(set(_documented_functions()))


def test_design_documents_exist():
    assert HLD.is_file(), "design/HLD.md is missing"
    assert LLD.is_file(), "design/LLD.md is missing"
    assert GUIDE.is_file(), "GUIDE.md is missing"


def test_signatures_were_actually_parsed():
    """Guard against the parser silently matching nothing."""
    assert len(DOCUMENTED) >= 15, f"only parsed {len(DOCUMENTED)} signatures from LLD.md"
    modules = {module for module, _ in DOCUMENTED}
    assert "segmentation/segmenter.py" in modules
    assert "maxflow/grid_maxflow.py" in modules


@pytest.mark.parametrize("module_path, function_name", DOCUMENTED)
def test_documented_function_exists(module_path, function_name):
    module = importlib.import_module(module_path[:-3].replace("/", "."))
    attribute = getattr(module, function_name, None)

    assert attribute is not None, (
        f"design/LLD.md documents {module_path}:{function_name}(), "
        "which no longer exists"
    )
    assert callable(attribute), f"{module_path}:{function_name} is documented as a function"


def test_documented_label_constants_match_the_code():
    """The label table in LLD section 1 states the actual values."""
    from maxflow.graph_builder import BACKGROUND, FOREGROUND, UNMARKED

    assert (UNMARKED, FOREGROUND, BACKGROUND) == (0, 1, 2)


def test_documented_parent_encoding_matches_the_code():
    """The BFS parent table in LLD section 1 states the actual values."""
    from maxflow import grid_maxflow as gm

    assert (gm.FROM_SOURCE, gm.FROM_WEST, gm.FROM_EAST) == (0, 1, 2)
    assert (gm.FROM_NORTH, gm.FROM_SOUTH, gm.UNREACHED) == (3, 4, -1)


def test_documented_default_grid_matches_the_code():
    """HLD and LLD both quote the 56px cut grid."""
    from segmentation.segmenter import DEFAULT_PROCESSING_MAX_DIM

    assert DEFAULT_PROCESSING_MAX_DIM == 56
    assert "56" in HLD.read_text()
    assert "56" in LLD.read_text()


@pytest.mark.parametrize("document", [HLD, LLD, GUIDE])
def test_cross_links_between_the_documents_resolve(document):
    text = document.read_text()
    for target in re.findall(r"\]\((?!https?:)([^)#][^)]*)\)", text):
        path = (document.parent / target.split("#")[0]).resolve()
        assert path.exists(), f"{document.name} links to a missing file: {target}"


@pytest.mark.parametrize("document", [HLD, LLD, GUIDE])
def test_documents_have_no_dashes(document):
    """House style for this repository."""
    text = document.read_text()
    assert "—" not in text, f"{document.name} contains an em dash"
    assert "–" not in text, f"{document.name} contains an en dash"


def test_guide_worked_example_still_matches_the_code():
    """GUIDE.md prints real solver output; recompute it and compare.

    The worked example is the part of the guide most likely to go quietly
    wrong, because it quotes exact numbers. If tuning changes any of them the
    guide must be regenerated, so this fails rather than letting it drift.
    """
    import numpy as np

    from maxflow.graph_builder import (
        BACKGROUND,
        FOREGROUND,
        UNMARKED,
        build_grid_capacities,
    )
    from maxflow.grid_maxflow import solve_grid_min_cut

    image = np.zeros((3, 4, 3), dtype=np.uint8)
    image[:, :2] = 200
    image[:, 2:] = 40

    scribbles = np.full((3, 4), UNMARKED, dtype=np.uint8)
    scribbles[1, 0] = FOREGROUND
    scribbles[1, 3] = BACKGROUND

    source, sink, east, south = build_grid_capacities(image, scribbles)
    flow, mask = solve_grid_min_cut(source, sink, east, south)

    text = GUIDE.read_text()
    for quoted in ["2.941", "153.941", "36.856", "3.497", "2.803", "0.00011068"]:
        assert quoted in text, f"GUIDE.md no longer quotes {quoted}"

    assert round(east[0, 1], 3) == 2.941, "weak N-link changed"
    assert round(east[0, 0], 3) == 50.0, "strong N-link changed"
    assert round(source[1, 0], 3) == 153.941, "hard constraint changed"
    assert round(source[0, 0], 3) == 3.497, "bright-pixel source capacity changed"
    assert round(source[0, 2], 3) == 2.803, "dark-pixel source capacity changed"
    assert round(flow, 3) == 36.856, "max flow changed"
    assert mask.tolist() == [[True, True, False, False]] * 3, "resulting mask changed"


def test_guide_cut_cost_equals_max_flow():
    """The guide claims the severed edges sum to the max flow. Verify it."""
    import numpy as np

    from maxflow.graph_builder import (
        BACKGROUND,
        FOREGROUND,
        UNMARKED,
        build_grid_capacities,
    )
    from maxflow.grid_maxflow import solve_grid_min_cut

    image = np.zeros((3, 4, 3), dtype=np.uint8)
    image[:, :2] = 200
    image[:, 2:] = 40
    scribbles = np.full((3, 4), UNMARKED, dtype=np.uint8)
    scribbles[1, 0] = FOREGROUND
    scribbles[1, 3] = BACKGROUND

    source, sink, east, south = build_grid_capacities(image, scribbles)
    flow, mask = solve_grid_min_cut(source, sink, east, south)

    severed = sum(
        east[y, x] for y in range(3) for x in range(3) if mask[y, x] and not mask[y, x + 1]
    )
    severed += sum(
        south[y, x] for y in range(2) for x in range(4) if mask[y, x] and not mask[y + 1, x]
    )
    severed += source[~mask].sum() + sink[mask].sum()

    assert severed == pytest.approx(flow), "cut cost should equal the max flow"
