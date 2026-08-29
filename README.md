# mincut-segmentation

Built an interactive image segmentation tool using max-flow/min-cut graph algorithms, achieving segmentation on user-marked images in under 2s.

**Try it live:** [singha105.github.io/mincut-segmentation](https://singha105.github.io/mincut-segmentation/)
*(no install needed — see [Try it in your browser](#try-it-in-your-browser) below. The Python/Streamlit app in this repo is the original reference implementation; see [Setup](#setup) to run it locally.)*

## Demo

Actual output of `segment()` on each sample image — a couple of scribbled
pixels in, a foreground/background mask out:

![Demo: original, scribbles, and segmented result for each sample image](docs/demo.png)

## Real-world examples

The synthetic shapes above are the easy case — clean solid colors, sharp
edges. Real photos are a genuinely harder test: textured fur, uneven
lighting, and backgrounds that share colors with the subject. Here's the
same pipeline run on four real animal photos, at each photo's own resolution:

![Demo: original, scribbles, and segmented result for a dog, hen, horse, and tiger photo](docs/demo_animals.png)

The dog and tiger each needed an extra scribble to get there, and the reason
is worth knowing: the foreground/background model is a single **grayscale**
intensity histogram per region, so wherever the subject and background share
a tonal range — a tiger's shadowed stripes against dark water, a dog's face
against an unlit room — two scribbles aren't enough information to separate
them. Adding a stroke in the confused area fixes it, which is exactly what
the [interactive tool](#try-it-in-your-browser) is for. A color (rather than
grayscale) model would need fewer corrections.

Source photos are from Wikimedia Commons: [dog](https://commons.wikimedia.org/wiki/File:Dog_(Unsplash).jpg) (Multa Media, CC0), [hen](https://commons.wikimedia.org/wiki/File:American_Game_hen.jpg) (Jean Ogden, CC BY 2.0), [horse](https://commons.wikimedia.org/wiki/File:Horse_grazing_in_the_shade_of_conifers_(Unsplash).jpg) (Annie Spratt, CC0), [tiger](https://commons.wikimedia.org/wiki/File:Wet_tiger_(Unsplash).jpg) (佑翔 賴, CC0). Full-size copies are in [`sample_images/`](sample_images/).

## Output quality: what went wrong, and the fix

Earlier versions of this project produced badly pixelated results. Both panels
below use the *same* photo, the *same* scribbles and the *same* algorithm —
the only thing that differs is what happens to the mask afterwards:

![Before and after: a blocky 30px cutout beside a full-resolution one](docs/quality_before_after.png)

### The problem

Edmonds-Karp is `O(V * E^2)`, and the number of augmenting paths it needs
grows with the pixel count — measured on a real photo, solving the cut
directly takes ~3.4s at 80×80 and ~27s at 150×150. Running it per-pixel on a
multi-megapixel image is not viable, so the image was downscaled to a 30px
grid before segmenting. That part was unavoidable.

The mistake was what came next: the cutout was composited **on that 30px grid**
and the tiny result was then scaled back up for display and download. So the
output resolution was the *solver's* resolution. You weren't looking at your
image — you were looking at the solver's working grid, every pixel of it
blown up ~25×.

### The fix

The `O(V * E^2)` constraint applies to the **cut**, not to the **image**, so
those two are now separated:

1. **Solve coarsely.** The min-cut still runs on a small grid. An object's
   *shape* is low-frequency and survives downscaling fine.
2. **Refine at full resolution.** The coarse mask is upsampled and passed
   through a [guided filter](https://kaiminghe.github.io/eccv10/) using the
   full-resolution image as the guide. Where the guide has an edge, the mask
   boundary snaps onto it — so the boundary recovers detail the coarse grid
   never contained. This is the step that turns a blocky staircase into a
   contour that follows fur and grass.
3. **Composite at full resolution.** The mask is applied to the *original*
   pixels, which are never resampled.

### The supporting changes

- **A faster solver, for a finer grid.** Profiling showed BFS was 3.37s of
  3.43s at 80×80 — inherent to the algorithm, not something tuning fixes. So
  [`maxflow/grid_maxflow.py`](maxflow/grid_maxflow.py) runs the same algorithm
  on the pixel grid using numpy arrays instead of a dict probe per edge, plus a
  vectorised step that saturates all `source → pixel → sink` paths at once.
  About 5× faster, which pays for a **64px** cut grid instead of 30px.
- **Scribbles are no longer downscaled on the way in.** They are rasterised at
  the image's own resolution and reduced with area-averaging that keeps any
  stroke touching a cell, so thin strokes can't vanish before the solve.
- **Better compositing.** Backgrounds can be black, dimmed, or a transparent
  PNG, and downloads come out at the input's resolution.

The result is the same size as the input, and foreground pixels are
bit-for-bit the originals — [`tests/test_segmentation.py`](tests/test_segmentation.py)
asserts exactly that, alongside a test that the recovered boundary is tight
rather than merely non-empty.

`segment_image()` is the entry point that does all of this; `segment()` still
solves directly at whatever resolution you hand it.

## Try it in your browser

The [live page](https://singha105.github.io/mincut-segmentation/) is a fully
working version of the tool, no install required:

- **Upload** any image, not just the samples above
- **Scribble** red for foreground / blue for background, with an adjustable brush size
- **Segment** — runs the same Edmonds-Karp min-cut pipeline as a JavaScript
  port ([`docs/segmentation.js`](docs/segmentation.js)), entirely client-side
- **Choose a background** — black, dimmed, or transparent PNG
- **Download** the result at your image's original resolution

There's no backend involved: image decoding, graph construction, the max-flow
solve, and the full-resolution refinement all happen in your browser via plain
JavaScript.

## Setup

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Run

```bash
streamlit run app.py
```

## Test

```bash
pytest
```

## How it works: Edmonds-Karp max-flow

Segmentation is framed as a min-cut problem: pixels are nodes in a flow
network, with a source terminal representing "foreground" and a sink
terminal representing "background." The min-cut that separates source
from sink also separates the image into foreground and background
regions — so finding that cut is the core of the algorithm.

[`maxflow/edmonds_karp.py`](maxflow/edmonds_karp.py) computes it via
Edmonds-Karp, the BFS variant of Ford-Fulkerson:

1. **Residual graph.** Every edge `(u, v)` with capacity `c` gets a
   reverse edge `(v, u)`, initially capacity 0, so that flow pushed
   along `(u, v)` can later be partially undone if a better routing is
   found.
2. **Find an augmenting path.** BFS from source to sink using only edges
   with positive residual capacity. BFS always finds the *shortest*
   (fewest-edge) such path, which is what bounds the algorithm to
   `O(V * E^2)` instead of degrading on adversarial graphs.
3. **Push flow.** The path's bottleneck is its smallest residual
   capacity. Subtract that amount from each forward edge on the path and
   add it to each reverse edge.
4. **Repeat** until BFS can no longer find a path from source to sink —
   at that point the flow is maximum.
5. **Recover the cut.** The nodes still reachable from the source in the
   final residual graph form one side of the min-cut; the original edges
   crossing from that side to the rest of the graph are the cut edges,
   and by the max-flow min-cut theorem their capacities sum to the
   max-flow value.

See [`tests/test_maxflow.py`](tests/test_maxflow.py) for hand-verified
examples, including a case that requires multiple augmenting-path
iterations.

### Two solvers, one algorithm

[`maxflow/edmonds_karp.py`](maxflow/edmonds_karp.py) is the readable
reference: a nested-dict graph walked one edge at a time. It is what the
hand-verified tests exercise, but a dict probe per edge makes it too slow to
segment on.

[`maxflow/grid_maxflow.py`](maxflow/grid_maxflow.py) runs the *same*
algorithm on the 4-connected pixel grid, keeping residual capacities in numpy
arrays so each BFS level expands as a few whole-array operations instead of
per-edge Python. It also saturates every `source → pixel → sink` path in one
vectorised step, since those are all shortest paths and share no edges. Same
asymptotics, far smaller constant — which is what buys the finer grid.

The two must agree exactly, and
[`tests/test_grid_maxflow.py`](tests/test_grid_maxflow.py) pins the fast one
to the reference on randomised graphs.

## Project layout

- `maxflow/` — the reference Edmonds-Karp solver, the fast grid solver, and image-to-graph construction
- `segmentation/` — orchestration: coarse cut → full-resolution refinement → mask and compositing
- `app.py` — Streamlit UI for marking foreground/background and viewing results
- `sample_images/` — synthetic shapes for manual testing, plus four real (CC0/CC-BY) animal photos for a harder real-world demo
- `tests/` — automated tests
- `docs/` — the [GitHub Pages site](https://singha105.github.io/mincut-segmentation/): `index.html`/`segmentation.js` (a client-side JS port with upload/scribble/segment/download) and `demo.png`/`demo_animals.png` (the static demo images)
