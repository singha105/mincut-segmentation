# mincut-segmentation

Built an interactive image segmentation tool using max-flow/min-cut graph algorithms, achieving segmentation on user-marked images in under 2s.

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

## Project layout

- `maxflow/` — Edmonds-Karp max-flow implementation and image-to-graph construction
- `segmentation/` — orchestration layer that turns a graph min-cut into a foreground/background mask
- `app.py` — Streamlit UI for marking foreground/background and viewing results
- `sample_images/` — small synthetic images for manual testing and demos
- `tests/` — automated tests
