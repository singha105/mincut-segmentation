"""
Graph construction for image segmentation via min-cut/max-flow.

Plan:
- Build a flow network from an input image plus user-provided foreground
  (fg) and background (bg) scribbles/markers.
- Each pixel becomes a node in the graph, plus two terminal nodes:
  source (S, representing foreground) and sink (T, representing
  background).
- N-links (pixel-to-pixel edges): connect each pixel to its neighbors
  (4- or 8-connectivity) with weights based on intensity/color similarity
  (e.g. a Gaussian on pixel-value difference), encouraging cuts along
  strong edges/boundaries in the image.
- T-links (pixel-to-terminal edges): connect each pixel to S and T with
  weights derived from how strongly it resembles the marked fg/bg regions
  (e.g. based on color histograms or distance to marked pixels). Pixels
  under user scribbles get hard-set (infinite) weights to enforce their
  label.
- Output: an adjacency/capacity representation consumable by
  edmonds_karp.py (e.g. adjacency dict or sparse matrix keyed by node id,
  with a mapping back to (row, col) pixel coordinates).
- Downsampling/superpixel grouping may be added later to keep the graph
  small enough for Edmonds-Karp to run within the target time budget.
"""
