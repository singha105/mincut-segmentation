"""
Top-level segmentation orchestration.

Plan:
- Expose a single entry point, e.g. segment(image, fg_markers, bg_markers),
  that ties together graph_builder and edmonds_karp:
    1. Build the flow network from the image and user markers
       (maxflow.graph_builder.build_graph).
    2. Run max-flow/min-cut on that network (maxflow.edmonds_karp).
    3. Map the resulting min-cut partition back to pixel coordinates to
       produce a binary foreground/background mask.
- Post-processing: optional cleanup of the mask (e.g. removing small
  isolated regions) before returning it.
- Return value: a mask (same H x W as the input image) plus any
  intermediate data useful for visualization (e.g. cut edges, timing
  info) so app.py can render results and confirm the <2s performance
  target.
- This module should stay independent of the Streamlit UI layer in
  app.py so the segmentation logic can be unit-tested in isolation
  (see tests/).
"""
