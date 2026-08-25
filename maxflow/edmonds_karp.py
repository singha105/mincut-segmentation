"""
Edmonds-Karp max-flow algorithm.

Plan:
- Implement BFS-based augmenting path search (Edmonds-Karp variant of
  Ford-Fulkerson) over a residual graph.
- Input: a flow network (see graph_builder.py) with source and sink nodes.
- Repeatedly find the shortest augmenting path (by edge count) from source
  to sink using BFS, then push flow equal to the path's bottleneck capacity,
  updating residual capacities (forward and reverse edges) after each pass.
- Terminate when no augmenting path remains; the accumulated flow equals
  the max-flow value.
- Once max-flow is reached, recover the min-cut by finding all nodes
  reachable from the source in the final residual graph; edges crossing
  from reachable to non-reachable nodes form the min-cut, which defines
  the foreground/background segmentation boundary.
- Complexity target: O(V * E^2), acceptable for the small/medium image
  graphs this tool will operate on.
"""
