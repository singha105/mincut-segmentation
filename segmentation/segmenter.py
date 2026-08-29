"""
Top-level segmentation orchestration.

Ties maxflow.graph_builder (image + scribbles -> flow network) and the
max-flow solvers (network -> min-cut) together into a mask.

Two entry points:

``segment(image, scribble_mask)``
    Runs the cut directly at whatever resolution it is handed. Simple and
    exact, but Edmonds-Karp is O(V * E^2), so it is only practical on small
    grids.

``segment_image(image, scribble_mask)``
    The one to use on real photos. It solves the cut on a downscaled copy,
    then upsamples the resulting mask back to the image's *full* resolution,
    refining the boundary against the original pixels so the output keeps the
    input's detail instead of inheriting the solver's coarse grid.

Why the two-stage approach
--------------------------
The number of augmenting paths Edmonds-Karp needs grows with the pixel count,
and each one costs a breadth-first search, so solving directly on a
multi-megapixel photo is not viable (measured: ~3.4s at 80x80, ~27s at
150x150). Segmentation quality, though, does not require a per-pixel cut:
the *shape* of the object is low-frequency and survives downscaling, while
the *precision of its boundary* is what needs full resolution. So the cut is
solved coarsely and the boundary is then snapped back onto real image edges
with a guided filter (He et al., "Guided Image Filtering"), which is exactly
the edge-aware joint-upsampling that step needs.

The practical consequence: the returned mask -- and any cutout made from it
-- is the same size as the image you passed in, at full quality.
"""

from typing import Optional, Tuple

import cv2
import numpy as np

from maxflow.graph_builder import (
    BACKGROUND,
    FOREGROUND,
    UNMARKED,
    _to_grayscale,
    build_grid_capacities,
)
from maxflow.grid_maxflow import solve_grid_min_cut

# Longest side of the grid the min-cut is actually solved on. Chosen from
# measured solver cost (see the module docstring): large enough to capture
# object shape, small enough to stay interactive.
DEFAULT_PROCESSING_MAX_DIM = 64


def segment(
    image: np.ndarray,
    scribble_mask: np.ndarray,
    n_link_weight: float = 50.0,
    num_bins: int = 32,
) -> np.ndarray:
    """Segment `image` into foreground/background given `scribble_mask`.

    Solves the min-cut at the resolution it is given and returns a boolean
    ``(H, W)`` mask that is True on the source side of the cut (foreground).
    """
    source, sink, east, south = build_grid_capacities(
        image, scribble_mask, n_link_weight=n_link_weight, num_bins=num_bins
    )
    _, mask = solve_grid_min_cut(source, sink, east, south)
    return mask


def _processing_size(height: int, width: int, max_dim: int) -> Tuple[int, int]:
    """Downscaled (height, width) with the longest side capped at `max_dim`."""
    scale = min(1.0, max_dim / max(height, width))
    return max(1, round(height * scale)), max(1, round(width * scale))


def _downscale_scribbles(
    scribble_mask: np.ndarray, target_height: int, target_width: int
) -> np.ndarray:
    """Shrink a scribble mask without losing thin strokes.

    Nearest-neighbour sampling would drop strokes narrower than one output
    cell. Instead each label is reduced with area-averaging and kept wherever
    *any* source pixel carried it, so a one-pixel scribble still survives.
    """
    size = (target_width, target_height)
    foreground = cv2.resize(
        (scribble_mask == FOREGROUND).astype(np.float32), size, interpolation=cv2.INTER_AREA
    )
    background = cv2.resize(
        (scribble_mask == BACKGROUND).astype(np.float32), size, interpolation=cv2.INTER_AREA
    )

    small = np.full((target_height, target_width), UNMARKED, dtype=np.uint8)
    small[background > 0] = BACKGROUND
    small[foreground > 0] = FOREGROUND  # foreground wins a contested cell
    return small


def _guided_filter(
    guide: np.ndarray, target: np.ndarray, radius: int, eps: float
) -> np.ndarray:
    """Edge-aware smoothing of `target` under the structure of `guide`.

    A local linear model ``target ~= a * guide + b`` is fitted in every window
    and the per-pixel averages of ``a``/``b`` are applied. Where the guide has
    an edge, the fit follows it; where the guide is flat, the output is simply
    smoothed. Used here to pull a coarse mask boundary onto the real edges of
    the full-resolution image.
    """
    guide = guide.astype(np.float32)
    target = target.astype(np.float32)
    kernel = (2 * radius + 1, 2 * radius + 1)

    mean_guide = cv2.boxFilter(guide, -1, kernel)
    mean_target = cv2.boxFilter(target, -1, kernel)
    mean_guide_target = cv2.boxFilter(guide * target, -1, kernel)
    mean_guide_sq = cv2.boxFilter(guide * guide, -1, kernel)

    covariance = mean_guide_target - mean_guide * mean_target
    variance = mean_guide_sq - mean_guide * mean_guide

    a = covariance / (variance + eps)
    b = mean_target - a * mean_guide

    return cv2.boxFilter(a, -1, kernel) * guide + cv2.boxFilter(b, -1, kernel)


def _refine_to_full_resolution(
    mask_small: np.ndarray,
    gray_full: np.ndarray,
    scribble_mask: np.ndarray,
    upscale_factor: float,
) -> np.ndarray:
    """Upsample a coarse mask to full resolution, snapping it to image edges."""
    height, width = gray_full.shape

    soft = cv2.resize(
        mask_small.astype(np.float32), (width, height), interpolation=cv2.INTER_LINEAR
    )

    # The window needs to span roughly one coarse cell, which is how far the
    # upsampled boundary can sit from the true edge.
    radius = int(max(2, round(upscale_factor)))
    guide = gray_full.astype(np.float32) / 255.0
    refined = _guided_filter(guide, soft, radius=radius, eps=1e-4)

    mask = refined >= 0.5

    # Hard constraints are part of the problem statement, so they must still
    # hold after refinement -- a scribbled pixel keeps the label the user gave it.
    mask[scribble_mask == FOREGROUND] = True
    mask[scribble_mask == BACKGROUND] = False
    return mask


def segment_image(
    image: np.ndarray,
    scribble_mask: np.ndarray,
    processing_max_dim: int = DEFAULT_PROCESSING_MAX_DIM,
    n_link_weight: float = 50.0,
    num_bins: int = 32,
) -> np.ndarray:
    """Segment a full-resolution image, returning a full-resolution mask.

    `image` is ``(H, W, 3)`` (or ``(H, W)``) and `scribble_mask` is ``(H, W)``
    at the same resolution. The min-cut is solved on a copy downscaled to
    `processing_max_dim`, then the mask is upsampled and refined against the
    original pixels.

    Returns a boolean ``(H, W)`` mask -- the same size as `image` -- so a
    cutout made from it loses none of the input's resolution.
    """
    gray_full = _to_grayscale(image)
    height, width = gray_full.shape

    proc_height, proc_width = _processing_size(height, width, processing_max_dim)

    # Already small enough to cut directly; no refinement step needed.
    if (proc_height, proc_width) == (height, width):
        return segment(image, scribble_mask, n_link_weight, num_bins)

    small_image = cv2.resize(image, (proc_width, proc_height), interpolation=cv2.INTER_AREA)
    small_scribbles = _downscale_scribbles(scribble_mask, proc_height, proc_width)

    mask_small = segment(small_image, small_scribbles, n_link_weight, num_bins)

    upscale_factor = max(height / proc_height, width / proc_width)
    return _refine_to_full_resolution(mask_small, gray_full, scribble_mask, upscale_factor)


def apply_mask(
    image: np.ndarray,
    mask: np.ndarray,
    background: str = "black",
    dim_factor: float = 0.25,
) -> np.ndarray:
    """Composite `image` against `mask` at full resolution.

    `background` selects what happens outside the mask: ``"black"`` blanks it,
    ``"dim"`` darkens it by `dim_factor` (keeps context visible), and
    ``"transparent"`` returns RGBA with the background alpha set to 0.
    """
    if background == "transparent":
        rgba = np.dstack([image[..., :3], np.where(mask, 255, 0).astype(np.uint8)])
        return rgba

    out = image.copy()
    if background == "dim":
        out[~mask] = (out[~mask] * dim_factor).astype(out.dtype)
    else:
        out[~mask] = 0
    return out


def cutout(
    image: np.ndarray,
    scribble_mask: np.ndarray,
    background: str = "black",
    processing_max_dim: int = DEFAULT_PROCESSING_MAX_DIM,
) -> Tuple[np.ndarray, np.ndarray]:
    """Convenience wrapper: segment at full resolution and composite.

    Returns ``(cutout_image, mask)``, both at the input image's resolution.
    """
    mask = segment_image(image, scribble_mask, processing_max_dim=processing_max_dim)
    return apply_mask(image, mask, background=background), mask
