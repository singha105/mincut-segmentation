"""Oxford-IIIT Pet data, simulated scribbles, and scoring.

Download images.tar.gz and annotations.tar.gz from
https://www.robots.ox.ac.uk/~vgg/data/pets/ and extract both into one folder.
"""

import zlib
from pathlib import Path
from typing import List, Tuple

import cv2
import numpy as np

from maxflow.graph_builder import BACKGROUND, FOREGROUND, UNMARKED

# Trimap values as the dataset ships them.
_TRIMAP_FOREGROUND = 1
_TRIMAP_BACKGROUND = 2

# Label values used here. The trimap's "border" band is a guess by the people
# who annotated it, so it is neither trained on nor scored.
LABEL_BACKGROUND = 0
LABEL_FOREGROUND = 1
LABEL_IGNORE = 255


def read_split(root: Path, split: str) -> List[str]:
    """Image names in the dataset's own ``trainval`` or ``test`` split."""
    with open(Path(root) / "annotations" / f"{split}.txt", encoding="utf-8") as handle:
        return [line.split()[0] for line in handle if line.strip()]


def load_images(root: Path, names: List[str], size: int) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """Return RGB images and labels resized to ``size``, skipping unreadable files."""
    root = Path(root)
    images, labels, kept = [], [], []
    for name in names:
        bgr = cv2.imread(str(root / "images" / f"{name}.jpg"), cv2.IMREAD_COLOR)
        trimap = cv2.imread(str(root / "annotations" / "trimaps" / f"{name}.png"), cv2.IMREAD_GRAYSCALE)
        if bgr is None or trimap is None:
            continue
        image = cv2.cvtColor(cv2.resize(bgr, (size, size), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2RGB)
        trimap = cv2.resize(trimap, (size, size), interpolation=cv2.INTER_NEAREST)

        label = np.full(trimap.shape, LABEL_IGNORE, dtype=np.uint8)
        label[trimap == _TRIMAP_FOREGROUND] = LABEL_FOREGROUND
        label[trimap == _TRIMAP_BACKGROUND] = LABEL_BACKGROUND
        images.append(image)
        labels.append(label)
        kept.append(name)
    return np.stack(images), np.stack(labels), kept


def has_both_labels(label: np.ndarray) -> bool:
    return bool((label == LABEL_FOREGROUND).any() and (label == LABEL_BACKGROUND).any())


def scribble_rng(name: str) -> np.random.Generator:
    """The same image always gets the same scribbles, whichever method is scored."""
    return np.random.default_rng(zlib.crc32(name.encode("utf-8")))


def simulate_scribbles(label: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Stand in for a user: strokes through a few random points of each region.

    Three points on the subject and four on the background, joined by a thin
    line that is clipped so it never crosses into the wrong region. Points come
    from the eroded region where possible, because people don't scribble right
    up against an edge.
    """
    scribbles = np.full(label.shape, UNMARKED, dtype=np.uint8)
    kernel = np.ones((7, 7), np.uint8)

    for value, mark, num_points in (
        (LABEL_BACKGROUND, BACKGROUND, 4),
        (LABEL_FOREGROUND, FOREGROUND, 3),
    ):
        region = (label == value).astype(np.uint8)
        eroded = cv2.erode(region, kernel)
        ys, xs = np.nonzero(eroded if eroded.any() else region)
        picks = rng.integers(0, len(xs), size=num_points)
        points = np.stack([xs[picks], ys[picks]], axis=1).reshape(-1, 1, 2).astype(np.int32)

        stroke = np.zeros(label.shape, dtype=np.uint8)
        cv2.polylines(stroke, [points], isClosed=False, color=1, thickness=2)
        scribbles[(stroke & region).astype(bool)] = mark

    return scribbles


def foreground_iou(prediction: np.ndarray, label: np.ndarray) -> float:
    """Intersection-over-union of the foreground, on pixels with a real label."""
    valid = label != LABEL_IGNORE
    truth = label == LABEL_FOREGROUND
    intersection = np.count_nonzero(prediction & truth & valid)
    union = np.count_nonzero((prediction | truth) & valid)
    return intersection / union if union else 1.0


def pixel_accuracy(prediction: np.ndarray, label: np.ndarray) -> float:
    valid = label != LABEL_IGNORE
    truth = label == LABEL_FOREGROUND
    return np.count_nonzero((prediction == truth) & valid) / max(1, np.count_nonzero(valid))
