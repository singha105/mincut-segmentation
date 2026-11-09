"""Does a learned regional term beat the scribble histograms?

Three methods, on the same images with the same simulated scribbles:

- the existing graph cut, whose regional term is scribble intensity histograms
- the U-Net alone, ignoring the scribbles
- the graph cut with the U-Net's probability as its regional term

Each graph-cut method gets its N-link weight tuned on the validation images.
The test split is then scored once, with those weights. Everything runs on a
128 px grid, which is finer than the app's default and so generous to the
histogram baseline. Masks are solved with PyMaxflow for speed; the tests pin it
to the from-scratch solver.

    pip install -r requirements-ml.txt -r requirements-fast.txt
    python benchmarks/compare_learned_unary.py --data /path/to/pets --checkpoint unet.pt
"""

import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch

from learned.pets import (
    foreground_iou,
    has_both_labels,
    load_images,
    pixel_accuracy,
    read_split,
    scribble_rng,
    simulate_scribbles,
)
from learned.unary import build_learned_capacities
from learned.unet import SmallUNet
from maxflow.fast_maxflow import solve_grid_min_cut_fast
from maxflow.graph_builder import build_grid_capacities
from train_unet import pick_device, split_trainval, to_input

N_LINK_WEIGHTS = [0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 50.0, 100.0, 200.0, 500.0, 1000.0]
HERE = os.path.dirname(os.path.abspath(__file__))


@torch.no_grad()
def predict(model, images, device, batch_size=64) -> np.ndarray:
    model.eval()
    output = []
    for start in range(0, len(images), batch_size):
        batch = to_input(torch.from_numpy(images[start : start + batch_size])).to(device)
        output.append(torch.sigmoid(model(batch)).squeeze(1).cpu().numpy())
    return np.concatenate(output)


def usable(images, labels, names, probabilities):
    keep = [i for i, label in enumerate(labels) if has_both_labels(label)]
    scribbles = [simulate_scribbles(labels[i], scribble_rng(names[i])) for i in keep]
    return images[keep], labels[keep], probabilities[keep], scribbles


def histogram_cut(image, scribble, _probability, weight):
    return solve_grid_min_cut_fast(*build_grid_capacities(image, scribble, n_link_weight=weight))[1]


def learned_cut(image, scribble, probability, weight):
    return solve_grid_min_cut_fast(*build_learned_capacities(image, scribble, probability, n_link_weight=weight))[1]


def score_cut(method, images, labels, probabilities, scribbles, weight):
    ious, accuracies = [], []
    for image, label, probability, scribble in zip(images, labels, probabilities, scribbles):
        mask = method(image, scribble, probability, weight)
        ious.append(foreground_iou(mask, label))
        accuracies.append(pixel_accuracy(mask, label))
    return float(np.mean(ious)), float(np.mean(accuracies))


def tune(method, images, labels, probabilities, scribbles):
    results = {w: score_cut(method, images, labels, probabilities, scribbles, w)[0] for w in N_LINK_WEIGHTS}
    best = max(results, key=results.get)
    return best, results


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True)
    parser.add_argument("--checkpoint", required=True)
    args = parser.parse_args()

    device = pick_device()
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    model = SmallUNet().to(device)
    model.load_state_dict(checkpoint["state_dict"])
    size = checkpoint["image_size"]

    _, val_names = split_trainval(read_split(args.data, "trainval"))
    val_images, val_labels, val_names = load_images(args.data, val_names, size)
    val = usable(val_images, val_labels, val_names, predict(model, val_images, device))

    histogram_weight, histogram_curve = tune(histogram_cut, *val)
    learned_weight, learned_curve = tune(learned_cut, *val)
    print("validation histogram:", {k: round(v, 4) for k, v in histogram_curve.items()}, flush=True)
    print("validation learned:  ", {k: round(v, 4) for k, v in learned_curve.items()}, flush=True)

    # Weights are fixed from here on. This is the only time the test split is read.
    test_images, test_labels, test_names = load_images(args.data, read_split(args.data, "test"), size)
    images, labels, probabilities, scribbles = usable(
        test_images, test_labels, test_names, predict(model, test_images, device)
    )

    start = time.time()
    histogram = score_cut(histogram_cut, images, labels, probabilities, scribbles, histogram_weight)
    learned = score_cut(learned_cut, images, labels, probabilities, scribbles, learned_weight)
    elapsed = time.time() - start
    unet_ious = [foreground_iou(p >= 0.5, l) for p, l in zip(probabilities, labels)]
    unet_accuracies = [pixel_accuracy(p >= 0.5, l) for p, l in zip(probabilities, labels)]
    unet = (float(np.mean(unet_ious)), float(np.mean(unet_accuracies)))

    report = {
        "test_images": len(images),
        "val_images": len(val[0]),
        "train_images": checkpoint["train_images"],
        "grid": size,
        "model_parameters": checkpoint["parameters"],
        "best_epoch": checkpoint["epoch"],
        "histogram_graph_cut": {"n_link_weight": histogram_weight, "iou": histogram[0], "accuracy": histogram[1]},
        "unet_alone": {"iou": unet[0], "accuracy": unet[1]},
        "learned_graph_cut": {"n_link_weight": learned_weight, "iou": learned[0], "accuracy": learned[1]},
        "solve_seconds_all_test_images": elapsed,
    }
    with open(os.path.join(HERE, "learned_unary.json"), "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)
        handle.write("\n")

    rows = [
        ("Graph cut, scribble histograms (existing)", histogram, f"{histogram_weight:g}"),
        ("U-Net alone, ignores scribbles", unet, "n/a"),
        ("Graph cut, U-Net regional term", learned, f"{learned_weight:g}"),
    ]
    lines = [
        "# Learned regional term vs scribble histograms",
        "",
        f"Oxford-IIIT Pet official test split, {len(images):,} images on a {size} px grid. Mean per-image",
        "foreground IoU and pixel accuracy, with the dataset's border band excluded.",
        "",
        "| Method | N-link weight (tuned on validation) | Foreground IoU | Pixel accuracy |",
        "| --- | ---: | ---: | ---: |",
    ]
    lines += [f"| {name} | {weight} | {score[0]:.3f} | {score[1]:.3f} |" for name, score, weight in rows]
    lines += [
        "",
        f"The U-Net has {checkpoint['parameters']:,} parameters and was trained on {checkpoint['train_images']:,} images",
        f"from the trainval split (best validation epoch: {checkpoint['epoch']}). Scribbles are simulated from the",
        "ground truth with a per-image seed, so every method sees the same ones. They are",
        "strokes through three random foreground points and four random background points.",
        "",
    ]
    with open(os.path.join(HERE, "learned_unary.md"), "w", encoding="utf-8") as handle:
        handle.write("\n".join(lines))
    print("\n".join(lines), flush=True)


if __name__ == "__main__":
    main()
