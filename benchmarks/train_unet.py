"""Train the small U-Net on Oxford-IIIT Pet.

The dataset's trainval split is divided 90/10 into training and validation by
a fixed seed. The test split is not touched here at all.

    pip install -r requirements-ml.txt
    python benchmarks/train_unet.py --data /path/to/pets --checkpoint unet.pt
"""

import argparse
import os
import random
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
import torch.nn.functional as F

from learned.pets import LABEL_FOREGROUND, LABEL_IGNORE, foreground_iou, load_images, read_split
from learned.unet import SmallUNet

IMAGE_SIZE = 128
SPLIT_SEED = 1234
VALIDATION_FRACTION = 0.1


def pick_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def split_trainval(names):
    shuffled = sorted(names)
    random.Random(SPLIT_SEED).shuffle(shuffled)
    cut = int(len(shuffled) * VALIDATION_FRACTION)
    return shuffled[cut:], shuffled[:cut]


def to_input(images: torch.Tensor) -> torch.Tensor:
    """``(B, H, W, 3)`` uint8 to the ``(B, 3, H, W)`` floats the model takes."""
    return images.permute(0, 3, 1, 2).float() / 255.0


def augment(batch: torch.Tensor, target: torch.Tensor, valid: torch.Tensor):
    flip = torch.rand(batch.shape[0], device=batch.device) < 0.5
    batch[flip] = batch[flip].flip(-1)
    target[flip] = target[flip].flip(-1)
    valid[flip] = valid[flip].flip(-1)
    brightness = 0.8 + 0.4 * torch.rand(batch.shape[0], 1, 1, 1, device=batch.device)
    return (batch * brightness).clamp(0.0, 1.0), target, valid


def segmentation_loss(logits: torch.Tensor, target: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
    bce = F.binary_cross_entropy_with_logits(logits, target, reduction="none")
    bce = (bce * valid).sum() / valid.sum().clamp(min=1.0)
    probability = torch.sigmoid(logits) * valid
    truth = target * valid
    dice = 1.0 - (2.0 * (probability * truth).sum() + 1.0) / (probability.sum() + truth.sum() + 1.0)
    return bce + dice


@torch.no_grad()
def mean_iou(model, images, labels, device, batch_size=64) -> float:
    model.eval()
    scores = []
    for start in range(0, len(images), batch_size):
        batch = to_input(torch.from_numpy(images[start : start + batch_size])).to(device)
        predictions = (torch.sigmoid(model(batch)) >= 0.5).squeeze(1).cpu().numpy()
        for prediction, label in zip(predictions, labels[start : start + batch_size]):
            scores.append(foreground_iou(prediction, label))
    return float(np.mean(scores))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    device = pick_device()

    train_names, val_names = split_trainval(read_split(args.data, "trainval"))
    start = time.time()
    train_images, train_labels, train_names = load_images(args.data, train_names, IMAGE_SIZE)
    val_images, val_labels, val_names = load_images(args.data, val_names, IMAGE_SIZE)
    print(f"loaded {len(train_names)} train, {len(val_names)} val in {time.time() - start:.0f}s on {device}", flush=True)

    model = SmallUNet().to(device)
    parameters = sum(p.numel() for p in model.parameters())
    print(f"model has {parameters:,} parameters", flush=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    images = torch.from_numpy(train_images)
    labels = torch.from_numpy(train_labels)
    best_iou = -1.0

    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_start = time.time()
        order = torch.randperm(len(images))
        total_loss, batches = 0.0, 0
        for first in range(0, len(order), args.batch_size):
            indices = order[first : first + args.batch_size]
            batch = to_input(images[indices]).to(device)
            label = labels[indices].to(device)
            target = (label == LABEL_FOREGROUND).float().unsqueeze(1)
            valid = (label != LABEL_IGNORE).float().unsqueeze(1)
            batch, target, valid = augment(batch, target, valid)

            optimizer.zero_grad()
            loss = segmentation_loss(model(batch), target, valid)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            batches += 1
        scheduler.step()

        iou = mean_iou(model, val_images, val_labels, device)
        print(
            f"epoch {epoch}/{args.epochs} loss {total_loss / batches:.4f} val_iou {iou:.4f} "
            f"({time.time() - epoch_start:.0f}s)",
            flush=True,
        )
        if iou > best_iou:
            best_iou = iou
            torch.save(
                {
                    "state_dict": {k: v.cpu() for k, v in model.state_dict().items()},
                    "image_size": IMAGE_SIZE,
                    "epoch": epoch,
                    "val_iou": iou,
                    "parameters": parameters,
                    "train_images": len(train_names),
                    "val_images": len(val_names),
                },
                args.checkpoint,
            )

    print(f"best val_iou {best_iou:.4f}; checkpoint at {args.checkpoint}", flush=True)


if __name__ == "__main__":
    main()
