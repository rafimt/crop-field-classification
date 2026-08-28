"""
Evaluate the trained model on the test set and write a detailed report:
a confusion matrix and per-class precision / recall / F1.

Run it AFTER training:
    python -m src.evaluate --config configs/train.yaml
"""

import argparse
import glob
import os

import numpy as np
import torch

from src.config_utils import load_config
from src.data.datamodule import CropDataModule
from src.models.resnet18_single import create_resnet18
from src.lit_module import CropClassifier


def find_checkpoint(output_dir):
    """Find the best.ckpt saved during training."""
    matches = glob.glob(os.path.join(output_dir, "**", "best.ckpt"), recursive=True)
    if not matches:
        raise FileNotFoundError("No best.ckpt found under " + output_dir)
    # If several, take the most recent.
    matches.sort(key=os.path.getmtime)
    return matches[-1]


def get_predictions(model, dataloader, device):
    """Run the model over all test data and collect true + predicted classes."""
    model.eval()
    model.to(device)

    true_labels = []
    predicted_labels = []

    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device)
            scores = model(images)                 # one score per class
            predictions = scores.argmax(dim=1)     # pick the highest-scoring class
            predicted_labels.extend(predictions.cpu().numpy().tolist())
            true_labels.extend(labels.numpy().tolist())

    return np.array(true_labels), np.array(predicted_labels)


def confusion_matrix(true_labels, predicted_labels, num_classes):
    """Build a simple confusion matrix (rows = true, columns = predicted)."""
    matrix = np.zeros((num_classes, num_classes), dtype=int)
    for t, p in zip(true_labels, predicted_labels):
        matrix[t, p] += 1
    return matrix


def per_class_scores(matrix):
    """From a confusion matrix, compute precision, recall and F1 per class."""
    num_classes = matrix.shape[0]
    scores = []
    for c in range(num_classes):
        true_positive = matrix[c, c]
        predicted_c = matrix[:, c].sum()   # how many times we predicted class c
        actual_c = matrix[c, :].sum()      # how many really are class c

        precision = true_positive / predicted_c if predicted_c > 0 else 0.0
        recall = true_positive / actual_c if actual_c > 0 else 0.0
        if precision + recall > 0:
            f1 = 2 * precision * recall / (precision + recall)
        else:
            f1 = 0.0
        scores.append((precision, recall, f1))
    return scores


def format_report(classes, matrix, scores):
    """Make a readable text block for the confusion matrix and scores."""
    lines = []
    lines.append("")
    lines.append("=== Detailed evaluation ===")
    lines.append("")
    lines.append("Confusion matrix (rows = true crop, columns = predicted crop):")

    # Header row with class names.
    header = "true \\ pred".ljust(12)
    for name in classes:
        header += name[:9].rjust(10)
    lines.append(header)

    for i, name in enumerate(classes):
        row = name[:11].ljust(12)
        for j in range(len(classes)):
            row += str(matrix[i, j]).rjust(10)
        lines.append(row)

    lines.append("")
    lines.append("Per-class scores:")
    lines.append("  class".ljust(14) + "precision".rjust(11)
                 + "recall".rjust(9) + "f1".rjust(9))
    for name, (p, r, f1) in zip(classes, scores):
        lines.append("  " + name[:11].ljust(12)
                     + str(round(p, 3)).rjust(11)
                     + str(round(r, 3)).rjust(9)
                     + str(round(f1, 3)).rjust(9))

    return "\n".join(lines) + "\n"


def main(train_config_path):
    train_cfg = load_config(train_config_path)
    data_cfg = load_config(train_cfg["data_config"])

    # Prepare the data (same settings as training).
    data = CropDataModule(
        data_config=data_cfg,
        batch_size=train_cfg["batch_size"],
        num_workers=train_cfg["num_workers"],
        augment=False,
    )
    data.setup()

    # Rebuild the model and load the trained weights.
    network = create_resnet18(
        num_input_channels=data.num_channels,
        num_classes=data.num_classes,
        pretrained=False,   # weights come from the checkpoint, not ImageNet
    )
    model = CropClassifier(network, num_classes=data.num_classes)

    ckpt_path = find_checkpoint(train_cfg["output_dir"])
    print("Loading checkpoint:", ckpt_path)
    checkpoint = torch.load(ckpt_path, map_location="cpu")
    model.load_state_dict(checkpoint["state_dict"])

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Predict on the test set.
    true_labels, predicted_labels = get_predictions(
        model, data.test_dataloader(), device
    )

    matrix = confusion_matrix(true_labels, predicted_labels, data.num_classes)
    scores = per_class_scores(matrix)
    report = format_report(data.classes, matrix, scores)

    print(report)

    # Append the detailed report to the existing results.txt.
    out_path = os.path.join(train_cfg["output_dir"], "results.txt")
    with open(out_path, "a", encoding="utf-8") as f:
        f.write(report)
    print("Appended detailed report to", out_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/train.yaml")
    args = parser.parse_args()
    main(args.config)
