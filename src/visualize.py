"""
Make visual outputs for the test set:
  1. A grid of test fields showing the RGB image + true vs predicted crop
     (green title = correct, red title = wrong).
  2. A confusion-matrix heatmap.

Run:
    python -m src.visualize --mode temporal   # Phase 2 model
    python -m src.visualize --mode single      # Phase 1 model

Images are saved to outputs/viz/.
"""

import argparse
import glob
import os

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")  # no screen needed, just save files
import matplotlib.pyplot as plt

from src.config_utils import load_config
from src.lit_module import CropClassifier


def load_setup(mode):
    """Build the right datamodule + model + checkpoint path for the mode."""
    if mode == "temporal":
        from src.data.datamodule_temporal import CropTemporalDataModule
        from src.models.resnet18_temporal import TemporalResNet18

        tcfg = load_config("configs/train_temporal.yaml")
        dcfg = load_config(tcfg["data_config"])
        data = CropTemporalDataModule(dcfg, batch_size=tcfg["batch_size"],
                                      num_workers=0, augment=False)
        data.setup()
        net = TemporalResNet18(data.num_channels, data.num_classes,
                               pretrained=False, hidden_size=tcfg["hidden_size"])
        ckpt_dir = "outputs/resnet18_temporal"
    else:
        from src.data.datamodule import CropDataModule
        from src.models.resnet18_single import create_resnet18

        tcfg = load_config("configs/train.yaml")
        dcfg = load_config(tcfg["data_config"])
        data = CropDataModule(dcfg, batch_size=tcfg["batch_size"],
                              num_workers=0, augment=False)
        data.setup()
        net = create_resnet18(data.num_channels, data.num_classes, pretrained=False)
        ckpt_dir = "outputs/resnet18"

    model = CropClassifier(net, num_classes=data.num_classes)
    ckpt = sorted(glob.glob(ckpt_dir + "/**/best.ckpt", recursive=True),
                  key=os.path.getmtime)[-1]
    model.load_state_dict(torch.load(ckpt, map_location="cpu")["state_dict"])
    model.eval()
    return data, model, dcfg


def predict_all(model, dataloader, device):
    """Return true labels and predicted labels for the whole test set (in order)."""
    model.to(device)
    y_true, y_pred = [], []
    with torch.no_grad():
        for images, labels in dataloader:
            scores = model(images.to(device))
            preds = scores.argmax(dim=1).cpu().numpy()
            y_pred.extend(preds.tolist())
            y_true.extend(labels.numpy().tolist())
    return np.array(y_true), np.array(y_pred)


def to_rgb(tensor, bands):
    """
    Turn a raw tensor into a displayable RGB image.
    tensor is (C, H, W) for single, or (T, C, H, W) for temporal (we pick a
    mid-season timestep). RGB = B04 (red), B03 (green), B02 (blue).
    """
    if tensor.ndim == 4:            # (T, C, H, W) -> pick a mid timestep
        tensor = tensor[len(tensor) // 2]

    r = tensor[bands.index("B04")]
    g = tensor[bands.index("B03")]
    b = tensor[bands.index("B02")]
    rgb = np.stack([r, g, b], axis=-1)  # (H, W, 3)

    # Contrast stretch so the field is visible (per-image 2-98 percentile).
    lo = np.percentile(rgb, 2)
    hi = np.percentile(rgb, 98)
    rgb = np.clip((rgb - lo) / (hi - lo + 1e-6), 0, 1)
    return rgb


def make_grid(data, dcfg, y_true, y_pred, mode, n_show=20):
    """Save a grid of test fields with true/predicted labels."""
    bands = dcfg["bands"]
    classes = data.classes
    test_ids = data.test_ids
    tensor_dir = dcfg["paths"]["tensors"]

    # Show a mix: some correct, some wrong, so the grid is informative.
    wrong = [i for i in range(len(test_ids)) if y_true[i] != y_pred[i]]
    right = [i for i in range(len(test_ids)) if y_true[i] == y_pred[i]]
    rng = np.random.default_rng(0)
    rng.shuffle(wrong)
    rng.shuffle(right)
    half = n_show // 2
    chosen = (wrong[:half] + right[:n_show - len(wrong[:half])])[:n_show]

    cols = 5
    rows = int(np.ceil(len(chosen) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 2.4, rows * 2.6))
    axes = np.array(axes).reshape(-1)

    for ax in axes:
        ax.axis("off")

    for slot, idx in enumerate(chosen):
        pid = test_ids[idx]
        tensor = np.load(os.path.join(tensor_dir, pid + ".npy"))
        rgb = to_rgb(tensor, bands)

        true_name = classes[y_true[idx]]
        pred_name = classes[y_pred[idx]]
        correct = (y_true[idx] == y_pred[idx])
        color = "green" if correct else "red"

        ax = axes[slot]
        ax.imshow(rgb)
        ax.set_title("true: " + true_name + "\npred: " + pred_name,
                     color=color, fontsize=8)

    title = "Test predictions (" + mode + ")  -  green = correct, red = wrong"
    fig.suptitle(title, fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.96])

    os.makedirs("outputs/viz", exist_ok=True)
    out = "outputs/viz/predictions_grid_" + mode + ".png"
    fig.savefig(out, dpi=130)
    plt.close(fig)
    print("Saved", out)


def make_confusion(data, y_true, y_pred, mode):
    """Save a confusion-matrix heatmap."""
    classes = data.classes
    n = len(classes)
    matrix = np.zeros((n, n), dtype=int)
    for t, p in zip(y_true, y_pred):
        matrix[t, p] += 1

    fig, ax = plt.subplots(figsize=(5.5, 5))
    im = ax.imshow(matrix, cmap="Blues")

    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(classes, rotation=45, ha="right")
    ax.set_yticklabels(classes)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("Confusion matrix (" + mode + ")")

    # Write the count in each cell.
    thresh = matrix.max() / 2 if matrix.max() > 0 else 0
    for i in range(n):
        for j in range(n):
            ax.text(j, i, str(matrix[i, j]), ha="center", va="center",
                    color="white" if matrix[i, j] > thresh else "black")

    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()

    os.makedirs("outputs/viz", exist_ok=True)
    out = "outputs/viz/confusion_matrix_" + mode + ".png"
    fig.savefig(out, dpi=130)
    plt.close(fig)
    print("Saved", out)


def main(mode):
    data, model, dcfg = load_setup(mode)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    y_true, y_pred = predict_all(model, data.test_dataloader(), device)

    acc = float((y_true == y_pred).mean())
    print(mode, "test accuracy:", round(acc, 4), "on", len(y_true), "fields")

    make_grid(data, dcfg, y_true, y_pred, mode)
    make_confusion(data, y_true, y_pred, mode)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["single", "temporal"], default="temporal")
    args = parser.parse_args()
    main(args.mode)
