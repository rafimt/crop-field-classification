"""
Train the ResNet18 crop classifier.

Run it like this (after the dataset is prepared):
    python -m src.train --config configs/train.yaml
"""

import argparse

import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import (
    ModelCheckpoint,
    EarlyStopping,
    LearningRateMonitor,
)
from pytorch_lightning.loggers import CSVLogger

from src.config_utils import load_config
from src.data.datamodule import CropDataModule
from src.models.resnet18_single import create_resnet18
from src.lit_module import CropClassifier


def main(train_config_path):
    # Load both config files.
    train_cfg = load_config(train_config_path)
    data_cfg = load_config(train_cfg["data_config"])

    # Make results repeatable.
    pl.seed_everything(data_cfg["random_seed"], workers=True)

    # 1. Prepare the data.
    data = CropDataModule(
        data_config=data_cfg,
        batch_size=train_cfg["batch_size"],
        num_workers=train_cfg["num_workers"],
        augment=train_cfg["augment"],
    )
    data.setup()
    print("Channels per image:", data.num_channels)
    print("Classes:", data.classes)
    print("Train / val / test sizes:",
          len(data.train_ids), len(data.val_ids), len(data.test_ids))

    # 2. Build the model, wrapped in the Lightning module.
    network = create_resnet18(
        num_input_channels=data.num_channels,
        num_classes=data.num_classes,
        pretrained=train_cfg["pretrained"],
    )
    model = CropClassifier(
        model=network,
        num_classes=data.num_classes,
        learning_rate=train_cfg["learning_rate"],
    )

    # 3. Set up logging + callbacks.
    logger = CSVLogger(train_cfg["output_dir"], name="resnet18")
    callbacks = [
        # Keep the checkpoint with the best validation macro-F1.
        ModelCheckpoint(monitor="val_f1", mode="max", save_top_k=1, filename="best"),
        # Stop early if validation stops improving for 7 epochs.
        EarlyStopping(monitor="val_f1", mode="max", patience=7),
        LearningRateMonitor(logging_interval="epoch"),
    ]

    # 4. Train.
    # Mixed precision (16-bit) only works on a GPU. If we are on CPU, fall
    # back to normal 32-bit so training does not crash.
    if torch.cuda.is_available():
        precision = train_cfg["precision"]
        print("GPU found - training on GPU with precision", precision)
    else:
        precision = "32-true"
        print("No GPU found - training on CPU with precision 32-true "
              "(slower). See README to install the GPU build of PyTorch.")

    trainer = pl.Trainer(
        max_epochs=train_cfg["max_epochs"],
        accelerator="auto",       # use the GPU if there is one
        devices="auto",
        precision=precision,
        logger=logger,
        callbacks=callbacks,
        log_every_n_steps=10,
    )
    trainer.fit(model, data)

    # 5. Test using the best checkpoint.
    results = trainer.test(model, data, ckpt_path="best")

    # 6. Save the results to a plain text file so they are easy to read later.
    save_results(train_cfg, data, results)


def save_results(train_cfg, data, results):
    import os
    import datetime

    os.makedirs(train_cfg["output_dir"], exist_ok=True)
    out_path = os.path.join(train_cfg["output_dir"], "results.txt")

    # results is a list with one dictionary of test metrics.
    metrics = results[0] if results else {}

    lines = []
    lines.append("Crop Type Classification - Test Results")
    lines.append("Date: " + datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    lines.append("")
    lines.append("Classes: " + ", ".join(data.classes))
    lines.append("Train / Val / Test sizes: "
                 + str(len(data.train_ids)) + " / "
                 + str(len(data.val_ids)) + " / "
                 + str(len(data.test_ids)))
    lines.append("Channels per image: " + str(data.num_channels))
    lines.append("")
    lines.append("Settings:")
    lines.append("  epochs:        " + str(train_cfg["max_epochs"]))
    lines.append("  batch size:    " + str(train_cfg["batch_size"]))
    lines.append("  learning rate: " + str(train_cfg["learning_rate"]))
    lines.append("  pretrained:    " + str(train_cfg["pretrained"]))
    lines.append("")
    lines.append("Test metrics:")
    for name, value in metrics.items():
        lines.append("  " + name + ": " + str(round(float(value), 4)))

    text = "\n".join(lines) + "\n"
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(text)

    print("\n" + text)
    print("Saved results to", out_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/train.yaml")
    args = parser.parse_args()
    main(args.config)
