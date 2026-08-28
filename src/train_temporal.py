"""
Train the Phase 2 temporal model (ResNet18 + GRU).

Run it AFTER preparing the temporal dataset:
    python -m src.train_temporal --config configs/train_temporal.yaml
"""

import argparse
import datetime
import os

import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import (
    ModelCheckpoint,
    EarlyStopping,
    LearningRateMonitor,
)
from pytorch_lightning.loggers import CSVLogger

from src.config_utils import load_config
from src.data.datamodule_temporal import CropTemporalDataModule
from src.models.resnet18_temporal import TemporalResNet18
from src.lit_module import CropClassifier


def save_results(train_cfg, data, results):
    os.makedirs(train_cfg["output_dir"], exist_ok=True)
    out_path = os.path.join(train_cfg["output_dir"], "results_temporal.txt")

    metrics = results[0] if results else {}
    lines = []
    lines.append("Crop Type Classification - TEMPORAL (Phase 2) Test Results")
    lines.append("Date: " + datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    lines.append("")
    lines.append("Classes: " + ", ".join(data.classes))
    lines.append("Timesteps per parcel: " + str(data.num_times))
    lines.append("Channels per image: " + str(data.num_channels))
    lines.append("Train / Val / Test sizes: "
                 + str(len(data.train_ids)) + " / "
                 + str(len(data.val_ids)) + " / "
                 + str(len(data.test_ids)))
    lines.append("")
    lines.append("Test metrics:")
    for name, value in metrics.items():
        lines.append("  " + name + ": " + str(round(float(value), 4)))

    text = "\n".join(lines) + "\n"
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(text)
    print("\n" + text)
    print("Saved results to", out_path)


def main(train_config_path):
    train_cfg = load_config(train_config_path)
    data_cfg = load_config(train_cfg["data_config"])

    pl.seed_everything(train_cfg["seed"], workers=True)

    data = CropTemporalDataModule(
        data_config=data_cfg,
        batch_size=train_cfg["batch_size"],
        num_workers=train_cfg["num_workers"],
        augment=train_cfg["augment"],
    )
    data.setup()
    print("Timesteps:", data.num_times, "| Channels:", data.num_channels)
    print("Classes:", data.classes)
    print("Train / val / test:",
          len(data.train_ids), len(data.val_ids), len(data.test_ids))

    network = TemporalResNet18(
        num_input_channels=data.num_channels,
        num_classes=data.num_classes,
        pretrained=train_cfg["pretrained"],
        hidden_size=train_cfg["hidden_size"],
    )
    model = CropClassifier(
        model=network,
        num_classes=data.num_classes,
        learning_rate=train_cfg["learning_rate"],
    )

    logger = CSVLogger(train_cfg["output_dir"], name="resnet18_temporal")
    callbacks = [
        ModelCheckpoint(monitor="val_f1", mode="max", save_top_k=1, filename="best"),
        EarlyStopping(monitor="val_f1", mode="max", patience=8),
        LearningRateMonitor(logging_interval="epoch"),
    ]

    if torch.cuda.is_available():
        precision = train_cfg["precision"]
        print("GPU found - training on GPU with precision", precision)
    else:
        precision = "32-true"
        print("No GPU found - training on CPU with precision 32-true (slower).")

    trainer = pl.Trainer(
        max_epochs=train_cfg["max_epochs"],
        accelerator="auto",
        devices="auto",
        precision=precision,
        logger=logger,
        callbacks=callbacks,
        log_every_n_steps=10,
    )
    trainer.fit(model, data)
    results = trainer.test(model, data, ckpt_path="best")
    save_results(train_cfg, data, results)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/train_temporal.yaml")
    args = parser.parse_args()
    main(args.config)
