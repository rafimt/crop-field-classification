"""
The PyTorch Lightning "wrapper" around our model.

Lightning handles the boring training loop for us. We just describe:
  - how to make a prediction (forward),
  - what happens in one training / validation / test step,
  - which optimizer to use.
"""

import torch
import torch.nn as nn
import pytorch_lightning as pl
from torchmetrics.classification import MulticlassAccuracy, MulticlassF1Score


class CropClassifier(pl.LightningModule):

    def __init__(self, model, num_classes, learning_rate=0.0003):
        super().__init__()
        self.model = model
        self.learning_rate = learning_rate

        # Cross-entropy is the standard loss for "pick one class" problems.
        self.loss_fn = nn.CrossEntropyLoss()

        # Metrics. Accuracy = % correct. Macro-F1 treats every class equally
        # (good when classes are not perfectly balanced).
        self.train_acc = MulticlassAccuracy(num_classes=num_classes)
        self.val_acc = MulticlassAccuracy(num_classes=num_classes)
        self.test_acc = MulticlassAccuracy(num_classes=num_classes)
        self.val_f1 = MulticlassF1Score(num_classes=num_classes, average="macro")
        self.test_f1 = MulticlassF1Score(num_classes=num_classes, average="macro")

    def forward(self, x):
        # Given a batch of images, return one score per class.
        return self.model(x)

    def training_step(self, batch, batch_index):
        images, labels = batch
        predictions = self(images)
        loss = self.loss_fn(predictions, labels)

        self.train_acc(predictions, labels)
        self.log("train_loss", loss, prog_bar=True)
        self.log("train_acc", self.train_acc, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_index):
        images, labels = batch
        predictions = self(images)
        loss = self.loss_fn(predictions, labels)

        self.val_acc(predictions, labels)
        self.val_f1(predictions, labels)
        self.log("val_loss", loss, prog_bar=True)
        self.log("val_acc", self.val_acc, prog_bar=True)
        self.log("val_f1", self.val_f1, prog_bar=True)

    def test_step(self, batch, batch_index):
        images, labels = batch
        predictions = self(images)

        self.test_acc(predictions, labels)
        self.test_f1(predictions, labels)
        self.log("test_acc", self.test_acc)
        self.log("test_f1", self.test_f1)

    def configure_optimizers(self):
        # Adam is a good default optimizer.
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)

        # Slowly lower the learning rate over training (cosine schedule).
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=self.trainer.max_epochs
        )
        return {"optimizer": optimizer, "lr_scheduler": scheduler}
