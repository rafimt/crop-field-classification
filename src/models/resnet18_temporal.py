"""
Temporal model for Phase 2.

Idea:
  - Use ONE ResNet18 as a "feature extractor" that turns a single image into a
    512-number summary.
  - Run it on each timestep (each period's image), giving one summary per time.
  - Feed that sequence of summaries into a small GRU (a recurrent network that
    reads sequences), so the model can learn the timing pattern of each crop.
  - A final layer turns the GRU output into class scores.

Input shape:  (batch, T, C, H, W)   -- T images per parcel
Output shape: (batch, num_classes)
"""

import torch
import torch.nn as nn

from src.models.resnet18_single import create_resnet18


class TemporalResNet18(nn.Module):

    def __init__(self, num_input_channels, num_classes, pretrained=True, hidden_size=128):
        super().__init__()

        # Build a ResNet18 and turn it into a feature extractor by removing its
        # final classification layer.
        encoder = create_resnet18(
            num_input_channels=num_input_channels,
            num_classes=1,          # placeholder; we remove this layer next
            pretrained=pretrained,
        )
        self.feature_size = encoder.fc.in_features  # 512 for ResNet18
        encoder.fc = nn.Identity()                  # now it outputs 512 features
        self.encoder = encoder

        # A small GRU reads the sequence of per-timestep features.
        self.gru = nn.GRU(
            input_size=self.feature_size,
            hidden_size=hidden_size,
            batch_first=True,
        )

        # Final layer: hidden state -> class scores.
        self.head = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        # x shape: (batch, T, C, H, W)
        batch_size, num_times, channels, height, width = x.shape

        # Merge batch and time so we can run the encoder on every image at once.
        x = x.view(batch_size * num_times, channels, height, width)
        features = self.encoder(x)                       # (batch*T, 512)

        # Split time back out: (batch, T, 512)
        features = features.view(batch_size, num_times, self.feature_size)

        # Run the GRU over the time axis.
        gru_output, _ = self.gru(features)               # (batch, T, hidden)

        # Use the last timestep's output (it has seen the whole sequence).
        last_step = gru_output[:, -1, :]                 # (batch, hidden)

        logits = self.head(last_step)                    # (batch, num_classes)
        return logits
