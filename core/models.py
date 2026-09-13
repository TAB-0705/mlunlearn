"""
Models  —  Layer 2.  Owner: Person A.

Purchase-100 is tabular (600 binary features, 100 classes), so the model is a
plain MLP. The IMAGE workflow (Fashion-MNIST, CIFAR-10) uses SmallCNN.

Kept deliberately capable of memorising: memorisation of the training set is
precisely the membership signal the auditor exploits, so we must NOT regularise
it away here.
"""
from __future__ import annotations

import torch
import torch.nn as nn


class MLP(nn.Module):
    def __init__(self, in_features: int = 600, n_classes: int = 100,
                 hidden=(256, 128)):
        super().__init__()
        layers = []
        prev = in_features
        for h in hidden:
            layers += [nn.Linear(prev, h), nn.ReLU()]
            prev = h
        layers.append(nn.Linear(prev, n_classes))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class SmallCNN(nn.Module):
    """Compact CNN for the image workflow (Fashion-MNIST 1x28x28, CIFAR-10 3x32x32).

    AdaptiveAvgPool fixes the flattened size (64x4x4) regardless of input H/W, so
    the SAME class handles 28x28 and 32x32 with no shape juggling. Like the MLP it
    is deliberately left able to MEMORISE (no dropout / weight decay) — memorisation
    of the forget set is exactly the membership signal the auditor exploits.
    """

    def __init__(self, in_channels: int = 1, n_classes: int = 10):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 32, 3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.AdaptiveAvgPool2d((4, 4)),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 4 * 4, 256), nn.ReLU(),
            nn.Linear(256, n_classes),
        )

    def forward(self, x):
        return self.classifier(self.features(x))


def build_model(name: str, in_features: int, n_classes: int) -> nn.Module:
    """Build a fresh model.

    NOTE on `in_features` for the CNN: the image workflow computes
    `in_features = X.shape[1]`, which for a 4-D image tensor [N, C, H, W] is the
    CHANNEL count (1 for Fashion-MNIST, 3 for CIFAR-10). So `in_features` doubles
    as `in_channels` here — this keeps train.py / LiRA calling build_model with an
    unchanged (name, in_features, n_classes) signature for both workflows.
    """
    if name == "mlp":
        return MLP(in_features=in_features, n_classes=n_classes)
    if name == "cnn":
        return SmallCNN(in_channels=in_features, n_classes=n_classes)
    raise ValueError(f"unknown model '{name}' (known: 'mlp', 'cnn')")
