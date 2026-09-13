"""
Training harness  —  Layer 3.  Owner: Person A.

Produces the two ANCHOR models every experiment is measured against:
  - ORIGINAL      : trained on retain + forget   (has seen the data to be deleted)
  - GOLD STANDARD : trained on retain only        (the answer key — truly "unlearned")

The gap between these two on the forget set IS the membership signal the auditor
must detect. An honest unlearning method should move the original toward the gold;
a forging one only pretends to.

Device is chosen automatically: CUDA on your RTX 3050, CPU here.
"""
from __future__ import annotations

from pathlib import Path
from typing import Tuple

import numpy as np
import torch
import torch.nn as nn

from core.config import RunConfig, set_seed
from core.models import build_model


def _device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _to_tensors(X: np.ndarray, y: np.ndarray, device):
    return (torch.from_numpy(X).float().to(device),
            torch.from_numpy(y).long().to(device))


def fit(model: nn.Module, X: np.ndarray, y: np.ndarray, *,
        epochs: int, lr: float, batch_size: int, seed: int) -> nn.Module:
    """Train an EXISTING model in place. Shared by fresh training AND fine-tuning
    (the fine-tune unlearning method reuses this exact loop). Deterministic."""
    set_seed(seed)
    device = _device()
    model = model.to(device)
    Xt, yt = _to_tensors(X, y, device)

    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()
    n = Xt.shape[0]

    model.train()
    for _ in range(epochs):
        perm = torch.randperm(n, device=device)
        for i in range(0, n, batch_size):
            idx = perm[i:i + batch_size]
            opt.zero_grad()
            loss = loss_fn(model(Xt[idx]), yt[idx])
            loss.backward()
            opt.step()
    return model


def train_model(X: np.ndarray, y: np.ndarray, config: RunConfig,
                in_features: int, n_classes: int) -> nn.Module:
    """Build a fresh model and train it. Deterministic under config.seed."""
    set_seed(config.seed)  # seed BEFORE build_model so weight init is reproducible
    model = build_model(config.model, in_features, n_classes)
    return fit(model, X, y, epochs=config.epochs, lr=config.lr,
               batch_size=config.batch_size, seed=config.seed)


@torch.no_grad()
def accuracy(model: nn.Module, X: np.ndarray, y: np.ndarray) -> float:
    model.eval()
    device = _device()
    Xt, yt = _to_tensors(X, y, device)
    preds = model(Xt).argmax(dim=1)
    return (preds == yt).float().mean().item()


def train_original(parts: dict, config: RunConfig,
                   in_features: int, n_classes: int) -> nn.Module:
    """Trained on retain + forget (concatenated)."""
    Xr, yr = parts["retain"]
    Xf, yf = parts["forget"]
    X = np.concatenate([Xr, Xf], axis=0)
    y = np.concatenate([yr, yf], axis=0)
    return train_model(X, y, config, in_features, n_classes)


def train_gold(parts: dict, config: RunConfig,
               in_features: int, n_classes: int) -> nn.Module:
    """The answer key: trained on retain ONLY — never sees the forget data."""
    Xr, yr = parts["retain"]
    return train_model(Xr, yr, config, in_features, n_classes)


def save_model(model: nn.Module, config: RunConfig, name: str,
               root: str = "core/results") -> Path:
    out = Path(root) / config.hash()
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"{name}.pt"
    torch.save(model.state_dict(), path)
    return path
