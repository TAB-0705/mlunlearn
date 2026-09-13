"""
The one place a verification signal touches a torch model  —  Layer 5.

Isolated here (and mirroring core.train's device handling exactly) so the rest
of the verify layer stays torch-free and unit-testable. Works for tabular
inputs [N, D] and image inputs [N, C, H, W] alike — the model decides.
"""
from __future__ import annotations

import numpy as np


def logits_of(model, X: np.ndarray, batch_size: int = 4096) -> np.ndarray:
    """Forward pass returning raw logits as a numpy array [N, n_classes].
    Batched so a large forget/test set can't blow up VRAM."""
    import torch
    from core.train import _device

    device = _device()
    model = model.to(device)
    model.eval()
    X = np.asarray(X, dtype=np.float32)
    outs = []
    with torch.no_grad():
        for i in range(0, len(X), batch_size):
            xb = torch.from_numpy(X[i:i + batch_size]).float().to(device)
            outs.append(model(xb).detach().cpu().numpy())
    return np.concatenate(outs, axis=0) if outs else np.empty((0, 0))
