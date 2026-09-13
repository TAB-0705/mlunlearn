"""
Unlearning-layer tests. Run:  python -m tests.test_unlearn

Proves the structural guarantees (fast, synthetic):
  1. unlearn() never mutates the caller's model.
  2. forging returns an unchanged model (same forget predictions as original).
  3. honest actually moves the model (predictions change vs original).
"""
import copy

import numpy as np

from core.config import RunConfig
from core.splits import DataSplitManager
from core.data.datasets import load_purchase100, slice_split
from core.train import train_original, accuracy
from core.unlearn import FineTuneUnlearn


def _setup():
    cfg = RunConfig(dataset="purchase100", seed=0, model="mlp",
                    epochs=15, batch_size=64, lr=1e-3)
    X, y = load_purchase100(synthetic=True, n_synth=1500)
    split = DataSplitManager().get(cfg.split_spec(len(X)))
    parts = slice_split(X, y, split)
    d, k = X.shape[1], int(y.max()) + 1
    original = train_original(parts, cfg, d, k)
    return parts, original


def _forget_preds(model, parts):
    import torch
    Xf, _ = parts["forget"]
    device = next(model.parameters()).device
    with torch.no_grad():
        x = torch.from_numpy(Xf).float().to(device)
        return model(x).argmax(1).cpu().numpy()


def test_no_mutation_and_forging_unchanged():
    parts, original = _setup()
    before = _forget_preds(original, parts)
    ft = FineTuneUnlearn(honest_epochs=10, lazy_epochs=2, seed=0)
    forged = ft.unlearn(original, parts["retain"], parts["forget"], "forging")
    after_original = _forget_preds(original, parts)  # original must be untouched
    after_forged = _forget_preds(forged, parts)
    assert np.array_equal(before, after_original), "unlearn() mutated the original!"
    assert np.array_equal(before, after_forged), "forging changed predictions"
    print("  [1] no mutation + forging leaves model unchanged  OK")


def test_honest_moves_model():
    parts, original = _setup()
    before = _forget_preds(original, parts)
    ft = FineTuneUnlearn(honest_epochs=15, lazy_epochs=2, seed=0)
    honest = ft.unlearn(original, parts["retain"], parts["forget"], "honest")
    after = _forget_preds(honest, parts)
    assert not np.array_equal(before, after), "honest unlearn didn't change anything"
    print("  [2] honest fine-tune actually moves the model     OK")


if __name__ == "__main__":
    print("Unlearning-layer tests:")
    test_no_mutation_and_forging_unchanged()
    test_honest_moves_model()
    print("\nAll unlearning tests passed.")
