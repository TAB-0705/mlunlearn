"""
Training-layer smoke tests. Run:  python -m tests.test_training

Fast (small synthetic, few epochs). Proves:
  1. Models actually learn (original memorises its training data).
  2. The membership gap exists (original >> gold on the forget set).
  3. Training is deterministic (same config -> same result).
"""
import numpy as np

from core.config import RunConfig
from core.splits import DataSplitManager
from core.data.datasets import load_purchase100, slice_split
from core.train import train_original, train_gold, accuracy


def _setup(seed=0):
    cfg = RunConfig(dataset="purchase100", seed=seed, model="mlp",
                    epochs=25, batch_size=64, lr=1e-3)
    X, y = load_purchase100(synthetic=True, n_synth=1500)
    split = DataSplitManager().get(cfg.split_spec(len(X)))
    parts = slice_split(X, y, split)
    return cfg, parts, X.shape[1], int(y.max()) + 1


def test_models_learn_and_gap_exists():
    cfg, parts, d, k = _setup()
    orig = train_original(parts, cfg, d, k)
    gold = train_gold(parts, cfg, d, k)
    Xr, yr = parts["retain"]
    Xf, yf = parts["forget"]
    orig_retain = accuracy(orig, Xr, yr)
    orig_forget = accuracy(orig, Xf, yf)
    gold_forget = accuracy(gold, Xf, yf)
    assert orig_retain > 0.5, f"original failed to learn retain ({orig_retain:.3f})"
    assert orig_forget > gold_forget + 0.1, \
        f"no membership gap: original {orig_forget:.3f} vs gold {gold_forget:.3f}"
    print(f"  [1] learn + gap: orig_retain={orig_retain:.3f} "
          f"forget gap={orig_forget - gold_forget:+.3f}  OK")


def test_training_deterministic():
    cfg, parts, d, k = _setup(seed=0)
    a = accuracy(train_gold(parts, cfg, d, k), *parts["retain"])
    b = accuracy(train_gold(parts, cfg, d, k), *parts["retain"])
    assert abs(a - b) < 1e-6, f"non-deterministic: {a} vs {b}"
    print(f"  [2] determinism: same config -> identical accuracy ({a:.4f})  OK")


if __name__ == "__main__":
    print("Training-layer tests:")
    test_models_learn_and_gap_exists()
    test_training_deterministic()
    print("\nAll training tests passed.")
