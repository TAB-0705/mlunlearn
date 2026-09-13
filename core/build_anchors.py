"""
Build the two anchor models and show the membership signal.

Run:  python -m core.build_anchors            (synthetic data)
      python -m core.build_anchors --real     (needs data/purchase100.npz)

Reports accuracy on retain / forget / test for both anchors. The number that
matters is FORGET accuracy: the original has trained on it (so it scores high =
"remembered"); the gold standard never saw it (so it scores near chance =
"forgotten"). That gap is exactly what an auditor tries to detect and a forging
provider tries to hide.
"""
from __future__ import annotations

import argparse

import numpy as np

from core.config import RunConfig
from core.splits import DataSplitManager
from core.data.datasets import load_purchase100, slice_split
from core.train import train_original, train_gold, accuracy, save_model


def main(real: bool = False, subsample: int = 0, epochs: int = 40):
    config = RunConfig(dataset="purchase100", seed=0, model="mlp",
                       epochs=epochs, batch_size=128, lr=1e-3)

    # Layer 1: load data (synthetic unless --real)
    X, y = load_purchase100(synthetic=not real, n_synth=6000)

    # optional subsample (literature trains Purchase-100 on ~15-20k points)
    if subsample and subsample < len(X):
        rng = np.random.default_rng(config.seed)
        keep = rng.choice(len(X), size=subsample, replace=False)
        X, y = X[keep], y[keep]

    in_features, n_classes = X.shape[1], int(y.max()) + 1

    # Layer 0: frozen split — the ONLY place data is split
    split = DataSplitManager().get(config.split_spec(len(X)))
    parts = slice_split(X, y, split)
    print(f"config hash: {config.hash()}   sizes: {split.sizes}")
    print(f"data: {'REAL Purchase-100' if real else 'synthetic (random labels)'}"
          f"  features={in_features} classes={n_classes}\n")

    # Layer 3: the two anchor models
    print("training ORIGINAL (retain + forget)...")
    original = train_original(parts, config, in_features, n_classes)
    print("training GOLD STANDARD (retain only)...")
    gold = train_gold(parts, config, in_features, n_classes)

    save_model(original, config, "original")
    save_model(gold, config, "gold")

    Xf, yf = parts["forget"]
    Xt, yt = parts["test"]
    Xr, yr = parts["retain"]

    print("\n                 retain    forget     test")
    print(f"ORIGINAL model   {accuracy(original, Xr, yr):.3f}    "
          f"{accuracy(original, Xf, yf):.3f}    {accuracy(original, Xt, yt):.3f}")
    print(f"GOLD  model      {accuracy(gold, Xr, yr):.3f}    "
          f"{accuracy(gold, Xf, yf):.3f}    {accuracy(gold, Xt, yt):.3f}")

    gap = accuracy(original, Xf, yf) - accuracy(gold, Xf, yf)
    chance = 1.0 / n_classes
    print(f"\nforget-set gap (original - gold): {gap:+.3f}   (chance = {chance:.3f})")
    print("This gap is the membership signal. On real Purchase-100 it is much larger;")
    print("on synthetic random labels it reflects pure memorisation by the original.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--real", action="store_true", help="use real Purchase-100")
    ap.add_argument("--subsample", type=int, default=0,
                    help="train on a random N-sample subset (0 = all)")
    ap.add_argument("--epochs", type=int, default=40)
    main(**vars(ap.parse_args()))
