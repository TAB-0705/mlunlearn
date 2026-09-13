"""
The first real audit: does fine-tune unlearning actually close the memorisation gap?

Run:  python -m core.demo_unlearn --real --subsample 15000 --epochs 25
      python -m core.demo_unlearn                 (synthetic)

Builds ORIGINAL and GOLD, then runs fine-tune unlearning at each honesty level and
reports forget-set accuracy + the RESIDUAL GAP vs gold (how much memorisation is
LEFT). Honest should shrink the gap; lazy less so; forging not at all — while all
of them look fine on plain test accuracy, which is the whole point.
"""
from __future__ import annotations

import argparse

import numpy as np

from core.config import RunConfig
from core.splits import DataSplitManager
from core.data.datasets import load_purchase100, slice_split
from core.train import train_original, train_gold, accuracy
from core.unlearn import FineTuneUnlearn


def main(real: bool = False, subsample: int = 0, epochs: int = 40):
    config = RunConfig(dataset="purchase100", seed=0, model="mlp",
                       epochs=epochs, batch_size=128, lr=1e-3)

    X, y = load_purchase100(synthetic=not real, n_synth=6000)
    if subsample and subsample < len(X):
        rng = np.random.default_rng(config.seed)
        keep = rng.choice(len(X), size=subsample, replace=False)
        X, y = X[keep], y[keep]
    d, k = X.shape[1], int(y.max()) + 1

    split = DataSplitManager().get(config.split_spec(len(X)))
    parts = slice_split(X, y, split)
    Xf, yf = parts["forget"]
    Xt, yt = parts["test"]

    print(f"data: {'REAL' if real else 'synthetic'}  sizes: {split.sizes}\n")
    print("training ORIGINAL and GOLD...")
    original = train_original(parts, config, d, k)
    gold = train_gold(parts, config, d, k)

    gold_forget = accuracy(gold, Xf, yf)

    def row(name, model):
        fa, ta = accuracy(model, Xf, yf), accuracy(model, Xt, yt)
        residual = fa - gold_forget          # memorisation still present vs gold
        print(f"{name:<20} {fa:.3f}     {ta:.3f}      {residual:+.3f}")

    print("\n                   forget    test     residual-gap(vs gold)")
    row("GOLD (target)", gold)
    row("ORIGINAL", original)

    ft = FineTuneUnlearn(honest_epochs=30, honest_lr=3e-3,
                         lazy_epochs=2, lazy_lr=1e-3, seed=config.seed)
    for honesty in ("honest", "lazy", "forging"):
        model = ft.unlearn(original, parts["retain"], parts["forget"], honesty)
        row(f"finetune-{honesty}", model)

    print("\nHow to read this:")
    print("- residual-gap = how much forget-set memorisation REMAINS vs gold.")
    print("  ~0 = genuinely forgotten;  large = still remembered.")
    print("- GOLD is the only model that forgets (residual 0) AND keeps test")
    print("  accuracy high. That is what true unlearning looks like.")
    print("- lazy / forging barely change: no real unlearning.")
    print("- honest fine-tune DOES erase memorisation (residual ~0) but pays for")
    print("  it in test accuracy — it damages the whole model, not just the forget")
    print("  set. Fine-tune cannot forget selectively the way gold does.")
    print("- Crucially, plain test accuracy MISLEADS: the forging model even looks")
    print("  'better' on test than the honest one. You cannot audit unlearning by")
    print("  eyeballing accuracy — you need the verification signals (next layer),")
    print("  which measure forget-set exposure directly and per-example.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--real", action="store_true")
    ap.add_argument("--subsample", type=int, default=0)
    ap.add_argument("--epochs", type=int, default=40)
    main(**vars(ap.parse_args()))
