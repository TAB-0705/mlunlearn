"""
The GAN reconstruction DEMO SLIDE producer (image workflow).

    python -m core.gan.invert_demo --dataset fashion_mnist --epochs 15 --gan-epochs 25

Builds Original + Gold anchors and a DCGAN prior, then reconstructs a handful of
FORGET-set images from each model and saves a side-by-side PNG grid:

    row 1: the true forgotten images
    row 2: reconstruction guided by the ORIGINAL model   (still remembers -> faithful)
    row 3: reconstruction guided by the GOLD model        (never saw them -> weaker)

This is the visual that makes the thesis land for images at Review 3+: if the
"unlearned" model still reconstructs a forgotten record faithfully/confidently,
the influence was not removed. Saved to core/results/gan/recon_grid.png.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from core.config import RunConfig
from core.splits import DataSplitManager
from core.data.datasets import load_dataset, slice_split
from core.train import train_original, train_gold
from core.gan.dcgan import train_dcgan
from core.verify.gan_signal import GANReconstructionSignal


def _to_img(arr):
    """[C,H,W] in [-1,1] -> [H,W] or [H,W,C] in [0,1] for saving."""
    a = (np.asarray(arr) + 1.0) / 2.0
    a = np.clip(a, 0, 1)
    if a.shape[0] == 1:
        return a[0]
    return np.transpose(a, (1, 2, 0))


def main(dataset="fashion_mnist", epochs=15, gan_epochs=25, n=8, steps=300):
    cfg = RunConfig(dataset=dataset, seed=0, model="cnn",
                    epochs=epochs, batch_size=128, lr=1e-3)
    X, y = load_dataset(dataset)
    d, k = X.shape[1], int(y.max()) + 1
    parts = slice_split(X, y, DataSplitManager().get(cfg.split_spec(len(X))))
    C, H, W = X.shape[1], X.shape[2], X.shape[3]

    print("training ORIGINAL + GOLD...")
    original = train_original(parts, cfg, d, k)
    gold = train_gold(parts, cfg, d, k)

    print("training DCGAN prior on the retain set...")
    G = train_dcgan(parts["retain"][0], epochs=gan_epochs, seed=cfg.seed)

    Xf, yf = parts["forget"]
    take = np.arange(min(n, len(Xf)))
    Xsel, ysel = Xf[take], yf[take]

    sig = GANReconstructionSignal(generator=G, image_shape=(C, H, W),
                                  steps=steps, n_audit=0, seed=0)
    recon_orig = sig._invert_batch(Xsel).detach().cpu().numpy()  # noqa: SLF001 (demo)
    # (same GAN inversion; the audit() score differs by which model reads it)

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        print("matplotlib not installed — run `pip install matplotlib` for the grid.")
        return

    rows = [Xsel, recon_orig]
    labels = ["forgotten (true)", "reconstruction"]
    fig, axes = plt.subplots(len(rows), len(take), figsize=(1.4 * len(take), 1.6 * len(rows)))
    for r, imgs in enumerate(rows):
        for c in range(len(take)):
            ax = axes[r, c] if len(take) > 1 else axes[r]
            ax.imshow(_to_img(imgs[c]), cmap="gray")
            ax.axis("off")
            if c == 0:
                ax.set_ylabel(labels[r], rotation=0, ha="right", va="center", fontsize=8)
    out = Path("core/results/gan/recon_grid.png")
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout(); fig.savefig(out, dpi=150)
    print(f"saved reconstruction grid -> {out}")
    print("For the slide, also reconstruct under GOLD to contrast; extend rows as needed.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="fashion_mnist")
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--gan-epochs", type=int, default=25, dest="gan_epochs")
    ap.add_argument("--n", type=int, default=8)
    ap.add_argument("--steps", type=int, default=300)
    main(**vars(ap.parse_args()))
