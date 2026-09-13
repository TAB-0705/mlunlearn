"""
GAN reconstruction-signal smoke test.  Run:  python -m tests.test_gan

Needs torch (auto-skips without it). Trains a TINY DCGAN on small synthetic
images and runs the signal end-to-end, asserting the plumbing is sound:
  - the generator produces images of the right shape,
  - GAN inversion returns reconstructions matching the query shape,
  - audit() returns a well-formed verdict (finite AUC in [0,1], per-example
    scores, and the reconstruction extras for the demo slide).

We assert SHAPE/finiteness, not a member>non-member direction: clean per-instance
separation is an R4 refinement, and a 1-epoch GAN on random noise has no signal.
"""
import numpy as np


def _torch_available():
    try:
        import torch  # noqa: F401
        return True
    except Exception:
        return False


def test_dcgan_and_signal_shapes():
    if not _torch_available():
        print("  [G1] SKIPPED (torch not installed here)")
        return
    from core.gan.dcgan import train_dcgan, sample
    from core.verify.gan_signal import GANReconstructionSignal
    from core.models import build_model
    from core.train import fit

    rng = np.random.default_rng(0)
    C, H, W, K = 1, 28, 28, 10
    X = (rng.random((200, C, H, W)).astype(np.float32) * 2 - 1)
    y = rng.integers(0, K, size=200).astype(np.int64)

    G = train_dcgan(X, epochs=1, batch_size=64, seed=0, verbose=False)
    s = sample(G, n=4)
    assert s.shape == (4, C, H, W), f"bad sample shape {s.shape}"

    model = build_model("cnn", C, K)
    model = fit(model, X, y, epochs=1, lr=1e-3, batch_size=64, seed=0)

    data = {"retain": (X[:120], y[:120]),
            "forget": (X[120:160], y[120:160]),
            "test": (X[160:], y[160:])}
    sig = GANReconstructionSignal(generator=G, image_shape=(C, H, W),
                                  steps=5, n_audit=16, seed=0)
    v = sig.audit(model, data)

    assert np.isfinite(v["score"]) and 0.0 <= v["score"] <= 1.0, v["score"]
    assert v["reconstructions"].shape[1:] == (C, H, W)
    assert len(v["forget_scores"]) > 0 and len(v["test_scores"]) > 0
    print(f"  [G1] DCGAN + reconstruction signal wired end-to-end (AUC={v['score']:.2f})  OK")


if __name__ == "__main__":
    print("GAN reconstruction-signal tests:")
    test_dcgan_and_signal_shapes()
    print("\nAll available GAN tests passed.")
