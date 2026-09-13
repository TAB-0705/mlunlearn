"""
GAN reconstruction signal  —  Layer 5 (IMAGE workflow only).  Owner: Person B.

The signal added in response to Review-2 feedback ("incorporate a GAN for the
image datasets"). It is an IMAGE-ONLY verification signal, plugged into the SAME
frozen VerificationSignal interface as MIA and LiRA — the multi-metric framework
gaining a metric, not a side experiment. The registry attaches it only to the
image workflow (nothing to reconstruct in tabular Purchase-100).

How it works (GAN-prior reconstruction / model-inversion, cf. arXiv 2405.20272):
  1. A DCGAN (core/gan/dcgan.py), trained on in-distribution images, gives a
     learned image manifold G(z).
  2. For each query image x, we invert the GAN: optimise the latent z so G(z)
     matches x -> a reconstruction R = G(z*) that lies ON the manifold of
     realistic images (off-manifold noise, which fools raw pixel attacks, is
     stripped away).
  3. The per-example score is the TARGET model's logit-scaled confidence on its
     own reconstruction, membership_scores(model(R), y). A model that memorised x
     is abnormally confident on x's on-manifold reconstruction; a model that
     never saw x is not. That difference is the membership evidence.

Why the panel wanted it: on images this also yields a VISUAL — reconstruct a
forgotten record from the "unlearned" model vs from Gold; if the unlearned model
still reconstructs it faithfully/confidently, the influence was not removed.
See core/gan/invert_demo.py for the before/after grid.

--------------------------------------------------------------------------------
SCORING MODES (the R3 -> R4 calibration story)

  score_mode="confidence"  (default, unchanged R3 behaviour)
      Raw logit-scaled confidence on the reconstruction. HONEST KNOWN LIMITATION:
      it over-flags. Confidence on a plausible in-class reconstruction is high for
      ANY competent model — including Gold — because it mostly measures how EASY
      the class is, not whether THIS record was a member. So absolute confidence
      reads high across the board and separates members from non-members poorly.

  score_mode="calibrated"  (first calibration; the honest step toward R4)
      Removes the per-class difficulty baseline — the same insight that makes
      LiRA work, applied to the GAN signal. For each class we estimate the
      confidence a NON-member reconstruction earns (from the held-out test set),
      then report each record's confidence as a z-score against its own class's
      non-member baseline. What survives is EXCESS confidence beyond "this class
      is easy" — i.e. genuine per-record memorisation. If a signal that looked
      strong under "confidence" collapses toward 0.5 here, that is itself the
      honest finding: the apparent separation was class difficulty, not
      membership. (Caveat kept for the panel: the test set here serves as both
      the class baseline and the non-member reference, a mild optimism; a fully
      held-out calibration split and a proper per-class likelihood ratio are the
      Review-4 refinement.)

Honesty rules kept: audit() never reads ground-truth membership — it uses only
the forget/test images it was handed; scoring vs truth happens in scoring.py.
The score reuses the same logit-scaled machinery as MIA, so it is directly
comparable to the population signal in the failure matrix.
"""
from __future__ import annotations

from typing import Any, Dict, Tuple

import numpy as np

from core.interfaces import VerificationSignal
from core.verify._common import membership_scores
from core.verify.scoring import verdict_from_scores


def _calibrate_by_class(forget_scores, forget_labels, test_scores, test_labels,
                        min_std: float = 1e-3):
    """Per-class difficulty removal (torch-free, unit-testable).

    Estimate (mean, std) of the NON-member (test) confidence within each class,
    then z-score both forget and test records against their own class baseline.
    Classes absent from the test set (no baseline) fall back to the global
    non-member (mean, std). Returns (forget_cal, test_cal)."""
    f = np.asarray(forget_scores, dtype=np.float64)
    t = np.asarray(test_scores, dtype=np.float64)
    yf = np.asarray(forget_labels).astype(int)
    yt = np.asarray(test_labels).astype(int)

    g_mean = float(t.mean()) if t.size else 0.0
    g_std = max(float(t.std()) if t.size else 1.0, min_std)

    stats = {}
    for c in np.unique(yt):
        vals = t[yt == c]
        stats[int(c)] = (float(vals.mean()), max(float(vals.std()), min_std))

    def z(scores, labels):
        out = np.empty_like(scores, dtype=np.float64)
        for i, (s, c) in enumerate(zip(scores, labels)):
            mu, sd = stats.get(int(c), (g_mean, g_std))
            out[i] = (s - mu) / sd
        return out

    return z(f, yf), z(t, yt)


class GANReconstructionSignal(VerificationSignal):
    name = "gan_reconstruction"

    def __init__(self, generator=None, image_shape=None, steps: int = 300,
                 lr: float = 0.05, restarts: int = 1, n_audit: int = 64,
                 score_mode: str = "confidence", seed: int = 0):
        # generator: a trained core.gan.dcgan.Generator (the image prior).
        self.generator = generator
        self.image_shape = image_shape  # (C, H, W)
        self.steps = steps
        self.lr = lr
        self.restarts = restarts
        self.n_audit = n_audit          # cap per set (inversion is the cost)
        if score_mode not in ("confidence", "calibrated"):
            raise ValueError(f"unknown score_mode '{score_mode}'")
        self.score_mode = score_mode
        self.seed = seed

    # ------------------------------------------------------------------ #
    def _invert_batch(self, X: np.ndarray):
        """Project a batch of images onto the GAN manifold: optimise z so
        G(z) ~ x. Returns reconstructions R as a torch tensor on device."""
        import torch
        from core.train import _device

        if self.generator is None:
            raise ValueError(
                "GANReconstructionSignal needs a trained generator. Build it via "
                "build_signal('gan_reconstruction', generator=G, image_shape=(C,H,W)).")

        device = _device()
        G = self.generator.to(device).eval()
        for p in G.parameters():
            p.requires_grad_(False)

        target = torch.from_numpy(np.asarray(X, np.float32)).to(device)
        b = target.size(0)
        best_R = None
        best_err = None
        torch.manual_seed(self.seed)
        for _ in range(max(1, self.restarts)):
            z = torch.randn(b, G.z_dim, device=device, requires_grad=True)
            opt = torch.optim.Adam([z], lr=self.lr)
            for _ in range(self.steps):
                opt.zero_grad()
                R = G(z)
                err = ((R - target) ** 2).flatten(1).mean(1)  # per-image MSE
                err.sum().backward()
                opt.step()
            with torch.no_grad():
                R = G(z)
                err = ((R - target) ** 2).flatten(1).mean(1)
                if best_err is None:
                    best_R, best_err = R, err
                else:
                    take = err < best_err
                    best_R[take] = R[take]
                    best_err[take] = err[take]
        return best_R  # [b, C, H, W], on device

    def _score_set(self, model, X, y):
        import torch
        recon = self._invert_batch(X)
        with torch.no_grad():
            logits = model(recon).detach().cpu().numpy()
        scores = membership_scores(logits, y)
        return scores, recon.detach().cpu().numpy()

    # ------------------------------------------------------------------ #
    def audit(self, model: Any,
              data: Dict[str, Tuple[np.ndarray, np.ndarray]]) -> Dict:
        rng = np.random.default_rng(self.seed)

        def cap(part):
            X, y = part
            if self.n_audit and len(X) > self.n_audit:
                idx = rng.choice(len(X), size=self.n_audit, replace=False)
                return X[idx], y[idx]
            return X, y

        Xf, yf = cap(data["forget"])
        Xt, yt = cap(data["test"])
        forget_scores, forget_recon = self._score_set(model, Xf, yf)
        test_scores, _ = self._score_set(model, Xt, yt)

        if self.score_mode == "calibrated":
            forget_scores, test_scores = _calibrate_by_class(
                forget_scores, yf, test_scores, yt)

        verdict = verdict_from_scores(self.name, forget_scores, test_scores)
        # extras for the demo slide (ignored by scoring.py):
        verdict["reconstructions"] = forget_recon
        verdict["originals"] = Xf
        verdict["score_mode"] = self.score_mode
        return verdict
