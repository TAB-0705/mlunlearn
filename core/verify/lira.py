"""
Per-example LiRA signal (offline / one-sided)  —  Layer 5.  Owner: Person B.

The thesis-critical auditor. Where aggregate MIA asks a population question, LiRA
asks, for EACH forgotten record individually: is the target model more confident
on this record than models that never trained on it would be?

Method (offline LiRA, Carlini et al. 2022 — the practical low-shadow variant):
  1. Train N cheap SHADOW models, each on a random half of a shadow pool.
  2. For every pooled example, collect its logit-scaled confidence (phi) from the
     shadows that did NOT train on it -> an OUT distribution (mean_out, global std).
     We keep only these phi values, never the shadow weights (the storage trick).
  3. For the TARGET model, z(example) = (phi_target - mean_out) / std. A high z
     means the target is abnormally confident on that record versus models that
     never saw it -> the record is still identifiable = still remembered.

Why it matters: aggregate MIA can read ~0.5 ("forgotten on average") while LiRA's
per-example z exposes a tail of records with large z. That gap IS thesis claim #1,
made measurable. Validate on a sanity case (known member high z, known non-member
low z) before trusting any verdict — see tests.

audit() never reads ground-truth membership; it only uses the forget/test sets it
was handed. Scoring against truth happens in scoring.py (Person C).
"""
from __future__ import annotations

from typing import Any, Dict, Tuple

import numpy as np

from core.interfaces import VerificationSignal
from core.verify._common import membership_scores
from core.verify._predict import logits_of
from core.verify.scoring import verdict_from_scores


class LiRA(VerificationSignal):
    name = "lira"

    def __init__(self, model_name: str, n_classes: int, in_features: int = 0,
                 n_shadows: int = 16, shadow_epochs: int = 20, lr: float = 1e-3,
                 batch_size: int = 128, subset_fraction: float = 0.5,
                 pool_cap: int = 6000, seed: int = 0):
        # in_features is used only by tabular models (MLP); image models ignore it.
        self.model_name = model_name
        self.n_classes = n_classes
        self.in_features = in_features
        self.n_shadows = n_shadows
        self.shadow_epochs = shadow_epochs
        self.lr = lr
        self.batch_size = batch_size
        self.subset_fraction = subset_fraction
        self.pool_cap = pool_cap
        self.seed = seed

    # ------------------------------------------------------------------ #
    def _build_pool(self, data):
        """Shadow pool = all forget + all test (so every audited record gets
        IN/OUT coverage) + a capped random sample of retain (to bound cost)."""
        Xr, yr = data["retain"]
        Xf, yf = data["forget"]
        Xt, yt = data["test"]
        rng = np.random.default_rng(self.seed)

        room = max(0, self.pool_cap - len(Xf) - len(Xt))
        if room < len(Xr):
            keep = rng.choice(len(Xr), size=room, replace=False)
            Xr, yr = Xr[keep], yr[keep]

        X = np.concatenate([Xf, Xr, Xt], axis=0)
        y = np.concatenate([yf, yr, yt], axis=0)
        forget_slice = slice(0, len(Xf))
        test_slice = slice(len(Xf) + len(Xr), len(Xf) + len(Xr) + len(Xt))
        return X, y, forget_slice, test_slice

    def _train_shadow(self, X, y, idx):
        from core.models import build_model  # torch imported lazily (kept off the import path)
        from core.train import fit
        model = build_model(self.model_name, self.in_features, self.n_classes)
        return fit(model, X[idx], y[idx], epochs=self.shadow_epochs, lr=self.lr,
                   batch_size=self.batch_size, seed=self.seed)

    # ------------------------------------------------------------------ #
    def _build_reference(self, data):
        """Train the shadow models ONCE and compute the per-example OUT
        distribution (mean_out, pooled std). This depends only on the DATA, not
        on the target model, so it is cached and reused across every model in the
        failure matrix — turning N_models x N_shadows trainings into N_shadows."""
        X, y, forget_slice, test_slice = self._build_pool(data)
        n = len(X)
        rng = np.random.default_rng(self.seed + 1)

        phi = np.full((n, self.n_shadows), np.nan, dtype=np.float64)  # keep phis, not weights
        is_in = np.zeros((n, self.n_shadows), dtype=bool)
        k = int(round(self.subset_fraction * n))
        for s in range(self.n_shadows):
            idx = rng.choice(n, size=k, replace=False)
            is_in[idx, s] = True
            shadow = self._train_shadow(X, y, idx)
            phi[:, s] = membership_scores(logits_of(shadow, X), y)

        mean_out = np.empty(n, dtype=np.float64)
        residuals = []
        for i in range(n):
            out = phi[i, ~is_in[i]]
            if out.size == 0:
                mean_out[i] = np.nanmean(phi[i])
            else:
                mean_out[i] = out.mean()
                residuals.append(out - mean_out[i])
        std = max(float(np.concatenate(residuals).std()) if residuals else 1.0, 1e-3)
        return {"X": X, "y": y, "forget_slice": forget_slice,
                "test_slice": test_slice, "mean_out": mean_out, "std": std}

    def audit(self, model: Any,
              data: Dict[str, Tuple[np.ndarray, np.ndarray]]) -> Dict:
        # cache the shadows/OUT-distribution on the instance (see _build_reference)
        if getattr(self, "_ref", None) is None:
            self._ref = self._build_reference(data)
        ref = self._ref

        phi_target = membership_scores(logits_of(model, ref["X"]), ref["y"])
        z = (phi_target - ref["mean_out"]) / ref["std"]
        return verdict_from_scores(self.name, z[ref["forget_slice"]], z[ref["test_slice"]])
