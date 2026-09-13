"""
Unlearning methods  —  Layer 4.  Owner: Person A.

The FIRST real unlearning method: fine-tune. A provider that wants to "forget" the
Delete pile takes the ORIGINAL model and keeps training it on the Keep pile only.
The hope is that continued training on retain data washes out the memorisation of
the forget data — a cheap approximation of retraining from scratch (the gold model).

This method implements the three PHASE-1 honesty levels:
  honest  — a genuine effort: fine-tune on retain for several epochs.
  lazy    — a token effort: fine-tune for very few epochs. Barely moves.
  forging — deceptive: claims full unlearning but changes nothing meaningful.
            (This is a hand-coded, signal-agnostic forge. The SMART, signal-
            targeted forge — one that actively suppresses whatever the auditor
            measures — is Person B's job once verification signals exist, and the
            LEARNED version is Phase 2.)

Note: fine-tune only ever reads the RETAIN data. It never touches forget — that is
exactly what makes it a plausible "we deleted it" story.
"""
from __future__ import annotations

import copy
from typing import Tuple

import numpy as np

from core.interfaces import UnlearnMethod
from core.train import fit


class FineTuneUnlearn(UnlearnMethod):
    name = "finetune"

    def __init__(self, honest_epochs: int = 30, honest_lr: float = 3e-3,
                 lazy_epochs: int = 2, lazy_lr: float = 1e-3,
                 batch_size: int = 128, seed: int = 0):
        # honest: strong enough to actually disrupt memorisation.
        # lazy:   a token effort that barely moves the weights.
        self.honest_epochs = honest_epochs
        self.honest_lr = honest_lr
        self.lazy_epochs = lazy_epochs
        self.lazy_lr = lazy_lr
        self.batch_size = batch_size
        self.seed = seed

    def unlearn(self, model,
                retain: Tuple[np.ndarray, np.ndarray],
                forget: Tuple[np.ndarray, np.ndarray],
                honesty: str):
        # Never mutate the caller's model — always work on a copy.
        model = copy.deepcopy(model)

        if honesty == "forging":
            # Deceptive provider: returns the (essentially unchanged) original and
            # claims it unlearned. Its membership gap will stay ~original.
            return model

        if honesty == "honest":
            epochs, lr = self.honest_epochs, self.honest_lr
        elif honesty == "lazy":
            epochs, lr = self.lazy_epochs, self.lazy_lr
        else:
            raise ValueError(f"unknown honesty '{honesty}'")

        Xr, yr = retain  # fine-tune on RETAIN only — forget is deliberately untouched
        return fit(model, Xr, yr, epochs=epochs, lr=lr,
                   batch_size=self.batch_size, seed=self.seed)
