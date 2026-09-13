"""
Config + seed system  —  Layer 0 (spine).  Owner: Person C (harness / methodology).

Two jobs:
  1. RunConfig  — the full, immutable description of one experiment run. Its hash()
     is a fingerprint stamped on every output so you always know which settings
     produced which numbers.
  2. set_seed   — makes all "random" operations repeatable across numpy / random / torch.

DESIGN NOTE (defensible at the panel):
  The DATA SPLIT must NOT depend on model/training fields (lr, epochs, ...).
  Changing the learning rate must not silently change which records are in the
  forget set. So the split identity is derived from SplitSpec (see splits.py),
  NOT from the full RunConfig. RunConfig.split_spec() extracts exactly the
  split-relevant fields.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict, field
import hashlib
import json
import random

import numpy as np


@dataclass(frozen=True)
class RunConfig:
    # --- split-relevant fields (these, and only these, define the data split) ---
    dataset: str = "purchase100"
    seed: int = 0
    test_fraction: float = 0.2      # held out first, never touched by training
    forget_fraction: float = 0.1    # fraction of the TRAIN POOL (after test held out)

    # --- model / training fields (added in later layers; do NOT affect the split) ---
    model: str = "mlp"
    epochs: int = 20
    batch_size: int = 128
    lr: float = 1e-3

    def hash(self) -> str:
        """Fingerprint of the ENTIRE run configuration."""
        payload = json.dumps(asdict(self), sort_keys=True).encode()
        return hashlib.sha256(payload).hexdigest()[:12]

    def split_spec(self, n_samples: int):
        """Extract ONLY the split-relevant fields, so the split is independent
        of model/training choices. Imported lazily to avoid a circular import."""
        from core.splits import SplitSpec
        return SplitSpec(
            dataset=self.dataset,
            n_samples=n_samples,
            seed=self.seed,
            test_fraction=self.test_fraction,
            forget_fraction=self.forget_fraction,
        )


def set_seed(seed: int) -> None:
    """Seed every RNG we use. Torch is seeded only if installed (spine stays light)."""
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        # deterministic cuDNN — slower but reproducible; the whole point of the spine
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except ImportError:
        pass  # numpy/random already seeded; torch will be seeded once installed
