"""
Interface contracts  —  Layer 0 (spine).

FREEZE THESE THREE SIGNATURES WITH A AND B ON DAY ONE.

They are the boundaries between the three owners. As long as everyone builds to
these, A, B, and C can work in parallel without touching each other's files or
blocking each other. Implementations live in later layers; these are the shapes.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, Tuple

import numpy as np


# --------------------------------------------------------------------------- #
# Owned by Person A — unlearning methods + training
# --------------------------------------------------------------------------- #
class UnlearnMethod(ABC):
    """Takes an ORIGINAL model (trained on retain+forget) and returns an
    'unlearned' model, given the retain and forget data and an honesty level.

    honesty ∈ {"honest", "lazy", "forging"}:
      honest  — genuinely tries to remove forget influence
      lazy    — a weak / half-hearted attempt
      forging — deliberately games the auditor's signal (defined per-signal by B)
    """

    name: str  # e.g. "finetune", "neggrad", "fisher", "sisa"

    @abstractmethod
    def unlearn(self, model: Any,
                retain: Tuple[np.ndarray, np.ndarray],
                forget: Tuple[np.ndarray, np.ndarray],
                honesty: str) -> Any:
        ...


# --------------------------------------------------------------------------- #
# Owned by Person B — verification signals + attacks + the thesis
# --------------------------------------------------------------------------- #
class VerificationSignal(ABC):
    """One independent line of evidence about whether unlearning happened.

    audit() runs the signal against a model and returns a verdict dict. It must
    NOT read ground-truth membership labels — it only sees model + data. Scoring
    against ground truth happens elsewhere (Person C), keeping the auditor honest.

    Returned dict must include at least:
      {"score": float,            # signal strength (higher = more 'still remembered')
       "per_example": np.ndarray  # optional; per-forget-sample exposure (LiRA)
      }
    """

    name: str  # e.g. "aggregate_mia", "lira", "backdoor", "distance"

    @abstractmethod
    def audit(self, model: Any, data: Dict[str, Tuple[np.ndarray, np.ndarray]]) -> Dict:
        ...


# --------------------------------------------------------------------------- #
# Owned by Person C — harness, scoring, methodology
# --------------------------------------------------------------------------- #
class Harness(ABC):
    """Drives one full run from a config and scores verdicts against ground truth."""

    @abstractmethod
    def run(self, config) -> str:
        """Execute the core measurement loop; return the results directory path."""
        ...

    @abstractmethod
    def score(self, verdict: Dict, ground_truth: Dict) -> Dict:
        """Turn a raw signal verdict into scored metrics (TP/FP, TPR@low-FPR, ...)."""
        ...
