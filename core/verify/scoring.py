"""
Verdict scoring  —  Layer 5.  Owner: Person C (harness / methodology).

The signals (Person B) produce raw verdicts WITHOUT looking at ground truth.
This module is the only place ground truth enters: it grades how well a signal's
per-example scores separated true members (the forget set — records that were in
the original model) from true non-members (the held-out test set).

Keeping this split is what makes the audit honest and defensible at the panel:
the auditor never sees the answer key; the experimenter scores it afterwards.

Torch-free (numpy + sklearn) so every reported metric is unit-testable.
"""
from __future__ import annotations

from typing import Dict, Optional

import numpy as np

from core.verify._common import attack_auc, tpr_at_fpr


def score_verdict(verdict: Dict, ground_truth: Optional[Dict] = None,
                  fprs=(0.001, 0.01)) -> Dict:
    """Turn a signal verdict into graded metrics.

    Expects the verdict to carry per-example scores for the forget set (true
    members) and the test set (true non-members):
        verdict["forget_scores"], verdict["test_scores"].
    `ground_truth` is accepted for interface symmetry (Harness.score) but the
    membership labels are already implied by which set each score came from —
    forget = member, test = non-member — which IS the ground truth we hold.
    """
    m = np.asarray(verdict["forget_scores"], dtype=np.float64)
    n = np.asarray(verdict["test_scores"], dtype=np.float64)

    out = {
        "signal": verdict.get("signal", "?"),
        "auc": attack_auc(m, n),
        "n_forget": int(len(m)),
        "n_test": int(len(n)),
    }
    for f in fprs:
        out[f"tpr@fpr={f:g}"] = tpr_at_fpr(m, n, f)
    return out


def verdict_from_scores(name: str, forget_scores: np.ndarray,
                        test_scores: np.ndarray) -> Dict:
    """Helper for signals: assemble the standard verdict dict. `score` is the
    attack AUC (higher = more 'still remembered'); `per_example` is the forget
    set's per-record exposure, which is what the per-example plots visualise."""
    forget_scores = np.asarray(forget_scores, dtype=np.float64)
    test_scores = np.asarray(test_scores, dtype=np.float64)
    return {
        "signal": name,
        "score": attack_auc(forget_scores, test_scores),
        "per_example": forget_scores,
        "forget_scores": forget_scores,
        "test_scores": test_scores,
    }
