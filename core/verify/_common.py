"""
Torch-free scoring math shared by every verification signal  —  Layer 5.

Kept deliberately dependency-light (numpy + sklearn only) so the *statistics*
that decide every headline number can be unit-tested without a GPU or torch.
The model-touching parts live in _predict.py; the maths lives here.

Central quantity: the logit-scaled membership score (Carlini et al., LiRA 2022).
For a model's probability p on an example's TRUE label,
        phi = log(p) - log(1 - p)
Higher phi  ==  the model is more confident on the true label  ==  the example
looks more like a training MEMBER. A model that has memorised the forget set
scores high phi on it; a model that never saw it scores low phi.
"""
from __future__ import annotations

import numpy as np


def softmax(logits: np.ndarray) -> np.ndarray:
    logits = np.asarray(logits, dtype=np.float64)
    z = logits - logits.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def membership_scores(logits: np.ndarray, labels: np.ndarray,
                      eps: float = 1e-6) -> np.ndarray:
    """Logit-scaled confidence on the TRUE label. Shape [N]. Higher = more
    member-like. This is the single per-example quantity every behavioural
    signal (MIA, LiRA) is built on."""
    probs = softmax(logits)
    labels = np.asarray(labels).astype(int)
    p_true = probs[np.arange(len(labels)), labels]
    p_true = np.clip(p_true, eps, 1.0 - eps)
    return np.log(p_true) - np.log(1.0 - p_true)


def attack_auc(member_scores: np.ndarray, nonmember_scores: np.ndarray) -> float:
    """AUC of the membership test: can we tell members (forget set) from
    known non-members (test set) by their score? ~0.5 = indistinguishable
    (looks forgotten); ->1.0 = still clearly identifiable (still remembered)."""
    from sklearn.metrics import roc_auc_score
    m = np.asarray(member_scores, dtype=np.float64)
    n = np.asarray(nonmember_scores, dtype=np.float64)
    if len(m) == 0 or len(n) == 0:
        return float("nan")
    y = np.concatenate([np.ones_like(m), np.zeros_like(n)])
    s = np.concatenate([m, n])
    if not np.isfinite(s).all():
        s = np.nan_to_num(s, nan=0.0, posinf=1e9, neginf=-1e9)
    return float(roc_auc_score(y, s))


def tpr_at_fpr(member_scores: np.ndarray, nonmember_scores: np.ndarray,
               fpr: float) -> float:
    """True-positive rate at a fixed, low false-positive rate.

    THIS is the number that matters for privacy (methodology note): AUC averages
    over all thresholds and hides the exposed tail, but harm lives in the tail.
    We fix a small FPR (e.g. 0.1%) by thresholding on the NON-member (test)
    scores, then measure what fraction of members (forget) clear that bar."""
    m = np.asarray(member_scores, dtype=np.float64)
    n = np.asarray(nonmember_scores, dtype=np.float64)
    if len(m) == 0 or len(n) == 0:
        return float("nan")
    # threshold = the (1 - fpr) quantile of non-member scores
    thr = np.quantile(n, 1.0 - fpr)
    return float((m > thr).mean())
