"""
Verification-layer tests.  Run:  python -m tests.test_verify

Split into two halves:
  A. Pure-maths tests (numpy + sklearn only) — always run. They pin the scoring
     behaviour that decides every headline number, with NO torch/GPU needed.
  B. End-to-end signal tests (need torch) — auto-skipped if torch isn't installed,
     so they run on your GPU machine but not on a bare box.

The maths sanity we assert:
  - membership scores that perfectly separate members from non-members -> AUC 1.
  - identical member/non-member distributions -> AUC ~ 0.5 (looks 'forgotten').
  - TPR at low FPR behaves monotonically.
The end-to-end sanity we assert (the real thesis check):
  - MIA/LiRA score the ORIGINAL model HIGH on the forget set and the GOLD model
    LOW — i.e. the signal detects memorisation and clears a truly-unlearned model.
"""
import numpy as np

from core.verify._common import membership_scores, attack_auc, tpr_at_fpr
from core.verify.scoring import verdict_from_scores, score_verdict


# ----------------------------- A. pure maths -------------------------------- #
def test_membership_scores_direction():
    # confident-on-true-label logits -> high phi; flat logits -> ~0 phi
    confident = np.array([[10.0, 0.0, 0.0]])
    flat = np.array([[0.0, 0.0, 0.0]])
    y = np.array([0])
    assert membership_scores(confident, y)[0] > membership_scores(flat, y)[0]
    print("  [A1] membership score increases with true-label confidence  OK")


def test_auc_separation():
    members = np.full(200, 5.0)       # members clearly more confident
    nonmembers = np.full(200, -5.0)
    assert attack_auc(members, nonmembers) > 0.99
    # identical distributions -> indistinguishable -> ~0.5
    rng = np.random.default_rng(0)
    a, b = rng.normal(size=500), rng.normal(size=500)
    assert 0.45 <= attack_auc(a, b) <= 0.55
    print("  [A2] AUC = 1 when separable, ~0.5 when identical             OK")


def test_tpr_at_fpr_monotone():
    rng = np.random.default_rng(1)
    members = rng.normal(2.0, 1.0, size=5000)
    nonmembers = rng.normal(0.0, 1.0, size=5000)
    assert tpr_at_fpr(members, nonmembers, 0.01) <= tpr_at_fpr(members, nonmembers, 0.1)
    print("  [A3] TPR at 1% FPR <= TPR at 10% FPR                         OK")


def test_score_verdict_shape():
    rng = np.random.default_rng(2)
    v = verdict_from_scores("aggregate_mia",
                            rng.normal(3, 1, 300), rng.normal(0, 1, 300))
    scored = score_verdict(v)
    assert scored["auc"] > 0.9
    assert "tpr@fpr=0.001" in scored and "tpr@fpr=0.01" in scored
    print("  [A4] score_verdict returns AUC + TPR@low-FPR                 OK")


# --------------------------- B. end-to-end (torch) -------------------------- #
def _torch_available():
    try:
        import torch  # noqa: F401
        return True
    except Exception:
        return False


def test_mia_detects_memorisation():
    if not _torch_available():
        print("  [B1] SKIPPED (torch not installed here)")
        return
    from core.config import RunConfig
    from core.splits import DataSplitManager
    from core.data.datasets import load_purchase100, slice_split
    from core.train import train_original, train_gold
    from core.verify.mia import AggregateMIA

    cfg = RunConfig(dataset="purchase100", seed=0, model="mlp",
                    epochs=15, batch_size=64, lr=1e-3)
    X, y = load_purchase100(synthetic=True, n_synth=2000)
    parts = slice_split(X, y, DataSplitManager().get(cfg.split_spec(len(X))))
    d, k = X.shape[1], int(y.max()) + 1
    original = train_original(parts, cfg, d, k)
    gold = train_gold(parts, cfg, d, k)

    mia = AggregateMIA()
    s_orig = mia.audit(original, parts)["score"]
    s_gold = mia.audit(gold, parts)["score"]
    assert s_orig > s_gold, f"MIA failed: original {s_orig:.3f} !> gold {s_gold:.3f}"
    print(f"  [B1] MIA: original={s_orig:.3f} > gold={s_gold:.3f}          OK")


def test_lira_detects_memorisation():
    if not _torch_available():
        print("  [B2] SKIPPED (torch not installed here)")
        return
    from core.config import RunConfig
    from core.splits import DataSplitManager
    from core.data.datasets import load_purchase100, slice_split
    from core.train import train_original, train_gold
    from core.verify.lira import LiRA

    cfg = RunConfig(dataset="purchase100", seed=0, model="mlp",
                    epochs=15, batch_size=64, lr=1e-3)
    X, y = load_purchase100(synthetic=True, n_synth=1500)
    parts = slice_split(X, y, DataSplitManager().get(cfg.split_spec(len(X))))
    d, k = X.shape[1], int(y.max()) + 1
    original = train_original(parts, cfg, d, k)
    gold = train_gold(parts, cfg, d, k)

    # a handful of cheap shadows is enough for the sanity direction to hold
    lira = LiRA(model_name="mlp", n_classes=k, in_features=d, n_shadows=8,
                shadow_epochs=15, batch_size=64, pool_cap=1500, seed=0)
    s_orig = lira.audit(original, parts)["score"]
    s_gold = lira.audit(gold, parts)["score"]
    assert s_orig > s_gold, f"LiRA failed: original {s_orig:.3f} !> gold {s_gold:.3f}"
    print(f"  [B2] LiRA: original={s_orig:.3f} > gold={s_gold:.3f}         OK")


if __name__ == "__main__":
    print("Verification-layer tests:")
    test_membership_scores_direction()
    test_auc_separation()
    test_tpr_at_fpr_monotone()
    test_score_verdict_shape()
    test_mia_detects_memorisation()
    test_lira_detects_memorisation()
    print("\nAll available verification tests passed.")
