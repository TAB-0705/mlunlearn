"""
The thesis figure  —  "aggregate metrics hide individuals" made visible.

Produces one report-ready, two-panel figure for a chosen model (default the
"claimed unlearned" finetune-honest model):

  LEFT  — log-scale ROC of aggregate MIA vs per-example LiRA. Privacy harm lives
          in the low-FPR tail, so we plot TPR against a LOG false-positive axis
          (the standard LiRA presentation) and mark the TPR @ 1% FPR operating
          point. LiRA sitting above MIA in the left of this plot IS the claim:
          the per-example auditor catches records the population average misses.

  RIGHT — per-record scatter on the forget set: x = aggregate-MIA confidence,
          y = per-example LiRA z-score. The dashed lines are each signal's 1%-FPR
          decision threshold (set from the non-member/test scores). Points in the
          upper-left region — LOW MIA (population says "forgotten") but HIGH LiRA
          (this record is still identifiable) — are thesis claim #1, one dot per
          exposed record.

Run:
    python -m core.plot_thesis --real --subsample 15000 --epochs 50 --n-shadows 32
    python -m core.plot_thesis --dataset fashion_mnist --subsample 4000 \
        --epochs 60 --n-shadows 6 --model "finetune-honest"

Writes core/results/plots/<dataset>_<model>_lira_vs_mia.png. Needs matplotlib.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


# --------------------------------------------------------------------------- #
#  Pure plotting (no torch) — takes score arrays, returns a saved figure path
# --------------------------------------------------------------------------- #
def _roc(member, nonmember):
    """ROC points (fpr, tpr) from member/non-member scores, higher = member."""
    from sklearn.metrics import roc_curve
    y = np.concatenate([np.ones_like(member), np.zeros_like(nonmember)])
    s = np.concatenate([member, nonmember])
    s = np.nan_to_num(s, nan=0.0, posinf=1e9, neginf=-1e9)
    fpr, tpr, _ = roc_curve(y, s)
    return fpr, tpr


def _tpr_at(member, nonmember, fpr=0.01):
    thr = np.quantile(nonmember, 1.0 - fpr)
    return float((member > thr).mean()), float(thr)


def make_figure(mia_forget, mia_test, lira_forget, lira_test, *,
                dataset, model_label, out_path, fpr_op=0.01):
    """Build the two-panel thesis figure. `*_forget` are per-record scores for the
    forget set (true members), `*_test` for the test set (true non-members).
    MIA and LiRA forget arrays must be in the SAME record order (both are in
    parts['forget'] order) so the scatter aligns per record."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.metrics import roc_auc_score

    # accessible two-hue scheme (blue = population MIA, orange = per-example LiRA)
    C_MIA, C_LIRA, C_HIT, C_BG = "#2b6cb0", "#dd6b20", "#c53030", "#a0aec0"

    def auc(m, n):
        y = np.concatenate([np.ones_like(m), np.zeros_like(n)])
        s = np.nan_to_num(np.concatenate([m, n]), nan=0.0, posinf=1e9, neginf=-1e9)
        return roc_auc_score(y, s)

    auc_mia, auc_lira = auc(mia_forget, mia_test), auc(lira_forget, lira_test)
    tpr_mia, thr_mia = _tpr_at(mia_forget, mia_test, fpr_op)
    tpr_lira, thr_lira = _tpr_at(lira_forget, lira_test, fpr_op)

    fig, (axL, axR) = plt.subplots(1, 2, figsize=(11, 4.6))

    # ---- LEFT: log-scale ROC -------------------------------------------- #
    for (m, n, c, name, a) in [
        (mia_forget, mia_test, C_MIA, "Aggregate MIA", auc_mia),
        (lira_forget, lira_test, C_LIRA, "Per-example LiRA", auc_lira)]:
        fpr, tpr = _roc(m, n)
        axL.plot(np.clip(fpr, 1e-4, 1), tpr, color=c, lw=2,
                 label=f"{name}  (AUC {a:.2f})")
    axL.plot([1e-4, 1], [1e-4, 1], ls=":", color=C_BG, lw=1, label="chance")
    axL.axvline(fpr_op, color="#718096", ls="--", lw=1)
    axL.scatter([fpr_op, fpr_op], [tpr_mia, tpr_lira],
                color=[C_MIA, C_LIRA], zorder=5, s=28)
    axL.set_xscale("log")
    axL.set_xlim(1e-3, 1)
    axL.set_ylim(0, 1)
    axL.set_xlabel("False-positive rate (log)")
    axL.set_ylabel("True-positive rate")
    axL.set_title(f"ROC — {model_label}\nTPR@{fpr_op:.0%}FPR: "
                  f"MIA {tpr_mia:.2f} vs LiRA {tpr_lira:.2f}")
    axL.legend(loc="lower right", fontsize=8, frameon=False)
    axL.grid(alpha=0.25, which="both")

    # ---- RIGHT: per-record scatter -------------------------------------- #
    exposed = (mia_forget < thr_mia) & (lira_forget > thr_lira)
    axR.scatter(mia_forget[~exposed], lira_forget[~exposed], s=14,
                color=C_BG, alpha=0.6, label="forget record")
    axR.scatter(mia_forget[exposed], lira_forget[exposed], s=22,
                color=C_HIT, alpha=0.9,
                label=f"MIA-forgotten, LiRA-flagged  (n={int(exposed.sum())})")
    axR.axvline(thr_mia, color=C_MIA, ls="--", lw=1)
    axR.axhline(thr_lira, color=C_LIRA, ls="--", lw=1)
    axR.set_xlabel("Aggregate-MIA confidence  (lower = 'forgotten')")
    axR.set_ylabel("Per-example LiRA z  (higher = 'remembered')")
    axR.set_title("Per-record exposure on the forget set\n"
                  "upper-left = population misses it, LiRA catches it")
    axR.legend(loc="upper right", fontsize=8, frameon=False)
    axR.grid(alpha=0.25)

    fig.suptitle(f"{dataset}: aggregate metrics hide individuals "
                 f"(thesis claim #1)", fontsize=12, y=1.02)
    fig.tight_layout()
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return {"path": str(out_path), "auc_mia": auc_mia, "auc_lira": auc_lira,
            "tpr_mia": tpr_mia, "tpr_lira": tpr_lira, "n_exposed": int(exposed.sum())}


# --------------------------------------------------------------------------- #
#  Pipeline wiring (torch) — build the model, run MIA + LiRA, plot
# --------------------------------------------------------------------------- #
def main(dataset="purchase100", real=False, subsample=15000, epochs=50,
         n_shadows=32, shadow_epochs=0, model="finetune-honest", seed=0):
    from core.config import RunConfig
    from core.splits import DataSplitManager
    from core.data.datasets import load_dataset, slice_split
    from core.train import train_original
    from core.unlearn import FineTuneUnlearn
    from core.verify import workflow_for_dataset, build_signal
    from core.verify.scoring import score_verdict

    workflow = workflow_for_dataset(dataset)
    model_name = "cnn" if workflow == "image" else "mlp"
    cfg = RunConfig(dataset=dataset, seed=seed, model=model_name,
                    epochs=epochs, batch_size=128, lr=1e-3)
    shadow_epochs = shadow_epochs or epochs

    synthetic = (not real) and (dataset == "purchase100")
    X, y = load_dataset(dataset, synthetic=synthetic)
    if subsample and subsample < len(X):
        rng = np.random.default_rng(seed)
        keep = rng.choice(len(X), size=subsample, replace=False)
        X, y = X[keep], y[keep]
    d, k = X.shape[1], int(y.max()) + 1
    split = DataSplitManager().get(cfg.split_spec(len(X)))
    parts = slice_split(X, y, split)

    print(f"training ORIGINAL then unlearning -> '{model}' ...")
    original = train_original(parts, cfg, d, k)
    if model in ("original", "ORIGINAL (before)"):
        target, model_label = original, "ORIGINAL (before)"
    else:
        ft = FineTuneUnlearn(honest_epochs=30, honest_lr=3e-3,
                             lazy_epochs=2, lazy_lr=1e-3, seed=cfg.seed)
        honesty = model.replace("finetune-", "") if model.startswith("finetune-") else "honest"
        target = ft.unlearn(original, parts["retain"], parts["forget"], honesty)
        model_label = f"finetune-{honesty}"

    mia = build_signal("aggregate_mia")
    lira = build_signal("lira", model_name=cfg.model, n_classes=k, in_features=d,
                        n_shadows=n_shadows, shadow_epochs=shadow_epochs,
                        lr=cfg.lr, batch_size=cfg.batch_size, seed=cfg.seed)

    print("running aggregate MIA + per-example LiRA ...")
    v_mia = mia.audit(target, parts)
    v_lira = lira.audit(target, parts)
    # print the headline numbers too, so they can go straight in the caption
    print("  MIA :", {k2: round(v, 3) for k2, v in score_verdict(v_mia).items()
                      if isinstance(v, float)})
    print("  LiRA:", {k2: round(v, 3) for k2, v in score_verdict(v_lira).items()
                      if isinstance(v, float)})

    out = f"core/results/plots/{dataset}_{model_label}_lira_vs_mia.png"
    info = make_figure(
        np.asarray(v_mia["forget_scores"]), np.asarray(v_mia["test_scores"]),
        np.asarray(v_lira["forget_scores"]), np.asarray(v_lira["test_scores"]),
        dataset=dataset, model_label=model_label, out_path=out)
    print(f"\nsaved figure -> {info['path']}")
    print(f"  claim-#1 records (MIA-forgotten, LiRA-flagged): {info['n_exposed']}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="purchase100")
    ap.add_argument("--real", action="store_true")
    ap.add_argument("--subsample", type=int, default=15000)
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--n-shadows", type=int, default=32, dest="n_shadows")
    ap.add_argument("--shadow-epochs", type=int, default=0, dest="shadow_epochs")
    ap.add_argument("--model", default="finetune-honest",
                    help="which model to audit: original | finetune-honest | "
                         "finetune-lazy | finetune-forging")
    ap.add_argument("--seed", type=int, default=0)
    main(**vars(ap.parse_args()))
