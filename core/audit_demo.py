"""
The Review-3 deliverable driver: run the verification signals and print the
FAILURE-MATRIX cells.

    python -m core.audit_demo --real --subsample 15000 --epochs 25          (tabular)
    python -m core.audit_demo                                               (synthetic)
    python -m core.audit_demo --dataset fashion_mnist ...                   (image)

Multi-seed (report-grade) mode — the methodology promises mean +/- std, so use this
for any number that goes in Chapter 4 or on a slide:

    python -m core.audit_demo --real --subsample 15000 --epochs 50 --seeds 3
    python -m core.audit_demo --dataset fashion_mnist --subsample 4000 \
        --epochs 60 --n-shadows 6 --gan-epochs 15 --seeds 3

--seeds N runs seeds 0..N-1 (or pass an explicit list, e.g. --seeds 0,7,42), reruns
the WHOLE pipeline per seed (split, anchors, unlearning, shadows, GAN), and reports
each cell as mean +/- std across seeds. A machine-readable dump (per-seed values +
aggregates) is written under core/results/audit/ so the report tables are copy-paste,
not hand-transcribed. With --seeds 1 (default) the original single-seed matrix is
printed unchanged.

It shows the "one framework, two workflows" seam in action: it asks the registry
which workflow the dataset belongs to and which signals that workflow runs, then
runs exactly those. Purchase-100 -> aggregate_mia + lira. Fashion-MNIST/CIFAR-10 ->
those plus gan_reconstruction.

For each unlearning outcome (gold, original, fine-tune honest/lazy/forging) it runs
every signal and reports the attack AUC and TPR at 1% FPR. Reading it:
  - AUC ~ 0.5  -> the signal thinks the record is FORGOTTEN.
  - AUC -> 1.0 -> the signal still IDENTIFIES the record = still remembered.
GOLD should read ~0.5 everywhere (truly forgotten); ORIGINAL and FORGING should read
high; HONEST should drop toward 0.5. Where a cheat reads ~0.5, that signal was
FOOLED — that is a red cell in the failure matrix, and the whole point of the study.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from core.config import RunConfig
from core.splits import DataSplitManager
from core.data.datasets import load_dataset, slice_split
from core.train import train_original, train_gold, accuracy
from core.unlearn import FineTuneUnlearn
from core.verify import (workflow_for_dataset, signal_names_for, build_signal)
from core.verify.scoring import score_verdict

FLAG_THRESHOLD = 0.55  # AUC above this = auditor says "still remembered"


def _load(dataset, real, subsample, seed):
    # synthetic fallback only applies to Purchase-100; image sets always load real.
    synthetic = (not real) and (dataset == "purchase100")
    X, y = load_dataset(dataset, synthetic=synthetic)
    if subsample and subsample < len(X):
        rng = np.random.default_rng(seed)
        keep = rng.choice(len(X), size=subsample, replace=False)
        X, y = X[keep], y[keep]
    return X, y


def _make_signals(names, cfg, in_features, n_classes, n_shadows, shadow_epochs,
                  generator=None, image_shape=None, gan_steps=300, gan_audit=64,
                  gan_score_mode="confidence"):
    signals = []
    for name in names:
        if name == "lira":
            signals.append(build_signal(
                "lira", model_name=cfg.model, n_classes=n_classes,
                in_features=in_features, n_shadows=n_shadows,
                shadow_epochs=shadow_epochs, lr=cfg.lr,
                batch_size=cfg.batch_size, seed=cfg.seed))
        elif name == "gan_reconstruction":
            signals.append(build_signal(
                "gan_reconstruction", generator=generator, image_shape=image_shape,
                steps=gan_steps, n_audit=gan_audit, seed=cfg.seed,
                score_mode=gan_score_mode))
        else:
            signals.append(build_signal(name))
    return signals


def run_once(dataset="purchase100", real=False, subsample=0, epochs=25,
             n_shadows=16, shadow_epochs=0, gan_epochs=25, gan_steps=300,
             gan_audit=64, gan_score_mode="confidence", seed=0, verbose=True):
    """Run the full pipeline for a SINGLE seed and return structured results.

    Returns a dict the printers/aggregator consume:
        {seed, workflow, model, names, synthetic, sizes, mem_gap:{forget,test,gap},
         cells: {model_label: {signal_name: {auc, tpr} | None}}}
    Every stochastic step keys off `seed` (via RunConfig.seed), so distinct seeds
    give genuinely independent runs — different split, anchors, shadows, GAN.
    """
    def say(*a):
        if verbose:
            print(*a)

    workflow = workflow_for_dataset(dataset)
    model_name = "cnn" if workflow == "image" else "mlp"
    cfg = RunConfig(dataset=dataset, seed=seed, model=model_name,
                    epochs=epochs, batch_size=128, lr=1e-3)
    shadow_epochs = shadow_epochs or epochs

    names = signal_names_for(dataset)
    say(f"[seed {seed}] dataset={dataset} -> workflow='{workflow}' "
        f"model='{model_name}' signals={names}")

    X, y = _load(dataset, real, subsample, cfg.seed)
    d, k = X.shape[1], int(y.max()) + 1
    split = DataSplitManager().get(cfg.split_spec(len(X)))
    parts = slice_split(X, y, split)
    is_synth = (not real) and (dataset == "purchase100")
    say(f"[seed {seed}] data: {'synthetic' if is_synth else 'REAL'}  sizes: {split.sizes}")

    say(f"[seed {seed}] training ORIGINAL + GOLD anchors...")
    original = train_original(parts, cfg, d, k)
    gold = train_gold(parts, cfg, d, k)

    # Memorisation diagnostic: the signals can only detect what the model memorised.
    Xf, yf = parts["forget"]; Xt, yt = parts["test"]
    of, ot = accuracy(original, Xf, yf), accuracy(original, Xt, yt)
    say(f"[seed {seed}] ORIGINAL memorisation gap: forget-acc={of:.3f} "
        f"test-acc={ot:.3f} gap={of - ot:+.3f}")

    ft = FineTuneUnlearn(honest_epochs=30, honest_lr=3e-3,
                         lazy_epochs=2, lazy_lr=1e-3, seed=cfg.seed)
    models = {
        "GOLD (forgotten)": gold,
        "ORIGINAL (before)": original,
        "finetune-honest": ft.unlearn(original, parts["retain"], parts["forget"], "honest"),
        "finetune-lazy": ft.unlearn(original, parts["retain"], parts["forget"], "lazy"),
        "finetune-forging": ft.unlearn(original, parts["retain"], parts["forget"], "forging"),
    }

    generator, image_shape = None, None
    if "gan_reconstruction" in names:
        from core.gan.dcgan import train_dcgan
        image_shape = (X.shape[1], X.shape[2], X.shape[3])
        say(f"[seed {seed}] training DCGAN prior on retain ({gan_epochs} epochs)...")
        generator = train_dcgan(parts["retain"][0], epochs=gan_epochs, seed=cfg.seed)

    signals = _make_signals(names, cfg, d, k, n_shadows, shadow_epochs,
                            generator=generator, image_shape=image_shape,
                            gan_steps=gan_steps, gan_audit=gan_audit,
                            gan_score_mode=gan_score_mode)

    cells = {}
    for label, model in models.items():
        cells[label] = {}
        for sig in signals:
            try:
                verdict = sig.audit(model, parts)
            except NotImplementedError:
                cells[label][sig.name] = None
                continue
            scored = score_verdict(verdict)
            cells[label][sig.name] = {
                "auc": float(scored["auc"]),
                "tpr": float(scored["tpr@fpr=0.01"]),
            }

    return {
        "seed": seed,
        "workflow": workflow,
        "model": model_name,
        "names": [s.name for s in signals],
        "synthetic": is_synth,
        "sizes": split.sizes,
        "mem_gap": {"forget": float(of), "test": float(ot), "gap": float(of - ot)},
        "cells": cells,
    }


# --------------------------------------------------------------------------- #
#  Printers
# --------------------------------------------------------------------------- #
def _print_single(res):
    """Original single-seed failure matrix (unchanged output)."""
    names = res["names"]
    print(f"\ndataset={res['workflow']}-workflow  model='{res['model']}'  signals={names}\n")
    print(f"data: {'synthetic' if res['synthetic'] else 'REAL'}  sizes: {res['sizes']}\n")
    mg = res["mem_gap"]
    print(f"ORIGINAL memorisation gap: forget-acc={mg['forget']:.3f}  "
          f"test-acc={mg['test']:.3f}  gap={mg['gap']:+.3f}  "
          f"(near 0 => nothing for the auditor to detect)\n")

    header = f"{'model':<20}" + "".join(f"{n:>22}" for n in names)
    print("FAILURE MATRIX  (AUC / TPR@1%FPR ; flag='R' remembered, '.' forgotten)")
    print(header)
    print("-" * len(header))
    for label, row_cells in res["cells"].items():
        row = f"{label:<20}"
        for n in names:
            c = row_cells.get(n)
            if c is None:
                row += f"{'(next step)':>22}"
                continue
            flag = "R" if c["auc"] > FLAG_THRESHOLD else "."
            cell = f"{c['auc']:.2f}/{c['tpr']:.2f}{flag}"
            row += f"{cell:>22}"
        print(row)
    _read_note()


def _agg(vals):
    """mean, std (sample std, ddof=1 when >1 value) over finite values."""
    a = np.asarray([v for v in vals if v is not None and np.isfinite(v)], dtype=float)
    if a.size == 0:
        return float("nan"), float("nan")
    if a.size == 1:
        return float(a[0]), 0.0
    return float(a.mean()), float(a.std(ddof=1))


def _print_aggregate(results, seeds):
    """Report-grade matrix: each cell is mean +/- std across seeds."""
    first = results[0]
    names = first["names"]
    print(f"\ndataset={first['workflow']}-workflow  model='{first['model']}'  "
          f"signals={names}")
    print(f"data: {'synthetic' if first['synthetic'] else 'REAL'}  "
          f"sizes: {first['sizes']}")
    print(f"seeds: {seeds}  (n={len(seeds)})\n")

    gaps = [r["mem_gap"]["gap"] for r in results]
    gm, gs = _agg(gaps)
    print(f"ORIGINAL memorisation gap: {gm:+.3f} +/- {gs:.3f}  "
          f"(near 0 => nothing for the auditor to detect)\n")

    labels = list(first["cells"].keys())

    def matrix(metric, title):
        header = f"{'model':<20}" + "".join(f"{n:>22}" for n in names)
        print(title)
        print(header)
        print("-" * len(header))
        for label in labels:
            row = f"{label:<20}"
            for n in names:
                vals = [r["cells"][label].get(n) for r in results]
                if all(v is None for v in vals):
                    row += f"{'(next step)':>22}"
                    continue
                m, s = _agg([v[metric] if v else None for v in vals])
                cell = f"{m:.2f}±{s:.2f}"
                row += f"{cell:>22}"
            print(row)
        print()

    print("FAILURE MATRIX  (mean +/- std across seeds)")
    matrix("auc", "  [AUC]")
    matrix("tpr", "  [TPR @ 1% FPR]")
    _read_note()


def _read_note():
    print("Read: GOLD should be all '.' (nothing to find). ORIGINAL/forging should")
    print("be 'R'. Any cheat that reads '.' FOOLED that signal — a red cell in the")
    print("thesis's failure matrix. Compare MIA (population) vs LiRA (per-example):")
    print("a record MIA calls forgotten but LiRA still flags is thesis claim #1.")


def _dump(results, seeds, dataset, out_root="core/results/audit"):
    """Write per-seed values + aggregates to JSON so report tables are copy-paste."""
    first = results[0]
    names = first["names"]
    labels = list(first["cells"].keys())
    agg = {}
    for label in labels:
        agg[label] = {}
        for n in names:
            for metric in ("auc", "tpr"):
                vals = [r["cells"][label].get(n) for r in results]
                m, s = _agg([v[metric] if v else None for v in vals])
                agg[label][f"{n}.{metric}"] = {"mean": m, "std": s}
    payload = {
        "dataset": dataset,
        "seeds": list(seeds),
        "workflow": first["workflow"],
        "model": first["model"],
        "sizes": first["sizes"],
        "mem_gap": {"mean": _agg([r["mem_gap"]["gap"] for r in results])[0],
                    "std": _agg([r["mem_gap"]["gap"] for r in results])[1]},
        "aggregate": agg,
        "per_seed": results,
    }
    out = Path(out_root)
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"{dataset}_seeds{'-'.join(map(str, seeds))}.json"
    path.write_text(json.dumps(payload, indent=2))
    print(f"\nwrote report-grade results -> {path}")
    return path


def _parse_seeds(spec):
    """'3' -> [0,1,2]  |  '0,7,42' -> [0,7,42]  |  '1' -> [0] (single-seed)."""
    spec = str(spec).strip()
    if "," in spec:
        return [int(x) for x in spec.split(",") if x.strip() != ""]
    n = int(spec)
    if n < 1:
        raise ValueError("--seeds must be >= 1 (or an explicit comma list)")
    return list(range(n))


def main(dataset="purchase100", real=False, subsample=0, epochs=25,
         n_shadows=16, shadow_epochs=0, gan_epochs=25, gan_steps=300, gan_audit=64,
         gan_score_mode="confidence", seeds="1"):
    seed_list = _parse_seeds(seeds)

    if len(seed_list) == 1:
        res = run_once(dataset=dataset, real=real, subsample=subsample, epochs=epochs,
                       n_shadows=n_shadows, shadow_epochs=shadow_epochs,
                       gan_epochs=gan_epochs, gan_steps=gan_steps, gan_audit=gan_audit,
                       gan_score_mode=gan_score_mode, seed=seed_list[0], verbose=True)
        _print_single(res)
        return

    results = []
    for s in seed_list:
        print(f"\n=========================  SEED {s}  =========================")
        results.append(run_once(
            dataset=dataset, real=real, subsample=subsample, epochs=epochs,
            n_shadows=n_shadows, shadow_epochs=shadow_epochs, gan_epochs=gan_epochs,
            gan_steps=gan_steps, gan_audit=gan_audit, gan_score_mode=gan_score_mode,
            seed=s, verbose=True))
    print("\n=====================  AGGREGATE (report-grade)  =====================")
    _print_aggregate(results, seed_list)
    _dump(results, seed_list, dataset)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="purchase100")
    ap.add_argument("--real", action="store_true")
    ap.add_argument("--subsample", type=int, default=0)
    ap.add_argument("--epochs", type=int, default=25)
    ap.add_argument("--n-shadows", type=int, default=16, dest="n_shadows")
    ap.add_argument("--shadow-epochs", type=int, default=0, dest="shadow_epochs")
    ap.add_argument("--gan-epochs", type=int, default=25, dest="gan_epochs")
    ap.add_argument("--gan-steps", type=int, default=300, dest="gan_steps")
    ap.add_argument("--gan-audit", type=int, default=64, dest="gan_audit")
    ap.add_argument("--gan-score-mode", default="confidence",
                    choices=["confidence", "calibrated"], dest="gan_score_mode",
                    help="GAN signal scoring: 'confidence' (R3 default) or "
                         "'calibrated' (per-class difficulty removed; reduces over-flagging)")
    ap.add_argument("--seeds", default="1", dest="seeds",
                    help="N (=> seeds 0..N-1) or an explicit list like 0,7,42. "
                         "N>1 prints mean+/-std and dumps JSON for the report.")
    main(**vars(ap.parse_args()))
