# Machine Unlearning Verification — Multi-Metric Framework

Auditing whether unlearning *verification* can be trusted. We build the verification
signals, then show two failures: aggregate metrics can report "forgotten" while
individual records remain recoverable, and a dishonest provider can game the test.

Central deliverable: a **failure matrix** (unlearning method × provider honesty)
× verification signals, each cell = did the auditor catch the cheat.

## Status: Layers 0–4 complete and tested (spine + data + models + training + fine-tune unlearning).

## Structure
```
core/
  config.py          # RunConfig + hashing + set_seed        [Person C]
  splits.py          # DataSplitManager — the ONLY splitter   [Person C]  ← most important file
  interfaces.py      # frozen contracts for A / B / C         [freeze together]
  models.py          # MLP (Purchase-100); CNN/ResNet later   [Person A]
  train.py           # fit() + original & gold anchors         [Person A]
  unlearn.py         # fine-tune unlearning (honest/lazy/forge)[Person A]
  build_anchors.py   # driver: membership gap (original vs gold)
  demo_unlearn.py    # driver: does unlearning close the gap?
  data/
    datasets.py      # loaders (Purchase-100 + synthetic)     [Person A]
  results/           # models + splits, keyed by config hash (auto-created)
phase2/              # gated learned-adversary extension (imports FROM core, never the reverse)
tests/
  test_spine.py      # determinism / integrity / independence
  test_training.py   # models learn + membership gap + determinism
```

## Setup
```bash
bash setup.sh                            # creates .venv, installs deps
# on the RTX 3050, install the CUDA torch build (see setup.sh)
```

## Verify (no GPU, no real data needed)
```bash
.venv/bin/python -m tests.test_spine     # foundation
.venv/bin/python -m tests.test_training  # models train + membership gap
.venv/bin/python -m core.build_anchors   # end-to-end demo on synthetic data
```

## What the demo shows
`build_anchors` trains the two anchor models and prints accuracy on retain/forget/test.
On synthetic random-label data the **original** memorises the forget set (~0.97) while the
**gold standard** stays at chance (~0.01) — a ~0.95 gap. **That gap is the membership
signal** the auditor detects and a forging provider hides. On real Purchase-100 the test
accuracy becomes meaningful and the gap persists.

## Getting real data

**Purchase-100** (197,324 × 600 binary features, 100 classes) — from the Shokri lab:
```bash
mkdir -p data && cd data
curl -L -o dataset_purchase.tgz \
  https://raw.githubusercontent.com/privacytrustlab/datasets/master/dataset_purchase.tgz
tar xzf dataset_purchase.tgz            # -> data/dataset_purchase (raw text)
cd ..
python -m core.data.make_purchase100    # -> data/purchase100.npz  (X float32, y int64 0..99)
```
Then: `python -m core.build_anchors --real --subsample 15000 --epochs 25`

**Fashion-MNIST** and **CIFAR-10** need no manual download — torchvision fetches them
(used in a later layer): `datasets.FashionMNIST("data", download=True)` /
`datasets.CIFAR10("data", download=True)`.

Until real data is in place, every loader accepts `synthetic=True` to build and test the
pipeline with a same-shaped fake dataset.

## The rule that keeps results honest
Data is split in **`core/splits.py` and nowhere else.** The split identity depends only
on `SplitSpec` (dataset, seed, fractions) — never on model/lr/epochs. So changing a
training hyperparameter cannot silently change which records are in the forget set.
`test_spine.py` proves this.

## Interface contracts — FREEZE WITH A AND B (day one)
- **A** implements `UnlearnMethod.unlearn(model, retain, forget, honesty) -> model`
- **B** implements `VerificationSignal.audit(model, data) -> verdict`  (must not read ground truth)
- **C** implements `Harness.run(config)` and `Harness.score(verdict, ground_truth)`

## Next layers (dependency order)
data loader → models (MLP first) → training (original + gold standard) →
fine-tune unlearning → aggregate MIA + LiRA → scorer → one plot = **the reviewable vertical slice**.
