# Chapter 4 — Implementation and Results — OUTLINE (for approval)

**Project:** A Multi-Metric Framework for Verifying Machine Unlearning in Privacy-Preserving AI Systems
**Course:** BCSE497J · Project-I · Review 3 (16 Sep 2026)
**Status of this file:** section-by-section outline only. Once approved, each bullet becomes full prose. Notes in _italics_ say what real content/figure fills the section.

---

## How this maps to the Review 3 rubric (your table)
- **1. Follow-up on Review 2 feedback** → §4.6 (GAN incorporation) + §4.9 (challenges).
- **2. Implementation & functional progress (~50%)** → §4.1–4.6.
- **3. Technical accuracy / best practices** → §4.2 (frozen interfaces, single-source split), §4.7 (reproducibility).
- **4. Interim testing, results & analysis** → §4.8 (results) + §4.9 (analysis).
- **5. Problem-solving & technical refinement** → §4.10 (challenges & fixes).
- **6. Report + Chapter 4** → this chapter.
- **7. Individual contribution** → §4.11.

---

## 4.1 Overview of the Implemented System
- One-paragraph recap: what was built this cycle — a reproducible harness that trains models, applies unlearning, and audits it with multiple verification signals, scoring each audit against known ground truth.
- The "one framework, two workflows" design: tabular (Purchase-100) and image (Fashion-MNIST/CIFAR-10) share one pipeline; only the signal set differs. _This is the concrete answer to the panel's GAN request._
- A small architecture diagram (modules + data flow). _To be drawn._

## 4.2 System Architecture and Design Principles
- Layered build (Layer 0 spine → Layer 5 signals); frozen interfaces (`UnlearnMethod`, `VerificationSignal`, `Harness`) that let three members work in parallel.
- **Single-source-of-truth split**: data split only in `DataSplitManager`; split identity depends on `(dataset, seed, fractions)`, never on model/hyperparameters — proven by `test_spine`. _The strongest methodology point; keep it._
- The dataset → workflow → signal registry (`core/verify/`). Table of which signals run per workflow.

## 4.3 Datasets and Models
- Purchase-100 (tabular, 600 features, 100 classes) → MLP.
- Fashion-MNIST / CIFAR-10 (image) → SmallCNN (adaptive pooling handles 28×28 and 32×32).
- Normalisation convention ([−1,1]) shared by classifier and GAN. Local data handling (torchvision auto-download).

## 4.4 The Core Measurement Loop
- Retain / Forget / Test; Original (retain+forget) vs Gold (retain-only = answer key); provider unlearns; auditor runs signals blind; score vs ground truth.
- "We play all three parties, so we know the truth and can grade the auditor." _Figure: loop diagram (reuse the Review-2 methodology slide)._

## 4.5 Unlearning Methods Implemented
- Fine-tune unlearning with three provider honesty levels: honest / lazy / forging. What each does; why fine-tune is the weak baseline.
- _State honestly: NegGrad, Fisher, SISA are designed against the same interface and scheduled for Review 4 — do not claim them as done._

## 4.6 Verification Signals (incl. Response to Review 2 Feedback)
- **Aggregate MIA** — population-level membership test (forget-vs-test AUC).
- **Per-example LiRA** — offline likelihood-ratio with cheap shadow models; keeps only per-example logits (storage trick); shadows trained once and reused across models.
- **GAN reconstruction signal (the Review-2 feedback item)** — DCGAN prior + latent-space model inversion; per-example score = model confidence on the on-manifold reconstruction. _Frame explicitly as: incorporated as an image-only verification signal through the existing frozen interface — "the multi-metric framework gaining a metric." Cite arXiv 2405.20272._
- _Honest scope line: backdoor-trigger and weight/posterior-distance signals are designed, implementation continues toward Review 4._

## 4.7 Experimental Setup and Reproducibility
- Hardware (RTX 3050 4 GB, i5, 16 GB); ~2 s/model on Purchase-100; deterministic seeds; config-hash-stamped outputs; the three green test suites (`test_spine`, `test_training`, `test_unlearn`) + new `test_verify`, `test_gan`.
- Commands used to produce the results (audit_demo / invert_demo).

## 4.8 Interim Results
- **4.8.1 Anchor / memorisation signal (Purchase-100):** Original forget-acc 0.997 vs Gold 0.771 → +0.226 gap. _Table._
- **4.8.2 Failure matrix — Purchase-100 (tabular workflow):** AUC / TPR@1%FPR per (honesty × signal). GOLD ~0.5; ORIGINAL/forging high; honest lowest. _Table (from audit_demo). Note: run final version at higher epochs + 3 seeds for mean±std before the deck._
- **4.8.3 Failure matrix — Fashion-MNIST (image workflow):** memorisation gap +0.121; MIA/LiRA order honest < lazy < forging ≈ original; LiRA > MIA. GAN column preliminary. _Table (from the run you just did)._
- **4.8.4 GAN reconstruction visual:** the true-vs-reconstructed forgotten-image grid. _Figure: `recon_grid.png`._

## 4.9 Analysis and Interpretation
- **Utility–forgetting trade-off:** only Gold forgets AND keeps utility; fine-tune either barely forgets or damages the whole model.
- **Plain accuracy misleads:** forging model looks "better" on test than honest — motivates real signals.
- **LiRA beats aggregate MIA** (higher AUC + TPR@low-FPR) → per-example evidence matters (thesis claim #1).
- **Overfitting-regime finding (methodological, honest):** membership signals exist only when a model memorises; a well-generalised model leaks nothing (all AUCs ~0.5) — image experiments are deliberately run in an overfitting regime. _This awareness is a strength; keep it._
- **GAN signal is a working prototype, not yet a clean discriminator** (over-flags Gold at current scale) — quantitative calibration is Review-4 work; its Review-3 value is the design + the reconstruction visual. _Be explicit — protects you in Q&A._

## 4.10 Challenges and Technical Refinement
- Device-safety bug in a test helper (CPU/GPU tensor mismatch) → device-aware helper.
- LiRA cost: shadows retrained per model → refactored to train once and reuse (~5× faster).
- No membership signal on well-generalised image models → discovered the overfitting-regime requirement; added a memorisation-gap diagnostic.
- Packaging / dependency notes (sklearn, torchvision, matplotlib).

## 4.11 Individual Contributions
- **Kanishka Mohankumar (Person A):** training pipeline, models, fine-tune unlearning (`train.py`, `models.py`, `unlearn.py`).
- **Vignesh Venkatesan (Person B):** verification signals + thesis + the GAN reconstruction signal (`verify/mia.py`, `verify/lira.py`, `verify/gan_signal.py`, `gan/dcgan.py`).
- **Brijesh T A (Person C):** harness, single-source split, config/reproducibility, scoring & statistical rigour, failure-matrix driver (`splits.py`, `config.py`, `verify/scoring.py`, `audit_demo.py`).
- _Adjust the A/B split between Kanishka and Vignesh if your real division differs._

## 4.12 Summary and Path to Review 4
- What is complete (harness, 2 workflows, 3 signals, first matrices + visual).
- Review-4 targets: NegGrad/Fisher/SISA; backdoor + distance signals; CIFAR-10 full runs; multi-seed error bars; GAN-signal calibration; dashboard; (gated) Phase-2 learned adversary.

---
### Open decisions for you before I write the prose
1. Confirm the member→role mapping in §4.11 (who is Person A vs B).
2. Should §4.8 use the current quick-run numbers, or will you do a higher-epoch/3-seed run first for the report figures? (I can add a `--seeds` mean±std mode.)
3. Any VIT-template section numbering I should match instead of 4.1–4.12?
