# Session Handoff — Machine Unlearning Verification: Layer 5 signals, GAN reconstruction, Review 3 deliverables

_2026-09-12_ · **Status:** Review 3 package complete (implementation + results + report + slides + Q&A). Remaining work is human — rehearsal, understanding, and a final seeded results run — before the panel on 16 Sep 2026.
_Continued from: `session-handoff.md` (proposal/scoping) and `session-handoff-project-I-1.md` (Layers 0–4 + Review 2 pivot)._

## Summary

This is BCSE497J Project-I (VIT, 3-person team, individually graded) — "A Multi-Metric Framework for Verifying Machine Unlearning in Privacy-Preserving AI Systems." Prior sessions scoped the thesis and built Layers 0–4 (harness, split spine, MLP, Original/Gold anchors, fine-tune unlearning) on Purchase-100. **This session responded to the Review 2 panel feedback ("incorporate a GAN for the image datasets") and built the Review 3 deliverables end to end:** the Layer 5 verification signals (aggregate MIA + per-example LiRA), the image workflow (SmallCNN + Fashion-MNIST/CIFAR-10 loaders), a DCGAN prior, and the GAN reconstruction signal — plus the full failure-matrix driver. It then produced the report (Chapter 4), a slide outline, an individual Q&A drill, and a finished PowerPoint deck. All code is committed to the user's repo and runs on their GPU (tests green, real failure matrices on two datasets, and a reconstruction visual).

## Resume Here

The build and results are done. If the user returns to continue, the live task is **the final report/rehearsal polish, not more building.** The single most useful next coding task available is adding a `--seeds` mean±std mode to `core/audit_demo.py` so the report tables carry error bars (the methodology promises this). Otherwise the agenda is the human checklist under "Plans / Next Steps." Do NOT start Review 4 work (new unlearning methods / signals) — that is explicitly out of scope for the 16 Sep review.

## Key Decision This Session: how the GAN was incorporated

The panel asked to "incorporate a GAN for the image datasets." The decision (Option A of the options presented) was to add the GAN as an **image-only verification signal** — a GAN-prior reconstruction/model-inversion attack — behind the existing frozen `VerificationSignal` interface, NOT as data augmentation or a separate pipeline. Rationale: it keeps the project one coherent "multi-metric" framework (the GAN is "one more metric") and it serves the verification thesis rather than bolting on an unrelated component. This framing is the answer to give in Q&A.

## Architecture: one framework, two workflows

A registry (`core/verify/__init__.py`) maps dataset → workflow → signals:
- **tabular** (Purchase-100): `aggregate_mia`, `lira`
- **image** (Fashion-MNIST, CIFAR-10): `aggregate_mia`, `lira`, `gan_reconstruction`

Same harness, same interfaces, same scoring; only the signal list differs. The GAN signal is image-only. `core/` never forks.

## Inputs to Load First (next session)

Work from the actual repo, not this document's prose. The user's repo is connected as a folder:
`C:\Users\tabri\OneDrive\Desktop\VIT\Proj - I\mlunlearn` (Windows). Key files added/changed this session are listed under "Files / Artifacts." The Python env is Python 3.12 in `.venv` with CUDA torch on an RTX 3050 (4 GB). Data (`data/purchase100.npz`) is already local; image datasets auto-download via torchvision.

## Files / Artifacts produced this session (all committed to the repo)

Verification layer (`core/verify/`): `_common.py` (torch-free scoring math — logit-scaled membership score, AUC, TPR@low-FPR), `scoring.py` (`score_verdict`, `verdict_from_scores`), `_predict.py` (`logits_of`), `mia.py` (`AggregateMIA`), `lira.py` (`LiRA` — offline shadow-model LiRA, shadows trained once and cached), `gan_signal.py` (`GANReconstructionSignal`), `__init__.py` (registry: `workflow_for_dataset`, `signal_names_for`, `build_signal`).

Image pipeline: `core/models.py` (added `SmallCNN` + `build_model('cnn')`; note `in_features` carries the channel count for CNNs), `core/data/datasets.py` (added `load_fashion_mnist`, `load_cifar10`, `load_dataset`).

GAN: `core/gan/dcgan.py` (Generator/Discriminator/`train_dcgan`/`sample`), `core/gan/invert_demo.py` (saves the reconstruction grid).

Driver + tests: `core/audit_demo.py` (failure-matrix driver; auto-trains the DCGAN for image datasets; prints AUC / TPR@1%FPR + a memorisation-gap diagnostic), `tests/test_verify.py`, `tests/test_gan.py`.

Report deliverables (in `report/`): `chapter4.md` (full Chapter 4), `review3-slides-outline.md`, `review3-qa-drill.md`, `review3-deck.pptx` (17-slide deck). Also `core/results/gan/recon_grid.png` (the reconstruction visual). Project docs: `claude/review3-build-status.md`.

## How to run (on the GPU laptop)

Dependencies to add to `.venv`: `pip install scikit-learn torchvision matplotlib` (torch/numpy already present).
- Tests: `python -m tests.test_verify` and `python -m tests.test_gan` (expect all OK on GPU).
- Tabular matrix: `python -m core.audit_demo --real --subsample 15000 --epochs 25` (raise epochs to 50–100 and `--n-shadows` to 32–64 for report figures).
- Image matrix (incl. GAN): `python -m core.audit_demo --dataset fashion_mnist --subsample 4000 --epochs 60 --n-shadows 6 --gan-epochs 15`.
- Reconstruction grid: `python -m core.gan.invert_demo --dataset fashion_mnist --gan-epochs 20 --n 8`.

## Interim Results (real data, quick-run — regenerate at higher epochs + seeds for the report)

Purchase-100 residual gap: Original forget-acc 0.997 vs Gold 0.771 = **+0.226** memorisation gap.
Purchase-100 failure matrix (AUC): Gold 0.50/0.51 (MIA/LiRA); Original 0.66/0.70; honest 0.57/0.58; lazy 0.63/0.66; forging 0.66/0.70. LiRA > MIA; forging = Original exactly.
Fashion-MNIST failure matrix (AUC, MIA/LiRA/GAN): Gold 0.53/0.52/0.58; Original 0.56/0.60/0.61; honest 0.54/0.55/0.60; lazy 0.55/0.58/0.60; forging 0.56/0.60/0.61. Memorisation gap +0.121.

## Guardrails / Out of Scope (do NOT do these)

- **Do NOT build Review 4 work now:** NegGrad/Fisher/SISA unlearning, backdoor/distance signals, CIFAR-10 at scale, the dashboard, or Phase 2. They are designed against the frozen interfaces and scheduled for Review 4; building them before 16 Sep optimises for the wrong review.
- **Do NOT oversell the GAN signal.** It is an honest working prototype: it trains, inverts, integrates, and produces convincing reconstructions, but at current scale it OVER-FLAGS (reads high even for the Gold model) and does not cleanly separate members from non-members. Present the design + the visual; state that quantitative calibration is Review 4. This caveat is in Chapter 4 (§4.9) and on deck slide 11.
- **Image experiments require an overfitting regime.** A well-generalised CNN produces no membership signal (all AUCs ~0.5, even on the Original) — that is a property of the regime, not a bug. Use small subsample + many epochs. The `audit_demo` memorisation-gap diagnostic makes this visible.
- **Keep the honesty discipline:** signals' `audit()` must never read ground truth or the Gold model; scoring against truth happens only in `scoring.py`.

## Assumptions (validate before relying on)

- Individual role mapping used in Chapter 4 / deck / Q&A: Kanishka Mohankumar (23BDS1007) = Person A (training/unlearning); Vignesh Venkatesan (23BDS1056) = Person B (verification + GAN + thesis); Brijesh T A (23BDS1122) = Person C (harness/methodology/rigour). Brijesh is the user (TAB). **Confirm the A/B split matches the real division of work.**
- Guide's name is a placeholder (`[name]`) throughout — fill it in.

## Plans / Next Steps (human checklist before 16 Sep)

- [ ] Review `chapter4.md`, confirm the role split, fill the guide's name, delete the draft note, and insert `recon_grid.png` as Figure 4.1.
- [ ] Finalise the deck (`review3-deck.pptx`): add the guide's name on slide 1; adjust the role split if needed.
- [ ] **Individual Q&A prep from `review3-qa-drill.md` — highest priority.** Each member must explain their own part cold: Brijesh (split + statistics), Vignesh (LiRA + GAN), Kanishka (training + fine-tune). It is individually graded and much of the code was AI-assisted.
- [ ] One cleaner results run (higher epochs, a couple of seeds) so report tables aren't quick-run numbers. Optional coding task: add a `--seeds` mean±std mode to `audit_demo.py`.
- [ ] Say the AI-use disclosure sentence aloud in the review (guidelines require it; it's on slide 16 and Chapter 4 §4.12).
- [ ] One timed dry-run with mock Q&A in the last day or two.

## The Thesis (do not lose)

Current unlearning verification produces a false sense of privacy. Two claims, each carries the project: (1) aggregate metrics hide individuals — per-example LiRA still identifies records the aggregate MIA calls forgotten; (2) a dishonest provider can game a known test. Deliverable: the failure matrix (unlearning method × provider honesty × verification signal → did the auditor catch it). Three memorised lines: "deleting the data isn't the hard part — removing its learned influence and verifying that is"; "unlearning claimed is not unlearning verified"; "we play all three parties, so we know the truth and can grade the auditor."

## Review schedule (from the official guidelines)

Review 3 (panel, 20 marks) — **16 Sep 2026** — ~50% implementation + interim results + Chapter 4. Review 4 (guide, 25) — 12–16 Oct. Review 5 (panel, 25, viva) — 21 Oct.

## References / Resources

GAN signal literature anchor: "Reconstruction Attacks on Machine Unlearning: Simple Models are Vulnerable" (arXiv 2405.20272); background GANMIA (IEEE), LOGAN. Thesis literature (Chapter 2): Zhang et al. ICML 2024 (verification is fragile); Xue et al. ACM Computing Surveys (behavioural/parametric taxonomy); Thudi et al. USENIX Security 2022 (unlearning unprovable at model level). Regulatory motivation: GDPR Right to Erasure, India DPDP Act, FTC algorithmic disgorgement.
