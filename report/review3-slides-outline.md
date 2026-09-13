# Review 3 — Slide Outline (Panel, 20 marks · 16 Sep 2026)

**Project:** A Multi-Metric Framework for Verifying Machine Unlearning in Privacy-Preserving AI Systems
**Team:** Kanishka Mohankumar (23BDS1007) · Vignesh Venkatesan (23BDS1056) · Brijesh T A (23BDS1122) · Guide: [name]

> This is a content outline (build it into your existing deck). Review 3 is about **progress, implementation, and results** — not re-teaching the design. Aim ~12–14 min. Each slide: **content** (what's on it) + **say** (speaker note). Rubric mapping is noted per slide.

---

### Slide 1 — Title
**Content:** Title, team, guide, "Project-I · Review 3". 
**Say:** "We're presenting our implementation progress and interim results, and the changes we made in response to the Review 2 feedback."

### Slide 2 — What we'll cover (agenda) *(rubric 6)*
**Content:** Response to Review 2 feedback → what we built → results → analysis → challenges → contributions → next.
**Say:** One line signalling you'll open with the panel's own feedback — shows you listened.

### Slide 3 — Response to Review 2 feedback: the GAN *(rubric 1 — do this early)*
**Content:** "Feedback: incorporate a GAN for the image datasets." → "What we did: added a GAN-based **reconstruction verification signal** for the image workflow, behind our existing interface." Small before/after: signals were {MIA, LiRA}; now image workflow = {MIA, LiRA, **GAN reconstruction**}.
**Say:** "We incorporated the GAN not as data processing but as a new image-specific verification signal — our multi-metric framework gaining a metric. It plugs into the same interface as our other signals." This is worth a full minute; it's the follow-up mark.

### Slide 4 — System at a glance: one framework, two workflows *(rubric 2, 3)*
**Content:** Diagram — tabular workflow (Purchase-100 → MLP → MIA, LiRA) and image workflow (Fashion-MNIST/CIFAR-10 → CNN → MIA, LiRA, GAN) sharing one harness. 
**Say:** "Same harness, same interfaces; the only difference is the signal set. The GAN is image-only, so it lives in the image workflow and nowhere else."

### Slide 5 — The measurement loop (recap) *(rubric 3)*
**Content:** Retain/Forget/Test → Original vs Gold (answer key) → provider unlearns (honest/lazy/forging) → auditor runs signals blind → score vs ground truth.
**Say:** "We play all three parties, so we know the truth and can grade the auditor. The auditor never sees ground truth — that's enforced in code."

### Slide 6 — What we implemented this cycle *(rubric 2)*
**Content:** Checklist: harness + single-source split; MLP + SmallCNN; fine-tune unlearning (3 honesty levels); **3 signals** (MIA, LiRA, GAN reconstruction); DCGAN prior; failure-matrix driver + scoring; 5 test suites green.
**Say:** "This is roughly the half-implementation Review 3 expects — and the design is frozen for the methods still to come."

### Slide 7 — Verification signals implemented *(rubric 3)*
**Content:** Three rows — Aggregate MIA (population), LiRA (per-example, shadow models), GAN reconstruction (image, model inversion). One line each on what it measures.
**Say:** Emphasise LiRA is the per-example signal that carries thesis claim #1; GAN is the image-native one from the feedback.

### Slide 8 — The GAN reconstruction signal (how it works) *(rubric 1, 3)*
**Content:** DCGAN trained on retain images → manifold G(z); invert G to match a query image → reconstruction; score = model's confidence on the reconstruction. Cite arXiv 2405.20272.
**Say:** "A model that memorised a record stays confident on its reconstruction; one that never saw it doesn't. It reuses the same confidence measure as MIA, so it's directly comparable in the matrix."

### Slide 9 — Results: Purchase-100 (the memorisation signal) *(rubric 4)*
**Content:** Residual-gap table: Original forget 0.997 vs Gold 0.771 = +0.226 gap; honest/lazy/forging residuals. 
**Say:** "Only Gold forgets AND keeps utility. And plain accuracy misleads — the forging model looks *better* on test than the honest one. That's why we need signals."

### Slide 10 — Results: Purchase-100 failure matrix *(rubric 4)*
**Content:** AUC table (Gold ~0.5; Original/forging high; honest lowest) for MIA and LiRA. 
**Say:** "Gold is clean, forging equals the original exactly, and LiRA is stronger than aggregate MIA — per-example evidence beats the average."

### Slide 11 — Results: Fashion-MNIST failure matrix *(rubric 4)*
**Content:** AUC table for MIA, LiRA, GAN across the honesty levels; note memorisation gap +0.121.
**Say:** "The ordering honest < lazy < forging ≈ original holds on images too. Note the GAN column is preliminary — I'll come back to that."

### Slide 12 — Results: GAN reconstruction (the visual) *(rubric 1, 4)*
**Content:** `recon_grid.png` — forgotten images vs reconstructions.
**Say:** "The generator reconstructs data the model was asked to forget. This is the visual basis of the reconstruction attack."

### Slide 13 — Analysis & honest findings *(rubric 4, 5)*
**Content:** (1) LiRA > MIA. (2) Accuracy rewards the cheater. (3) **Memorisation regime**: signals exist only when a model memorises. (4) GAN signal is a working prototype, not yet a clean discriminator — calibration is Review 4.
**Say:** State the GAN caveat proactively — it shows you understand your own method. The regime finding shows methodological maturity.

### Slide 14 — Challenges solved *(rubric 5)*
**Content:** Device-safety bug fixed; LiRA shadows trained once and reused (~5× faster); discovered + handled the overfitting-regime requirement.
**Say:** Frame as evidence of real engineering and refinement, not smooth sailing.

### Slide 15 — Individual contributions *(rubric 7)*
**Content:** A (Kanishka): training + unlearning. B (Vignesh): verification signals + GAN + thesis. C (Brijesh): harness + split + scoring + rigour.
**Say:** Each member states their own line — panel grades individually.

### Slide 16 — Path to Review 4 + AI-use disclosure *(rubric 6)*
**Content:** Next: NegGrad/Fisher/SISA, backdoor + distance signals, CIFAR-10 at scale, multi-seed error bars, GAN calibration, dashboard. Plus one line: AI tools used for design guidance, scaffolding, debugging; decisions/understanding/validation are ours.
**Say:** Say the AI-disclosure sentence out loud — the guidelines require it.

### Slide 17 — Thank you / Q&A
**Content:** "Thank you — questions welcome." **Say:** hand to Q&A (use the drill sheet).

---
**Speaker split (individual grading):** Brijesh — slides 4, 5, 9, 10, 13, 14 (architecture, methodology, results, rigour). Vignesh — slides 3, 7, 8, 11, 12 (signals, GAN, thesis). Kanishka — slides 6 (implementation), 9-part (training), 15 method parts. Whoever opens takes 1–2.
**Timing:** never rush slides 8, 10, 13 — marks live there. If long, compress 5 and 14.
