# Review 3 — Individual Q&A Drill Sheet

**Why this matters:** Review 3 Q&A is graded **individually** — the panel targets specific members — and much of the code was AI-assisted, so each of you must explain **your own part** cold. Practice these out loud. Answers are crisp on purpose; expand naturally.

> Three lines everyone should have memorised:
> 1. "Deleting the data isn't the hard part — removing its *learned influence* from the weights, and *verifying* that, is."
> 2. "Unlearning claimed is not unlearning verified."
> 3. "We play all three parties, so we know the ground truth — which lets us actually score how often the auditor is right."

---

## A. The GAN (Review 2 feedback) — Vignesh owns; everyone should know the one-liner

**Q: How did you incorporate the GAN the panel asked for?**
As an image-only **verification signal**, not data processing. A DCGAN learns a manifold of realistic images; we invert it to reconstruct a query image and score the target model's confidence on that reconstruction. It plugs into the same `VerificationSignal` interface as MIA and LiRA — so it's our multi-metric framework gaining a metric.

**Q: Why is a GAN reconstruction a *membership* signal?**
A model that memorised a record stays abnormally confident on its on-manifold reconstruction; a model that never saw it doesn't. Projecting onto the GAN manifold strips off-manifold noise, so the confidence read is cleaner than raw pixels. It follows the reconstruction-attack literature (arXiv 2405.20272).

**Q: Your GAN column reads high even for the Gold model — isn't that wrong?**
Yes, at this scale it over-flags and doesn't yet cleanly separate members from non-members. It's a working prototype: the prior trains, the inversion runs, it integrates through the interface, and it produces convincing reconstructions. Turning it into a calibrated per-record discriminator is our Review 4 work. We're presenting the design, the pipeline, and the visual — honestly.

**Q: Why not just use the GAN to generate more training images (augmentation)?**
That wouldn't serve verification, which is the whole project. Framing it as a verification signal keeps it inside the thesis and gives us the reconstruction evidence.

---

## B. Verification & thesis — Vignesh

**Q: What does LiRA actually compute?**
For each record, we train shadow models on random data subsets and build the distribution of confidence that models which *never saw it* would give. The target model's confidence is scored as a z-score against that OUT distribution. High z = the model is abnormally confident on that record = still remembered. We keep only the per-record confidences, not shadow weights.

**Q: Why LiRA on top of aggregate MIA?**
MIA is an average and can say "forgotten on average" while a tail of records stays exposed. LiRA is per-example, so it catches that tail. In our results LiRA scores higher than MIA on the same models — evidence for exactly that (thesis claim #1).

**Q: If every signal is defeatable, why use several?**
Any single signal has a blind spot; defeating all of them at once — behavioural and parametric — is much harder. The failure matrix measures how often a provider can, which quantifies residual risk.

**Q: Isn't a "forging" provider unrealistic?**
It's the worst case, and security research plans for the worst case. A provider with a commercial or legal incentive to appear compliant is exactly who verification must catch.

---

## C. Harness, split & rigour — Brijesh (you)

**Q: How do you know your results aren't leaking / how is the split reproducible?**
Data is split in exactly one place — `DataSplitManager` — and the split depends only on (dataset, seed, fractions), never on model or hyperparameters. So changing epochs or learning rate cannot move a record between retain and forget. `test_spine` asserts the three sets are disjoint and cover every sample once.

**Q: How do you know a measured gap isn't just noise?**
We fix seeds and report across them; for the final numbers we run multiple seeds and report mean ± std, and we don't claim a gap unless it's several std wide. Purchase-100 trains in ~2 s, so many seeds are cheap.

**Q: Why report TPR at a low FPR, not just AUC?**
AUC averages over all thresholds and hides the exposed tail; privacy harm lives in that tail. TPR at a low FPR is the operating point that matters.

**Q: Why did your image matrix first come out all ~0.5?**
Because the CNN generalised and didn't memorise — so there was no membership signal for *any* auditor, even on the Original. Membership signals only exist when a model memorises, so we run the image experiments in a deliberate overfitting regime, and treat a well-generalised model as a benign baseline. We added a memorisation-gap diagnostic to make the regime visible.

**Q: What's the Gold model and why is the auditor forbidden from seeing it?**
Gold is trained on retain only — the ideal "perfectly forgotten" model, our answer key. A real auditor never has it; if our auditor read it, the audit would be circular. It's used only afterwards, by us, to score whether the verdict was right.

---

## D. Training & unlearning — Kanishka

**Q: Why fine-tune if it's the weak baseline?**
It's the honest first thing a provider reaches for, and showing it fails to forget *selectively* (our real result) motivates the stronger methods. It's the control.

**Q: What do honest/lazy/forging mean?**
Honest = genuine effort (many epochs on retain); lazy = a token effort (few epochs); forging = claims unlearning but returns an essentially unchanged model. Forging's signal stays equal to the Original — which our matrix shows exactly.

**Q: What are the five methods and how do they differ?**
Retrain = ideal reference (perfect, expensive); fine-tune = cheap, weak (implemented); NegGrad = gradient ascent to push the forget set out; Fisher = uses parameter importance; SISA = retrain only affected shards. The last three are designed against our interface and scheduled for Review 4.

---

## E. Anyone may get these

**Q: Is your implementation about 50% done?**
Yes — harness, both workflows, three signals, unlearning with three honesty levels, and results on two datasets are complete. Remaining methods and signals are designed against frozen interfaces for Review 4.

**Q: Did you use AI tools?**
Yes, disclosed per course policy: design guidance, code scaffolding, and debugging. The decisions, the understanding, the interpretation, and the validation are ours — which this Q&A demonstrates.

**Q: Aren't you teaching people to evade audits?**
We build the forging provider only to measure and defeat it — defensive research. The output helps auditors set thresholds, not cheaters pass them. We use only public benchmark datasets, no real user data.

**Q: What's the single most important result so far?**
That plain accuracy rewards the cheater while our signals don't — and that per-example LiRA detects residual influence the aggregate metric misses. That's the project's thesis, measured.
