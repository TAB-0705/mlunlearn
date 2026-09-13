> **Draft note (delete before submission).** Assumptions used while drafting, per the outline: individual roles are Kanishka = Person A (training/unlearning), Vignesh = Person B (verification signals + GAN), Brijesh T A = Person C (harness/methodology/rigor); the result tables below are the current interim ("quick-run") figures and should be regenerated at higher epochs with 3 seeds (mean ± std) for the final report; section numbering is 4.1–4.12. Replace `[Guide name]` where it appears.

# Chapter 4 — Implementation and Results

## 4.1 Overview of the Implemented System

This chapter describes the working system built for the second half of Project-I and the interim results it produces. The system implements the complete measurement loop proposed in Chapter 3: it trains machine-learning models, applies machine-unlearning methods to them under different provider behaviours, audits the result with several independent verification signals, and scores each audit against ground truth that the experimenters — but not the auditor — hold. The output is a *failure matrix* that records, for every combination of unlearning method, provider honesty, and verification signal, whether the auditor correctly detected the true state of unlearning.

A central design decision, and the one that directly answers the Review 2 panel feedback, is that the framework runs as **one system with two workflows**. A *tabular* workflow (Purchase-100) and an *image* workflow (Fashion-MNIST and CIFAR-10) share the same harness, the same frozen interfaces, and the same scoring; the only difference between them is the set of verification signals each runs. The image workflow additionally runs a **GAN-based reconstruction signal**, which has no meaning for tabular data. Incorporating the GAN this way — as one more verification signal behind the existing interface rather than as a separate pipeline — keeps the project a single coherent "multi-metric" framework while satisfying the panel's request to apply a GAN to the image datasets.

At the time of this review the harness, both workflows, three verification signals, one unlearning method with three honesty levels, and the GAN prior are implemented, tested, and producing results on real data. This constitutes the approximately-half implementation expected at Review 3, with the remaining unlearning methods and signals designed against the same interfaces and scheduled for Review 4.

## 4.2 System Architecture and Design Principles

The system is organised as a sequence of layers built on a fixed "spine." The spine (Layer 0) defines three abstract interfaces that act as contracts between the three team members: `UnlearnMethod`, which takes an original model and returns an unlearned one; `VerificationSignal`, whose `audit(model, data)` method returns a verdict without ever reading membership ground truth; and `Harness`, which drives a run and scores verdicts. Because every component is written to these fixed shapes, the three members can develop the training pipeline, the verification signals, and the harness in parallel without editing one another's files.

The most important design principle, and the one most worth defending at the panel, is that **data is split in exactly one place and nowhere else**. The `DataSplitManager` derives the Retain/Forget/Test partition purely from a `SplitSpec` consisting of the dataset name, a random seed, and the split fractions. Crucially, the split does *not* depend on any model or training hyperparameter, so changing the learning rate or number of epochs cannot silently move a record from the retain set into the forget set. This property is asserted automatically by the `test_spine` test suite, which checks that the three partitions are disjoint and together cover every sample exactly once. This guarantees that any measured difference between models is attributable to unlearning and not to an accidental change in which data was used.

The two workflows are expressed through a small registry that maps each dataset to a workflow and each workflow to its list of signals. The tabular workflow runs the aggregate-MIA and LiRA signals; the image workflow runs those two plus the GAN reconstruction signal. Adding a signal is a matter of registering a class, not modifying the harness, which keeps the codebase from forking into two parallel copies.

## 4.3 Datasets and Models

Purchase-100 is a tabular dataset of roughly 197,000 records, each a 600-dimensional binary feature vector belonging to one of 100 classes. Because it is high-dimensional, sparse, and easily memorised, it is the fastest dataset to iterate on (a model trains in about two seconds) and it produces the strongest membership signal, which makes it the primary quantitative dataset. Its model is a multilayer perceptron.

Fashion-MNIST (60,000 greyscale 28×28 images, 10 classes) and CIFAR-10 (50,000 colour 32×32 images, 10 classes) are the image datasets. Both are handled by a single compact convolutional network, `SmallCNN`, whose adaptive pooling layer makes it work unchanged on both 28×28 and 32×32 inputs. All image data is normalised to the range [−1, 1], which is deliberately the same range the GAN generator produces, so that the classifier and the generator share one convention. Image datasets are downloaded automatically to a local cache on first use; no manual data preparation is required. In keeping with the project's aim, neither the MLP nor the CNN is regularised to prevent memorisation, because memorisation of the forget set is precisely the signal the auditor is designed to detect.

## 4.4 The Core Measurement Loop

For each run the data is split into Retain (about 90% of the training pool), Forget (about 10%, the records "to be deleted"), and Test (held out entirely, serving as a pool of known non-members). Two anchor models are trained: the **Original**, trained on Retain plus Forget, representing the model before deletion; and the **Gold** model, trained on Retain only, representing the ideal "perfectly forgotten" state. The Gold model is the experimental answer key — a real auditor never has it, but the experimenters do, which is what makes it possible to score the auditor objectively.

A provider then takes the Original model and applies an unlearning method under a chosen honesty level. The resulting model is passed to the auditor, which runs each verification signal **without seeing which records were truly members**. Finally the harness compares each signal's verdict against the known ground truth and records the outcome. The experimenters play all three roles — data owner, provider, and auditor — so the correct answer is always known and the auditor's accuracy can be measured like the sensitivity and specificity of a diagnostic test.

## 4.5 Unlearning Methods Implemented

The unlearning method implemented in this cycle is **fine-tuning**: the provider continues training the Original model on the Retain set only, in the hope that further training washes out the memorisation of the Forget set. It is a cheap approximation of retraining from scratch and is deliberately the weakest realistic method, which makes it a useful baseline. It is implemented at three provider honesty levels: *honest* (a genuine effort — many epochs of fine-tuning), *lazy* (a token effort — very few epochs), and *forging* (a deceptive provider that claims to have unlearned but returns an essentially unchanged model). The forging level is signal-agnostic at this stage; a smarter, signal-targeted forger is future work.

The three stronger, targeted methods named in the objectives — NegGrad (gradient ascent on the forget set), a Fisher-information-based method, and SISA (sharded retraining) — are designed against the same `UnlearnMethod` interface and are scheduled for Review 4. They are not claimed as complete here.

## 4.6 Verification Signals, including the Response to Review 2 Feedback

Three verification signals are implemented, each an independent line of evidence.

**Aggregate membership inference (MIA)** is the cheap, population-level baseline. It compares the model's logit-scaled confidence on the forget set against its confidence on the held-out test set, and reports the AUC of distinguishing the two. On the Original model this AUC is high (the forget records are still identifiable); on the Gold model it falls to about 0.5 (the forget records look like strangers). Its blind spot is that it is an average, so it can report "forgotten on average" while a minority of records remain fully exposed.

**Per-example LiRA** addresses that blind spot. It trains a set of cheap shadow models on random halves of a data pool and, for each record, builds a distribution of the confidence that models which never saw it would assign; the target model's confidence is then scored against that distribution as a per-record z-score. Only the per-example confidences are stored, never the shadow weights, keeping the memory cost small. To keep the cost manageable across the whole failure matrix, the shadow models are trained once and reused for every audited model, since they depend only on the data and not on the target.

**GAN reconstruction (the Review 2 feedback item)** is the image-only signal. A DCGAN is trained on in-distribution images to provide a learned manifold of realistic images, G(z). For each query image the generator is inverted — the latent z is optimised so that G(z) matches the image — producing an on-manifold reconstruction; the per-record score is the target model's confidence on that reconstruction. The intuition, following recent work on reconstruction attacks against unlearning, is that a model which memorised a record remains abnormally confident on its reconstruction, whereas a model that never saw it does not. This signal is incorporated deliberately as one more `VerificationSignal` behind the existing interface, so it extends the multi-metric framework rather than sitting beside it, and it additionally yields a visual demonstration (Section 4.8.4). The two remaining signals from the objectives — a backdoor-trigger signal and a weight/posterior-distance signal — are designed and scheduled for Review 4.

## 4.7 Experimental Setup and Reproducibility

All experiments run on a laptop with an NVIDIA RTX 3050 (4 GB), an Intel i5 processor, and 16 GB of RAM. Every run is deterministic: a single seeding function fixes the random state across NumPy, Python, and PyTorch, and cuDNN is set to deterministic mode. Each run's full configuration is hashed and stamped on its outputs, so any reported number can be traced back to the exact settings that produced it. Correctness is guarded by test suites: `test_spine` (split integrity and independence), `test_training` (models learn and the membership gap appears), `test_unlearn` (unlearning does not mutate the caller's model and behaves per honesty level), `test_verify` (the scoring mathematics, plus an on-GPU check that both MIA and LiRA score the Original above the Gold), and `test_gan` (the DCGAN and reconstruction signal run end to end). Results in this chapter were produced by the harness's driver scripts on real data.

## 4.8 Interim Results

### 4.8.1 The membership signal on Purchase-100

The two anchor models generalise almost identically on held-out test data, but differ sharply on the forget set. The Original scores about 0.997 accuracy on the forget records it memorised, while the Gold model scores about 0.771 — the accuracy of a model that treats those records as strangers. The resulting **+0.226 gap is the memorisation the auditor must detect and a forging provider must hide.** That the two models are otherwise equally good confirms the experimental setup is sound.

| Model | Forget acc | Test acc | Residual gap vs Gold |
|---|---|---|---|
| Gold (ideal target) | 0.771 | 0.771 | +0.000 |
| Original (before unlearning) | 0.997 | 0.793 | +0.226 |
| Fine-tune — honest | 0.757 | 0.659 | −0.013 |
| Fine-tune — lazy | 0.958 | 0.756 | +0.187 |
| Fine-tune — forging | 0.997 | 0.793 | +0.226 |

Two findings already emerge. First, only the Gold model both forgets and keeps its utility; honest fine-tuning erases memorisation but damages overall test accuracy (0.79 → 0.66), and lazy/forging barely forget at all — fine-tuning cannot forget *selectively*. Second, plain accuracy misleads: the forging model looks *better* on test accuracy than the honest one, so judging unlearning by accuracy would rank the cheater above the honest provider. This is the strongest motivation for dedicated verification signals.

### 4.8.2 Failure matrix — Purchase-100 (tabular workflow)

Running the two tabular signals across the honesty levels gives the first failure-matrix cells (AUC; higher = still identifiable).

| Model | Aggregate MIA | LiRA |
|---|---|---|
| Gold (forgotten) | 0.50 | 0.51 |
| Original (before) | 0.66 | 0.70 |
| Fine-tune — honest | 0.57 | 0.58 |
| Fine-tune — lazy | 0.63 | 0.66 |
| Fine-tune — forging | 0.66 | 0.70 |

The Gold model reads ≈0.5 on both signals (nothing to find), while the Original and the forging provider read high, and forging equals the Original exactly — a forging provider changed nothing, as intended. LiRA is consistently stronger than aggregate MIA, which supports the thesis that per-example evidence detects residual influence that a population average can miss.

### 4.8.3 Failure matrix — Fashion-MNIST (image workflow)

On images the model must be pushed into an overfitting regime before any membership signal exists (Section 4.9). In that regime the Original shows a forget-vs-test accuracy gap of +0.121, and the three signals give:

| Model | Aggregate MIA | LiRA | GAN reconstruction |
|---|---|---|---|
| Gold (forgotten) | 0.53 | 0.52 | 0.58 |
| Original (before) | 0.56 | 0.60 | 0.61 |
| Fine-tune — honest | 0.54 | 0.55 | 0.60 |
| Fine-tune — lazy | 0.55 | 0.58 | 0.60 |
| Fine-tune — forging | 0.56 | 0.60 | 0.61 |

The MIA and LiRA columns reproduce the expected ordering — honest < lazy < forging ≈ original — with LiRA again the stronger signal. The GAN reconstruction column currently reads high for every model including Gold, indicating that, at this scale, it does not yet cleanly separate members from non-members; it is a working prototype whose quantitative calibration is Review 4 work (Section 4.9).

### 4.8.4 GAN reconstruction visual

The clearest demonstration of the GAN signal at this stage is qualitative. Figure 4.1 shows eight forgotten Fashion-MNIST images (top row) beside their DCGAN reconstructions (bottom row). The reconstructions are recognisably the same garments, showing that the generator has learned a manifold rich enough to reconstruct data the model was asked to forget — the visual foundation of the reconstruction attack.

*(Figure 4.1: `core/results/gan/recon_grid.png` — true forgotten images vs their GAN reconstructions.)*

## 4.9 Analysis and Interpretation

The results support the project's two central claims and surface one honest methodological point. The **utility–forgetting trade-off** shows that cheap fine-tuning cannot match the Gold model, motivating the stronger methods to come. The finding that **plain accuracy rewards the cheater** is the clearest argument for verification signals at all. The observation that **LiRA outperforms aggregate MIA** is direct evidence for the thesis that aggregate metrics can under-report individual exposure.

A methodological finding worth stating plainly is that **membership signals exist only when a model memorises.** A convolutional network trained on the full Fashion-MNIST set generalises so well that its confidence on forget and test images is nearly identical, and every signal reads ≈0.5 — including on the Original, before any unlearning. This is not a failure of the signals; it is a property of the regime. The image experiments are therefore deliberately run in an overfitting regime (fewer images, more epochs), and a well-generalised model is understood as a benign baseline that leaks nothing. A memorisation-gap diagnostic was added to the harness to make this regime visible at a glance.

Finally, the **GAN reconstruction signal is presented as a working prototype rather than a finished discriminator.** It trains, inverts, and integrates through the frozen interface, and it produces convincing reconstructions, but at the current scale it over-flags the Gold model and does not yet yield a clean quantitative separation. Its value at Review 3 is the design and the visual; calibrating it into a reliable per-record signal is scheduled for Review 4. Stating this openly is deliberate: it reflects an accurate understanding of what the method does and does not yet achieve.

## 4.10 Challenges and Technical Refinement

Several concrete problems were identified and resolved during implementation. A device-safety bug, in which a test helper built an input tensor on the CPU while the model was on the GPU, was fixed by making the helper query the model's device. LiRA was initially retraining its shadow models for every audited model, which was acceptable on fast tabular data but prohibitively slow on image CNNs; it was refactored to train the shadow set once and reuse it across all models, an approximately fivefold speed-up. The discovery that image models produced no signal led to the overfitting-regime analysis above and to the memorisation-gap diagnostic. Dependency and packaging issues (scikit-learn, torchvision, and matplotlib requirements) were identified and documented so the pipeline runs cleanly on a fresh environment.

## 4.11 Individual Contributions

**Kanishka Mohankumar (23BDS1007) — Person A, training and unlearning.** Implemented the model definitions (the MLP and `SmallCNN`), the training loop and the two anchor models, and the fine-tune unlearning method with its three honesty levels (`models.py`, `train.py`, `unlearn.py`).

**Vignesh Venkatesan (23BDS1056) — Person B, verification and the thesis.** Implemented the verification signals — aggregate MIA and per-example LiRA — and, in response to the Review 2 feedback, the DCGAN prior and the GAN reconstruction signal, and owns the intellectual argument of the project (`verify/mia.py`, `verify/lira.py`, `verify/gan_signal.py`, `gan/dcgan.py`).

**Brijesh T A (23BDS1122) — Person C, harness, methodology and rigour.** Implemented the configuration and seeding system, the single-source-of-truth data split, the verdict scoring (AUC and TPR at a fixed low false-positive rate), and the failure-matrix driver, and owns the experimental methodology and statistical rigour (`config.py`, `splits.py`, `verify/scoring.py`, `audit_demo.py`).

## 4.12 Disclosure of AI-Tool Use

In accordance with the course guidelines, the team discloses the use of AI assistance (Claude) during this project for design guidance, code scaffolding, and debugging. All design decisions, the understanding of the methods, the interpretation of results, and the validation of the system are the team's own, as demonstrated in the review and viva.

## 4.13 Summary and Path to Review 4

The half-way implementation is complete and validated: a reproducible harness, two workflows under one framework, three verification signals, a fine-tune unlearning method with three provider behaviours, and the first failure matrices and reconstruction visual on real data. The path to Review 4 is to add the remaining unlearning methods (NegGrad, Fisher, SISA) and signals (backdoor and distance), extend to CIFAR-10 at scale, report results with multiple seeds and error bars, calibrate the GAN reconstruction signal into a reliable discriminator, and complete the full failure matrix across method, provider, and signal.
