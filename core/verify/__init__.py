"""
Verification layer (Layer 5) — the "one overall framework, two workflows" seam.

Every dataset maps to a WORKFLOW, and each workflow declares which verification
signals apply to it:

    tabular  (Purchase-100)             -> aggregate_mia, lira
    image    (Fashion-MNIST, CIFAR-10)  -> aggregate_mia, lira, gan_reconstruction

Same harness, same frozen VerificationSignal interface, same scoring. The ONLY
difference between the two workflows is this signal list — the GAN reconstruction
signal is image-only (nothing to reconstruct in tabular data), so it is attached
to the image workflow and nowhere else. That is exactly the separation the panel
asked for, kept inside one model instead of forking the codebase.

Person A / B / C never edit each other's files to add a signal: they register a
class here and it flows through the whole harness.
"""
from __future__ import annotations

from typing import Dict, List

# ---- which workflow each dataset belongs to ---------------------------------- #
DATASET_WORKFLOW: Dict[str, str] = {
    "purchase100": "tabular",
    "fashion_mnist": "image",
    "cifar10": "image",
}

# ---- which signals each workflow runs (image = tabular signals + the GAN) ----- #
WORKFLOW_SIGNALS: Dict[str, List[str]] = {
    "tabular": ["aggregate_mia", "lira"],
    "image": ["aggregate_mia", "lira", "gan_reconstruction"],
}


def workflow_for_dataset(dataset: str) -> str:
    try:
        return DATASET_WORKFLOW[dataset]
    except KeyError:
        raise ValueError(
            f"unknown dataset '{dataset}'. Known: {sorted(DATASET_WORKFLOW)}"
        )


def signal_names_for(dataset: str) -> List[str]:
    """The signal names this dataset's workflow should run."""
    return list(WORKFLOW_SIGNALS[workflow_for_dataset(dataset)])


def build_signal(name: str, **kwargs):
    """Factory. Signals with heavy deps are imported lazily so the tabular
    workflow never imports GAN code, and vice-versa."""
    if name == "aggregate_mia":
        from core.verify.mia import AggregateMIA
        return AggregateMIA()
    if name == "lira":
        from core.verify.lira import LiRA
        return LiRA(**kwargs)
    if name == "gan_reconstruction":
        from core.verify.gan_signal import GANReconstructionSignal
        return GANReconstructionSignal(**kwargs)
    raise ValueError(f"unknown signal '{name}'")
