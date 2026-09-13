"""
DataSplitManager  —  Layer 0 (spine).  Owner: Person C.

THE most important file in the repo. The entire lab setup depends on knowing
EXACTLY which records are Retain / Forget / Test for every seed. A single
indexing slip here corrupts every downstream number and fails SILENTLY.

Rule enforced by this module: data is split HERE and NOWHERE ELSE.
  - Retain : the model is allowed to keep this (~90% of the train pool)
  - Forget : the data the user asked to delete (~10% of the train pool)
  - Test   : held out entirely; never seen by any training. Serves as the
             pool of known non-members for the auditor.

Ground truth = we know precisely which indices are 'forget'. The auditor role
pretends not to; scoring compares its verdict against this known truth.

Every split is deterministic (seeded), persisted to disk (so a run is
reproducible + auditable), and validated for disjointness + full coverage.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path
import hashlib
import json

import numpy as np


@dataclass(frozen=True)
class SplitSpec:
    """The ONLY inputs that define a split. Note: no model/lr/epochs here."""
    dataset: str
    n_samples: int
    seed: int
    test_fraction: float = 0.2
    forget_fraction: float = 0.1  # of the train pool (after test is held out)

    def hash(self) -> str:
        return hashlib.sha256(
            json.dumps(asdict(self), sort_keys=True).encode()
        ).hexdigest()[:12]


@dataclass
class Split:
    retain_idx: np.ndarray
    forget_idx: np.ndarray
    test_idx: np.ndarray
    spec: SplitSpec

    def assert_valid(self, n_samples: int) -> None:
        r = set(self.retain_idx.tolist())
        f = set(self.forget_idx.tolist())
        t = set(self.test_idx.tolist())
        assert r.isdisjoint(f), "INTEGRITY VIOLATION: retain/forget overlap"
        assert r.isdisjoint(t), "INTEGRITY VIOLATION: retain/test overlap"
        assert f.isdisjoint(t), "INTEGRITY VIOLATION: forget/test overlap"
        assert len(r) + len(f) + len(t) == n_samples, \
            "INTEGRITY VIOLATION: splits do not cover every sample exactly once"

    @property
    def sizes(self) -> dict:
        return {
            "retain": len(self.retain_idx),
            "forget": len(self.forget_idx),
            "test": len(self.test_idx),
        }


class DataSplitManager:
    """Single source of truth for all splits. Caches to disk keyed on the spec hash."""

    def __init__(self, cache_dir: str = "core/results/splits"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def get(self, spec: SplitSpec) -> Split:
        """Return the frozen split for this spec, creating + caching it if new."""
        npz_path = self.cache_dir / f"{spec.dataset}_{spec.hash()}.npz"
        manifest_path = self.cache_dir / f"{spec.dataset}_{spec.hash()}.json"

        if npz_path.exists():
            d = np.load(npz_path)
            split = Split(d["retain_idx"], d["forget_idx"], d["test_idx"], spec)
        else:
            split = self._make(spec)
            np.savez(
                npz_path,
                retain_idx=split.retain_idx,
                forget_idx=split.forget_idx,
                test_idx=split.test_idx,
            )
            manifest_path.write_text(json.dumps(
                {"spec": asdict(spec), "hash": spec.hash(), "sizes": split.sizes},
                indent=2,
            ))

        split.assert_valid(spec.n_samples)  # validated on every load, not just creation
        return split

    @staticmethod
    def _make(spec: SplitSpec) -> Split:
        rng = np.random.default_rng(spec.seed)          # seeded => deterministic
        perm = rng.permutation(spec.n_samples)

        n_test = int(round(spec.test_fraction * spec.n_samples))
        test_idx = perm[:n_test]

        pool = perm[n_test:]                            # the train pool
        n_forget = int(round(spec.forget_fraction * len(pool)))
        forget_idx = pool[:n_forget]
        retain_idx = pool[n_forget:]

        # sort for stable, human-readable, comparison-friendly index arrays
        return Split(
            retain_idx=np.sort(retain_idx),
            forget_idx=np.sort(forget_idx),
            test_idx=np.sort(test_idx),
            spec=spec,
        )
