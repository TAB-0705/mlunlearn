"""
Spine tests. Run:  python -m tests.test_spine   (from the repo root)

These prove the four properties the whole project rests on:
  1. Determinism      — same spec => byte-identical splits + same hash
  2. Sensitivity      — different seed => different split
  3. Integrity        — retain/forget/test are disjoint and cover everything
  4. Independence      — changing lr/epochs does NOT change the data split
"""
import numpy as np

from core.config import RunConfig
from core.splits import DataSplitManager, SplitSpec
from core.data.datasets import load_purchase100, slice_split


def test_determinism():
    spec = SplitSpec("purchase100", n_samples=20000, seed=0)
    a = DataSplitManager().get(spec)
    b = DataSplitManager().get(spec)
    assert np.array_equal(a.retain_idx, b.retain_idx)
    assert np.array_equal(a.forget_idx, b.forget_idx)
    assert np.array_equal(a.test_idx, b.test_idx)
    assert spec.hash() == SplitSpec("purchase100", 20000, 0).hash()
    print("  [1] determinism: same spec -> identical splits + hash  OK")


def test_seed_sensitivity():
    s0 = DataSplitManager().get(SplitSpec("purchase100", 20000, seed=0))
    s1 = DataSplitManager().get(SplitSpec("purchase100", 20000, seed=1))
    assert not np.array_equal(s0.forget_idx, s1.forget_idx)
    print("  [2] sensitivity: different seed -> different split       OK")


def test_integrity():
    spec = SplitSpec("purchase100", n_samples=20000, seed=0,
                     test_fraction=0.2, forget_fraction=0.1)
    split = DataSplitManager().get(spec)
    split.assert_valid(spec.n_samples)  # raises on any overlap / gap
    # expected sizes: test=4000, pool=16000, forget=1600, retain=14400
    assert split.sizes == {"retain": 14400, "forget": 1600, "test": 4000}, split.sizes
    print(f"  [3] integrity: disjoint + full coverage {split.sizes}  OK")


def test_split_independent_of_model_params():
    """The panel question: 'if you change the learning rate, does the data split
    change?'  Answer must be NO."""
    n = 20000
    c_lr_low = RunConfig(dataset="purchase100", seed=0, lr=1e-4, epochs=10)
    c_lr_high = RunConfig(dataset="purchase100", seed=0, lr=1e-1, epochs=99)
    assert c_lr_low.split_spec(n).hash() == c_lr_high.split_spec(n).hash()
    # ...but the FULL run hashes differ (they are genuinely different runs)
    assert c_lr_low.hash() != c_lr_high.hash()
    print("  [4] independence: lr/epochs change -> same split, diff run  OK")


def test_slicing_end_to_end():
    X, y = load_purchase100(synthetic=True, n_synth=20000)
    spec = SplitSpec("purchase100", n_samples=len(X), seed=0)
    split = DataSplitManager().get(spec)
    parts = slice_split(X, y, split)
    assert parts["retain"][0].shape[0] == 14400
    assert parts["forget"][0].shape[0] == 1600
    assert parts["test"][0].shape[0] == 4000
    print("  [5] slicing: synthetic data cut cleanly by frozen split  OK")


if __name__ == "__main__":
    print("Spine tests:")
    test_determinism()
    test_seed_sensitivity()
    test_integrity()
    test_split_independent_of_model_params()
    test_slicing_end_to_end()
    print("\nAll spine tests passed.")
