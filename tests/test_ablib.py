"""Tests for the EDT AB-test helper library."""
import experiments.edt_ab.ablib as ablib  # noqa: F401  (import smoke test)


def test_package_imports():
    assert ablib is not None


import torch
from fractus.continuous_engine import ContinuousThoughtEngine


def test_build_engine_is_deterministic_and_13m():
    a = ablib.build_engine(seed=42)
    b = ablib.build_engine(seed=42)
    c = ablib.build_engine(seed=7)
    assert isinstance(a, ContinuousThoughtEngine)
    # Same seed → bit-identical weights.
    for pa, pb in zip(a.parameters(), b.parameters()):
        assert torch.equal(pa, pb)
    # Different seed → different weights.
    differs = any(not torch.equal(pa, pc)
                  for pa, pc in zip(a.parameters(), c.parameters()))
    assert differs
    n = sum(p.numel() for p in a.parameters())
    assert 12_000_000 <= n <= 14_000_000, f"expected ~13M params, got {n}"
