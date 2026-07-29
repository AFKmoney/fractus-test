"""Tests for the EDT 1B AB-test helper library. Runs on CPU at reduced config."""
import experiments.edt_1b_ab.ablib_1b as ablib_1b  # noqa: F401


def test_package_imports():
    assert ablib_1b is not None


import torch
from fractus1B.model_1b import Fractus1B

# Reduced config for fast CPU tests. The full config (1B params) is only for GPU runs.
REDUCED = dict(d_model=128, n_layers=2, n_heads=2, d_head=64, n_levels=2,
               n_experts=4, top_k=2, expert_d_ff=128, siren_rank=16, max_seq_len=64)


def test_build_engine_1b_deterministic_reduced():
    a = ablib_1b.build_engine_1b(seed=42, **REDUCED)
    b = ablib_1b.build_engine_1b(seed=42, **REDUCED)
    c = ablib_1b.build_engine_1b(seed=7, **REDUCED)
    assert isinstance(a, Fractus1B)
    for pa, pb in zip(a.parameters(), b.parameters()):
        assert torch.equal(pa, pb)
    assert any(not torch.equal(pa, pc) for pa, pc in zip(a.parameters(), c.parameters()))


def test_build_engine_1b_lm_head_tied():
    """lm_head.weight must BE embed.tok_embed.weight (tied)."""
    eng = ablib_1b.build_engine_1b(seed=42, **REDUCED)
    assert eng.lm_head.weight is eng.embed.tok_embed.weight
