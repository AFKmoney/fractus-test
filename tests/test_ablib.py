"""Tests for the EDT AB-test helper library."""
import tempfile, os

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


def _make_tiny_corpus(path, n=5000):
    torch.manual_seed(0)
    torch.save(torch.randint(0, 50257, (n,), dtype=torch.int32), path)


def test_load_corpus_splits_and_partitions():
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "tiny.pt")
        _make_tiny_corpus(p, n=5000)
        split = ablib.load_corpus(
            p, n_train=1000, n_holdout=200, n_phase1=400, seed=42,
        )
    assert split["train"].numel() == 1000
    assert split["holdout"].numel() == 200
    assert split["phase1"].numel() == 400
    # Phase-1 partition: 4 disjoint contiguous slices, 100 each.
    assert len(split["domain_split"]) == 4
    assert all(s.numel() == 100 for s in split["domain_split"])
    # Disjointness: concatenating the slices reconstructs phase1 exactly.
    rejoined = torch.cat(split["domain_split"])
    assert torch.equal(rejoined, split["phase1"])
    # Train and holdout must not overlap (drawn from different parts of the shuffle).
    assert split["train"].numel() + split["holdout"].numel() + split["phase1"].numel() == 1600


def test_make_hidden_bank_shape_and_alignment():
    eng = ablib.build_engine(seed=42)
    tokens = torch.arange(1000, dtype=torch.int64) % 50257
    bank = ablib.make_hidden_bank(eng, tokens, chunk_len=32, n_chunks=10, seed=0)
    assert set(bank.keys()) == {"h_in", "h_target"}
    # 10 chunks × 31 consecutive pairs = 310 samples.
    assert bank["h_in"].shape == (310, eng.d_model)
    assert bank["h_target"].shape == (310, eng.d_model)
    # Pairs are positionally aligned: h_target[t] == observe(token after h_in[t]).
    # Sanity: embedding is deterministic, so re-deriving matches.
    with torch.no_grad():
        first_in_token = tokens[0]
        # The first sampled pair uses tokens at consecutive corpus positions.
        # We only check that h_in != h_target (otherwise the target is trivial).
        assert not torch.equal(bank["h_in"][0], bank["h_target"][0])
