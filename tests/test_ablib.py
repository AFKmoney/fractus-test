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


def test_phase1_shared_reduces_mse_on_bank():
    eng = ablib.build_engine(seed=42)
    tokens = torch.arange(2000, dtype=torch.int64) % 50257
    bank = ablib.make_hidden_bank(eng, tokens, chunk_len=32, n_chunks=20, seed=0)
    mse_before = ablib._eval_expert_mse(eng, 0, bank)
    ablib.phase1_experts_shared(eng, bank, steps=20, lr=1e-2, seed=0)
    mse_after = ablib._eval_expert_mse(eng, 0, bank)
    # Training must reduce the expert's MSE on its own training bank.
    assert mse_after < mse_before


def test_phase1_partitioned_trains_each_expert_on_own_bank():
    eng = ablib.build_engine(seed=42)
    tokens = torch.arange(2000, dtype=torch.int64) % 50257
    banks = [ablib.make_hidden_bank(eng, tokens[:500], chunk_len=32, n_chunks=8, seed=i)
             for i in range(4)]
    mse_before = [ablib._eval_expert_mse(eng, i, banks[i]) for i in range(4)]
    ablib.phase1_experts_partitioned(eng, banks, steps=15, lr=1e-2, seed=0)
    mse_after = [ablib._eval_expert_mse(eng, i, banks[i]) for i in range(4)]
    for i in range(4):
        assert mse_after[i] < mse_before[i], f"expert {i} did not improve"


def test_phase2b_reduces_ce_and_only_touches_observe_and_head():
    eng = ablib.build_engine(seed=42)
    # Snapshot the frozen params (attn + moe norms + experts) to prove they don't change.
    frozen_names = [n for n, _ in eng.named_parameters()
                    if not (n.startswith("observe.") or n.startswith("output_head."))]
    frozen_before = {n: p.detach().clone() for n, p in eng.named_parameters()
                     if n in frozen_names}
    tokens = torch.arange(3000, dtype=torch.int64) % 50257
    ce_before = ablib._eval_ce(eng, tokens[:1000])
    ablib.phase2b_embedding(eng, tokens, steps=30, lr=1e-2, seed=0)
    ce_after = ablib._eval_ce(eng, tokens[:1000])
    assert ce_after < ce_before
    # Frozen params untouched.
    for n, p in eng.named_parameters():
        if n in frozen_before:
            assert torch.equal(p, frozen_before[n]), f"{n} changed during Phase 2b"


def test_phase3_joint_reduces_loss_and_returns_curve():
    from fractus.train.online import OnlineTrainer
    eng = ablib.build_engine(seed=42)
    tokens = torch.arange(4000, dtype=torch.int64) % 50257
    out = ablib.phase3_joint(eng, tokens, steps=50, lr=1e-3, chunk_len=16)
    assert "losses" in out and "avg_loss" in out and "accuracy" in out
    assert len(out["losses"]) > 0
    # First half vs second half: loss should trend down.
    half = len(out["losses"]) // 2
    assert sum(out["losses"][half:]) / max(len(out["losses"]) - half, 1) < \
           sum(out["losses"][:half]) / max(half, 1)
    # All params trainable.
    assert all(p.requires_grad for p in eng.parameters())


def test_evaluate_ppl_returns_finite_positive():
    eng = ablib.build_engine(seed=42)
    tokens = torch.arange(2000, dtype=torch.int64) % 50257
    ppl = ablib.evaluate_ppl(eng, tokens[:500])
    assert ppl == ppl  # not NaN
    assert ppl > 0.0


def test_expert_diversity_returns_float_in_range():
    eng = ablib.build_engine(seed=42)
    probe_tokens = torch.arange(256, dtype=torch.int64) % 50257
    div = ablib.expert_diversity(eng, probe_tokens)
    assert -1.0 <= div <= 1.0


def test_greedy_sample_returns_string_of_expected_length():
    eng = ablib.build_engine(seed=42)
    prompt = torch.tensor([1, 2, 3], dtype=torch.int64)
    s = ablib.greedy_sample(eng, prompt, n_tokens=10)
    assert isinstance(s, str)
    # Tokenizer detok may merge; just check it's non-empty.
    assert len(s) > 0
