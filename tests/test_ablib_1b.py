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


import tempfile, os


def _make_tiny_corpus(path, n=5000):
    torch.manual_seed(0)
    torch.save(torch.randint(0, 50257, (n,), dtype=torch.int32), path)


def test_load_corpus_1b_splits_and_partitions():
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "tiny.pt")
        _make_tiny_corpus(p, n=5000)
        split = ablib_1b.load_corpus_1b(p, n_train=1000, n_holdout=200,
                                        n_phase1=512, n_experts=8, seed=42)
    assert split["train"].numel() == 1000
    assert split["holdout"].numel() == 200
    assert split["phase1"].numel() == 512
    assert len(split["domain_split"]) == 8
    assert all(s.numel() == 64 for s in split["domain_split"])
    assert torch.equal(torch.cat(split["domain_split"]), split["phase1"])


def test_make_hidden_bank_1b_block0_uses_embedding():
    eng = ablib_1b.build_engine_1b(seed=42, **REDUCED)
    tokens = torch.arange(200, dtype=torch.int64) % 50257
    bank0 = ablib_1b.make_hidden_bank_1b(eng, tokens, after_block=0,
                                         chunk_len=16, n_chunks=5, seed=0)
    eng.eval()
    with torch.no_grad():
        emb = eng.embed(tokens[:17].unsqueeze(0))  # (1, 17, D)
    assert bank0["h_in"].shape[1] == eng.d_model
    assert not torch.equal(bank0["h_in"][0], bank0["h_target"][0])


def test_make_hidden_bank_1b_deeper_block_differs():
    """Bank at block 1 must differ from block 0 (it passed through block 0)."""
    eng = ablib_1b.build_engine_1b(seed=42, **REDUCED)
    tokens = torch.arange(200, dtype=torch.int64) % 50257
    b0 = ablib_1b.make_hidden_bank_1b(eng, tokens, after_block=0, chunk_len=16, n_chunks=3, seed=0)
    b1 = ablib_1b.make_hidden_bank_1b(eng, tokens, after_block=1, chunk_len=16, n_chunks=3, seed=0)
    assert not torch.allclose(b0["h_in"][0], b1["h_in"][0], atol=1e-5)
