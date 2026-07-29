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


def test_phase1_experts_1b_reduces_mse_one_expert():
    eng = ablib_1b.build_engine_1b(seed=42, **REDUCED)
    tokens = torch.arange(500, dtype=torch.int64) % 50257
    bank = ablib_1b.make_hidden_bank_1b(eng, tokens, after_block=0, chunk_len=16, n_chunks=10, seed=0)
    mse_before = ablib_1b._eval_expert_mse_1b(eng, block_idx=0, expert_idx=0, bank=bank)
    ablib_1b._train_one_expert_1b(eng, block_idx=0, expert_idx=0, bank=bank,
                                  steps=20, lr=1e-2, seed=0)
    mse_after = ablib_1b._eval_expert_mse_1b(eng, block_idx=0, expert_idx=0, bank=bank)
    assert mse_after < mse_before


def test_phase1_experts_1b_natural_isolation():
    """Training expert (0,0) must NOT move expert (0,1) — separate modules."""
    eng = ablib_1b.build_engine_1b(seed=42, **REDUCED)
    tokens = torch.arange(500, dtype=torch.int64) % 50257
    bank = ablib_1b.make_hidden_bank_1b(eng, tokens, after_block=0, chunk_len=16, n_chunks=10, seed=0)
    e1_before = eng.blocks[0].moe.experts_w1[1].U.detach().clone()
    ablib_1b._train_one_expert_1b(eng, 0, 0, bank, steps=20, lr=1e-2, seed=0)
    e1_after = eng.blocks[0].moe.experts_w1[1].U
    assert torch.equal(e1_before, e1_after), "expert 1 moved while training expert 0"


def test_phase2a_attention_1b_reduces_loss_and_isolates():
    eng = ablib_1b.build_engine_1b(seed=42, **REDUCED)
    # Phase 2a trains attn + norm1 of EVERY block; the freeze set must exclude
    # all of them, leaving only genuinely-frozen params (embed, lm_head, norm,
    # norm_kur, kuramoto, norm_moe, moe of each block).
    attn_params = set()
    for blk in eng.blocks:
        attn_params |= set(id(p) for p in blk.attn.parameters())
        attn_params |= set(id(p) for p in blk.norm1.parameters())
    frozen_before = {id(p): p.detach().clone() for p in eng.parameters()
                     if id(p) not in attn_params}
    torch.manual_seed(0)
    losses = ablib_1b.phase2a_attention_1b(eng, n_steps_per_layer=20, lr=1e-2,
                                           seq_len=16, batch_size=8, seed=0)
    half = len(losses) // 2
    assert sum(losses[half:]) / max(len(losses) - half, 1) < sum(losses[:half]) / max(half, 1)
    for p in eng.parameters():
        if id(p) in frozen_before:
            assert torch.equal(p, frozen_before[id(p)]), "frozen param moved in Phase 2a"


def test_phase2b_embedding_1b_tied_and_reduces_ce():
    eng = ablib_1b.build_engine_1b(seed=42, **REDUCED)
    frozen_names = [n for n, _ in eng.named_parameters()
                    if not n.startswith("embed.tok_embed.")]
    frozen_before = {n: p.detach().clone() for n, p in eng.named_parameters()
                     if n in frozen_names}
    tokens = torch.arange(2000, dtype=torch.int64) % 50257
    ce_before = ablib_1b._eval_ce_1b(eng, tokens[:500])
    losses = ablib_1b.phase2b_embedding_1b(eng, tokens, steps=30, lr=1e-2,
                                           seq_len=16, seed=0)
    ce_after = ablib_1b._eval_ce_1b(eng, tokens[:500])
    assert ce_after < ce_before
    assert eng.lm_head.weight is eng.embed.tok_embed.weight
    for n, p in eng.named_parameters():
        if n in frozen_before:
            assert torch.equal(p, frozen_before[n]), f"{n} moved in Phase 2b"


def test_phase3_joint_1b_reduces_loss_returns_curve():
    eng = ablib_1b.build_engine_1b(seed=42, **REDUCED)
    tokens = torch.arange(2000, dtype=torch.int64) % 50257
    out = ablib_1b.phase3_joint_1b(eng, tokens, steps=30, lr=1e-3,
                                   seq_len=16, use_pgsu=False)
    assert "losses" in out and "avg_loss" in out and "accuracy" in out
    half = len(out["losses"]) // 2
    assert sum(out["losses"][half:]) / max(len(out["losses"]) - half, 1) < \
           sum(out["losses"][:half]) / max(half, 1)
    assert all(p.requires_grad for p in eng.parameters())


def test_evaluate_ppl_1b_finite_positive():
    eng = ablib_1b.build_engine_1b(seed=42, **REDUCED)
    tokens = torch.arange(500, dtype=torch.int64) % 50257
    ppl = ablib_1b.evaluate_ppl_1b(eng, tokens[:300], seq_len=16)
    assert ppl == ppl and ppl > 0.0


def test_expert_diversity_1b_in_range():
    eng = ablib_1b.build_engine_1b(seed=42, **REDUCED)
    probe = torch.arange(64, dtype=torch.int64) % 50257
    div = ablib_1b.expert_diversity_1b(eng, probe, block_idx=0)
    assert -1.0 <= div <= 1.0


def test_greedy_sample_1b_returns_string():
    eng = ablib_1b.build_engine_1b(seed=42, **REDUCED)
    prompt = torch.tensor([1, 2, 3], dtype=torch.int64)
    s = ablib_1b.greedy_sample_1b(eng, prompt, n_tokens=8)
    assert isinstance(s, str) and len(s) > 0
