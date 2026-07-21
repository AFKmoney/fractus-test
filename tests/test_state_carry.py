"""Tests of the L8 state-carrying attention and lightweight trainer."""

import torch


def test_attention_carry_matches_whole_sequence():
    """L8 CRITERION: processing a sequence as 2 chunks (carrying S,z) must give
    the SAME output as processing it whole (within atol=1e-3 — bf16-ish drift).

    This is the core proof that state-carry is mathematically equivalent to
    full-sequence attention for LINEAR attention (impossible for softmax).
    """
    from fractus.nn.attention import FractalLinearAttention
    from fractus.nn.stats import elu_plus_one
    torch.manual_seed(0)
    attn = FractalLinearAttention(d_model=16, n_heads=2, d_head=8, n_levels=2)
    attn.eval()
    B, L, Dm, H, D, nlev = 2, 12, 16, 2, 8, 2
    x = torch.randn(B, L, Dm)

    # Whole-sequence reference output.
    y_whole = attn(x)

    # Chunked with state carry: rebuild the attention internals manually,
    # passing the (S,z) state between chunks.
    q_all = torch.einsum("bld,de->ble", x, attn.w_qkv[0]) + attn.b_qkv[0]
    k_all = torch.einsum("bld,de->ble", x, attn.w_qkv[1]) + attn.b_qkv[1]
    v_all = torch.einsum("bld,de->ble", x, attn.w_qkv[2]) + attn.b_qkv[2]
    q_all = q_all.view(B, L, H, D)
    k_all = k_all.view(B, L, H, D)
    v_all = v_all.view(B, L, H, D)
    offsets = attn.level_offsets

    chunk_len = 5
    # Initialize the carried state to zeros so the first chunk returns a tuple.
    S0 = torch.zeros(B * nlev * H, D, D)  # (B*nlev*H, D, D)
    z0 = torch.zeros(B * nlev * H, D)
    outputs = []
    for start in range(0, L, chunk_len):
        end = min(start + chunk_len, L)
        Lc = end - start
        qc = q_all[:, start:end]
        kc = k_all[:, start:end]
        vc = v_all[:, start:end]
        qf = elu_plus_one(qc.unsqueeze(1) + offsets.view(nlev, 1, 1, 1), alpha=1.0)
        kf = elu_plus_one(kc.unsqueeze(1) + offsets.view(nlev, 1, 1, 1), alpha=1.0)
        vf = vc.unsqueeze(1).expand(B, nlev, Lc, H, D)
        qf = qf.permute(0, 1, 3, 2, 4).reshape(B * nlev * H, Lc, D)
        kf = kf.permute(0, 1, 3, 2, 4).reshape(B * nlev * H, Lc, D)
        vf = vf.permute(0, 1, 3, 2, 4).reshape(B * nlev * H, Lc, D)
        yc, (S0, z0) = attn._linear_attention_causal_vectorized(qf, kf, vf, carry=(S0, z0))
        yc = yc.reshape(B, nlev, H, Lc, D).permute(0, 1, 3, 2, 4).reshape(B, nlev, Lc, H * D)
        outputs.append(yc)
    y_chunked = torch.cat(outputs, dim=2)  # (B, nlev, L, H*D)
    level_weights = torch.softmax(attn.level_logits, dim=-1)
    y_chunked_full = (y_chunked * level_weights.view(1, nlev, 1, 1)).sum(dim=1)
    y_chunked_out = y_chunked_full @ attn.w_out + attn.b_out

    assert torch.allclose(y_whole, y_chunked_out, atol=1e-3), \
        f"state-carry output differs from whole-sequence: " \
        f"max diff {(y_whole - y_chunked_out).abs().max()}"


def test_attention_carry_chunks_differ_from_no_carry():
    """Sanity: with carry, chunk 2 sees chunk 1's state → different from
    processing chunk 2 in isolation."""
    from fractus.nn.attention import FractalLinearAttention
    from fractus.nn.stats import elu_plus_one
    torch.manual_seed(1)
    attn = FractalLinearAttention(d_model=8, n_heads=2, d_head=4, n_levels=1)
    attn.eval()
    B, H, D, nlev = 1, 2, 4, 1
    # Two chunks of 4 tokens each.
    q1 = torch.randn(B * nlev * H, 4, D)
    k1 = torch.randn(B * nlev * H, 4, D)
    v1 = torch.randn(B * nlev * H, 4, D)
    q2 = torch.randn(B * nlev * H, 4, D)
    k2 = torch.randn(B * nlev * H, 4, D)
    v2 = torch.randn(B * nlev * H, 4, D)
    q1 = elu_plus_one(q1); k1 = elu_plus_one(k1)
    q2 = elu_plus_one(q2); k2 = elu_plus_one(k2)

    # Chunk 2 isolated (zero initial state).
    z_init = torch.zeros(B * nlev * H, D)
    S_init = torch.zeros(B * nlev * H, D, D)
    y2_iso, _ = attn._linear_attention_causal_vectorized(q2, k2, v2, carry=(S_init, z_init))
    # Chunk 1 then chunk 2 with carry.
    y1, state = attn._linear_attention_causal_vectorized(q1, k1, v1, carry=(S_init, z_init))
    y2_carried, _ = attn._linear_attention_causal_vectorized(q2, k2, v2, carry=state)
    # The carried output must differ from isolated (the state adds context).
    assert not torch.allclose(y2_iso, y2_carried, atol=1e-4), \
        "carrying state should change chunk-2 output vs isolated"


def test_lightweight_trainer_runs():
    """The LightweightTrainer runs end-to-end and the loss is finite."""
    from fractus.train import LightweightTrainer
    import torch.nn as nn
    torch.manual_seed(0)

    class TinyModel(nn.Module):
        """Returns (logits (B, vocab), aux) — matches the trainer contract."""
        def __init__(self):
            super().__init__()
            self.lin = nn.Linear(4, 8)
        def forward(self, x):
            return self.lin(x.float()), torch.tensor(0.0)

    model = TinyModel()
    trainer = LightweightTrainer(model, lr=1e-2, warmup_steps=2, t_max=10)
    info = trainer.info()
    assert info["threads"] >= 1

    inputs = torch.randint(0, 4, (6, 4))   # (B=6, in=4)
    targets = torch.randint(0, 8, (6,))    # (B=6,) class indices
    m = trainer.train_step(inputs, targets, vocab_size=8)
    assert torch.isfinite(torch.tensor(m["ce"]))
    # A few more steps to exercise the scheduler.
    for _ in range(5):
        trainer.train_step(inputs, targets, vocab_size=8)
    assert trainer.step_count == 6
