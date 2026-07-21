"""Equivalence tests: vectorized attention == looped attention.

CRITERION: the vectorized version must give EXACTLY the same outputs as the
looped version (within 1e-5), to guarantee that no bug was introduced during
the optimization.
"""

import torch


def test_vectorized_matches_looped_small():
    """On a small case (B=2, L=8, D=4), vectorized == looped."""
    from fractus.nn.attention import FractalLinearAttention
    torch.manual_seed(0)
    attn = FractalLinearAttention(d_model=8, n_heads=2, d_head=4, n_levels=1)
    attn.eval()
    x = torch.randn(2, 8, 8)
    out_looped = attn(x)
    # The vectorized version is called via _linear_attention_causal_vectorized.
    # We compare it to the looped version on the same q,k,v.
    # For that, we reproduce the projection + feature map.
    from fractus.nn.stats import elu_plus_one
    B, L, _ = x.shape
    q_all = torch.einsum("bld,de->ble", x, attn.w_qkv[0]) + attn.b_qkv[0]
    k_all = torch.einsum("bld,de->ble", x, attn.w_qkv[1]) + attn.b_qkv[1]
    v_all = torch.einsum("bld,de->ble", x, attn.w_qkv[2]) + attn.b_qkv[2]
    # One head, level 0.
    q = q_all.view(B, L, 2, 4).transpose(1, 2)[:, 0]
    k = k_all.view(B, L, 2, 4).transpose(1, 2)[:, 0]
    v = v_all.view(B, L, 2, 4).transpose(1, 2)[:, 0]
    q = elu_plus_one(q + attn.level_offsets[0])
    k = elu_plus_one(k + attn.level_offsets[0])
    out_looped_one = attn._linear_attention_causal_one_head(q, k, v)
    out_vec_one = attn._linear_attention_causal_vectorized(q, k, v)
    assert torch.allclose(out_looped_one, out_vec_one, atol=1e-5), \
        f"Vectorized != looped: max diff {(out_looped_one - out_vec_one).abs().max()}"


def test_vectorized_matches_looped_larger():
    """On a larger case (B=4, L=32, D=8)."""
    from fractus.nn.attention import FractalLinearAttention
    from fractus.nn.stats import elu_plus_one
    torch.manual_seed(1)
    attn = FractalLinearAttention(d_model=16, n_heads=2, d_head=8, n_levels=1)
    attn.eval()
    x = torch.randn(4, 32, 16)
    B, L, _ = x.shape
    q_all = torch.einsum("bld,de->ble", x, attn.w_qkv[0]) + attn.b_qkv[0]
    k_all = torch.einsum("bld,de->ble", x, attn.w_qkv[1]) + attn.b_qkv[1]
    v_all = torch.einsum("bld,de->ble", x, attn.w_qkv[2]) + attn.b_qkv[2]
    q = q_all.view(B, L, 2, 8).transpose(1, 2)[:, 0]
    k = k_all.view(B, L, 2, 8).transpose(1, 2)[:, 0]
    v = v_all.view(B, L, 2, 8).transpose(1, 2)[:, 0]
    q = elu_plus_one(q + attn.level_offsets[0])
    k = elu_plus_one(k + attn.level_offsets[0])
    out_looped = attn._linear_attention_causal_one_head(q, k, v)
    out_vec = attn._linear_attention_causal_vectorized(q, k, v)
    assert torch.allclose(out_looped, out_vec, atol=1e-5)


def test_vectorized_preserves_causality():
    """The vectorized version must preserve causality."""
    from fractus.nn.attention import FractalLinearAttention
    torch.manual_seed(0)
    attn = FractalLinearAttention(d_model=8, n_heads=2, d_head=4, n_levels=1)
    attn.eval()
    x = torch.randn(1, 6, 8)
    out1 = attn(x)
    x_mod = x.clone()
    x_mod[0, 4:] = torch.randn(2, 8)
    out2 = attn(x_mod)
    assert torch.allclose(out1[0, :4], out2[0, :4], atol=1e-5), \
        "Vectorized must remain causal"


def test_vectorized_faster_than_looped():
    """The vectorized version must be faster than the looped one.
    We do not run a strict benchmark, just a check that it is significantly
    faster (factor > 2)."""
    import time
    from fractus.nn.attention import FractalLinearAttention
    from fractus.nn.stats import elu_plus_one
    torch.manual_seed(0)
    attn = FractalLinearAttention(d_model=16, n_heads=2, d_head=8, n_levels=1)
    attn.eval()
    B, L, D = 4, 64, 16
    x = torch.randn(B, L, D)
    q_all = torch.einsum("bld,de->ble", x, attn.w_qkv[0]) + attn.b_qkv[0]
    k_all = torch.einsum("bld,de->ble", x, attn.w_qkv[1]) + attn.b_qkv[1]
    v_all = torch.einsum("bld,de->ble", x, attn.w_qkv[2]) + attn.b_qkv[2]
    q = q_all.view(B, L, 2, 8).transpose(1, 2)[:, 0]
    k = k_all.view(B, L, 2, 8).transpose(1, 2)[:, 0]
    v = v_all.view(B, L, 2, 8).transpose(1, 2)[:, 0]
    q = elu_plus_one(q + attn.level_offsets[0])
    k = elu_plus_one(k + attn.level_offsets[0])

    # Warmup.
    attn._linear_attention_causal_one_head(q, k, v)
    attn._linear_attention_causal_vectorized(q, k, v)

    # Looped.
    t0 = time.time()
    for _ in range(3):
        attn._linear_attention_causal_one_head(q, k, v)
    t_looped = (time.time() - t0) / 3
    # Vectorized.
    t0 = time.time()
    for _ in range(3):
        attn._linear_attention_causal_vectorized(q, k, v)
    t_vec = (time.time() - t0) / 3
    speedup = t_looped / max(t_vec, 1e-9)
    print(f"\nLooped: {t_looped*1000:.1f}ms, Vectorized: {t_vec*1000:.1f}ms, speedup: {speedup:.1f}x")
    assert speedup > 2.0, f"Vectorized should be >2x faster, got {speedup:.1f}x"
