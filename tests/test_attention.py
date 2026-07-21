"""Tests of FractalLinearAttention: shape, causality, differentiability.

Katharopoulos causal linear attention (O(L·d2) instead of O(L2·d)).
Faithfully ported from the original system's src/attention.rs.
"""

import torch
import pytest


def test_attention_shape():
    """Input (B, L, d_model) → output (B, L, d_model)."""
    from fractus.nn.attention import FractalLinearAttention
    attn = FractalLinearAttention(d_model=32, n_heads=4, d_head=8, n_levels=2)
    x = torch.randn(2, 10, 32)
    out = attn(x)
    assert out.shape == (2, 10, 32)


def test_attention_is_finite():
    from fractus.nn.attention import FractalLinearAttention
    attn = FractalLinearAttention(d_model=32, n_heads=4, d_head=8, n_levels=2)
    x = torch.randn(2, 10, 32)
    out = attn(x)
    assert torch.isfinite(out).all()


def test_attention_causality():
    """The attention is CAUSAL: changing the token at position j >= t must not
    affect the output at position t < j."""
    from fractus.nn.attention import FractalLinearAttention
    torch.manual_seed(0)
    attn = FractalLinearAttention(d_model=16, n_heads=2, d_head=8, n_levels=1)
    attn.eval()  # disable any dropout
    x = torch.randn(1, 6, 16)
    out1 = attn(x)
    # Modifying position 4 (and after) must not change the outputs at positions 0..3.
    x_modified = x.clone()
    x_modified[0, 4:] = torch.randn(2, 16)  # break positions 4 and 5
    out2 = attn(x_modified)
    # The first 4 positions must be identical (strict causality).
    assert torch.allclose(out1[0, :4], out2[0, :4], atol=1e-5), \
        "The attention is not causal: a future token affected a past output"


def test_attention_backward_propagates():
    """L2a CRITERION: backward() must propagate a finite AND non-zero gradient to
    EVERY parameter. This is exactly the test that the original system failed."""
    from fractus.nn.attention import FractalLinearAttention
    attn = FractalLinearAttention(d_model=32, n_heads=4, d_head=8, n_levels=2)
    x = torch.randn(2, 8, 32)
    out = attn(x)
    loss = out.pow(2).sum()
    loss.backward()

    params = list(attn.named_parameters())
    assert len(params) > 0
    for name, p in params:
        assert p.requires_grad, f"{name} should requires_grad=True"
        assert p.grad is not None, f"{name} received no gradient"
        assert torch.isfinite(p.grad).all(), f"{name} has a non-finite gradient"
        assert p.grad.abs().sum().item() > 0, f"{name} received a zero gradient"


def test_attention_multi_levels_changes_output():
    """ORTHOgonal test of the multi-level effect (corrects the previous
    quasi-tautological version that drew two modules from the same RNG stream).

    Idea: with n_levels=2, if we force level_logits = [+inf, -inf] (so the
    softmax puts all the weight on level 0, none on level 1), the output must
    be EXACTLY that of an n_levels=1 module with the same level-0 offset. And
    conversely with [-inf, +inf] (all on level 1). This isolates the multi-level
    offset effect independently of the random init.
    """
    from fractus.nn.attention import FractalLinearAttention
    torch.manual_seed(0)
    x = torch.randn(1, 8, 16)

    # Single-level module (reference).
    attn_ref = FractalLinearAttention(d_model=16, n_heads=2, d_head=8, n_levels=1)

    # Two-level module with the SAME Q/K/V/out weights (explicit copy).
    attn2 = FractalLinearAttention(d_model=16, n_heads=2, d_head=8, n_levels=2)
    attn2.w_qkv.data = attn_ref.w_qkv.data.clone()
    attn2.b_qkv.data = attn_ref.b_qkv.data.clone()
    attn2.w_out.data = attn_ref.w_out.data.clone()
    attn2.b_out.data = attn_ref.b_out.data.clone()

    out_ref = attn_ref(x)

    # Case 1: all the weight on level 0 → must reproduce attn_ref exactly.
    with torch.no_grad():
        attn2.level_logits.data = torch.tensor([1e9, -1e9])
    out_level0_only = attn2(x)
    assert torch.allclose(out_level0_only, out_ref, atol=1e-4), \
        "Forcing level_logits=[+inf,-inf] should reproduce n_levels=1 exactly"

    # Case 2: all the weight on level 1 → must DIFFER from attn_ref
    # (because the offset ω_1 = (φ2)^{-1} ≠ ω_0 = 1, so the feature map differs).
    with torch.no_grad():
        attn2.level_logits.data = torch.tensor([-1e9, 1e9])
    out_level1_only = attn2(x)
    assert not torch.allclose(out_level1_only, out_ref, atol=1e-5), \
        "Forcing level_logits=[-inf,+inf] (level-1 offset ≠ level-0 offset) should " \
        "give a different output than level 0 — proof that the multi-level offsets " \
        "actually change the computation"


def test_attention_d_model_constraint():
    """d_model must be divisible by n_heads (otherwise error)."""
    from fractus.nn.attention import FractalLinearAttention
    with pytest.raises(ValueError):
        FractalLinearAttention(d_model=30, n_heads=4, d_head=8, n_levels=1)
        # 4 * 8 = 32 ≠ 30
