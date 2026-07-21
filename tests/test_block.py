"""Tests of FractalBlock: assembly LayerNorm → attention → residual.

The critical test (test_block_backward_every_param) is the culmination of L2a:
it proves that the whole block is differentiable and that backward propagates a
finite AND non-zero gradient to EVERY parameter. This is what the original system
could not do.
"""

import torch


def test_block_shape():
    from fractus.nn.block import FractalBlock
    block = FractalBlock(d_model=32, n_heads=4, d_head=8, n_levels=2)
    x = torch.randn(2, 10, 32)
    out = block(x)
    assert out.shape == (2, 10, 32)


def test_block_is_finite():
    from fractus.nn.block import FractalBlock
    block = FractalBlock(d_model=32, n_heads=4, d_head=8, n_levels=2)
    x = torch.randn(2, 10, 32) * 3  # somewhat large values
    out = block(x)
    assert torch.isfinite(out).all()


def test_block_residual_connection():
    """The block has a residual connection: with a good init, the output is
    close to the input (no explosion)."""
    from fractus.nn.block import FractalBlock
    torch.manual_seed(0)
    block = FractalBlock(d_model=32, n_heads=4, d_head=8, n_levels=2)
    block.eval()
    x = torch.randn(1, 8, 32)
    out = block(x)
    # The residual guarantees out ≈ x + small attn(x). We only check that
    # the output is of the same order of magnitude (no explosion).
    assert out.std().item() < 10.0 * x.std().item()


def test_block_backward_every_param():
    """L2a CRITERION: backward() must propagate a finite AND non-zero gradient to
    EVERY block parameter. This is exactly what the original system failed to do
    (training.rs:399 used random noise instead of a gradient)."""
    from fractus.nn.block import FractalBlock
    block = FractalBlock(d_model=32, n_heads=4, d_head=8, n_levels=2)
    x = torch.randn(2, 8, 32)
    out = block(x)
    loss = out.pow(2).sum()
    loss.backward()

    params = list(block.named_parameters())
    assert len(params) > 0, "The block has no parameter"
    for name, p in params:
        assert p.requires_grad, f"{name} should requires_grad=True"
        assert p.grad is not None, f"{name} received no gradient (dead parameter)"
        assert torch.isfinite(p.grad).all(), f"{name} has a non-finite gradient (NaN/Inf)"
        grad_l1 = p.grad.abs().sum().item()
        assert grad_l1 > 0, (
            f"{name} received a zero gradient — autodiff does not propagate "
            f"to this parameter (grad L1 = {grad_l1})"
        )


def test_block_full_shape_and_finite():
    from fractus.nn.block import FractalBlockFull
    block = FractalBlockFull(
        d_model=32, n_heads=4, d_head=8, n_levels=2,
        n_oscillators=8, coupling_rank=4,
        n_experts=4, top_k=2, kappa=4.0,
    )
    x = torch.randn(2, 8, 32)
    out, lb_loss = block(x)
    assert out.shape == (2, 8, 32)
    assert torch.isfinite(out).all()
    assert torch.isfinite(lb_loss)


def test_block_full_backward_every_param():
    """L2b CRITERION: FractalBlockFull (attn + Kuramoto + MoE) must propagate a
    finite AND non-zero gradient to EVERY parameter. The ultimate proof that the
    entire fractal pipeline is differentiable end-to-end."""
    from fractus.nn.block import FractalBlockFull
    block = FractalBlockFull(
        d_model=32, n_heads=4, d_head=8, n_levels=2,
        n_oscillators=8, coupling_rank=4,
        n_experts=4, top_k=2, kappa=4.0,
    )
    x = torch.randn(2, 8, 32)
    out, lb_loss = block(x)
    loss = out.pow(2).sum() + 0.1 * lb_loss
    loss.backward()

    params = list(block.named_parameters())
    assert len(params) > 0
    for name, p in params:
        assert p.requires_grad, f"{name} should requires_grad=True"
        assert p.grad is not None, f"{name} received no gradient"
        assert torch.isfinite(p.grad).all(), f"{name} has a non-finite gradient"
        assert p.grad.abs().sum().item() > 0, f"{name} received a zero gradient"
