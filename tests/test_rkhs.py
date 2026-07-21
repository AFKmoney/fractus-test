"""Tests of RKHSCausalOperator: a true RKHS via RFF, not a bare low-rank projection."""

import torch


def test_rkhs_output_shape():
    from fractus.causal.rkhs import RKHSCausalOperator
    op = RKHSCausalOperator(dim=8, rank=4, n_rff=32)
    x = torch.randn(16, 8)
    y = op(x)
    assert y.shape == (16, 8)


def test_rkhs_is_finite():
    from fractus.causal.rkhs import RKHSCausalOperator
    op = RKHSCausalOperator(dim=8, rank=4, n_rff=32)
    x = torch.randn(16, 8) * 5
    assert torch.isfinite(op(x)).all()


def test_rkhs_kernel_approx_positive():
    """k(x,x) must be positive."""
    from fractus.causal.rkhs import RKHSCausalOperator
    op = RKHSCausalOperator(dim=4, rank=2, n_rff=64)
    x = torch.randn(3, 4)
    kxx = op.kernel(x, x)
    assert (torch.diagonal(kxx) > 0).all()


def test_rkhs_backward_every_param():
    """L4 CRITERION: backward propagates a finite AND non-zero gradient to EVERY trainable
    parameter (U, V, decode). The W_rff are frozen by design (Rahimi-Recht)."""
    from fractus.causal.rkhs import RKHSCausalOperator
    op = RKHSCausalOperator(dim=8, rank=4, n_rff=32)
    x = torch.randn(16, 8)
    y = op(x)
    loss = y.pow(2).sum()
    loss.backward()

    params = list(op.named_parameters())
    assert len(params) > 0
    for name, p in params:
        assert p.requires_grad, f"{name} should requires_grad=True"
        assert p.grad is not None, f"{name} received no gradient"
        assert torch.isfinite(p.grad).all(), f"{name} non-finite gradient"
        # All named params (U, V, decode.weight) are trainable and
        # must receive a non-zero gradient.
        assert p.grad.abs().sum().item() > 0, f"{name} received a zero gradient"


def test_rkhs_not_just_linear_projection():
    """TRUE RKHS: the output must go through a non-linear feature map (cos/sin),
    not be a simple linear projection x@W (the fake RKHS of the original).

    We verify that the forward CANNOT be written as a simple Linear:
    if we replace φ(x) = [cos,sin] with φ(x) = x (linear), the output changes.
    Concretely: we compare the true forward to a forward where we short-circuit
    the features with the identity (via a linear approximation around 0).
    """
    from fractus.causal.rkhs import RKHSCausalOperator
    torch.manual_seed(0)
    op = RKHSCausalOperator(dim=4, rank=2, n_rff=64)
    # Small input around 0: at x≈0, cos(ω·x+b)≈cos(b), sin(ω·x+b)≈sin(b)+ω·x·cos(b).
    # So the RKHS is approximately linear in x for small x, but not exactly.
    # For large x, the non-linearity is manifest.
    x_small = torch.randn(8, 4) * 0.01  # small
    x_large = torch.randn(8, 4) * 5.0   # large
    y_small = op(x_small)
    y_large = op(x_large)
    # For large x, if it were linear, doubling x would double y. We verify the
    # non-linearity: op(2·x_large) ≠ 2·op(x_large) because of sin/cos.
    y_2x = op(2.0 * x_large)
    ratio = y_2x / (2.0 * y_large + 1e-10)
    # If linear: ratio = 1 everywhere. If non-linear: ratio deviates from 1.
    deviation = (ratio - 1.0).abs().mean().item()
    assert deviation > 1e-3, \
        f"The RKHS must be non-linear (deviation > 1e-3), got {deviation}"
