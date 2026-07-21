"""Tests of notears_penalty: differentiable, =0 for DAG, >0 for cycle."""

import torch


def test_notears_zero_for_dag():
    """h(W) ≈ 0 if W is an obvious DAG (strict lower-triangular)."""
    from fractus.causal.notears import notears_penalty
    W = torch.tensor([
        [0.0, 0.0, 0.0],
        [0.5, 0.0, 0.0],
        [0.3, 0.4, 0.0],
    ])
    h = notears_penalty(W)
    assert abs(h.item()) < 1e-3, f"h(DAG) should be ~0, got {h.item()}"


def test_notears_positive_for_cycle():
    """h(W) > 0 if W contains a cycle."""
    from fractus.causal.notears import notears_penalty
    W = torch.tensor([
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
        [1.0, 0.0, 0.0],
    ])
    h = notears_penalty(W)
    assert h.item() > 0.5, f"h(cycle) should be > 0.5, got {h.item()}"


def test_notears_zero_for_zero_matrix():
    """h(0) = 0 (the zero matrix is trivially acyclic)."""
    from fractus.causal.notears import notears_penalty
    W = torch.zeros(4, 4)
    h = notears_penalty(W)
    assert abs(h.item()) < 1e-6


def test_notears_is_differentiable():
    """h(W) must be differentiable (gradient w.r.t. W)."""
    from fractus.causal.notears import notears_penalty
    W = torch.randn(3, 3, requires_grad=True)
    h = notears_penalty(W)
    h.backward()
    assert W.grad is not None
    assert torch.isfinite(W.grad).all()


def test_notears_shape_scalar():
    from fractus.causal.notears import notears_penalty
    W = torch.randn(5, 5)
    h = notears_penalty(W)
    assert h.dim() == 0


def test_notears_larger_cycle_detected():
    """A size-4 cycle must be detected.

    Note: NOTEARS is sensitive to the AMPLITUDE of the cycle weights (h measures
    intensity, not just presence). With weights 1.0, h ≈ 0.17; with weights
    2.0, h ≈ 49. We therefore use weights 1.5 and a threshold > 0.1.
    """
    from fractus.causal.notears import notears_penalty
    W = torch.zeros(4, 4)
    W[0, 1] = W[1, 2] = W[2, 3] = W[3, 0] = 1.5  # moderate weights
    h = notears_penalty(W)
    assert h.item() > 0.1, f"h(4-node cycle, weights 1.5) should be > 0.1, got {h.item()}"
