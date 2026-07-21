"""Tests of numerical utilities: elu_plus_one, stable_softmax."""

import torch


def test_elu_plus_one_positive_branch():
    """For x > 0: elu_plus_one(x) = x + 1."""
    from fractus.nn.stats import elu_plus_one
    assert abs(elu_plus_one(torch.tensor(2.0)).item() - 3.0) < 1e-6


def test_elu_plus_one_at_zero():
    """elu_plus_one(0, α=1) = 1 (else branch: α(e^0-1)+1 = 1)."""
    from fractus.nn.stats import elu_plus_one
    assert abs(elu_plus_one(torch.tensor(0.0)).item() - 1.0) < 1e-6


def test_elu_plus_one_strictly_positive():
    """elu_plus_one is strictly positive (required for linear attention)."""
    from fractus.nn.stats import elu_plus_one
    xs = torch.linspace(-10, 10, 100)
    out = elu_plus_one(xs)
    assert (out > 0).all()


def test_elu_plus_one_vectorized():
    """Works on a tensor of arbitrary shape (differentiable)."""
    from fractus.nn.stats import elu_plus_one
    x = torch.randn(4, 8, requires_grad=True)
    out = elu_plus_one(x)
    assert out.shape == x.shape
    loss = out.sum()
    loss.backward()
    assert x.grad is not None and torch.isfinite(x.grad).all()


def test_stable_softmax_sums_to_one():
    from fractus.nn.stats import stable_softmax
    logits = torch.tensor([1.0, 2.0, 3.0])
    p = stable_softmax(logits, dim=-1)
    assert abs(p.sum().item() - 1.0) < 1e-6
    assert (p >= 0).all()


def test_stable_softmax_large_values_no_overflow():
    """Stable softmax: no overflow even with large values."""
    from fractus.nn.stats import stable_softmax
    logits = torch.tensor([1000.0, 1001.0, 1002.0])
    p = stable_softmax(logits, dim=-1)
    assert torch.isfinite(p).all()
    assert abs(p.sum().item() - 1.0) < 1e-5
