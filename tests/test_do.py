"""Tests of do_intervention: true Pearl do-calculus, not column-zeroing."""

import torch


def test_do_intervention_clamps_value():
    """do(X_i = v) must fix column i to v for all rows."""
    from fractus.causal.do import do_intervention
    x = torch.randn(4, 3)
    intervened = do_intervention(x, var_idx=1, value=5.0)
    assert torch.allclose(intervened[:, 1], torch.full((4,), 5.0))
    assert torch.allclose(intervened[:, 0], x[:, 0])


def test_do_intervention_other_cols_unchanged():
    from fractus.causal.do import do_intervention
    x = torch.randn(4, 3)
    intervened = do_intervention(x, var_idx=0, value=-2.0)
    assert torch.allclose(intervened[:, 1], x[:, 1])
    assert torch.allclose(intervened[:, 2], x[:, 2])


def test_do_intervention_preserves_shape():
    from fractus.causal.do import do_intervention
    x = torch.randn(8, 5)
    intervened = do_intervention(x, var_idx=2, value=0.0)
    assert intervened.shape == x.shape


def test_do_intervention_is_differentiable():
    from fractus.causal.do import do_intervention
    x = torch.randn(4, 3, requires_grad=True)
    intervened = do_intervention(x, var_idx=1, value=2.0)
    loss = intervened.sum()
    loss.backward()
    assert x.grad is not None
    assert torch.isfinite(x.grad).all()


def test_do_intervention_not_zeroing():
    """L4 CRITERION: do(X_i = v) must NOT just zero the column (the prior fake
    implementation did so). It must set it to v (which can be non-zero)."""
    from fractus.causal.do import do_intervention
    x = torch.randn(4, 3)
    intervened = do_intervention(x, var_idx=1, value=7.7)
    assert not torch.allclose(intervened[:, 1], torch.zeros(4)), \
        "do(X_i=v) must not zero the column (the prior bug set it to 0)"
    assert torch.allclose(intervened[:, 1], torch.full((4,), 7.7))
