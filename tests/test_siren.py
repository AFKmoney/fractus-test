"""Tests of TorusSirenWeight: a true SIREN sin(ω0·), not SiLU."""

import inspect
import torch


def test_siren_uses_sin_not_silu():
    """L3 CRITERION: the SIREN must use torch.sin as the non-linearity,
    NOT nn.SiLU. This is exactly the prior falsehood (torus_siren.py:15,17 used SiLU).

    We verify this via REAL source inspection of the module code (AST),
    not via string search (which would match the docstring that explains
    the correction)."""
    import ast
    from fractus.nn import siren as siren_mod

    # 1. torch.sin must be called in the forward.
    src = inspect.getsource(siren_mod)
    assert 'torch.sin(' in src, "The SIREN must use torch.sin(ω₀·)"

    # 2. Parse the source and verify that no Attribute with attr='SiLU'
    #    is used as a call (nn.SiLU()).
    tree = ast.parse(src)
    silu_calls = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
            if node.func.attr == 'SiLU':
                silu_calls.append(node)
    assert not silu_calls, \
        f"No SiLU() call may appear in the code (prior bug). Found: {len(silu_calls)}"


def test_siren_omega0_is_30_not_56():
    """ω0 = 30 (justified by Sitzmann 2020), NOT 56 (unjustified, inherited from the original)."""
    from fractus.nn.siren import TorusSirenWeight
    s = TorusSirenWeight(out_h=16, out_w=16, hidden=32)
    assert abs(s.omega0 - 30.0) < 1e-6, f"ω₀ should be 30.0, got {s.omega0}"


def test_siren_output_shape():
    """The SIREN evaluated on the grid produces a matrix (out_h, out_w)."""
    from fractus.nn.siren import TorusSirenWeight
    s = TorusSirenWeight(out_h=16, out_w=16, hidden=32)
    W = s()
    assert W.shape == (16, 16)


def test_siren_is_finite():
    from fractus.nn.siren import TorusSirenWeight
    s = TorusSirenWeight(out_h=16, out_w=16, hidden=32)
    W = s()
    assert torch.isfinite(W).all()


def test_siren_backward_propagates():
    """L3 CRITERION: backward propagates a finite AND non-zero gradient to EVERY parameter."""
    from fractus.nn.siren import TorusSirenWeight
    s = TorusSirenWeight(out_h=16, out_w=16, hidden=32)
    W = s()
    loss = W.pow(2).sum()
    loss.backward()

    params = list(s.named_parameters())
    assert len(params) > 0
    for name, p in params:
        assert p.requires_grad, f"{name} should requires_grad=True"
        assert p.grad is not None, f"{name} received no gradient"
        assert torch.isfinite(p.grad).all(), f"{name} has a non-finite gradient"
        assert p.grad.abs().sum().item() > 0, f"{name} received a zero gradient"


def test_siren_fewer_params_than_dense():
    """The SIREN must have FEWER parameters than the equivalent dense matrix."""
    from fractus.nn.siren import TorusSirenWeight
    s = TorusSirenWeight(out_h=32, out_w=32, hidden=16)
    n_siren = sum(p.numel() for p in s.parameters())
    n_dense = 32 * 32
    assert n_siren < n_dense, \
        f"SIREN ({n_siren} params) should be < dense ({n_dense})"
