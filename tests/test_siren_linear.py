"""Tests of SirenLinear: behaves like nn.Linear but W = SIREN(grid)."""

import torch


def test_siren_linear_shape():
    """SirenLinear(in, out) behaves like nn.Linear: (B, in) → (B, out)."""
    from fractus.nn.siren_linear import SirenLinear
    layer = SirenLinear(in_features=16, out_features=16, hidden=32)
    x = torch.randn(4, 16)
    y = layer(x)
    assert y.shape == (4, 16)


def test_siren_linear_is_finite():
    from fractus.nn.siren_linear import SirenLinear
    layer = SirenLinear(in_features=16, out_features=16, hidden=32)
    x = torch.randn(4, 16)
    assert torch.isfinite(layer(x)).all()


def test_siren_linear_backward_propagates():
    """L3 CRITERION: backward propagates a finite AND non-zero gradient to EVERY parameter
    of the SIREN (which IS the weight matrix, in the graph)."""
    from fractus.nn.siren_linear import SirenLinear
    layer = SirenLinear(in_features=16, out_features=16, hidden=32)
    x = torch.randn(4, 16)
    y = layer(x)
    loss = y.pow(2).sum()
    loss.backward()

    params = list(layer.named_parameters())
    assert len(params) > 0
    for name, p in params:
        assert p.requires_grad
        assert p.grad is not None, f"{name} received no gradient"
        assert torch.isfinite(p.grad).all()
        assert p.grad.abs().sum().item() > 0, f"{name} received a zero gradient"


def test_siren_linear_has_no_dense_weight():
    """SirenLinear must NOT have a separate dense nn.Parameter — the matrix
    comes entirely from the SIREN."""
    from fractus.nn.siren_linear import SirenLinear
    layer = SirenLinear(in_features=16, out_features=16, hidden=32)
    # All params must come from the SIREN (prefix 'siren.') or be the bias.
    param_names = [n for n, _ in layer.named_parameters()]
    for n in param_names:
        assert n.startswith("siren.") or n == "bias", \
            f"Unexpected param (should be siren.* or bias): {n}"
