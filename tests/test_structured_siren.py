"""Tests of StructuredSirenLinear (L9 core innovation)."""

import torch


def test_structured_siren_shape():
    """Output shape matches nn.Linear."""
    from fractus.nn.structured_siren import StructuredSirenLinear
    layer = StructuredSirenLinear(in_features=64, out_features=128, rank=32, siren_hidden=32)
    x = torch.randn(4, 64)
    y = layer(x)
    assert y.shape == (4, 128)


def test_structured_siren_is_finite():
    from fractus.nn.structured_siren import StructuredSirenLinear
    layer = StructuredSirenLinear(in_features=64, out_features=128, rank=32, siren_hidden=32)
    x = torch.randn(4, 64) * 3
    assert torch.isfinite(layer(x)).all()


def test_structured_siren_backward():
    """Gradients flow to BOTH low-rank params (U, V) AND the SIREN residual."""
    from fractus.nn.structured_siren import StructuredSirenLinear
    layer = StructuredSirenLinear(in_features=64, out_features=128, rank=32, siren_hidden=32)
    x = torch.randn(4, 64)
    y = layer(x)
    loss = y.pow(2).sum()
    loss.backward()
    for name, p in layer.named_parameters():
        assert p.grad is not None, f"{name} got no gradient"
        assert torch.isfinite(p.grad).all(), f"{name} non-finite grad"


def test_structured_siren_compression_ratio():
    """L9 CRITERION: the compression ratio must be > 1 (fewer params than dense).

    For 256x256, rank=64, siren_hidden=32:
    - Dense: 256*256 = 65536
    - Low-rank: (256+256)*64 = 32768
    - SIREN: 2*32 + 32*32 + 32*1 + biases ≈ 1184
    - Total actual ≈ 33952 + bias 256
    - Ratio ≈ 65536/34208 ≈ 1.9×

    We check it's > 1.5 (proven compression). Higher ranks + bigger matrices
    give better ratios.
    """
    from fractus.nn.structured_siren import StructuredSirenLinear
    layer = StructuredSirenLinear(in_features=256, out_features=256, rank=32, siren_hidden=32)
    ratio = layer.compression_ratio
    assert ratio > 1.5, f"Compression ratio should be > 1.5, got {ratio}"


def test_structured_siren_large_matrix_high_compression():
    """On a large matrix (1024x1024) with low rank, compression should be high."""
    from fractus.nn.structured_siren import StructuredSirenLinear
    layer = StructuredSirenLinear(in_features=1024, out_features=1024, rank=128, siren_hidden=64)
    ratio = layer.compression_ratio
    # Dense: 1024*1024 = 1,048,576
    # Low-rank: (1024+1024)*128 = 262,144
    # SIREN: 2*64 + 64*64 + 64 + biases ≈ 4352
    # Total ≈ 266,496 + bias 1024
    # Ratio ≈ 1M/267K ≈ 3.9
    assert ratio > 3.0, f"Large-matrix compression should be > 3, got {ratio}"


def test_structured_siren_can_fit_target():
    """The layer must be able to FIT a target matrix (both low-rank + SIREN
    contribute). Overfit a random 128x128 target."""
    from fractus.nn.structured_siren import StructuredSirenLinear
    torch.manual_seed(42)
    layer = StructuredSirenLinear(in_features=128, out_features=128, rank=64, siren_hidden=64)
    opt = torch.optim.Adam(layer.parameters(), lr=1e-2)
    # Random target matrix to fit.
    target_W = torch.randn(128, 128) * 0.5
    x = torch.eye(128)  # identity input → output = W + bias
    initial_loss = None
    for step in range(200):
        opt.zero_grad()
        out = layer(x)
        loss = ((out - target_W) ** 2).mean()
        if initial_loss is None:
            initial_loss = loss.item()
        loss.backward()
        opt.step()
    final_loss = loss.item()
    assert final_loss < initial_loss * 0.3, \
        f"StructuredSirenLinear failed to fit target: {initial_loss:.4f} -> {final_loss:.4f}"
