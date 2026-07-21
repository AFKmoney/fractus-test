"""Tests of the fractal embedding: char features, Fourier basis, FractalEmbedding.

The critical criterion (inherited from the original system, which failed here): the
forward pass must be differentiable and backward() must propagate finite gradients
everywhere.
"""

import torch
import pytest


# ---------------------------------------------------------------------------
# CharClassFeatures (16 morphological features)
# ---------------------------------------------------------------------------

def test_char_features_shape():
    """16 features for every token id."""
    from fractus.nn.char_features import CharClassFeatures
    f = CharClassFeatures.extract(ord("a"))
    assert f.shape == (16,)


def test_char_features_vowel():
    """'a' is a vowel (feature 0 = 1)."""
    from fractus.nn.char_features import CharClassFeatures
    f = CharClassFeatures.extract(ord("a"))
    assert f[0].item() == 1.0  # is_vowel


def test_char_features_digit_value():
    """'5' is a digit of value 5 (feature 11)."""
    from fractus.nn.char_features import CharClassFeatures
    f = CharClassFeatures.extract(ord("5"))
    assert f[2].item() == 1.0   # is_digit
    assert f[11].item() == 5.0  # digit_value


def test_char_features_batch_consistency():
    """The same letter yields the same feature vector."""
    from fractus.nn.char_features import CharClassFeatures
    f1 = CharClassFeatures.extract(ord("z"))
    f2 = CharClassFeatures.extract(ord("z"))
    assert torch.equal(f1, f2)


# ---------------------------------------------------------------------------
# MandelbrotFourierBasis (Fourier basis with (φ2)^{-k} decay)
# ---------------------------------------------------------------------------

def test_fourier_basis_shape():
    """For vocab 128 and 32 frequencies: matrix (vocab, 2·n_freq) (sin+cos)."""
    from fractus.nn.fourier import MandelbrotFourierBasis
    basis = MandelbrotFourierBasis(vocab_size=128, n_frequencies=32)
    M = basis.matrix()  # (vocab, 2*n_freq)
    assert M.shape == (128, 64)  # 2*32 columns (sin+cos)


def test_fourier_basis_is_finite():
    from fractus.nn.fourier import MandelbrotFourierBasis
    basis = MandelbrotFourierBasis(vocab_size=128, n_frequencies=16)
    M = basis.matrix()
    assert torch.isfinite(M).all()


def test_fourier_frequencies_decay():
    """The frequencies ω_k = (φ2)^{-k} must decay geometrically."""
    from fractus.nn.fourier import MandelbrotFourierBasis
    basis = MandelbrotFourierBasis(vocab_size=10, n_frequencies=4)
    # ω_0 = 1.0, ω_1 = 1/φ2, ω_2 = 1/φ4, ...
    phi_sq = ((1 + 5 ** 0.5) / 2) ** 2
    expected = [phi_sq ** (-k) for k in range(4)]
    for k, exp in enumerate(expected):
        assert abs(basis.frequencies[k].item() - exp) < 1e-5, \
            f"freq[{k}] = {basis.frequencies[k].item()}, expected {exp}"


def test_fourier_matrix_is_deterministic():
    """Two calls yield the same matrix (no randomness)."""
    from fractus.nn.fourier import MandelbrotFourierBasis
    b1 = MandelbrotFourierBasis(vocab_size=64, n_frequencies=16)
    b2 = MandelbrotFourierBasis(vocab_size=64, n_frequencies=16)
    assert torch.allclose(b1.matrix(), b2.matrix())


# ---------------------------------------------------------------------------
# FractalEmbedding (assembly + trainable projection)
# ---------------------------------------------------------------------------

def test_fractal_embedding_shape():
    """Output (N, d_model) for input (N,) of ids."""
    from fractus.nn.embedding import FractalEmbedding
    emb = FractalEmbedding(vocab_size=128, d_model=64, n_frequencies=16)
    ids = torch.tensor([0, 1, 2, 65, 97])  # mix
    out = emb(ids)
    assert out.shape == (5, 64)


def test_fractal_embedding_is_finite():
    from fractus.nn.embedding import FractalEmbedding
    emb = FractalEmbedding(vocab_size=128, d_model=64, n_frequencies=16)
    ids = torch.arange(128)
    out = emb(ids)
    assert torch.isfinite(out).all()


def test_fractal_embedding_backward_propagates():
    """CRITICAL: backward() must propagate a finite AND non-zero gradient to EVERY parameter.

    This is exactly the test that the original system failed (training.rs:399 used random
    noise instead of a gradient). Here, the Linear projection is in the autodiff graph,
    so the gradients must be non-zero and finite.

    We check EVERY parameter individually (not just "at least one"), because a dead
    parameter (zero gradient) in the vortex MLP for instance would indicate a silently
    broken autodiff — exactly the defect this rebuild must eliminate.
    """
    from fractus.nn.embedding import FractalEmbedding
    emb = FractalEmbedding(vocab_size=128, d_model=64, n_frequencies=16)
    ids = torch.tensor([0, 1, 2, 3, 4])
    out = emb(ids)
    loss = out.pow(2).sum()
    loss.backward()

    params = list(emb.named_parameters())
    assert len(params) > 0, "The model has no trainable parameter"
    for name, p in params:
        assert p.requires_grad, f"{name} should requires_grad=True"
        assert p.grad is not None, f"{name} received no gradient (dead parameter)"
        assert torch.isfinite(p.grad).all(), f"{name} has a non-finite gradient (NaN/Inf)"
        grad_l1 = p.grad.abs().sum().item()
        assert grad_l1 > 0, (
            f"{name} received a zero gradient — autodiff does not propagate "
            f"to this parameter (grad L1 = {grad_l1})"
        )


def test_fractal_embedding_respects_vocab_bounds():
    """An id >= vocab_size must raise an error (no silent crash)."""
    from fractus.nn.embedding import FractalEmbedding
    emb = FractalEmbedding(vocab_size=100, d_model=32, n_frequencies=8)
    with pytest.raises(IndexError):
        emb(torch.tensor([100]))  # out of bounds
