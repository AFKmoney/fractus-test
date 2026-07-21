"""Tests of KuramotoLayer: encode/decode, RK4, phase_loss, backward."""

import math
import torch


def test_kuramoto_output_shape():
    """Output phases (B, L, N_osc) for input (B, L, d_model)."""
    from fractus.nn.phase_ode import KuramotoLayer
    layer = KuramotoLayer(d_model=16, n_oscillators=8, rank=4)
    x = torch.randn(2, 10, 16)
    phases = layer(x)
    assert phases.shape == (2, 10, 8)


def test_kuramoto_phases_in_unit_circle():
    """All phases ∈ [0, 2π) (modular wrapping after RK4)."""
    from fractus.nn.phase_ode import KuramotoLayer
    layer = KuramotoLayer(d_model=16, n_oscillators=8, rank=4)
    x = torch.randn(2, 10, 16) * 10
    phases = layer(x)
    assert (phases >= 0).all() and (phases < 2 * math.pi).all()


def test_kuramoto_is_finite():
    from fractus.nn.phase_ode import KuramotoLayer
    layer = KuramotoLayer(d_model=16, n_oscillators=8, rank=4)
    x = torch.randn(2, 10, 16)
    phases = layer(x)
    assert torch.isfinite(phases).all()


def test_kuramoto_backward_every_param():
    """L2b CRITERION: backward propagates a finite AND non-zero gradient to EVERY parameter."""
    from fractus.nn.phase_ode import KuramotoLayer
    layer = KuramotoLayer(d_model=16, n_oscillators=8, rank=4)
    x = torch.randn(2, 10, 16)
    phases = layer(x)
    loss = phases.sum()
    loss.backward()

    params = list(layer.named_parameters())
    assert len(params) > 0
    for name, p in params:
        assert p.requires_grad, f"{name} should requires_grad=True"
        assert p.grad is not None, f"{name} received no gradient"
        assert torch.isfinite(p.grad).all(), f"{name} has a non-finite gradient"
        assert p.grad.abs().sum().item() > 0, f"{name} received a zero gradient"


def test_kuramoto_phase_loss_shape_and_finite():
    """phase_loss(phases) returns a finite scalar."""
    from fractus.nn.phase_ode import KuramotoLayer
    layer = KuramotoLayer(d_model=16, n_oscillators=8, rank=4)
    x = torch.randn(2, 10, 16)
    phases = layer(x)
    loss = layer.phase_loss(phases)
    assert loss.dim() == 0
    assert torch.isfinite(loss)


def test_kuramoto_decode_to_bias_shape():
    """decode_to_bias(phases, d_model) returns (B, L, d_model)."""
    from fractus.nn.phase_ode import KuramotoLayer
    layer = KuramotoLayer(d_model=16, n_oscillators=8, rank=4)
    phases = torch.rand(2, 10, 8) * 2 * math.pi
    bias = layer.decode_to_bias(phases, d_model=16)
    assert bias.shape == (2, 10, 16)
    assert torch.isfinite(bias).all()


def test_rk4_vectorized_matches_reference():
    """L8 CRITERION: the unrolled RK4 (single end-wrap) must match the looped
    reference (per-step wrap) within atol=1e-5. Proves the optimization didn't
    change the dynamics, only removed Python overhead.

    With dt=0.1, n_steps=4 the accumulated drift from delayed wrapping is tiny;
    we assert it stays well within the linear-attention noise floor.
    """
    from fractus.nn.phase_ode import KuramotoLayer
    torch.manual_seed(0)
    # Use n_steps=4 (the default in FractalBlockFull) to exercise the unroll.
    layer = KuramotoLayer(d_model=16, n_oscillators=8, rank=4, n_steps=4, dt=0.1)
    x = torch.randn(3, 12, 16)
    theta_init = layer._encode_from_hidden(x)

    out_vec = layer._rk4_integrate(theta_init)
    out_ref = layer._rk4_integrate_looped(theta_init)

    max_diff = (out_vec - out_ref).abs().max().item()
    assert torch.allclose(out_vec, out_ref, atol=1e-6), \
        f"unrolled RK4 differs from looped reference: max diff {max_diff}"


def test_rk4_single_step_n_steps_1():
    """n_steps=1 must still integrate one full RK4 step (not a no-op)."""
    from fractus.nn.phase_ode import KuramotoLayer
    torch.manual_seed(0)
    layer = KuramotoLayer(d_model=8, n_oscillators=4, rank=2, n_steps=1, dt=0.1)
    x = torch.randn(2, 5, 8)
    theta_init = layer._encode_from_hidden(x)
    out = layer._rk4_integrate(theta_init)
    # Must differ from the init (integration happened).
    assert not torch.allclose(out, theta_init, atol=1e-6)
    assert torch.isfinite(out).all()
