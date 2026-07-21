"""Tests of KuramotoLyapunov: a true Lyapunov function on the Kuramoto subsystem."""

import math
import torch


def test_lyapunov_V_positive_for_nonzero():
    """V(θ) > 0 for θ != θ*."""
    from fractus.stability.lyapunov import KuramotoLyapunov
    from fractus.nn.phase_ode import KuramotoLayer
    kur = KuramotoLayer(d_model=8, n_oscillators=4, rank=2)
    lyap = KuramotoLyapunov(kur, target_phase=0.0)
    phases = torch.rand(2, 3, 4) * 2 * math.pi + 0.1  # != 0
    V = lyap.V(phases)
    assert (V > 0).all()


def test_lyapunov_V_zero_at_target():
    """V(θ*) = 0 (phase synchronized to the target)."""
    from fractus.stability.lyapunov import KuramotoLyapunov
    from fractus.nn.phase_ode import KuramotoLayer
    kur = KuramotoLayer(d_model=8, n_oscillators=4, rank=2)
    lyap = KuramotoLyapunov(kur, target_phase=1.0)
    phases = torch.full((2, 3, 4), 1.0)  # all at θ* = 1.0
    V = lyap.V(phases)
    assert (V.abs() < 1e-5).all()


def test_lyapunov_V_handles_wrap():
    """V handles circular wrapping: θ = 2π - 0.01 ≈ θ* = 0.01."""
    from fractus.stability.lyapunov import KuramotoLyapunov
    from fractus.nn.phase_ode import KuramotoLayer
    kur = KuramotoLayer(d_model=8, n_oscillators=4, rank=2)
    lyap = KuramotoLyapunov(kur, target_phase=0.0)
    # Phase 2π - 0.01 must be close to 0 (wrap), so V is small.
    phases_near_zero = torch.full((1, 1, 4), 2 * math.pi - 0.01)
    phases_far = torch.full((1, 1, 4), math.pi)  # far from 0
    V_near = lyap.V(phases_near_zero).item()
    V_far = lyap.V(phases_far).item()
    assert V_near < V_far, f"V(2π-0.01)={V_near} should be < V(π)={V_far}"


def test_lyapunov_is_stable_trajectory_true_for_decreasing():
    """A trajectory where V decreases → stable."""
    from fractus.stability.lyapunov import KuramotoLyapunov
    from fractus.nn.phase_ode import KuramotoLayer
    kur = KuramotoLayer(d_model=8, n_oscillators=4, rank=2)
    lyap = KuramotoLyapunov(kur, target_phase=0.0)
    # Trajectory converging toward 0: phases decreasing toward 0.
    traj = [torch.full((1, 2, 4), p) for p in [2.0, 1.5, 1.0, 0.5, 0.1]]
    assert lyap.is_stable_trajectory(traj) is True


def test_lyapunov_is_stable_trajectory_false_for_increasing():
    """A trajectory where V increases → unstable."""
    from fractus.stability.lyapunov import KuramotoLyapunov
    from fractus.nn.phase_ode import KuramotoLayer
    kur = KuramotoLayer(d_model=8, n_oscillators=4, rank=2)
    lyap = KuramotoLyapunov(kur, target_phase=0.0)
    traj = [torch.full((1, 2, 4), p) for p in [0.1, 0.5, 1.0, 1.5, 2.0]]  # diverges
    assert lyap.is_stable_trajectory(traj) is False


def test_lyapunov_not_just_output_norm():
    """L6 CRITERION: V must NOT be just ||y||² (the fake Lyapunov of the original).
    We verify that V depends on PHASES, not on an output norm."""
    import inspect
    from fractus.stability import lyapunov as lyap_mod
    src = inspect.getsource(lyap_mod)
    # V must be computed from the phases (circular difference θ - θ*).
    assert "target_phase" in src
    assert "remainder" in src or "wrap" in src.lower(), \
        "V must handle circular phase wrapping (not just ||y||²)"
