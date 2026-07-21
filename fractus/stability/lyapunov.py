"""KuramotoLyapunov: Lyapunov function of the Kuramoto subsystem.

A TRUE Lyapunov function V(theta) = 0.5 * sum(theta_i - theta*)^2 on the Kuramoto
subsystem (the only true dynamical system in the model). Not ||y||^2.
"""

import math
import torch
import torch.nn as nn

from ..nn.phase_ode import KuramotoLayer


class KuramotoLyapunov(nn.Module):
    """Lyapunov function of the Kuramoto subsystem.

    Args:
        kuramoto : a KuramotoLayer (whose stability we want to measure).
        target_phase : synchronized target phase θ* (0.0 by default).
    """

    def __init__(self, kuramoto: KuramotoLayer, target_phase: float = 0.0):
        super().__init__()
        self.kuramoto = kuramoto
        self.target_phase = target_phase

    def V(self, phases: torch.Tensor) -> torch.Tensor:
        """V(θ) = 1⁄2·Σi (θi − θ*)2 (positive definite, = 0 if synchronized).

        phases : (..., N). Returns a scalar.
        """
        # Circular distance: min(|θ-θ*|, 2π - |θ-θ*|) to handle wrapping.
        diff = phases - self.target_phase
        # Wrap into [-π, π].
        diff = torch.remainder(diff + math.pi, 2 * math.pi) - math.pi
        return 0.5 * (diff ** 2).sum(dim=-1)

    def is_stable_trajectory(self, phases_trajectory: list) -> bool:
        """Checks that V decreases along the trajectory.

        phases_trajectory : list of tensors (..., N), one per time step.
        Returns True if V is monotonically non-increasing (within an epsilon).
        """
        if len(phases_trajectory) < 2:
            return True
        Vs = [self.V(p).mean().item() for p in phases_trajectory]
        eps = 1e-6
        for i in range(len(Vs) - 1):
            if Vs[i + 1] > Vs[i] + eps:
                return False
        return True
