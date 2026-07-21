"""CognitiveModes: Kuramoto phases as a detector of mental state.

THE INNOVATION. The Kuramoto oscillators aren't just a routing mechanism —
they're a DYNAMICAL SYSTEM whose phase pattern reflects the current "cognitive
mode" of the engine. This module:

    1. Extracts features from the phase vector (synchronization, clustering).
    2. Maps phase patterns to named cognitive modes (analytical, creative,
       focused, exploratory).
    3. Lets the engine ADAPT its behavior based on its current mode.

This is what makes Fractus feel ALIVE — it has mental states that change
how it processes information, like a human shifting between focused work
and creative brainstorming.

Usage:
    modes = CognitiveModes(n_oscillators=8, mode_names=[
        "analytical", "creative", "focused", "exploratory",
        "verbal", "spatial", "procedural", "memory"
    ])
    mode = modes.classify(phases)  # → {"mode": "analytical", "confidence": 0.82}
"""

import math
import torch
import torch.nn as nn


class CognitiveModes(nn.Module):
    """Classify the Kuramoto phase state into cognitive modes.

    Args:
        n_oscillators: number of Kuramoto oscillators.
        mode_names: list of cognitive mode names.
        n_modes: number of modes (defaults to len(mode_names)).
    """

    def __init__(
        self,
        n_oscillators: int = 8,
        mode_names: list = None,
        n_modes: int = None,
    ):
        super().__init__()
        self.n_oscillators = n_oscillators
        if mode_names is None:
            mode_names = [
                "analytical", "creative", "focused", "exploratory",
                "verbal", "spatial", "procedural", "memory",
            ]
        self.mode_names = mode_names[:n_modes or len(mode_names)]
        self.n_modes = len(self.mode_names)

        # Learnable classifier: phase features → mode logits.
        # Input features: synchronization (1) + mean phase (1) +
        # phase variance (1) + per-oscillator sin/cos (2*N).
        n_features = 3 + 2 * n_oscillators
        self.classifier = nn.Sequential(
            nn.Linear(n_features, 64),
            nn.ReLU(),
            nn.Linear(64, self.n_modes),
        )

    def extract_features(self, phases: torch.Tensor) -> torch.Tensor:
        """Extract cognitive features from the phase vector.

        Args:
            phases: (..., N) oscillator phases in [0, 2π).
        Returns:
            features: (..., 3 + 2*N) feature vector.
        """
        # Flatten leading dims.
        *leading, N = phases.shape
        phases_flat = phases.reshape(-1, N)  # (B, N)

        sin_p = torch.sin(phases_flat)
        cos_p = torch.cos(phases_flat)

        # Feature 1: order parameter r (synchronization degree).
        # r = |mean(e^{iθ})| = sqrt(mean(cos)² + mean(sin)²)
        r = torch.sqrt(cos_p.mean(dim=-1) ** 2 + sin_p.mean(dim=-1) ** 2)  # (B,)

        # Feature 2: mean phase.
        mean_phase = torch.atan2(sin_p.mean(dim=-1), cos_p.mean(dim=-1))  # (B,)

        # Feature 3: phase variance (how spread out the phases are).
        phase_var = sin_p.var(dim=-1) + cos_p.var(dim=-1)  # (B,)

        # Features 4+: per-oscillator sin/cos.
        osc_features = torch.cat([sin_p, cos_p], dim=-1)  # (B, 2N)

        features = torch.cat([
            r.unsqueeze(-1),
            mean_phase.unsqueeze(-1),
            phase_var.unsqueeze(-1),
            osc_features,
        ], dim=-1)  # (B, 3 + 2N)

        return features.reshape(*leading, features.shape[-1])

    def classify(self, phases: torch.Tensor) -> dict:
        """Classify the current cognitive mode.

        Args:
            phases: (N,) or (1, N) oscillator phases.
        Returns:
            dict with "mode" (str), "confidence" (float), and "all_modes" (dict).
        """
        if phases.dim() == 1:
            phases = phases.unsqueeze(0)
        features = self.extract_features(phases)  # (1, features)
        logits = self.classifier(features)  # (1, n_modes)
        probs = torch.softmax(logits, dim=-1)  # (1, n_modes)

        top_idx = probs.argmax(dim=-1).item()
        top_prob = probs[0, top_idx].item()
        mode_name = self.mode_names[top_idx]

        all_modes = {
            self.mode_names[i]: probs[0, i].item()
            for i in range(self.n_modes)
        }

        return {
            "mode": mode_name,
            "confidence": top_prob,
            "all_modes": all_modes,
        }

    def mode_loss(self, phases: torch.Tensor, target_mode: str) -> torch.Tensor:
        """Cross-entropy loss to TRAIN the mode classifier.

        Args:
            phases: (N,) oscillator phases.
            target_mode: the name of the mode this should be.
        Returns:
            loss: scalar.
        """
        if target_mode not in self.mode_names:
            return torch.tensor(0.0)
        target_idx = self.mode_names.index(target_mode)

        features = self.extract_features(phases.unsqueeze(0))
        logits = self.classifier(features)
        target = torch.tensor([target_idx], device=phases.device)
        return torch.nn.functional.cross_entropy(logits, target)

    def info(self) -> dict:
        return {
            "n_oscillators": self.n_oscillators,
            "modes": self.mode_names,
            "n_features": 3 + 2 * self.n_oscillators,
        }
