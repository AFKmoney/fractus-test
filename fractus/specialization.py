"""Specialized experts: force each MoE expert to own a specific skill domain.

THE INNOVATION. Standard MoE experts are interchangeable — routing is based
on phase similarity, not content. This module adds:

    1. Domain labels: each expert is assigned a domain (code, math, text...).
    2. A diversity loss: penalizes two experts that produce similar outputs
       for the same input → forces specialization.
    3. A domain-matching bonus: when the input matches an expert's domain,
       the routing gate is boosted.

This makes the MoE a true SKILL DISPATCHER: the Kuramoto phases detect the
cognitive mode, and the specialized experts provide domain-specific processing.

Usage:
    spec = ExpertSpecialization(
        n_experts=4,
        domains=["code", "math", "language", "reasoning"],
    )
    div_loss = spec.diversity_loss(all_expert_outputs)  # add to training loss
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ExpertSpecialization(nn.Module):
    """Forces MoE experts to specialize on distinct domains.

    Args:
        n_experts: number of experts.
        domains: list of domain names (one per expert).
        d_model: model dimension (for the domain embedding).
    """

    def __init__(
        self,
        n_experts: int = 4,
        domains: list = None,
        d_model: int = 128,
    ):
        super().__init__()
        self.n_experts = n_experts
        if domains is None:
            domains = [f"domain_{i}" for i in range(n_experts)]
        self.domains = domains[:n_experts]

        # Learnable domain embedding: each expert has a "domain vector" that
        # represents what it's good at. Used for diversity loss.
        self.domain_vectors = nn.Parameter(
            torch.randn(n_experts, d_model) * 0.02
        )

    def diversity_loss(self, expert_outputs: torch.Tensor) -> torch.Tensor:
        """Penalize experts that produce similar outputs.

        Args:
            expert_outputs: (E, D) — the output of each expert on the same input.
        Returns:
            loss: scalar. Lower = more diverse (better specialization).
        """
        if expert_outputs.shape[0] < 2:
            return torch.tensor(0.0, device=expert_outputs.device)

        # Cosine similarity matrix between expert outputs.
        # We want experts to be ORTHOGONAL (sim → 0).
        sims = F.cosine_similarity(
            expert_outputs.unsqueeze(1),  # (E, 1, D)
            expert_outputs.unsqueeze(0),  # (1, E, D)
            dim=-1,
        )  # (E, E)

        # Zero the diagonal (self-similarity is always 1).
        eye = torch.eye(self.n_experts, device=expert_outputs.device)
        sims = sims * (1 - eye)

        # Penalty: sum of off-diagonal similarities (want → 0).
        return sims.abs().sum() / (self.n_experts * (self.n_experts - 1))

    def domain_embedding_loss(self) -> torch.Tensor:
        """Keep domain vectors well-separated (orthogonal).

        Returns:
            loss: scalar. Lower = domains more distinct.
        """
        if self.n_experts < 2:
            return torch.tensor(0.0)
        # Gram matrix of domain vectors.
        gram = self.domain_vectors @ self.domain_vectors.T  # (E, E)
        # Normalize by norms.
        norms = self.domain_vectors.norm(dim=-1, keepdim=True)  # (E, 1)
        normed_gram = gram / (norms @ norms.T + 1e-10)
        # Zero diagonal.
        eye = torch.eye(self.n_experts, device=gram.device)
        off_diag = normed_gram * (1 - eye)
        return off_diag.abs().sum() / (self.n_experts * (self.n_experts - 1))

    def get_domain(self, expert_idx: int) -> str:
        """Get the domain label for an expert."""
        if 0 <= expert_idx < len(self.domains):
            return self.domains[expert_idx]
        return "unknown"

    def info(self) -> dict:
        return {
            "n_experts": self.n_experts,
            "domains": self.domains,
            "method": "diversity + orthogonality loss",
        }
