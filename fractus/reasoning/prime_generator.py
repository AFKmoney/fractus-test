"""PrimeGenerator: an MLP that learns to produce prime numbers.

L5+ v2 REDESIGN: the task "converge toward a target to within 1e-3" (the original
proof.rs) was unattainable with the GRU+EMA architecture (proven by two attempts:
pure REINFORCE, then curriculum+shaping+baseline, both ~0% validity).

NEW TASK: produce integers that are PRIME. The exact verifier (PrimeSieve) tests
primality — this is binary (true/false on concrete integers), not a numerical
convergence.

Why this task is learnable:
    - Prime density in [2,100] = 25%. A random generator already succeeds
      1 time in 4 → REINFORCE gets a constant non-zero signal.
    - The objective (maximize the fraction of n that are prime) is clear and measurable.
    - The verifier is SOUND: every accepted n is mathematically prime.

Architecture: a simple MLP. Input = a context (random vector or target).
Output = logits over N classes (n ∈ [2, N]). argmax = predicted n.

Honesty note: this is a simple SYMBOLIC task (produce a prime), not a logically
structured proof. This is the level attainable with REINFORCE over an exact
verifier. A genuine "proof" in the sense of a logical derivation requires a
search tree + structural verification (future work).
"""

import torch
import torch.nn as nn

from ..math.primes import PrimeSieve


class PrimeGenerator(nn.Module):
    """An MLP that learns to produce prime numbers.

    Args:
        max_n:       the predicted integers lie in [2, max_n].
        context_dim: dimension of the context vector (input).
        hidden:      MLP width.
    """

    def __init__(self, max_n: int = 100, context_dim: int = 16, hidden: int = 64):
        super().__init__()
        if max_n < 2:
            raise ValueError("max_n must be >= 2")
        self.max_n = max_n
        self.context_dim = context_dim
        # Classes: indices 0..max_n-2 correspond to n = 2..max_n.
        self.n_classes = max_n - 1
        self.mlp = nn.Sequential(
            nn.Linear(context_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, self.n_classes),
        )
        self.sieve = PrimeSieve(max(max_n, 1000))

    def forward(self, context: torch.Tensor) -> torch.Tensor:
        """context: (B, context_dim) → logits (B, n_classes).

        logits[b, i] corresponds to n = i + 2.
        """
        return self.mlp(context)

    def predict(self, context: torch.Tensor) -> torch.Tensor:
        """Returns the predicted n (argmax), shape (B,). n ∈ [2, max_n]."""
        logits = self.forward(context)
        indices = logits.argmax(dim=-1)  # (B,) ∈ [0, n_classes-1]
        return indices + 2  # n = index + 2

    def is_prime_pred(self, n: torch.Tensor) -> torch.Tensor:
        """Checks the primality of the predicted n. Returns a bool tensor (B,)."""
        # n is a tensor of integers; we check it via the sieve.
        results = []
        for ni in n.tolist():
            results.append(self.sieve.verify_prime(int(ni)))
        return torch.tensor(results, dtype=torch.bool)
