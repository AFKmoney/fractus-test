"""Numerical utilities: sigmoid, cosine_similarity.

Ported from the original system (src/math/stats.rs). Pure PyTorch.
"""

import torch


def sigmoid(x: torch.Tensor) -> torch.Tensor:
    """σ(x) = 1 / (1 + exp(-x)). Numerically stable for large negative x."""
    return torch.sigmoid(x)


def cosine_similarity(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Cosine similarity: dot(a,b) / (||a|| · ||b||). 0 if a norm < 1e-10."""
    dot = (a * b).sum()
    norm_a = a.norm()
    norm_b = b.norm()
    if norm_a < 1e-10 or norm_b < 1e-10:
        return torch.tensor(0.0, dtype=a.dtype, device=a.device)
    return dot / (norm_a * norm_b)
