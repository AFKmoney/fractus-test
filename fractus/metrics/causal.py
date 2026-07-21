"""Honest causal metrics: Structural Hamming Distance, causal accuracy.

SHD and causal_accuracy are MEASURED on a true DAG, with no clamp.
"""

import torch


def structural_hamming_distance(
    true_W: torch.Tensor,
    pred_W: torch.Tensor,
    threshold: float = 0.3,
) -> int:
    """SHD: number of mispredicted edges after binarization.

    Args:
        true_W : true adjacency matrix (n, n).
        pred_W : predicted matrix (n, n).
        threshold : binarization threshold (|W_ij| > threshold → edge present).
    Returns:
        shd : integer >= 0. 0 = perfect prediction.
    """
    true_bin = (true_W.abs() > threshold).float()
    pred_bin = (pred_W.abs() > threshold).float()
    diff = (true_bin != pred_bin).sum().item()
    return int(diff)


def causal_accuracy(
    true_W: torch.Tensor,
    pred_W: torch.Tensor,
    threshold: float = 0.3,
) -> float:
    """Fraction of adjacency-matrix entries predicted correctly. NO clamp
    (unlike the original system, which capped it at 0.98).

    Args:
        true_W, pred_W : matrices (n, n).
        threshold : binarization threshold.
    Returns:
        accuracy ∈ [0, 1].
    """
    true_bin = (true_W.abs() > threshold).float()
    pred_bin = (pred_W.abs() > threshold).float()
    correct = (true_bin == pred_bin).float().mean().item()
    return correct
