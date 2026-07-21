"""Metrics subpackage: honest measurements (compression, causal, perplexity).

L3: compression (real measurement, no hardcoding).
L4: causal (SHD, causal accuracy, no clamp).
L6: perplexity (true exp(val_loss), not a proxy).
"""

from .compression import measure_compression_ratio
from .causal import structural_hamming_distance, causal_accuracy
from .perplexity import honest_perplexity

__all__ = [
    "measure_compression_ratio",
    "structural_hamming_distance",
    "causal_accuracy",
    "honest_perplexity",
]
