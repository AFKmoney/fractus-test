"""Honest measurement of a model's compression ratio.

The ratio is MEASURED: we count the parameters actually used and compare them
to the size the matrices would have if they were dense. No hardcoded values.
"""

import torch
import torch.nn as nn

from ..nn.siren_linear import SirenLinear


def _count_params(module: nn.Module) -> int:
    """Total number of trainable parameters in a module."""
    return sum(p.numel() for p in module.parameters() if p.requires_grad)


def measure_compression_ratio(model: nn.Module) -> float:
    """MEASURES the compression ratio of a model.

    Args:
        model: an nn.Module that may contain SirenLinear and/or nn.Linear.
    Returns:
        ratio > 0. ratio = 1.0 if the model is 100% dense.
        ratio > 1 if the model contains SirenLinear (effective compression).
        ratio < 1 is possible but rare (SIREN larger than the matrix).

    KNOWN LIMITATION: only handles SirenLinear, nn.Linear, nn.LayerNorm, nn.Embedding.
    Any other module (Conv, BatchNorm, RNN, etc.) is silently ignored —
    the ratio would then be underestimated. For general use, extend the list or
    emit a warning. Sufficient for the L3 demo (MLP) and the fractus transformer.
    """
    total_dense_equivalent = 0
    total_actual_params = 0

    for module in model.modules():
        if isinstance(module, SirenLinear):
            # Dense-equivalent size: in·out + out (matrix + bias).
            dense_eq = module.in_features * module.out_features
            if module.bias is not None:
                dense_eq += module.out_features
            actual = _count_params(module)
            total_dense_equivalent += dense_eq
            total_actual_params += actual
        elif isinstance(module, nn.Linear) and not isinstance(module, SirenLinear):
            # nn.Linear: dense_eq == actual (no compression).
            dense_eq = module.in_features * module.out_features
            if module.bias is not None:
                dense_eq += module.out_features
            actual = _count_params(module)
            total_dense_equivalent += dense_eq
            total_actual_params += actual
        elif isinstance(module, (nn.LayerNorm, nn.Embedding)):
            # Other modules: no compression (counted at their real size).
            actual = _count_params(module)
            total_dense_equivalent += actual
            total_actual_params += actual

    if total_actual_params == 0:
        return 1.0
    return total_dense_equivalent / total_actual_params
