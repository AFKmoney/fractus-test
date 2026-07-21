"""Tests of measure_compression_ratio: REAL measurement, no hardcoding."""

import inspect
import torch


def test_compression_no_hardcoded_204():
    """L3 CRITERION: the measurement CODE must NOT hardcode a return ratio.
    We tolerate the mention of '20.4×' in docstrings (which explain the corrected
    prior bug), but forbid any 'return 20.4' or fixed numeric ratio in the logic."""
    import re
    from fractus.metrics import compression
    src = inspect.getsource(compression)
    # Strip comments/docstrings before searching.
    code_lines = []
    in_docstring = False
    for line in src.split('\n'):
        stripped = line.strip()
        if stripped.startswith('"""') or stripped.startswith("'''"):
            in_docstring = not in_docstring or stripped.count('"""') == 2
            continue
        if in_docstring:
            continue
        if stripped.startswith('#'):
            continue
        code_lines.append(line)
    code_only = '\n'.join(code_lines)
    # No literal 20.4 in the executable code.
    assert not re.search(r'\b20\.4\b', code_only), \
        "The literal 20.4 is forbidden in executable code (prior bug)"


def test_compression_pure_dense_returns_one():
    """A 100% dense model (no SirenLinear) → ratio ~1.0.

    Note: the ratio for a pure nn.Linear is not EXACTLY 1.0 because we
    count the bias in the actual params (in·out + out) and in the dense
    equivalent (in·out + out as well) — so ~1.0 within epsilon.
    """
    from fractus.metrics.compression import measure_compression_ratio
    m = torch.nn.Linear(16, 16)
    ratio = measure_compression_ratio(m)
    assert abs(ratio - 1.0) < 1e-6, f"Expected pure-dense ratio 1.0, got {ratio}"


def test_compression_with_siren_gt_one():
    """A model with SirenLinear → ratio > 1 (fewer params than the dense equivalent)."""
    from fractus.nn.siren_linear import SirenLinear
    from fractus.metrics.compression import measure_compression_ratio
    m = torch.nn.Sequential(
        SirenLinear(32, 32, hidden=16),
        torch.nn.ReLU(),
        torch.nn.Linear(32, 10),
    )
    ratio = measure_compression_ratio(m)
    assert ratio > 1.0, f"Expected ratio > 1, got {ratio}"


def test_compression_returns_finite():
    from fractus.metrics.compression import measure_compression_ratio
    m = torch.nn.Linear(8, 8)
    r = measure_compression_ratio(m)
    assert isinstance(r, float)
    assert r > 0
