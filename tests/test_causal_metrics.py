"""Tests of structural_hamming_distance: honest measurement, no 0.98 clamp."""

import inspect
import torch


def test_shd_perfect_match_zero():
    from fractus.metrics.causal import structural_hamming_distance
    W = torch.tensor([
        [0.0, 0.0, 0.0],
        [0.5, 0.0, 0.0],
        [0.3, 0.4, 0.0],
    ])
    shd = structural_hamming_distance(W, W, threshold=0.1)
    assert shd == 0


def test_shd_counts_missing_edges():
    from fractus.metrics.causal import structural_hamming_distance
    true_W = torch.tensor([
        [0.0, 0.0, 0.0],
        [0.5, 0.0, 0.0],
        [0.3, 0.4, 0.0],
    ])
    pred_W = torch.zeros(3, 3)
    shd = structural_hamming_distance(true_W, pred_W, threshold=0.1)
    assert shd == 3


def test_shd_counts_extra_edges():
    from fractus.metrics.causal import structural_hamming_distance
    true_W = torch.zeros(3, 3)
    pred_W = torch.tensor([
        [0.0, 0.5, 0.3],
        [0.0, 0.0, 0.4],
        [0.0, 0.0, 0.0],
    ])
    shd = structural_hamming_distance(true_W, pred_W, threshold=0.1)
    assert shd == 3


def test_shd_threshold_filters_small_values():
    from fractus.metrics.causal import structural_hamming_distance
    true_W = torch.tensor([
        [0.0, 0.0, 0.0],
        [0.5, 0.0, 0.0],
        [0.0, 0.0, 0.0],
    ])
    pred_W = torch.tensor([
        [0.0, 0.0, 0.0],
        [0.05, 0.0, 0.0],  # < threshold, ignored
        [0.0, 0.0, 0.0],
    ])
    shd = structural_hamming_distance(true_W, pred_W, threshold=0.1)
    assert shd == 1


def test_shd_no_clamp_to_098():
    """L4 CRITERION: the EXECUTABLE code must not contain a 0.98 clamp
    (the prior bug: benchmarks.py:43-46 used min(causal_acc, 0.98)).

    We tolerate '0.98' in docstrings (which explain the corrected bug),
    but forbid it in Python expressions outside comments/docstrings."""
    from fractus.metrics import causal as causal_mod

    src = inspect.getsource(causal_mod)
    code_lines = []
    in_docstring = False
    for line in src.split('\n'):
        stripped = line.strip()
        if stripped.startswith('"""') or stripped.startswith("'''"):
            if stripped.count('"""') == 2 or stripped.count("'''") == 2:
                continue  # single-line docstring
            in_docstring = not in_docstring
            continue
        if in_docstring:
            continue
        if stripped.startswith('#'):
            continue
        code_lines.append(line)
    code_only = '\n'.join(code_lines)
    assert "0.98" not in code_only, \
        "The literal 0.98 is forbidden in executable code (prior bug)"


def test_causal_accuracy_no_clamp():
    """causal_accuracy must not be clamped."""
    from fractus.metrics.causal import causal_accuracy
    true_W = torch.eye(3)
    pred_W = torch.eye(3) * 2.0
    acc = causal_accuracy(true_W, pred_W, threshold=0.5)
    assert abs(acc - 1.0) < 1e-6
