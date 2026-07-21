"""Tests of the Farey sequence and expert phase selection."""

import math


def test_farey_sequence_basic():
    """F_3 = {0/1, 1/3, 1/2, 2/3, 1/1} (5 terms)."""
    from fractus.nn.farey import farey_sequence
    seq = farey_sequence(3)
    assert seq == [(0, 1), (1, 3), (1, 2), (2, 3), (1, 1)]


def test_farey_sequence_order_1():
    """F_1 = {0/1, 1/1}."""
    from fractus.nn.farey import farey_sequence
    assert farey_sequence(1) == [(0, 1), (1, 1)]


def test_farey_sequence_sorted():
    """The fractions must be ascending (a Farey property)."""
    from fractus.nn.farey import farey_sequence
    seq = farey_sequence(5)
    values = [p / q for (p, q) in seq]
    assert values == sorted(values)


def test_farey_sequence_all_denominators_le_n():
    """In F_n, all denominators are <= n."""
    from fractus.nn.farey import farey_sequence
    seq = farey_sequence(6)
    for (p, q) in seq:
        assert q <= 6


def test_expert_phases_count():
    """expert_phases(n) returns exactly n angles."""
    from fractus.nn.farey import expert_phases
    for n in [4, 8, 16]:
        phases = expert_phases(n)
        assert len(phases) == n


def test_expert_phases_in_unit_circle():
    """All angles ∈ [0, 2π)."""
    from fractus.nn.farey import expert_phases
    phases = expert_phases(8)
    for theta in phases:
        assert 0.0 <= theta < 2 * math.pi


def test_expert_phases_distinct():
    """The expert phases must be distinct (otherwise routing degenerates)."""
    from fractus.nn.farey import expert_phases
    phases = expert_phases(8)
    for i in range(len(phases)):
        for j in range(i + 1, len(phases)):
            assert abs(phases[i] - phases[j]) > 1e-6, \
                f"phases[{i}]={phases[i]} == phases[{j}]={phases[j]}"
