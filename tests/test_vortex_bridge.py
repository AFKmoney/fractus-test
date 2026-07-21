"""Python bridge tests for the 2-adic vortex functions.

Verifies that the Rust wrappers are properly exposed and return correct
values. These tests do NOT do advanced math (that's in Rust) — they only
validate the PyO3 bridge.
"""


def test_collatz_hash_is_deterministic():
    """Same input → same output (property required for conditioning)."""
    from fractus import _core
    h1 = _core.collatz_hash(7, 10)
    h2 = _core.collatz_hash(7, 10)
    assert h1 == h2


def test_collatz_hash_zero_stays_zero():
    """Convention: 0 → 0."""
    from fractus import _core
    assert _core.collatz_hash(0, 100) == 0


def test_collatz_hash_returns_u64():
    """The hash must be a non-negative integer compatible with PyTorch indexing."""
    from fractus import _core
    h = _core.collatz_hash(42, 5)
    assert isinstance(h, int)
    assert h >= 0


def test_ultrametric_distance_self_is_zero():
    """d(a, a) = 0."""
    from fractus import _core
    assert _core.ultrametric_distance(42, 42) == 0.0


def test_ultrametric_distance_symmetric():
    """d(a, b) = d(b, a)."""
    from fractus import _core
    for a, b in [(1, 2), (7, 56), (100, 200)]:
        assert _core.ultrametric_distance(a, b) == _core.ultrametric_distance(b, a)


def test_ultrametric_distance_in_unit_interval():
    """For a != b, d(a,b) ∈ (0, 1] (p-adic norm)."""
    from fractus import _core
    for a, b in [(1, 2), (7, 56), (100, 200), (3, 9)]:
        d = _core.ultrametric_distance(a, b)
        assert 0.0 < d <= 1.0, f"d({a},{b}) = {d} outside (0, 1]"


def test_norm_2adic_basic():
    """||x||_2 = 2^{-v_2(x)}, verified on a few known values."""
    from fractus import _core
    assert _core.norm_2adic(0) == 0.0
    assert _core.norm_2adic(1) == 1.0   # v_2(1)=0 → 2^0
    assert _core.norm_2adic(2) == 0.5   # v_2(2)=1 → 2^-1
    assert _core.norm_2adic(8) == 0.125  # v_2(8)=3 → 2^-3


def test_ultrametric_strong_triangle_in_python():
    """The strong ultrametric property must hold via the Python bridge.
    This is the pivot test that distinguishes 2^{-v} (correct) from 2^{+v} (the original bug)."""
    from fractus import _core
    # The triplet (7, 56, 13) discriminates: passes with -v, fails with +v.
    x, y, z = 7, 56, 13
    d_xy = _core.ultrametric_distance(x, y)
    d_yz = _core.ultrametric_distance(y, z)
    d_xz = _core.ultrametric_distance(x, z)
    assert d_xz <= max(d_xy, d_yz) + 1e-9
