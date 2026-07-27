"""Tests for the EDT AB-test helper library."""
import experiments.edt_ab.ablib as ablib  # noqa: F401  (import smoke test)


def test_package_imports():
    assert ablib is not None
