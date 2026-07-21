"""Smoke test: proves that the Python → PyTorch → Rust plumbing holds.

These tests do NOT validate any mathematical logic — only that the building
blocks communicate. If any of these tests fail, nothing else can work.
"""


def test_torch_available():
    """PyTorch is installed and functional."""
    import torch
    t = torch.tensor([1.0, 2.0, 3.0])
    assert t.sum().item() == 6.0


def test_numpy_available():
    """NumPy is installed (needed for the tensor bridge)."""
    import numpy as np
    a = np.array([1, 2, 3])
    assert a.sum() == 6


def test_rust_bridge_import():
    """The native module fractus._core is well-built and importable."""
    from fractus import _core
    assert hasattr(_core, "add")


def test_rust_bridge_add():
    """Python can call Rust and recover the correct result."""
    from fractus import _core
    assert _core.add(2, 3) == 5
    assert _core.add(-10, 4) == -6


def test_torch_numpy_interop():
    """PyTorch and numpy exchange tensors (needed for the Rust bridge)."""
    import numpy as np
    import torch
    arr = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    t = torch.from_numpy(arr)
    assert t.dtype == torch.float32
    # Back to numpy
    back = t.numpy()
    assert np.allclose(back, arr)
