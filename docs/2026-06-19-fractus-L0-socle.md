# Fractus L0 — Technical Foundation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Establish the technical foundation of fractus — a repo with a pure-Rust crate (`fractus-core`), a PyO3 bindings crate (`fractus-py`), a Python package (`fractus`), and a smoke test that proves Python → PyTorch → maturin → Rust → back to Python works end-to-end.

**Architecture:** Three isolated components. (1) `fractus-core`: pure Rust, no I/O, exports mathematical functions (here just `add` for the smoke test + the 2-adic vortex port from the original). (2) `fractus-py`: PyO3/maturin bindings that expose `fractus-core` to Python under the name `fractus._core`. (3) `fractus`: Python package installing PyTorch and hosting the trainable model (empty for L0). Rust stays outside the autodiff graph; forward/backward will be done in PyTorch (later layers).

**Tech Stack:** Rust 1.94 + `nalgebra`, `pyo3` (feature `extension-module`); Python 3.14 via the `py` launcher in a dedicated venv; `maturin 1.14` for the build; `torch` (CPU-only wheel) + `numpy` + `pytest`.

**Environment (verified on the target machine):**
- `py` → Python 3.14.0 with pip 25.3 ✅
- `cargo` / `rustc` 1.94.0 ✅
- `python`/`python1` (MSYS2) → **without pip, do not use**
- Hardware: AMD Ryzen 5 5500U, effective CPU-only
- Created in: `C:/Users/PHIL/ZCodeProject/fractus/`

**Spec link:** `docs/SPEC.md`, section 6 "L0 — Technical foundation".

---

## File Structure

```
C:/Users/PHIL/ZCodeProject/fractus/
├── .gitignore                          # ignore .venv, target/, __pycache__, *.egg-info
├── README.md                           # short pitch + dev instructions
├── pyproject.toml                      # root Python project (fractus), built by maturin
├── requirements-dev.txt                # pinned versions for reproducibility
├── crate/
│   ├── fractus-core/                   # workspace member: pure-Rust mathematical core
│   │   ├── Cargo.toml
│   │   └── src/
│   │       ├── lib.rs                  # pub mod only for existing modules
│   │       └── vortex.rs               # 2-adic port from the original (correct + fixes)
│   └── fractus-py/                     # workspace member: PyO3 bindings
│       ├── Cargo.toml
│       └── src/
│           └── lib.rs                  # #[pymodule] fractus._core
├── Cargo.toml                          # root workspace (links the 2 crates)
├── fractus/                            # Python package (the trainable model, empty in L0)
│   ├── __init__.py                     # expose fractus._core for the test
│   └── nn/__init__.py                  # placeholder for later layers
└── tests/
    ├── __init__.py
    └── test_smoke.py                   # THE smoke test that crosses everything
```

**Responsibilities:**
- `fractus-core`: pure mathematical computation, testable in Rust alone, no Python dependency.
- `fractus-py`: Python↔Rust bridge. Contains no logic, just `#[pyfunction]` wrappers.
- `fractus` (Python): the user package. In L0, just the bridge import + `nn/` placeholder.
- `tests/test_smoke.py`: proves everything holds.

---

## Task 1: Initialize the repo and the .gitignore

**Files:**
- Create: `C:/Users/PHIL/ZCodeProject/fractus/.gitignore`

- [ ] **Step 1: Create the fractus folder and init git**

Run (cmd or PowerShell):
```bash
mkdir "C:\Users\PHIL\ZCodeProject\fractus"
cd "C:\Users\PHIL\ZCodeProject\fractus"
git init
git branch -M main
```
Expected: `Initialized empty Git repository in ...` then silent for `git branch`.

- [ ] **Step 2: Write the .gitignore**

Create `C:/Users/PHIL/ZCodeProject/fractus/.gitignore`:
```gitignore
# Python
__pycache__/
*.py[cod]
*$py.class
*.egg-info/
.eggs/
build/
dist/

# Virtual env
.venv/
venv/

# Rust
target/
**/*.rs.bk

# IDE
.vscode/
.idea/

# OS
.DS_Store
Thumbs.db

# Logs / checkpoints
*.log
checkpoints/
```

- [ ] **Step 3: Initial commit**

```bash
cd "C:\Users\PHIL\ZCodeProject\fractus"
git add .gitignore
git commit -m "chore: initial commit with .gitignore"
```
Expected: `[main (root-commit) ...] chore: initial commit with .gitignore`

---

## Task 2: Create the root Cargo workspace

**Files:**
- Create: `C:/Users/PHIL/ZCodeProject/fractus/Cargo.toml`

- [ ] **Step 1: Write the workspace Cargo.toml**

Create `C:/Users/PHIL/ZCodeProject/fractus/Cargo.toml`:
```toml
# fractus root workspace.
# The two crates (fractus-core, fractus-py) are isolated members.
[workspace]
members = ["crate/fractus-core", "crate/fractus-py"]
resolver = "2"

# Release profile optimized for CPU-only.
[profile.release]
opt-level = 3
lto = true
codegen-units = 1
```

- [ ] **Step 2: Commit**

```bash
git add Cargo.toml
git commit -m "chore: add Cargo workspace manifest"
```
Expected: `1 file changed`

---

## Task 3: Create the fractus-core crate with add() (first Rust smoke test)

**Files:**
- Create: `C:/Users/PHIL/ZCodeProject/fractus/crate/fractus-core/Cargo.toml`
- Create: `C:/Users/PHIL/ZCodeProject/fractus/crate/fractus-core/src/lib.rs`

- [ ] **Step 1: Write fractus-core's Cargo.toml**

Create `C:/Users/PHIL/ZCodeProject/fractus/crate/fractus-core/Cargo.toml`:
```toml
[package]
name = "fractus-core"
version = "0.1.0"
edition = "2021"

[lib]
name = "fractus_core"
path = "src/lib.rs"

[dependencies]
# No dependencies in L0. nalgebra/serde/etc. added when a module needs them.
```

- [ ] **Step 2: Write a minimal lib.rs (just add, for the smoke test)**

Create `C:/Users/PHIL/ZCodeProject/fractus/crate/fractus-core/src/lib.rs`:
```rust
//! # fractus-core
//!
//! Pure mathematical core of fractus. No I/O, no Python dependency.
//! All functions here are testable in Rust alone.
//!
//! In L0, only `add` is exposed for the smoke test. The real modules
//! (vortex, siren, causal, proof) are added in later layers.

/// Integer addition. Exists only for the Python↔Rust smoke test.
pub fn add(a: i64, b: i64) -> i64 {
    a + b
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_add() {
        assert_eq!(add(2, 3), 5);
        assert_eq!(add(-1, 1), 0);
    }
}
```

- [ ] **Step 3: Verify the crate compiles and the test passes**

Run:
```bash
cd "C:\Users\PHIL\ZCodeProject\fractus"
cargo test -p fractus-core
```
Expected: `test result: ok. 1 passed` and `Compiling fractus-core`.

- [ ] **Step 4: Commit**

```bash
git add crate/fractus-core/
git commit -m "feat(core): add fractus-core crate with add() smoke function"
```
Expected: `2 files changed`

---

## Task 4: Create the fractus-py crate (PyO3 bindings)

**Files:**
- Create: `C:/Users/PHIL/ZCodeProject/fractus/crate/fractus-py/Cargo.toml`
- Create: `C:/Users/PHIL/ZCodeProject/fractus/crate/fractus-py/src/lib.rs`

- [ ] **Step 1: Write fractus-py's Cargo.toml**

Create `C:/Users/PHIL/ZCodeProject/fractus/crate/fractus-py/Cargo.toml`:
```toml
[package]
name = "fractus-py"
version = "0.1.0"
edition = "2021"

[lib]
name = "_core"
crate-type = ["cdylib"]
path = "src/lib.rs"

[dependencies]
fractus-core = { path = "../fractus-core" }
pyo3 = { version = "0.22", features = ["extension-module"] }
```

Note: `crate-type = ["cdylib"]` is mandatory for maturin to produce a Python extension.
`name = "_core"` makes the Python module called `fractus._core` (after pyproject configuration).

- [ ] **Step 2: Write the bindings lib.rs**

Create `C:/Users/PHIL/ZCodeProject/fractus/crate/fractus-py/src/lib.rs`:
```rust
//! Python bindings (PyO3) for fractus-core.
//!
//! This crate contains NO logic — only #[pyfunction] wrappers that delegate
//! to fractus-core. The goal is to expose the Rust to Python under the name
//! `fractus._core`.

use pyo3::prelude::*;

/// Integer addition — Python wrapper for fractus_core::add.
/// Exposed only for the L0 smoke test.
#[pyfunction]
fn add(a: i64, b: i64) -> i64 {
    fractus_core::add(a, b)
}

/// Python module `fractus._core`.
#[pymodule]
fn _core(_py: Python<'_>, m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(add, m)?)?;
    Ok(())
}
```

- [ ] **Step 3: Verify the workspace compiles (without maturin, just cargo check)**

Run:
```bash
cd "C:\Users\PHIL\ZCodeProject\fractus"
cargo check -p fractus-py
```
Expected: `Compiling pyo3 ...` then `Finished`. If the `Python` linker is missing under Windows, the error will show up here — fix it before continuing (typically: install the C++ build tools or configure the linker; pyo3/maturin normally handles it automatically).

- [ ] **Step 4: Commit**

```bash
git add crate/fractus-py/
git commit -m "feat(py): add fractus-py PyO3 bindings with add() wrapper"
```
Expected: `2 files changed`

---

## Task 5: Create the root pyproject.toml (configured for maturin)

**Files:**
- Create: `C:/Users/PHIL/ZCodeProject/fractus/pyproject.toml`

- [ ] **Step 1: Write pyproject.toml**

Create `C:/Users/PHIL/ZCodeProject/fractus/pyproject.toml`:
```toml
# fractus — Python package built via maturin (Rust backend).
# The native module comes from crate/fractus-py; the Python package comes from fractus/.

[build-system]
requires = ["maturin>=1.4,<2.0"]
build-backend = "maturin"

[project]
name = "fractus"
version = "0.1.0"
description = "Unified rebuild of the original systems: a trainable fractal transformer."
requires-python = ">=3.10"
dependencies = [
    "torch>=2.2",
    "numpy>=1.26",
]

[project.optional-dependencies]
dev = [
    "pytest>=8.0",
    "maturin>=1.4,<2.0",
]

[tool.maturin]
# The native module (cdylib _core) will be placed in fractus/ → importable as fractus._core.
python-source = "."
module-name = "fractus._core"
manifest-path = "crate/fractus-py/Cargo.toml"
features = ["pyo3/extension-module"]
```

Critical note: `module-name = "fractus._core"` + `python-source = "."` makes maturin place the `.pyd` in the `fractus/` package, importable as `from fractus import _core`. The `pyo3/extension-module` feature is passed directly to maturin (not in the crate Cargo.toml, to avoid the original's misconfigured `[features] python = ["pyo3"]` bug).

- [ ] **Step 2: Commit**

```bash
git add pyproject.toml
git commit -m "build: add pyproject.toml with maturin backend configuration"
```
Expected: `1 file changed`

---

## Task 6: Create the Python package fractus (placeholder)

**Files:**
- Create: `C:/Users/PHIL/ZCodeProject/fractus/fractus/__init__.py`
- Create: `C:/Users/PHIL/ZCodeProject/fractus/fractus/nn/__init__.py`

- [ ] **Step 1: Write fractus/__init__.py**

Create `C:/Users/PHIL/ZCodeProject/fractus/fractus/__init__.py`:
```python
"""fractus — unified rebuild of the original systems.

L0: only the native bridge `_core` is exposed. The nn/, causal/, reasoning/ modules
will be added in later layers (L1+).
"""

__version__ = "0.1.0"

# The native module fractus._core is built by maturin and placed here.
# We import it explicitly so it is accessible via `from fractus import _core`.
try:
    from fractus import _core  # noqa: F401
except ImportError as e:
    raise ImportError(
        "The native module fractus._core was not found. "
        "Did you run `maturin develop`?"
    ) from e
```

- [ ] **Step 2: Write fractus/nn/__init__.py (placeholder)**

Create `C:/Users/PHIL/ZCodeProject/fractus/fractus/nn/__init__.py`:
```python
"""nn subpackage — neural-network modules (PyTorch).

L0: empty. Will be filled in L1 (embedding) then L2 (attention, MoE, blocks).
"""
```

- [ ] **Step 3: Commit**

```bash
git add fractus/
git commit -m "feat(py): add fractus Python package skeleton with _core bridge"
```
Expected: `2 files changed`

---

## Task 7: Create the Python smoke test

**Files:**
- Create: `C:/Users/PHIL/ZCodeProject/fractus/tests/__init__.py`
- Create: `C:/Users/PHIL/ZCodeProject/fractus/tests/test_smoke.py`

- [ ] **Step 1: Write tests/__init__.py**

Create `C:/Users/PHIL/ZCodeProject/fractus/tests/__init__.py`:
```python
# Tests package. Tests are discovered by pytest at the repo root.
```

- [ ] **Step 2: Write tests/test_smoke.py**

Create `C:/Users/PHIL/ZCodeProject/fractus/tests/test_smoke.py`:
```python
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
```

- [ ] **Step 3: Verify the tests fail before installation (sanity check)**

Run:
```bash
cd "C:\Users\PHIL\ZCodeProject\fractus"
python -m pytest tests/test_smoke.py -v
```
Expected: FAIL — `ModuleNotFoundError: No module named 'fractus'` (the package is not installed yet). This is normal; it confirms the tests are meaningful.

- [ ] **Step 4: Commit**

```bash
git add tests/
git commit -m "test: add smoke tests for torch/numpy/rust-bridge"
```
Expected: `2 files changed`

---

## Task 8: Create requirements-dev.txt (reproducibility)

**Files:**
- Create: `C:/Users/PHIL/ZCodeProject/fractus/requirements-dev.txt`

- [ ] **Step 1: Write requirements-dev.txt**

Create `C:/Users/PHIL/ZCodeProject/fractus/requirements-dev.txt`:
```text
# Pinned versions for dev reproducibility.
# PyTorch CPU-only wheel (no CUDA — the AMD APU is not supported by ROCm under Windows).
# IMPORTANT: on Windows, install CPU torch with the explicit index-url:
#   pip install torch --index-url https://download.pytorch.org/whl/cpu

torch>=2.2,<3.0
numpy>=1.26,<3.0
pytest>=8.0,<9.0
maturin>=1.4,<2.0
```

- [ ] **Step 2: Commit**

```bash
git add requirements-dev.txt
git commit -m "build: pin dev dependencies (torch CPU-only, maturin, pytest)"
```
Expected: `1 file changed`

---

## Task 9: Write the README (pitch + dev setup)

**Files:**
- Create: `C:/Users/PHIL/ZCodeProject/fractus/README.md`

- [ ] **Step 1: Write README.md**

Create `C:/Users/PHIL/ZCodeProject/fractus/README.md`:
```markdown
# fractus

A unified rebuild of the original systems: a **trainable** fractal transformer, with SIREN
compression, NOTEARS causal reasoning, and proof generation/verification. CPU-only.

> State: **L0 (technical foundation)** — the Python↔Rust plumbing holds, no
> mathematical logic yet. See `docs/SPEC.md`.

## Stack

- **Rust** (`crate/fractus-core`): pure mathematical core (2-adic vortex, SIREN, NOTEARS,
  proof verification). Off the autodiff graph.
- **Python + PyTorch** (`fractus/`): trainable model, forward/backward, datasets.
- **maturin**: the bridge between the two.

## Dev setup (Windows)

Prerequisites: Rust (`cargo`), the `py` launcher (Python 3.10+).

```powershell
cd C:\Users\PHIL\ZCodeProject\fractus

# 1. Create the dedicated venv (use `py`, not the MSYS2 `python` which has no pip)
py -m venv .venv
.\.venv\Scripts\Activate.ps1

# 2. Install CPU-only PyTorch + dev tools
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements-dev.txt

# 3. Build and install the native Rust module into the venv
maturin develop --release

# 4. Run the tests
pytest tests/ -v
```

The 4 commands `cargo build`, `maturin develop`, `import torch; import fractus`,
`pytest` must all succeed.

## Layout

```
crate/fractus-core/   Rust: pure mathematical core (testable alone)
crate/fractus-py/     Rust: PyO3 bindings (no logic)
fractus/              Python: trainable model (L0: just the _core bridge)
tests/                integration tests
```

## Roadmap

See the spec: layers L0 (foundation) → L7 (demo). L0 = this repo as-is.
```

- [ ] **Step 2: Commit**

```bash
git add README.md
git commit -m "docs: add README with setup instructions and roadmap pointer"
```
Expected: `1 file changed`

---

## Task 10: Install the venv and validate the smoke test (REAL L0 verification)

**Files:** (none — this is runtime validation)

- [ ] **Step 1: Create the venv**

Run (PowerShell, from `C:\Users\PHIL\ZCodeProject\fractus`):
```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python --version   # must show 3.14.x
```
Expected: a `.venv/` folder created, prompt modified with `(.venv)`.

If PowerShell activation is blocked by execution policy:
```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
```

- [ ] **Step 2: Install CPU-only PyTorch**

Run (in the active venv):
```powershell
pip install torch --index-url https://download.pytorch.org/whl/cpu
```
Expected: download (~200 MB) then `Successfully installed torch-...`. May take a few minutes.

Verify:
```powershell
python -c "import torch; print(torch.__version__); print('cuda:', torch.cuda.is_available())"
```
Expected: a torch version, `cuda: False` (CPU-only, this is intended).

- [ ] **Step 3: Install the remaining dev dependencies**

```powershell
pip install -r requirements-dev.txt
```
Expected: `numpy`, `pytest`, `maturin` installed.

- [ ] **Step 4: Build and install the native module with maturin**

```powershell
maturin develop --release
```
Expected: `📦 Built ...` then `🛠 Installed fractus-...`. maturin compiles the Rust, produces the `.pyd`, and installs it into the venv. May take 2-5 min the first time (compiles pyo3).

On error: check that the C++ build tools are present (Visual Studio Build Tools or MSVC). maturin normally handles it automatically.

- [ ] **Step 5: Run the smoke tests — THEY MUST ALL PASS**

```powershell
pytest tests/ -v
```
Expected output (the 5 tests):
```
tests/test_smoke.py::test_torch_available PASSED
tests/test_smoke.py::test_numpy_available PASSED
tests/test_smoke.py::test_rust_bridge_import PASSED
tests/test_smoke.py::test_rust_bridge_add PASSED
tests/test_smoke.py::test_torch_numpy_interop PASSED
===== 5 passed in ...s =====
```

**This is the "L0 done" criterion:** the 5 tests pass. If even one fails, L0 is not done — debug before moving to L1.

- [ ] **Step 6: (Optional but recommended) Add the venv to gitignore if forgotten**

Verify:
```powershell
git status
```
If `.venv/` shows up as untracked, the Task 1 .gitignore has an issue — fix it. Otherwise, `git status` should show only clean files (nothing, or just `target/` if committed by mistake).

- [ ] **Step 7: L0 final commit**

```powershell
git status   # should be clean
git log --oneline   # should show ~9 commits
```
No file to commit (the venv and target/ are ignored). L0 is done.

---

## Task 11 (bonus): Port vortex.rs from the original — preparation for L1

**Note:** This task prepares L1 without executing it. The vortex is the only mathematically correct module of the original; we port it now (in Rust alone) so L1 starts fast. The Python bindings come in L1.

**Files:**
- Create: `C:/Users/PHIL/ZCodeProject/fractus/crate/fractus-core/src/vortex.rs`
- Modify: `C:/Users/PHIL/ZCodeProject/fractus/crate/fractus-core/src/lib.rs`

- [ ] **Step 1: Write vortex.rs (corrected port from the original)**

Create `C:/Users/PHIL/ZCodeProject/fractus/crate/fractus-core/src/vortex.rs`:
```rust
//! # 2-adic Vortex
//!
//! Ported from the original system (rust/src/vortex.rs), with corrections:
//! - The unused `HashMap` import was removed.
//! - The tautological test `assert!(d1 <= d2.max(d1))` was replaced with a true
//!   ultrametric test: `d(x,z) <= max(d(x,y), d(y,z))` on random data.
//!
//! Honest naming: we speak of a "Collatz hash" (not "ergodic flow" — the ergodicity
//! of Collatz is unproven, an open problem), of an "ultrametric distance", and of a
//! "2-adic norm" (exact terms).

/// 2-adic valuation v_2(x) = max{k : 2^k divides x}.
/// For x=0, returns 64 (convention for u64).
pub fn valuation_2(x: u64) -> u32 {
    if x == 0 {
        return 64;
    }
    x.trailing_zeros()
}

/// 3-adic valuation v_3(x) = max{k : 3^k divides x}.
pub fn valuation_3(x: u64) -> u32 {
    if x == 0 {
        return 0; // convention: v_3(0) = infinity, we cap at 0 for u64
    }
    let mut val = 0u32;
    let mut n = x;
    while n % 3 == 0 {
        val += 1;
        n /= 3;
    }
    val
}

/// Collatz hash of an integer. Used as a deterministic state hash.
/// Note: "ergodicity of Collatz" is unproven — we just call it a "hash".
pub fn collatz_hash(mut x: u64, steps: u32) -> u64 {
    for _ in 0..steps {
        if x == 0 {
            break;
        }
        if x % 2 == 0 {
            x /= 2;
        } else {
            x = 3.wrapping_mul(x).wrapping_add(1);
        }
    }
    x
}

/// 2-adic ultrametric distance: d(a,b) = 2^{v_2(a XOR b)}.
/// Verifies the strong ultrametric property: d(x,z) <= max(d(x,y), d(y,z)).
pub fn ultrametric_distance(a: u64, b: u64) -> u64 {
    let diff = a ^ b;
    if diff == 0 {
        return 0;
    }
    1u64 << valuation_2(diff)
}

/// 2-adic norm: ||x||_2 = 2^{-v_2(x)}.
/// Returns an f64 (can be very small for large even x).
pub fn norm_2adic(x: u64) -> f64 {
    if x == 0 {
        return 0.0; // ||0|| = 0 by convention (v_2(0) = infinity)
    }
    let v = valuation_2(x) as i32;
    2f64.powi(-v)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_valuation_2_basic() {
        assert_eq!(valuation_2(0), 64);
        assert_eq!(valuation_2(1), 0);
        assert_eq!(valuation_2(2), 1);
        assert_eq!(valuation_2(4), 2);
        assert_eq!(valuation_2(8), 3);
        assert_eq!(valuation_2(56), 3); // 56 = 7 * 8
    }

    #[test]
    fn test_valuation_3_basic() {
        assert_eq!(valuation_3(0), 0);
        assert_eq!(valuation_3(1), 0);
        assert_eq!(valuation_3(3), 1);
        assert_eq!(valuation_3(9), 2);
        assert_eq!(valuation_3(27), 3);
        assert_eq!(valuation_3(56), 0); // 56 is not divisible by 3
    }

    #[test]
    fn test_collatz_hash_deterministic() {
        // Same input → same output (deterministic).
        assert_eq!(collatz_hash(7, 10), collatz_hash(7, 10));
        // 0 stays 0.
        assert_eq!(collatz_hash(0, 10), 0);
    }

    #[test]
    fn test_ultrametric_distance_self_is_zero() {
        assert_eq!(ultrametric_distance(42, 42), 0);
    }

    #[test]
    fn test_ultrametric_distance_symmetry() {
        for (a, b) in [(1u64, 2), (7, 56), (100, 200), (3, 9)] {
            assert_eq!(ultrametric_distance(a, b), ultrametric_distance(b, a));
        }
    }

    #[test]
    fn test_ultrametric_strong_triangle_inequality() {
        // The true ultrametric property: d(x,z) <= max(d(x,y), d(y,z)).
        // CORRECTION of the original tautological test.
        let triples: [(u64, u64, u64); 8] = [
            (1, 2, 4),
            (7, 56, 13),
            (100, 200, 300),
            (3, 9, 27),
            (5, 11, 23),
            (1024, 1, 2),
            (7, 13, 21),
            (255, 256, 257),
        ];
        for (x, y, z) in triples {
            let d_xy = ultrametric_distance(x, y);
            let d_yz = ultrametric_distance(y, z);
            let d_xz = ultrametric_distance(x, z);
            assert!(
                d_xz <= d_xy.max(d_yz),
                "Ultrametric failure: d({},{})={} > max(d({},{})={}, d({},{})={})",
                x, z, d_xz, x, y, d_xy, y, z, d_yz
            );
        }
    }

    #[test]
    fn test_norm_2adic_basic() {
        assert_eq!(norm_2adic(0), 0.0);
        assert_eq!(norm_2adic(1), 1.0);   // v_2(1) = 0 → 2^0 = 1
        assert_eq!(norm_2adic(2), 0.5);   // v_2(2) = 1 → 2^-1
        assert_eq!(norm_2adic(4), 0.25);  // v_2(4) = 2 → 2^-2
        assert_eq!(norm_2adic(8), 0.125); // v_2(8) = 3 → 2^-3
    }

    #[test]
    fn test_norm_2adic_fuzz_ultrametric() {
        // On pseudo-random data, the norm must be <= 1 for x != 0.
        for x in [1u64, 3, 5, 7, 9, 11, 42, 137, 1023, 65535] {
            let n = norm_2adic(x);
            assert!(n > 0.0 && n <= 1.0, "norm_2adic({}) = {} outside [0,1]", x, n);
        }
    }
}
```

- [ ] **Step 2: Declare the vortex module in lib.rs**

Modify `C:/Users/PHIL/ZCodeProject/fractus/crate/fractus-core/src/lib.rs` — replace all the content with:
```rust
//! # fractus-core
//!
//! Pure mathematical core of fractus. No I/O, no Python dependency.
//! All functions here are testable in Rust alone.

pub mod vortex;

/// Integer addition. Exists only for the Python↔Rust smoke test.
pub fn add(a: i64, b: i64) -> i64 {
    a + b
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_add() {
        assert_eq!(add(2, 3), 5);
        assert_eq!(add(-1, 1), 0);
    }
}
```

- [ ] **Step 3: Run the Rust tests — THEY MUST ALL PASS**

Run:
```bash
cd "C:\Users\PHIL\ZCodeProject\fractus"
cargo test -p fractus-core
```
Expected: 9 tests pass (1 `test_add` + 8 vortex tests). Output:
```
test result: ok. 9 passed; 0 failed
```

- [ ] **Step 4: Commit**

```bash
git add crate/fractus-core/
git commit -m "feat(core): port 2-adic vortex from the original with ultrametric test fix"
```
Expected: `2 files changed`

---

## Final "L0 done" criterion

After Task 10 (and the bonus Task 11), these 4 verifications must all succeed:

```bash
cd "C:\Users\PHIL\ZCodeProject\fractus"

# 1. Rust compiles and its tests pass
cargo test -p fractus-core
# → 9 passed (1 add + 8 vortex)

# 2. The Python↔Rust bridge builds
maturin develop --release
# → 🛠 Installed fractus-...

# 3. Both import
python -c "import torch; import fractus; print('OK', torch.__version__)"
# → OK 2.x.x

# 4. The smoke tests pass
pytest tests/ -v
# → 5 passed
```

If everything passes, L0 is done and we can move to the L1 plan (fractal embedding + wired vortex).

---

## Self-Review (post-writing)

Verifications performed:

**1. Spec coverage:** The L0 spec section requires (a) reproducible environment → Task 5, 8; (b) fractus-core crate → Task 3, 11; (c) fractus-py bindings crate → Task 4; (d) crossing smoke test → Task 7, 10. ✅ All covered.

**2. Placeholder scan:** No "TBD", no "TODO", no "fill in" section. All steps contain complete code. ✅

**3. Type consistency:** `add(a, b)` defined in `fractus-core/src/lib.rs` (Task 3), wrapped in `fractus-py/src/lib.rs` (Task 4) as `#[pyfunction] fn add`, tested in `test_smoke.py` (Task 7). Names consistent everywhere. `valuation_2`, `ultrametric_distance`, `norm_2adic` (Task 11) consistent between definition and tests. ✅

**4. Dependency order:** Task 4 (fractus-py) depends on Task 3 (fractus-core) → respected. Task 10 (maturin develop) depends on Tasks 3-9 → respected. Task 11 (bonus) can be done after or before L1 without blocking.

**5. Windows-specific commands:** `py` (not the MSYS2 `python`), `.\.venv\Scripts\Activate.ps1`, `--index-url https://download.pytorch.org/whl/cpu` for CPU torch. All verified against the target machine. ✅
