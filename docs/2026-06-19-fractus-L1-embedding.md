# Fractus L1 — Fractal embedding + 2-adic vortex wiring Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Wire the 2-adic vortex from the original system into the neural network for the first time, using it as **conditioning** for a trainable PyTorch MLP. Add the fractal codepoint embedding (Fourier basis with Mandelbrot decay + 16 morphological features), all in pure PyTorch for autodiff. Fix two original system errors: the orphaned vortex (never imported) and the misnamed "Mandelbrot frequencies".

**Architecture:** (1) `fractus-core` (Rust) exposes `collatz_hash(token_id)` and `ultrametric_distance(a,b)` — exact computation, outside the autodiff graph, called from Python to precompute the conditioning. (2) `fractus/nn/embedding.py` (PyTorch) contains three composed modules: `CharClassFeatures` (16 deterministic morphological features, ported from the original), `MandelbrotFourierBasis` (Fourier basis with `(φ2)^{-k}` decay, honestly named), and `FractalEmbedding` (final assembly: the morpho features + the Fourier basis + the vortex-conditioned phases are projected to `d_model` by a trainable `nn.Linear`). The forward pass is differentiable end-to-end.

**Tech Stack:** Rust 1.94 + pyo3 0.29 (already installed in L0); Python 3.14 + torch 2.12 CPU + numpy (already installed in L0); pytest. The native module must be rebuilt via `maturin develop` after the Rust modification.

**Spec link:** `docs/SPEC.md`, section "L1 — Fractal embedding + 2-adic vortex wiring". Validated key decision: **option B** — the exact 2-adic hash (Rust, off-graph) conditions a trainable MLP (PyTorch, in the graph).

**Prerequisites:** L0 done (repo `C:/Users/PHIL/ZCodeProject/fractus/` operational, venv `.venv` with torch + maturin installed, 9 Rust tests + 5 Python tests pass).

**Honest vocabulary applied in this plan:**
- "Mandelbrot frequencies" → "Mandelbrot-decayed Fourier basis" (the `(φ2)^{-k}` decay is real and justified, but this is not the Mandelbrot set).
- "Collatz ergodic flow" → "Collatz hash" (the ergodicity of Collatz is unproven, an open problem).
- We speak of a "2-adic norm" and "ultrametric distance" (exact mathematical terms).

---

## File Structure

```
C:/Users/PHIL/ZCodeProject/fractus/
├── crate/fractus-core/src/
│   ├── lib.rs                  # unchanged (already declares pub mod vortex)
│   └── vortex.rs               # MODIFY: also expose norm_2adic (already defined, just bind it)
├── crate/fractus-py/src/
│   └── lib.rs                  # MODIFY: add #[pyfunction] wrappers for collatz_hash,
│                               #   ultrametric_distance, norm_2adic
├── fractus/nn/
│   ├── __init__.py             # MODIFY: export the public classes
│   ├── char_features.py        # CREATE: 16 morphological features (ported from the original)
│   ├── fourier.py              # CREATE: Mandelbrot-decayed Fourier basis
│   └── embedding.py            # CREATE: FractalEmbedding (assembly + Linear projection)
└── tests/
    ├── test_vortex_bridge.py   # CREATE: tests for the Python bridge of the 2-adic functions
    └── test_embedding.py       # CREATE: tests for the fractal embedding
```

**Responsibilities (one file = one responsibility):**
- `char_features.py`: only the 16 morphological features (deterministic, no parameters).
- `fourier.py`: only the Mandelbrot-decayed Fourier basis (deterministic, no parameters).
- `embedding.py`: the assembly + the trainable projection (the only place with `nn.Parameter`).
- `test_vortex_bridge.py`: verifies the Rust functions are callable from Python and return correct values (bridge operational).
- `test_embedding.py`: verifies shapes, finiteness, and above all that `backward()` propagates finite gradients (the critical criterion inherited from the original, which failed here).

---

## Task 1: Expose collatz_hash, ultrametric_distance, norm_2adic in the Python bindings

**Files:**
- Modify: `C:/Users/PHIL/ZCodeProject/fractus/crate/fractus-py/src/lib.rs`

- [ ] **Step 1: Rewrite lib.rs with the new wrappers**

Replace the entire content of `C:/Users/PHIL/ZCodeProject/fractus/crate/fractus-py/src/lib.rs` with:

```rust
//! Python bindings (PyO3) for fractus-core.
//!
//! This crate contains NO logic — only #[pyfunction] wrappers that delegate
//! to fractus-core. The goal is to expose the Rust to Python under the name
//! `fractus._core`.

use pyo3::prelude::*;

/// Integer addition — Python wrapper for fractus_core::add.
/// Exposed only for the smoke test.
#[pyfunction]
fn add(a: i64, b: i64) -> i64 {
    fractus_core::add(a, b)
}

/// Collatz hash of a token id. Wrapper for fractus_core::vortex::collatz_hash.
/// Used as deterministic conditioning (outside the autodiff graph) for the
/// fractal embedding (spec L1, option B).
#[pyfunction]
fn collatz_hash(x: u64, steps: u32) -> u64 {
    fractus_core::vortex::collatz_hash(x, steps)
}

/// 2-adic ultrametric distance: d(a,b) = 2^{-v_2(a ⊕ b)}.
/// Wrapper for fractus_core::vortex::ultrametric_distance. Lies in (0, 1].
#[pyfunction]
fn ultrametric_distance(a: u64, b: u64) -> f64 {
    fractus_core::vortex::ultrametric_distance(a, b)
}

/// 2-adic norm: ||x||_2 = 2^{-v_2(x)}. Wrapper for fractus_core::vortex::norm_2adic.
#[pyfunction]
fn norm_2adic(x: u64) -> f64 {
    fractus_core::vortex::norm_2adic(x)
}

/// Python module `fractus._core`.
///
/// pyo3 0.29 signature: the module is received as `&Bound<'_, PyModule>`.
/// The `.add_function(...)` methods come from the `PyModuleMethods` trait
/// (re-exported by `pyo3::prelude`).
#[pymodule]
fn _core(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(add, m)?)?;
    m.add_function(wrap_pyfunction!(collatz_hash, m)?)?;
    m.add_function(wrap_pyfunction!(ultrametric_distance, m)?)?;
    m.add_function(wrap_pyfunction!(norm_2adic, m)?)?;
    Ok(())
}
```

- [ ] **Step 2: Rebuild the native module**

Run (from `C:/Users/PHIL/ZCodeProject/fractus`, active venv):
```powershell
.venv\Scripts\python.exe -m maturin develop --release
```
Expected: `🛠 Installed fractus-0.1.0`. May take 30-60s (recompiles pyo3 if the hash changed).

- [ ] **Step 3: Quick check that the new functions are exposed**

```powershell
.venv\Scripts\python.exe -c "from fractus import _core; print('hash(7,5)=', _core.collatz_hash(7,5)); print('dist(1,2)=', _core.ultrametric_distance(1,2)); print('norm(8)=', _core.norm_2adic(8))"
```
Expected: three numeric values without error (e.g. `hash(7,5)= 52`, `dist(1,2)= 0.5`, `norm(8)= 0.125`).

- [ ] **Step 4: Commit**

```bash
cd "C:\Users\PHIL\ZCodeProject\fractus"
git add crate/fractus-py/src/lib.rs
git commit -m "feat(py): expose collatz_hash, ultrametric_distance, norm_2adic to Python"
```
Expected: `1 file changed`.

---

## Task 2: Python bridge tests for the 2-adic vortex functions (TDD)

**Files:**
- Create: `C:/Users/PHIL/ZCodeProject/fractus/tests/test_vortex_bridge.py`

- [ ] **Step 1: Write the tests**

Create `C:/Users/PHIL/ZCodeProject/fractus/tests/test_vortex_bridge.py`:
```python
"""Python bridge tests for the 2-adic vortex functions.

Verifies that the Rust wrappers are properly exposed and return correct
values. These tests do NOT do advanced math (that's in Rust) — they only
validate the PyO3 bridge.
"""

import pytest


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
```

- [ ] **Step 2: Run the tests — THEY MUST ALL PASS**

```powershell
.venv\Scripts\python.exe -m pytest tests/test_vortex_bridge.py -v
```
Expected: 8 passed. If a test fails, the PyO3 bridge is broken — check that Task 1 rebuilt the module.

- [ ] **Step 3: Commit**

```bash
git add tests/test_vortex_bridge.py
git commit -m "test(vortex): 8 Python bridge tests for collatz_hash/ultrametric/norm"
```
Expected: `1 file changed`.

---

## Task 3: Implement the 16 morphological features (CharClassFeatures)

**Files:**
- Create: `C:/Users/PHIL/ZCodeProject/fractus/fractus/nn/char_features.py`

- [ ] **Step 1: Write the failing test**

Create `C:/Users/PHIL/ZCodeProject/fractus/tests/test_embedding.py` (initial content — will be extended in later tasks):
```python
"""Tests of the fractal embedding: char features, Fourier basis, FractalEmbedding.

The critical criterion (inherited from the original system, which failed here): the
forward pass must be differentiable and backward() must propagate finite gradients
everywhere.
"""

import torch
import pytest


# ---------------------------------------------------------------------------
# Task 3: CharClassFeatures (16 morphological features)
# ---------------------------------------------------------------------------

def test_char_features_shape():
    """16 features for every token id."""
    from fractus.nn.char_features import CharClassFeatures
    f = CharClassFeatures.extract(ord("a"))
    assert f.shape == (16,)


def test_char_features_vowel():
    """'a' is a vowel (feature 0 = 1)."""
    from fractus.nn.char_features import CharClassFeatures
    f = CharClassFeatures.extract(ord("a"))
    assert f[0].item() == 1.0  # is_vowel


def test_char_features_digit_value():
    """'5' is a digit of value 5 (feature 11)."""
    from fractus.nn.char_features import CharClassFeatures
    f = CharClassFeatures.extract(ord("5"))
    assert f[2].item() == 1.0   # is_digit
    assert f[11].item() == 5.0  # digit_value


def test_char_features_batch_consistency():
    """The same letter yields the same feature vector."""
    from fractus.nn.char_features import CharClassFeatures
    f1 = CharClassFeatures.extract(ord("z"))
    f2 = CharClassFeatures.extract(ord("z"))
    assert torch.equal(f1, f2)
```

- [ ] **Step 2: Run the test to verify it fails**

```powershell
.venv\Scripts\python.exe -m pytest tests/test_embedding.py -v
```
Expected: FAIL — `ModuleNotFoundError: No module named 'fractus.nn.char_features'`. This is normal.

- [ ] **Step 3: Implement CharClassFeatures**

Create `C:/Users/PHIL/ZCodeProject/fractus/fractus/nn/char_features.py`:
```python
"""16 deterministic morphological features per token.

Ported from the original system (src/embedding.rs, CharClassFeatures). The token id is
interpreted as a Unicode codepoint; for ids < 128 these are ASCII characters, and beyond
that we derive features from the numeric value.

These features have NO trainable parameters — they are computed deterministically, then
concatenated with the Fourier basis in FractalEmbedding.
"""

import torch


class CharClassFeatures:
    """Extraction of 16 morphological features from a token id.

    Features (index: meaning):
        0  : is_vowel          (a, e, i, o, u)
        1  : is_consonant      (non-vowel letter)
        2  : is_digit          (0-9)
        3  : is_space          (0x20)
        4  : is_uppercase
        5  : is_lowercase
        6  : is_punctuation    (!"#$%...)
        7  : is_alphabetic
        8  : is_numeric        (alias of is_digit here)
        9  : is_whitespace     (space, tab, newline)
        10 : is_control        (codepoint < 32 or == 127)
        11 : digit_value       (0-9, or 0 if not a digit)
        12 : char_category     (simplified Unicode category as float)
        13 : position_in_alphabet (0-25, or -1 if not a letter; we encode -1→0)
        14 : is_ascii          (codepoint < 128)
        15 : parity            (even token id = 1, odd = 0)
    """

    N_FEATURES = 16

    VOWELS = frozenset(b"aeiouAEIOU")

    @staticmethod
    def extract(token_id: int) -> torch.Tensor:
        """Returns a float32 tensor of shape (16,)."""
        f = torch.zeros(CharClassFeatures.N_FEATURES, dtype=torch.float32)

        # We interpret the low byte as a potential character.
        as_byte = (token_id & 0xFF)

        # 0: is_vowel
        is_vowel_bool = as_byte in CharClassFeatures.VOWELS
        f[0] = float(is_vowel_bool)

        # 1: is_consonant (alphabetic non-vowel letter)
        is_alpha = (
            (0x41 <= as_byte <= 0x5A) or  # A-Z
            (0x61 <= as_byte <= 0x7A)     # a-z
        )
        f[1] = float(is_alpha and not is_vowel_bool)

        # 2: is_digit
        is_digit = 0x30 <= as_byte <= 0x39
        f[2] = float(is_digit)

        # 3: is_space (0x20)
        f[3] = float(as_byte == 0x20)

        # 4: is_uppercase
        f[4] = float(0x41 <= as_byte <= 0x5A)

        # 5: is_lowercase
        f[5] = float(0x61 <= as_byte <= 0x7A)

        # 6: is_punctuation (ASCII punctuation ranges)
        is_punct = (
            (0x21 <= as_byte <= 0x2F) or
            (0x3A <= as_byte <= 0x40) or
            (0x5B <= as_byte <= 0x60) or
            (0x7B <= as_byte <= 0x7E)
        )
        f[6] = float(is_punct)

        # 7: is_alphabetic
        f[7] = float(is_alpha)

        # 8: is_numeric (alias of is_digit here)
        f[8] = float(is_digit)

        # 9: is_whitespace (space, tab 0x09, newline 0x0A, CR 0x0D)
        f[9] = float(as_byte in (0x09, 0x0A, 0x0D, 0x20))

        # 10: is_control (codepoint < 32 or == 127)
        f[10] = float(as_byte < 0x20 or as_byte == 0x7F)

        # 11: digit_value
        f[11] = float(as_byte - 0x30) if is_digit else 0.0

        # 12: char_category simplified: 1.0 letter, 2.0 digit, 3.0 punctuation,
        #     4.0 space, 5.0 control, 0.0 other.
        if is_alpha:
            f[12] = 1.0
        elif is_digit:
            f[12] = 2.0
        elif is_punct:
            f[12] = 3.0
        elif f[9] > 0.5:
            f[12] = 4.0
        elif f[10] > 0.5:
            f[12] = 5.0

        # 13: position_in_alphabet (0-25, or 0 if not a letter)
        if 0x41 <= as_byte <= 0x5A:
            f[13] = float(as_byte - 0x41)
        elif 0x61 <= as_byte <= 0x7A:
            f[13] = float(as_byte - 0x61)

        # 14: is_ascii
        f[14] = float(token_id < 128)

        # 15: parity (even token id)
        f[15] = float((token_id % 2) == 0)

        return f
```

- [ ] **Step 4: Run the tests — THEY MUST PASS**

```powershell
.venv\Scripts\python.exe -m pytest tests/test_embedding.py -v
```
Expected: 4 passed (the 4 char-features tests).

- [ ] **Step 5: Commit**

```bash
git add fractus/nn/char_features.py tests/test_embedding.py
git commit -m "feat(nn): port CharClassFeatures (16 morphological features) from the original"
```
Expected: `2 files changed`.

---

## Task 4: Implement the Mandelbrot-decayed Fourier basis

**Files:**
- Create: `C:/Users/PHIL/ZCodeProject/fractus/fractus/nn/fourier.py`

- [ ] **Step 1: Add the tests to tests/test_embedding.py**

Append to `C:/Users/PHIL/ZCodeProject/fractus/tests/test_embedding.py` (after the existing char-features tests):
```python


# ---------------------------------------------------------------------------
# Task 4: MandelbrotFourierBasis (Fourier basis with (φ2)^{-k} decay)
# ---------------------------------------------------------------------------

def test_fourier_basis_shape():
    """For vocab 128 and 32 frequencies: matrix (vocab, 2·n_freq) (sin+cos)."""
    from fractus.nn.fourier import MandelbrotFourierBasis
    basis = MandelbrotFourierBasis(vocab_size=128, n_frequencies=32)
    M = basis.matrix()  # (vocab, 2*n_freq)
    assert M.shape == (128, 64)  # 2*32 columns (sin+cos)


def test_fourier_basis_is_finite():
    from fractus.nn.fourier import MandelbrotFourierBasis
    basis = MandelbrotFourierBasis(vocab_size=128, n_frequencies=16)
    M = basis.matrix()
    assert torch.isfinite(M).all()


def test_fourier_frequencies_decay():
    """The frequencies ω_k = (φ2)^{-k} must decay geometrically."""
    from fractus.nn.fourier import MandelbrotFourierBasis
    basis = MandelbrotFourierBasis(vocab_size=10, n_frequencies=4)
    # ω_0 = 1.0, ω_1 = 1/φ2, ω_2 = 1/φ4, ...
    phi_sq = ((1 + 5 ** 0.5) / 2) ** 2
    expected = [phi_sq ** (-k) for k in range(4)]
    for k, exp in enumerate(expected):
        assert abs(basis.frequencies[k].item() - exp) < 1e-5, \
            f"freq[{k}] = {basis.frequencies[k].item()}, expected {exp}"


def test_fourier_matrix_is_deterministic():
    """Two calls yield the same matrix (no randomness)."""
    from fractus.nn.fourier import MandelbrotFourierBasis
    b1 = MandelbrotFourierBasis(vocab_size=64, n_frequencies=16)
    b2 = MandelbrotFourierBasis(vocab_size=64, n_frequencies=16)
    assert torch.allclose(b1.matrix(), b2.matrix())
```

- [ ] **Step 2: Run to verify the new tests fail**

```powershell
.venv\Scripts\python.exe -m pytest tests/test_embedding.py -v
```
Expected: 4 passed (char features), 4 failed/error (fourier — module absent).

- [ ] **Step 3: Implement MandelbrotFourierBasis**

Create `C:/Users/PHIL/ZCodeProject/fractus/fractus/nn/fourier.py`:
```python
"""Fourier basis with Mandelbrot decay for the fractal embedding.

Inspired by the original system (src/math/mandelbrot.rs + src/embedding.rs) but renamed
honestly: the original called these "Mandelbrot frequencies" in reference to the Mandelbrot
set, but it is really just a geometric decay of base φ2 (the square of the golden ratio).
We therefore call it the "Mandelbrot-decayed Fourier basis" — the decay is real and
justified (multi-level scale separation), but the link to the Mandelbrot set is nil.

Mathematics:
    φ = (1 + √5) / 2  ≈ 1.618
    φ2 ≈ 2.618
    ω_k = (φ2)^{-k}    for k = 0, 1, ..., n_freq-1

The Fourier basis associates with each token id t and each frequency k the pair
(sin, cos) of ω_k · t :
    M[t, 2k]   = sin(ω_k · t)
    M[t, 2k+1] = cos(ω_k · t)

We store n_freq frequencies; the produced matrix has 2·n_freq columns (sin+cos per
frequency). The caller (FractalEmbedding) handles the final projection.

NO trainable parameters here: everything is deterministic, computed once.
"""

import math
import torch


class MandelbrotFourierBasis:
    """Deterministic Fourier basis with (φ2)^{-k} decay.

    Attributes:
        vocab_size   : number of covered token ids (0 .. vocab_size-1)
        n_frequencies : number of frequencies ω_k
        frequencies  : tensor (n_frequencies,) of ω_k, in float32
    """

    def __init__(self, vocab_size: int, n_frequencies: int):
        if vocab_size <= 0 or n_frequencies <= 0:
            raise ValueError("vocab_size and n_frequencies must be > 0")
        self.vocab_size = vocab_size
        self.n_frequencies = n_frequencies

        phi = (1.0 + math.sqrt(5.0)) / 2.0
        phi_sq = phi * phi  # ≈ 2.618
        ks = torch.arange(n_frequencies, dtype=torch.float32)
        # ω_k = (φ2)^{-k}
        self.frequencies = phi_sq ** (-ks)

        # Precompute the matrix (vocab_size, 2·n_frequencies).
        self._matrix = self._build_matrix()

    def _build_matrix(self) -> torch.Tensor:
        """Builds the matrix M[t, :] = [sin(ω_k·t), cos(ω_k·t)] for all k."""
        t = torch.arange(self.vocab_size, dtype=torch.float32).unsqueeze(1)  # (V, 1)
        omega = self.frequencies.unsqueeze(0)  # (1, K)
        phases = omega * t  # (V, K) broadcast
        sin_part = torch.sin(phases)  # (V, K)
        cos_part = torch.cos(phases)  # (V, K)
        # Interleave sin/cos: columns 0,2,4,... = sin; 1,3,5,... = cos
        M = torch.empty(self.vocab_size, 2 * self.n_frequencies, dtype=torch.float32)
        M[:, 0::2] = sin_part
        M[:, 1::2] = cos_part
        return M

    def matrix(self) -> torch.Tensor:
        """Returns the precomputed matrix (vocab_size, 2·n_frequencies)."""
        return self._matrix
```

- [ ] **Step 4: Run all the tests — THEY MUST PASS**

```powershell
.venv\Scripts\python.exe -m pytest tests/test_embedding.py -v
```
Expected: 8 passed (4 char features + 4 fourier).

- [ ] **Step 5: Commit**

```bash
git add fractus/nn/fourier.py tests/test_embedding.py
git commit -m "feat(nn): add MandelbrotFourierBasis (honestly named Fourier basis)"
```
Expected: `2 files changed`.

---

## Task 5: Implement FractalEmbedding (assembly + trainable projection)

**Files:**
- Create: `C:/Users/PHIL/ZCodeProject/fractus/fractus/nn/embedding.py`
- Modify: `C:/Users/PHIL/ZCodeProject/fractus/fractus/nn/__init__.py`

- [ ] **Step 1: Add the FractalEmbedding tests**

Append to `C:/Users/PHIL/ZCodeProject/fractus/tests/test_embedding.py`:
```python


# ---------------------------------------------------------------------------
# Task 5: FractalEmbedding (assembly + trainable projection)
# ---------------------------------------------------------------------------

def test_fractal_embedding_shape():
    """Output (N, d_model) for input (N,) of ids."""
    from fractus.nn.embedding import FractalEmbedding
    emb = FractalEmbedding(vocab_size=128, d_model=64, n_frequencies=16)
    ids = torch.tensor([0, 1, 2, 65, 97])  # mix
    out = emb(ids)
    assert out.shape == (5, 64)


def test_fractal_embedding_is_finite():
    from fractus.nn.embedding import FractalEmbedding
    emb = FractalEmbedding(vocab_size=128, d_model=64, n_frequencies=16)
    ids = torch.arange(128)
    out = emb(ids)
    assert torch.isfinite(out).all()


def test_fractal_embedding_backward_propagates():
    """CRITICAL: backward() must propagate a finite AND non-zero gradient to EVERY parameter.

    This is exactly the test that the original system failed (training.rs:399 used random
    noise instead of a gradient). Here, the Linear projection is in the autodiff graph,
    so the gradients must be non-zero and finite.

    We check EVERY parameter individually (not just "at least one"), because a dead
    parameter (zero gradient) in the vortex MLP for instance would indicate a silently
    broken autodiff — exactly the defect this rebuild must eliminate.
    """
    from fractus.nn.embedding import FractalEmbedding
    emb = FractalEmbedding(vocab_size=128, d_model=64, n_frequencies=16)
    ids = torch.tensor([0, 1, 2, 3, 4])
    out = emb(ids)
    loss = out.pow(2).sum()
    loss.backward()

    params = list(emb.named_parameters())
    assert len(params) > 0, "The model has no trainable parameter"
    for name, p in params:
        assert p.requires_grad, f"{name} should requires_grad=True"
        assert p.grad is not None, f"{name} received no gradient (dead parameter)"
        assert torch.isfinite(p.grad).all(), f"{name} has a non-finite gradient (NaN/Inf)"
        grad_l1 = p.grad.abs().sum().item()
        assert grad_l1 > 0, (
            f"{name} received a zero gradient — autodiff does not propagate "
            f"to this parameter (grad L1 = {grad_l1})"
        )


def test_fractal_embedding_respects_vocab_bounds():
    """An id >= vocab_size must raise an error (no silent crash)."""
    from fractus.nn.embedding import FractalEmbedding
    emb = FractalEmbedding(vocab_size=100, d_model=32, n_frequencies=8)
    with pytest.raises(IndexError):
        emb(torch.tensor([100]))  # out of bounds
```

- [ ] **Step 2: Run to verify the FractalEmbedding tests fail**

```powershell
.venv\Scripts\python.exe -m pytest tests/test_embedding.py -v
```
Expected: 8 passed (char + fourier), 4 failed/error (FractalEmbedding absent).

- [ ] **Step 3: Implement FractalEmbedding**

Create `C:/Users/PHIL/ZCodeProject/fractus/fractus/nn/embedding.py`:
```python
"""FractalEmbedding: a trainable fractal codepoint embedding.

Combines three feature sources for each token id t:

    (A) 16 deterministic morphological features (CharClassFeatures)
    (B) Mandelbrot-decayed Fourier basis (MandelbrotFourierBasis)
    (C) Vortex conditioning: a 2-adic hash (Collatz, computed in Rust,
        outside the autodiff graph) is projected into phases via a trainable
        MLP (PyTorch, in the graph). The 2-adic vortex influences learning
        without pretending to be differentiable.

The final projection to d_model is a trainable nn.Linear. The entire forward
pass is differentiable end-to-end. The deterministic parts (A, B, and the
hash of C) are precomputed as buffers outside the graph; only the MLP of C
and the final projection carry trainable parameters.

Corrections vs the original systems:
- The original did not learn (training.rs:399 = noise) → here backward() works (test).
- The original: the 2-adic vortex was orphaned (never imported by Python) →
  here it genuinely conditions the embedding.
- The original: the "Mandelbrot frequencies" were misnamed → here we say
  "Mandelbrot-decayed Fourier basis" (see fourier.py).
"""

import torch
import torch.nn as nn

from .char_features import CharClassFeatures
from .fourier import MandelbrotFourierBasis


class FractalEmbedding(nn.Module):
    """Trainable fractal embedding.

    Args:
        vocab_size:    number of token ids covered.
        d_model:       output dimension.
        n_frequencies: number of frequencies for the Fourier basis.
        vortex_hidden: width of the MLP that projects the Collatz hash into phases.
        collatz_steps: number of Collatz iterations for the hash (deterministic).
    """

    def __init__(
        self,
        vocab_size: int,
        d_model: int,
        n_frequencies: int = 16,
        vortex_hidden: int = 32,
        collatz_steps: int = 7,
    ):
        super().__init__()
        if vocab_size <= 0 or d_model <= 0:
            raise ValueError("vocab_size and d_model must be > 0")

        self.vocab_size = vocab_size
        self.d_model = d_model
        self.collatz_steps = collatz_steps

        # (A) Morphological features: deterministic precomputation, outside graph.
        char_matrix = torch.stack(
            [CharClassFeatures.extract(t) for t in range(vocab_size)], dim=0
        )
        self.register_buffer("char_features", char_matrix)

        # (B) Mandelbrot-decayed Fourier basis: deterministic precomputation.
        self.fourier = MandelbrotFourierBasis(vocab_size, n_frequencies)
        fourier_matrix = self.fourier.matrix()
        self.register_buffer("fourier_features", fourier_matrix)

        # (C) Vortex conditioning: Collatz hash precomputed (outside graph),
        # then projected by a trainable MLP (in the graph).
        try:
            from fractus import _core
        except ImportError as e:
            raise ImportError(
                "fractus._core not found. Run `maturin develop`."
            ) from e
        hashes = torch.tensor(
            [_core.collatz_hash(t, collatz_steps) for t in range(vocab_size)],
            dtype=torch.float32,
        )
        max_h = hashes.max().item() + 1.0
        hashes_norm = hashes / max_h
        self.register_buffer("vortex_hashes", hashes_norm)

        self.vortex_phase_dim = vortex_hidden
        self.vortex_mlp = nn.Sequential(
            nn.Linear(1, vortex_hidden),
            nn.Tanh(),
            nn.Linear(vortex_hidden, vortex_hidden),
        )

        in_dim = 16 + fourier_matrix.shape[1] + vortex_hidden
        self.proj = nn.Linear(in_dim, d_model)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        """token_ids: (N,) or (N, L) of integers in [0, vocab_size).

        Returns (N, d_model) or (N, L, d_model).
        """
        if token_ids.max() >= self.vocab_size or token_ids.min() < 0:
            raise IndexError(
                f"token_id outside [0, {self.vocab_size}): "
                f"min={int(token_ids.min())}, max={int(token_ids.max())}"
            )

        original_shape = token_ids.shape
        flat = token_ids.reshape(-1)

        char = self.char_features[flat]
        fourier = self.fourier_features[flat]

        h = self.vortex_hashes[flat].unsqueeze(1)
        vortex_phases = self.vortex_mlp(h)

        x = torch.cat([char, fourier, vortex_phases], dim=1)
        out = self.proj(x)

        return out.reshape(*original_shape, self.d_model)
```

- [ ] **Step 4: Update fractus/nn/__init__.py**

Replace the entire content of `C:/Users/PHIL/ZCodeProject/fractus/fractus/nn/__init__.py` with:
```python
"""nn subpackage: neural-network modules (PyTorch).

L1: fractal embedding (FractalEmbedding).
"""

from .char_features import CharClassFeatures
from .fourier import MandelbrotFourierBasis
from .embedding import FractalEmbedding

__all__ = ["CharClassFeatures", "MandelbrotFourierBasis", "FractalEmbedding"]
```

- [ ] **Step 5: Run all the tests — THEY MUST PASS**

```powershell
.venv\Scripts\python.exe -m pytest tests/test_embedding.py -v
```
Expected: 12 passed (4 char + 4 fourier + 4 FractalEmbedding). The `test_fractal_embedding_backward_propagates` test is the critical criterion: it proves the embedding truly learns.

- [ ] **Step 6: Commit**

```bash
git add fractus/nn/embedding.py fractus/nn/__init__.py tests/test_embedding.py
git commit -m "feat(nn): add FractalEmbedding with 2-adic vortex conditioning

- Assembly: 16 char features + Mandelbrot Fourier basis + vortex
  conditioning (Collatz hash in Rust → trainable MLP in PyTorch).
- Spec L1 option B: the exact 2-adic vortex (off-graph) conditions a
  differentiable MLP. The vortex is NOT claimed to be differentiable.
- Critical criterion validated: backward() propagates finite gradients to
  every parameter (the test the original system failed)."
```
Expected: `3 files changed`.

---

## Task 6: Interactive L1 demo — prove the embedding learns

**Files:**
- Create: `C:/Users/PHIL/ZCodeProject/fractus/scripts/demo_embedding.py`

- [ ] **Step 1: Write the demo**

Create `C:/Users/PHIL/ZCodeProject/fractus/scripts/demo_embedding.py`:
```python
"""Demo L1: proves that FractalEmbedding truly learns.

Goal: overfit a fixed random embedding target in a few Adam steps,
and show that the loss drops. This is the minimal proof that autodiff flows
through the fractal embedding (which the original system could not do).

Run:
    python scripts/demo_embedding.py
"""

import torch
from fractus.nn import FractalEmbedding


def main():
    torch.manual_seed(42)

    vocab = 64
    d_model = 32
    emb = FractalEmbedding(vocab_size=vocab, d_model=d_model, n_frequencies=12)
    print(f"Trainable parameters: {sum(p.numel() for p in emb.parameters())}")

    # Fixed random target: the goal is to overfit this target.
    target = torch.randn(vocab, d_model)

    opt = torch.optim.Adam(emb.parameters(), lr=1e-2)

    initial_loss = None
    for step in range(200):
        opt.zero_grad()
        ids = torch.arange(vocab)
        out = emb(ids)
        loss = ((out - target) ** 2).mean()
        if initial_loss is None:
            initial_loss = loss.item()
        loss.backward()
        opt.step()
        if step % 40 == 0 or step == 199:
            print(f"step {step:3d}  loss = {loss.item():.4f}")

    final_loss = loss.item()
    print()
    print(f"Initial loss: {initial_loss:.4f}")
    print(f"Final loss  : {final_loss:.4f}")
    print(f"Reduction   : {(1 - final_loss / initial_loss) * 100:.1f}%")

    if final_loss < initial_loss * 0.5:
        print("\n✓ SUCCESS: the fractal embedding learns (loss divided by >2).")
    else:
        print("\n✗ FAILURE: the loss does not drop enough — investigate.")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run the demo**

```powershell
.venv\Scripts\python.exe scripts\demo_embedding.py
```
Expected output (approximate — exact values depend on the seed):
```
Trainable parameters: ~3000
step   0  loss = 1.2xxx
step  40  loss = 0.6xxx
step  80  loss = 0.3xxx
step 120  loss = 0.15xxx
step 160  loss = 0.08xxx
step 199  loss = 0.05xxx

Initial loss: 1.2xxx
Final loss  : 0.05xxx
Reduction   : ~95%

✓ SUCCESS: the fractal embedding learns (loss divided by >2).
```

If the demo shows FAILURE, the bug is serious — debug before L2.

- [ ] **Step 3: Commit**

```bash
git add scripts/demo_embedding.py
git commit -m "demo(L1): prove FractalEmbedding learns (overfit target, loss drops)"
```
Expected: `1 file changed`.

---

## Final "L1 done" criterion

After Task 6, these verifications must all succeed:

```powershell
cd "C:\Users\PHIL\ZCodeProject\fractus"

# 1. Rust compiles and its tests pass (unchanged since L0)
cargo test -p fractus-core
# → 9 passed

# 2. The native module is up to date (includes the new bindings)
.venv\Scripts\python.exe -m maturin develop --release
# → 🛠 Installed fractus-0.1.0

# 3. All Python tests pass
.venv\Scripts\python.exe -m pytest tests/ -v
# → 5 (smoke) + 8 (vortex bridge) + 12 (embedding) = 25 passed

# 4. The demo proves learning
.venv\Scripts\python.exe scripts\demo_embedding.py
# → "✓ SUCCESS: the fractal embedding learns"
```

If everything passes, L1 is done and we can move to the L2 plan (fractal transformer block: linear attention + Kuramoto + Farey MoE).

---

## Self-Review (post-writing)

**1. Spec coverage:**
- Spec L1 requires (a) ported CharClassFeatures → Task 3 ✅;
  (b) MandelbrotFourierBasis (renamed honestly) → Task 4 ✅;
  (c) option B bridge (hash conditions a trainable MLP) → Task 5 ✅;
  (d) "backward() propagates finite gradients" criterion → `test_fractal_embedding_backward_propagates` Task 5 ✅;
  (e) ultrametric tested from Python → `test_ultrametric_strong_triangle_in_python` Task 2 ✅.
- Complete coverage.

**2. Placeholder scan:** no "TBD/TODO/fill in". Every step has complete code or exact commands. ✅

**3. Type consistency:** `CharClassFeatures.extract(int) → Tensor(16,)`. `MandelbrotFourierBasis.matrix() → Tensor(V, 2K)`, `.frequencies → Tensor(K,)`. `FractalEmbedding(vocab_size, d_model, n_frequencies, vortex_hidden, collatz_steps).forward(Tensor) → Tensor`. Consistent between definitions (Tasks 3/4/5) and tests. ✅

**4. Coherent imports:** `from fractus.nn import FractalEmbedding` works thanks to the updated `fractus/nn/__init__.py` (Task 5 Step 4). The hash comes from `from fractus import _core` (Task 5 Step 3, in the module `__init__`). ✅

**5. Dependency on L0:** The `_core` bridge (L0) is required for Task 1 (new wrappers) and Task 5 (Collatz hash). `cargo test -p fractus-core` unchanged (vortex.rs is not modified in L1, just exposed differently). ✅

**6. YAGNI:** No MoE, no attention, no causality — just the embedding. Everything else comes in L2+. ✅
