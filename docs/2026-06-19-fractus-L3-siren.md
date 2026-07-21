# Fractus L3 — True SIREN + honestly measured compression Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Fix the most visible falsehood of the original system: the fake SIREN (it used `nn.SiLU` instead of `sin(ω0·)`) and the hardcoded 20.4× compression ratio. We implement a **true SIREN** (sinusoids as the non-linearity, ω0=30 justified by Sitzmann 2020), apply it to the fractal transformer's weight matrices, and **honestly measure** the obtained compression ratio.

**Honest scientific position (validated decision):** A SIREN represents **smooth** functions well (images, scalar fields). But the weights of a trained network are essentially **dense structured noise**. We therefore expect a **low** compression ratio (~1× to 3×), **not** 20.4×. The L3 documentation will say frankly why, and the demo will measure the truth. This is exactly the opposite of the original's falsehood.

**Architecture:** (1) `fractus/nn/siren.py` — `TorusSirenWeight`: a true SIREN `sin(ω0·(Wx+b))` on the torus T2 = [0,1)2, which regenerates a matrix W[h,w] from a coordinate grid. (2) `SirenLinear`: an `nn.Module` that behaves like `nn.Linear` but whose weight matrix is produced by a SIREN (SIREN parameters trainable). (3) `fractus/metrics/compression.py`: `measure_compression_ratio(model)` genuinely measures the ratio (equivalent dense size / SIREN params). (4) Demo: we replace the attention projections with `SirenLinear`, train, and measure.

**Tech Stack:** PyTorch 2.12 CPU, pytest.

**Spec link:** `docs/SPEC.md`, section "L3 — True SIREN compression + honest measurement".

**Prerequisites:** L2 done (62 tests pass, FractalBlockFull works).

**Reference SIREN math (Sitzmann et al. 2020):**
- Non-linearity: `sin(ω0 · (Wx + b))` (NOT SiLU, NOT ReLU).
- ω0 = 30.0 (SIREN paper empirical value — **not** 56, which is unjustified).
- Special layer init: first layer `U(-1/ω0, 1/ω0)`; subsequent layers `U(-√(6/ω02·fan_in), √(6/ω02·fan_in))`.
- For weight compression: evaluate the SIREN on a grid (h,w) → matrix W[h,w].

---

## File Structure

```
C:/Users/PHIL/ZCodeProject/fractus/
├── fractus/nn/
│   ├── __init__.py             # MODIFY: export TorusSirenWeight, SirenLinear
│   ├── siren.py                # CREATE: TorusSirenWeight (true sin(ω0·) SIREN)
│   └── siren_linear.py         # CREATE: SirenLinear (Linear whose W comes from a SIREN)
├── fractus/metrics/
│   ├── __init__.py             # CREATE
│   └── compression.py          # CREATE: measure_compression_ratio (honest measurement)
└── tests/
    ├── test_siren.py           # CREATE: SIREN tests (sin present, no SiLU, backward)
    ├── test_siren_linear.py    # CREATE: SirenLinear tests (shape, backward)
    └── test_compression.py     # CREATE: test measure_compression_ratio (no hardcode)
```

**Responsibilities:**
- `siren.py`: the pure SIREN (represents a scalar field on T2).
- `siren_linear.py`: an `nn.Module` adapter that produces a `nn.Linear`-like layer whose W = SIREN(grid).
- `metrics/compression.py`: measures the real ratio, NO hardcoded literal.

---

## Task 1: TorusSirenWeight (true sin(ω0·) SIREN)

**Files:**
- Create: `C:/Users/PHIL/ZCodeProject/fractus/fractus/nn/siren.py`

- [ ] **Step 1: Write the tests**

Create `C:/Users/PHIL/ZCodeProject/fractus/tests/test_siren.py`:
```python
"""Tests of TorusSirenWeight: a true SIREN sin(ω0·), not SiLU."""

import inspect
import torch


def test_siren_uses_sin_not_silu():
    """L3 CRITERION: the SIREN must use torch.sin, NOT nn.SiLU.
    This is exactly the original falsehood (torus_siren.py:15,17 used SiLU)."""
    from fractus.nn import siren
    src = inspect.getsource(siren)
    assert 'torch.sin(' in src or 'sin(' in src, "The SIREN must use sin(ω₀·)"
    assert 'SiLU' not in src and 'silu' not in src.lower(), \
        "No more SiLU (the original falsehood)"


def test_siren_omega0_is_30_not_56():
    """ω0 = 30 (justified by Sitzmann 2020), NOT 56 (unjustified, inherited from the original)."""
    from fractus.nn.siren import TorusSirenWeight
    s = TorusSirenWeight(out_h=16, out_w=16, hidden=32)
    assert abs(s.omega0 - 30.0) < 1e-6, f"ω₀ should be 30.0, got {s.omega0}"


def test_siren_output_shape():
    """The SIREN evaluated on the grid produces a matrix (out_h, out_w)."""
    from fractus.nn.siren import TorusSirenWeight
    s = TorusSirenWeight(out_h=16, out_w=16, hidden=32)
    W = s()
    assert W.shape == (16, 16)


def test_siren_is_finite():
    from fractus.nn.siren import TorusSirenWeight
    s = TorusSirenWeight(out_h=16, out_w=16, hidden=32)
    W = s()
    assert torch.isfinite(W).all()


def test_siren_backward_propagates():
    """L3 CRITERION: backward propagates a finite AND non-zero gradient to EVERY parameter."""
    from fractus.nn.siren import TorusSirenWeight
    s = TorusSirenWeight(out_h=16, out_w=16, hidden=32)
    W = s()
    loss = W.pow(2).sum()
    loss.backward()

    params = list(s.named_parameters())
    assert len(params) > 0
    for name, p in params:
        assert p.requires_grad, f"{name} should requires_grad=True"
        assert p.grad is not None, f"{name} received no gradient"
        assert torch.isfinite(p.grad).all(), f"{name} has a non-finite gradient"
        assert p.grad.abs().sum().item() > 0, f"{name} received a zero gradient"


def test_siren_fewer_params_than_dense():
    """The SIREN must have FEWER parameters than the equivalent dense matrix
    (otherwise there is no compression). For (32,32) with hidden=16:
    dense = 1024 params, SIREN ≈ 2·16 + 16·16 + 16·1 + biases ≈ 300-400.
    So ratio > 2 expected AT THE PARAMETER LEVEL. (But reconstruction quality
    is another question — see the demo.)"""
    from fractus.nn.siren import TorusSirenWeight
    s = TorusSirenWeight(out_h=32, out_w=32, hidden=16)
    n_siren = sum(p.numel() for p in s.parameters())
    n_dense = 32 * 32
    assert n_siren < n_dense, \
        f"SIREN ({n_siren} params) should be < dense ({n_dense})"
```

- [ ] **Step 2: Run to verify the tests fail**

```powershell
.venv\Scripts\python.exe -m pytest tests/test_siren.py -v
```
Expected: FAIL — module absent.

- [ ] **Step 3: Implement TorusSirenWeight**

Create `C:/Users/PHIL/ZCodeProject/fractus/fractus/nn/siren.py`:
```python
"""TorusSirenWeight: a true SIREN sin(ω0·) to represent a weight matrix.

CORRECTION OF THE ORIGINAL SYSTEM'S LIE:
- The original used nn.SiLU (torus_siren.py:15,17) → here we use torch.sin(ω0·(Wx+b)),
  the TRUE SIREN non-linearity (Sitzmann et al. 2020).
- The original used the unjustified ω0=56 → here ω0=30.0 (SIREN paper empirical value,
  which shows ω0≈30 is optimal for representing continuous functions).
- The original commented "Simple reconstruction: sum of harmonics (real implementation uses
  Fourier)" (torus_siren.py:39) → here the reconstruction is REAL (SIREN forward
  on a 2D grid).

HONEST SCIENTIFIC POSITION:
A SIREN represents smooth functions well (images, scalar fields).
The weights of a trained network are essentially dense structured noise.
We therefore expect a LOW compression ratio (~1× to 3×), NOT 20.4×.
The ratio is MEASURED (metrics/compression.py), never hardcoded.

Math (Sitzmann 2020):
    Non-linearity: sin(ω0 · (Wx + b)) for each hidden layer.
    Output layer: linear (no sin).
    Init: first layer U(-1/ω0, 1/ω0); subsequent U(-√(6/(ω02·fan_in)), ...).

The SIREN takes coords (u,v) ∈ [0,1)2 on the torus T2 as input and produces
a scalar W[u,v]. Evaluated on an h×w grid, it regenerates the matrix W.
"""

import math
import torch
import torch.nn as nn


class TorusSirenWeight(nn.Module):
    """SIREN that represents a weight matrix W[out_h, out_w] as a scalar
    field over the torus T2 = [0,1)2.

    Args:
        out_h, out_w : dimensions of the matrix to regenerate.
        hidden       : width of the SIREN hidden layers.
        omega0       : fundamental frequency (30.0 by default, Sitzmann 2020).
    """

    def __init__(
        self,
        out_h: int,
        out_w: int,
        hidden: int = 32,
        omega0: float = 30.0,
    ):
        super().__init__()
        if out_h < 1 or out_w < 1 or hidden < 1:
            raise ValueError("out_h, out_w, hidden must be >= 1")
        self.out_h = out_h
        self.out_w = out_w
        self.hidden = hidden
        self.omega0 = omega0

        # Layers: Linear(2 → hidden) → sin → Linear(hidden → hidden) → sin → Linear(hidden → 1).
        # Three layers total (as in the SIREN paper for scalar fields).
        self.fc1 = nn.Linear(2, hidden)
        self.fc2 = nn.Linear(hidden, hidden)
        self.fc3 = nn.Linear(hidden, 1)
        self._init_siren_weights()

        # Precomputed grid (off-graph because it is constant).
        grid = self._build_grid(out_h, out_w)  # (out_h·out_w, 2)
        self.register_buffer("grid", grid)

    def _init_siren_weights(self):
        """SIREN-specific init (Sitzmann 2020, section 3.2)."""
        # First layer: U(-1/ω0, 1/ω0).
        with torch.no_grad():
            nn.init.uniform_(self.fc1.weight, -1.0 / self.omega0, 1.0 / self.omega0)
            nn.init.zeros_(self.fc1.bias)
            # Subsequent layers: U(-√(6/(ω02·fan_in)), √(6/(ω02·fan_in))).
            for layer in [self.fc2, self.fc3]:
                fan_in = layer.weight.shape[1]
                bound = math.sqrt(6.0 / (self.omega0 ** 2 * fan_in))
                nn.init.uniform_(layer.weight, -bound, bound)
                nn.init.zeros_(layer.bias)

    @staticmethod
    def _build_grid(h: int, w: int) -> torch.Tensor:
        """Grid of coords (u,v) ∈ [0,1)2 on the torus, shape (h·w, 2)."""
        u = torch.linspace(0, 1, h, dtype=torch.float32)
        v = torch.linspace(0, 1, w, dtype=torch.float32)
        grid = torch.stack(torch.meshgrid(u, v, indexing="ij"), dim=-1)  # (h, w, 2)
        return grid.reshape(-1, 2)  # (h·w, 2)

    def forward(self) -> torch.Tensor:
        """Evaluates the SIREN on the grid → matrix W[out_h, out_w].

        This is the 'decompression': we regenerate W from the SIREN params.
        """
        x = self.grid  # (h·w, 2)
        # Layer 1 + sin(ω0·).
        x = torch.sin(self.omega0 * self.fc1(x))
        # Layer 2 + sin(ω0·).
        x = torch.sin(self.omega0 * self.fc2(x))
        # Output layer: linear (no sin).
        x = self.fc3(x)  # (h·w, 1)
        return x.squeeze(-1).reshape(self.out_h, self.out_w)
```

- [ ] **Step 4: Run the tests — THEY MUST PASS**

```powershell
.venv\Scripts\python.exe -m pytest tests/test_siren.py -v
```
Expected: 6 passed. The `test_siren_uses_sin_not_silu` test is the critical L3 criterion.

- [ ] **Step 5: Commit**

```bash
cd "C:\Users\PHIL\ZCodeProject\fractus"
git add fractus/nn/siren.py tests/test_siren.py
git commit -m "feat(nn): add TorusSirenWeight (real sin(ω0·) SIREN, ω0=30, Sitzmann init)"
```

---

## Task 2: SirenLinear (Linear whose W comes from a SIREN)

**Files:**
- Create: `C:/Users/PHIL/ZCodeProject/fractus/fractus/nn/siren_linear.py`

- [ ] **Step 1: Write the tests**

Create `C:/Users/PHIL/ZCodeProject/fractus/tests/test_siren_linear.py`:
```python
"""Tests of SirenLinear: behaves like nn.Linear but W = SIREN(grid)."""

import torch


def test_siren_linear_shape():
    """SirenLinear(in, out) behaves like nn.Linear: (B, in) → (B, out)."""
    from fractus.nn.siren_linear import SirenLinear
    layer = SirenLinear(in_features=16, out_features=16, hidden=32)
    x = torch.randn(4, 16)
    y = layer(x)
    assert y.shape == (4, 16)


def test_siren_linear_is_finite():
    from fractus.nn.siren_linear import SirenLinear
    layer = SirenLinear(in_features=16, out_features=16, hidden=32)
    x = torch.randn(4, 16)
    assert torch.isfinite(layer(x)).all()


def test_siren_linear_backward_propagates():
    """L3 CRITERION: backward propagates a finite AND non-zero gradient to EVERY parameter
    of the SIREN (which IS the weight matrix, in the graph)."""
    from fractus.nn.siren_linear import SirenLinear
    layer = SirenLinear(in_features=16, out_features=16, hidden=32)
    x = torch.randn(4, 16)
    y = layer(x)
    loss = y.pow(2).sum()
    loss.backward()

    params = list(layer.named_parameters())
    assert len(params) > 0
    for name, p in params:
        assert p.requires_grad
        assert p.grad is not None, f"{name} received no gradient"
        assert torch.isfinite(p.grad).all()
        assert p.grad.abs().sum().item() > 0, f"{name} received a zero gradient"


def test_siren_linear_has_no_dense_weight():
    """SirenLinear must NOT have a separate dense nn.Parameter — the matrix
    comes entirely from the SIREN."""
    from fractus.nn.siren_linear import SirenLinear
    layer = SirenLinear(in_features=16, out_features=16, hidden=32)
    # The only params must be the SIREN's + the bias.
    param_names = [n for n, _ in layer.named_parameters()]
    assert not any("dense" in n or "weight" in n.lower() and "siren" not in n.lower()
                   for n in param_names), \
        f"SirenLinear should not have a separate dense weight: {param_names}"
```

- [ ] **Step 2: Run to verify the tests fail**

```powershell
.venv\Scripts\python.exe -m pytest tests/test_siren_linear.py -v
```
Expected: FAIL — module absent.

- [ ] **Step 3: Implement SirenLinear**

Create `C:/Users/PHIL/ZCodeProject/fractus/fractus/nn/siren_linear.py`:
```python
"""SirenLinear: an nn.Linear-like layer whose weight matrix is produced
by a SIREN.

CORRECTION vs the original: in the original, the decompressed matrix W was computed then
DISCARDED (training_loop.py:30-37 applied a mirror to W then ran on the raw input).
Here, the SIREN IS the matrix: we evaluate the SIREN at each forward to obtain W,
then we do y = x @ W + b. Everything is in the autodiff graph.

Usage: replace some nn.Linear layers with SirenLinear to compress their
weights via SIREN. The trade-off: fewer parameters (compression) but a more
expensive forward (SIREN evaluation at each call) and potentially reduced
expressiveness (SIREN weights are smooth, not dense — see the L3 demo).
"""

import torch
import torch.nn as nn

from .siren import TorusSirenWeight


class SirenLinear(nn.Module):
    """Linear layer whose matrix W = SIREN(grid).

    Args:
        in_features, out_features : dimensions (as in nn.Linear).
        hidden : width of the SIREN that produces W.
        omega0 : SIREN frequency.
        bias   : if True, adds a trainable bias (as in nn.Linear).
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        hidden: int = 32,
        omega0: float = 30.0,
        bias: bool = True,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        # The weight matrix comes from a SIREN evaluated on a grid
        # (in_features, out_features).
        self.siren = TorusSirenWeight(
            out_h=in_features, out_w=out_features, hidden=hidden, omega0=omega0
        )
        # Separate trainable bias (not compressed — this is a vector, not a matrix).
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features))
        else:
            self.register_parameter("bias", None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x : (..., in_features) → (..., out_features).

        W = self.siren() : (in_features, out_features), in the autodiff graph.
        y = x @ W + bias.
        """
        W = self.siren()  # (in_features, out_features), differentiable
        y = x @ W
        if self.bias is not None:
            y = y + self.bias
        return y
```

- [ ] **Step 4: Run the tests — THEY MUST PASS**

```powershell
.venv\Scripts\python.exe -m pytest tests/test_siren_linear.py -v
```
Expected: 4 passed.

- [ ] **Step 5: Commit**

```bash
git add fractus/nn/siren_linear.py tests/test_siren_linear.py
git commit -m "feat(nn): add SirenLinear (Linear whose W comes from a SIREN, in-graph)"
```

---

## Task 3: Honest compression-ratio measurement

**Files:**
- Create: `C:/Users/PHIL/ZCodeProject/fractus/fractus/metrics/__init__.py`
- Create: `C:/Users/PHIL/ZCodeProject/fractus/fractus/metrics/compression.py`

- [ ] **Step 1: Write the tests**

Create `C:/Users/PHIL/ZCodeProject/fractus/tests/test_compression.py`:
```python
"""Tests of measure_compression_ratio: REAL measurement, no hardcoding."""

import inspect
import torch


def test_compression_no_hardcoded_204():
    """L3 CRITERION: the measurement code must NOT contain the literal 20.4
    (the original's hardcoded falsehood in training_loop.py:52)."""
    from fractus.metrics import compression
    src = inspect.getsource(compression)
    assert "20.4" not in src, "The literal 20.4 is forbidden (original falsehood)"


def test_compression_pure_dense_returns_one():
    """A 100% dense model (no SirenLinear) → ratio 1.0."""
    from fractus.metrics.compression import measure_compression_ratio
    m = torch.nn.Linear(16, 16)
    ratio = measure_compression_ratio(m)
    assert abs(ratio - 1.0) < 1e-6


def test_compression_with_siren_gt_one():
    """A model with SirenLinear → ratio > 1 (fewer params than the dense equivalent)."""
    from fractus.nn.siren_linear import SirenLinear
    from fractus.metrics.compression import measure_compression_ratio
    m = torch.nn.Sequential(
        SirenLinear(32, 32, hidden=16),  # SIREN instead of Linear(32,32)
        torch.nn.ReLU(),
        torch.nn.Linear(32, 10),  # classic dense
    )
    ratio = measure_compression_ratio(m)
    # The ratio must be > 1 (SIRENs have fewer params than the equivalent dense
    # matrix). The exact value depends on hidden, but > 1 is guaranteed.
    assert ratio > 1.0, f"Expected ratio > 1, got {ratio}"


def test_compression_returns_finite():
    from fractus.metrics.compression import measure_compression_ratio
    m = torch.nn.Linear(8, 8)
    r = measure_compression_ratio(m)
    assert isinstance(r, float)
    assert r > 0
```

- [ ] **Step 2: Run to verify the tests fail**

```powershell
.venv\Scripts\python.exe -m pytest tests/test_compression.py -v
```
Expected: FAIL — module absent.

- [ ] **Step 3: Implement metrics/compression.py**

Create `C:/Users/PHIL/ZCodeProject/fractus/fractus/metrics/__init__.py`:
```python
"""metrics subpackage: honest measurements (compression, causal, perplexity).

L3: compression (real measurement, no hardcoding).
"""
```

Create `C:/Users/PHIL/ZCodeProject/fractus/fractus/metrics/compression.py`:
```python
"""HONEST measurement of a model's compression ratio.

CORRECTION OF THE ORIGINAL SYSTEM'S LIE:
- The original hardcoded "compression_ratio": 20.4 in training_loop.py:52.
- Here, the ratio is MEASURED: we count the parameters actually used and compare them
  to the size the matrices would have if they were dense.

Ratio definition:
    ratio = (sum of equivalent dense sizes of SirenLinear) /
            (sum of SIREN params + remaining dense params)

For a SirenLinear(in, out, hidden=h):
    - equivalent dense size = in·out (the matrix it replaces)
    - SIREN params = 2·h + h·h + h·1 + biases ≈ h2 + 3h
    The ratio of THIS layer = in·out / params_SIREN.

For a mixed model (SirenLinear + nn.Linear), the global ratio is:
    (Σ equivalent dense sizes) / (Σ total params).

We do NOT claim 20.4×. We measure. The L3 demo will show the true figure.
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
        model : an nn.Module that may contain SirenLinear and/or nn.Linear.
    Returns:
        ratio > 0. ratio = 1.0 if the model is 100% dense.
        ratio > 1 if the model contains SirenLinear (effective compression).
        ratio < 1 is possible but rare (SIREN larger than the matrix).
    """
    total_dense_equivalent = 0  # size the matrices would have if dense
    total_actual_params = 0     # params actually used

    for module in model.modules():
        if isinstance(module, SirenLinear):
            # Equivalent dense size: in·out (the matrix W it replaces).
            dense_eq = module.in_features * module.out_features
            # Actual params: SIREN params + bias.
            actual = _count_params(module)
            total_dense_equivalent += dense_eq
            total_actual_params += actual
        elif isinstance(module, nn.Linear) and not isinstance(module, SirenLinear):
            # nn.Linear: dense_eq == actual (no compression).
            dense_eq = module.in_features * module.out_features
            actual = _count_params(module)
            total_dense_equivalent += dense_eq
            total_actual_params += actual
        elif isinstance(module, (nn.LayerNorm, nn.Embedding, nn.Parameter)):
            # Other modules: no compression (counted at their real size).
            actual = _count_params(module)
            total_dense_equivalent += actual
            total_actual_params += actual

    if total_actual_params == 0:
        return 1.0
    return total_dense_equivalent / total_actual_params
```

- [ ] **Step 4: Update fractus/nn/__init__.py**

Modify `C:/Users/PHIL/ZCodeProject/fractus/fractus/nn/__init__.py` — add:
```python
from .siren import TorusSirenWeight
from .siren_linear import SirenLinear
```
And extend `__all__` with `"TorusSirenWeight"`, `"SirenLinear"`.

- [ ] **Step 5: Run the tests — THEY MUST PASS**

```powershell
.venv\Scripts\python.exe -m pytest tests/test_compression.py tests/test_siren.py tests/test_siren_linear.py -v
```
Expected: 4 + 6 + 4 = 14 passed.

- [ ] **Step 6: Commit**

```bash
git add fractus/metrics/ fractus/nn/__init__.py tests/test_compression.py
git commit -m "feat(metrics): add measure_compression_ratio (honest, no 20.4 hardcode)"
```

---

## Task 4: L3 demo — measure the true compression on the transformer

**Files:**
- Create: `C:/Users/PHIL/ZCodeProject/fractus/scripts/demo_siren_compression.py`

- [ ] **Step 1: Write the demo**

Create `C:/Users/PHIL/ZCodeProject/fractus/scripts/demo_siren_compression.py`:
```python
"""Demo L3: measure the TRUE SIREN compression on a model.

We build two variants of a mini-MLP:
    (A) 100% dense (nn.Linear)
    (B) Hidden layers in SirenLinear, last layer dense.

We measure:
    - The compression ratio (SIREN params vs dense equivalent).
    - The learning capacity: can we overfit a target with (B) as well
      as with (A)?

HONEST SCIENTIFIC POSITION:
We expect a MODEST ratio (~2× to 5×) and a loss in learning quality
(SIREN weights are smooth, which limits the capacity to express arbitrary
functions). This is the truth — to be compared with the original's false 20.4×.

Run:
    python scripts/demo_siren_compression.py
"""

import torch
import torch.nn as nn
from fractus.nn import SirenLinear
from fractus.metrics.compression import measure_compression_ratio


def make_dense_model(d_in, d_hidden, d_out):
    return nn.Sequential(
        nn.Linear(d_in, d_hidden), nn.ReLU(),
        nn.Linear(d_hidden, d_hidden), nn.ReLU(),
        nn.Linear(d_hidden, d_out),
    )


def make_siren_model(d_in, d_hidden, d_out, siren_hidden=16):
    return nn.Sequential(
        SirenLinear(d_in, d_hidden, hidden=siren_hidden), nn.ReLU(),
        SirenLinear(d_hidden, d_hidden, hidden=siren_hidden), nn.ReLU(),
        nn.Linear(d_hidden, d_out),  # last layer dense
    )


def train_and_eval(model, X, Y, n_steps=300, lr=1e-2):
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    initial = None
    for step in range(n_steps):
        opt.zero_grad()
        pred = model(X)
        loss = ((pred - Y) ** 2).mean()
        if initial is None:
            initial = loss.item()
        loss.backward()
        opt.step()
    final = loss.item()
    return initial, final


def main():
    torch.manual_seed(42)
    d_in, d_hidden, d_out = 16, 32, 8
    n_samples = 64

    # Target: a non-trivial function (sinusoid at a non-aligned frequency).
    X = torch.randn(n_samples, d_in)
    Y = torch.sin(X[:, :d_out] * 1.3) + 0.5 * torch.cos(X[:, :d_out] * 0.7)

    dense = make_dense_model(d_in, d_hidden, d_out)
    siren = make_siren_model(d_in, d_hidden, d_out, siren_hidden=16)

    n_dense = sum(p.numel() for p in dense.parameters())
    n_siren = sum(p.numel() for p in siren.parameters())
    ratio_dense = measure_compression_ratio(dense)
    ratio_siren = measure_compression_ratio(siren)

    print("=== Measured compression ===")
    print(f"Dense model : {n_dense} params, ratio = {ratio_dense:.2f}×")
    print(f"SIREN model : {n_siren} params, ratio = {ratio_siren:.2f}×")
    print(f"Savings     : {(1 - n_siren/n_dense)*100:.1f}% fewer params")
    print()

    print("=== Learning capacity (overfit sinusoidal target) ===")
    i_d, f_d = train_and_eval(dense, X, Y)
    i_s, f_s = train_and_eval(siren, X, Y)
    print(f"Dense : loss {i_d:.4f} → {f_d:.4f}  (drop {(1-f_d/i_d)*100:.1f}%)")
    print(f"SIREN : loss {i_s:.4f} → {f_s:.4f}  (drop {(1-f_s/i_s)*100:.1f}%)")
    print()

    print("=== Honest verdict ===")
    print(f"Real compression ratio: {ratio_siren:.2f}×")
    print(f"  (compare to the '20.4×' hardcoded in the original design, which was false)")
    print(f"Learning quality loss: {(f_s - f_d):.4f} (SIREN - Dense)")
    if ratio_siren > 1.5 and f_s < i_s * 0.5:
        print("\n✓ The SIREN compresses (>1.5×) AND learns — honest and useful.")
    elif ratio_siren > 1.5:
        print("\n~ The SIREN compresses but learns less well — a trade-off to document.")
    else:
        print("\n~ Weak compression (<1.5×) — the SIREN is not suited to these weights.")
    print("\nConclusion: the original's '20.4× without loss' claim is not reproduced.")
    print("The SIREN is useful for smooth functions, not for dense weights.")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run the demo**

```powershell
.venv\Scripts\python.exe scripts\demo_siren_compression.py
```
Expected: measured ratio between 1.5× and 5× (NOT 20.4×). SIREN learning quality <= Dense.

- [ ] **Step 3: Commit**

```bash
git add scripts/demo_siren_compression.py
git commit -m "demo(L3): measure honest SIREN compression ratio (not 20.4x)"
```

---

## Final "L3 done" criterion

```powershell
.venv\Scripts\python.exe -m pytest tests/ -v
# → 62 (L0+L1+L2) + 6 (siren) + 4 (siren_linear) + 4 (compression) = 76 passed

.venv\Scripts\python.exe scripts\demo_siren_compression.py
# → measured ratio (not 20.4×), honest verdict displayed
```

L3 done → we have a true SIREN, an honest measurement, and the frank documentation of why 20.4× was a falsehood. We then move to L4 (NOTEARS causality).

---

## Self-Review

**1. Spec coverage:** (a) TorusSirenWeight true sin(ω0·) → Task 1 ✅; (b) SirenLinear (W in the graph) → Task 2 ✅; (c) measure_compression_ratio honest → Task 3 ✅; (d) demo measuring the true ratio → Task 4 ✅; (e) critical tests (sin not SiLU, ω0=30 not 56, no 20.4 hardcode) → Task 1 Step 1 + Task 3 Step 1 ✅.

**2. Placeholder scan:** no TBD. ✅

**3. Mathematical honesty:** ω0=30 justified (Sitzmann 2020), sin instead of SiLU, correct SIREN init, ratio measured not hardcoded, demo explicitly documents the scientific position (SIREN on dense weights = low compression). ✅

**4. Type consistency:** `TorusSirenWeight(out_h, out_w, hidden, omega0).forward() → Tensor(out_h, out_w)`. `SirenLinear(in, out, hidden, omega0).forward((..., in)) → (..., out)`. `measure_compression_ratio(nn.Module) → float`. Consistent. ✅
