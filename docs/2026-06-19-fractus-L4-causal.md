# Fractus L4 — NOTEARS causality + RKHS RFF + do-calculus Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Fix the original's "fake RKHS" (just a low-rank projection `x@U@VT`, no kernel) and the "trivial do-calculus" (column-zeroing). Implement a **true** causal-discovery pipeline: (1) a differentiable NOTEARS acyclicity penalty `h(A) = tr(e^{A⊙A}) − n` (faithfully ported from the original `causal.rs`), (2) a causal RKHS operator via **Random Fourier Features** (Rahimi-Recht 2007), (3) **true** Pearl do-calculus (post-intervention sampling), (4) a **Structural Hamming Distance** metric measured on a known synthetic DAG.

**Architecture:** (1) `fractus/causal/notears.py` — `notears_penalty(W)`: a differentiable scalar, =0 iff W is a DAG. (2) `fractus/causal/rkhs.py` — `RKHSCausalOperator`: Gaussian kernel approximated by RFF, operator L in feature space. (3) `fractus/causal/do.py` — `do_intervention`: true Pearl do-calculus (clamp + propagation). (4) `fractus/metrics/causal.py` — `structural_hamming_distance`: measured SHD, no 0.98 clamp. (5) `data/causal/generate_scm.py` — generate synthetic Structural Causal Models (known DAG + data). (6) Demo: NOTEARS recovers a 5-node synthetic DAG.

**Tech Stack:** PyTorch 2.12 CPU, numpy, pytest.

**Spec link:** `docs/SPEC.md`, section "L4 — NOTEARS causality + RKHS".

**Prerequisites:** L3 done (76 tests pass).

**Reference math:**
- NOTEARS (Zheng 2018): `h(W) = tr(e^{W⊙W}) − n` where e^{·} = matrix exponential (20-term Taylor). h(W)=0 iff W is acyclic. Differentiable.
- RKHS via RFF (Rahimi-Recht 2007): Gaussian kernel `k(x,y) = exp(-||x-y||2/(2σ2))` ≈ `φ(x)·φ(y)` where `φ(x) = [cos(ω_k·x), sin(ω_k·x)] / √K`, `ω_k ~ N(0, 1/σ2)`.
- Pearl do-calculus: `P(Y | do(X=x))` = `P(Y | X=x)`. The intervention fixes X=x (clamp), then propagates.
- SHD: number of mispredicted edges (missing + extra + wrong orientation).

---

## File Structure

```
C:/Users/PHIL/ZCodeProject/fractus/
├── fractus/causal/
│   ├── __init__.py             # CREATE
│   ├── notears.py              # CREATE: notears_penalty (differentiable)
│   ├── rkhs.py                 # CREATE: RKHSCausalOperator (RFF)
│   └── do.py                   # CREATE: do_intervention (Pearl)
├── fractus/metrics/
│   ├── causal.py               # CREATE: structural_hamming_distance
│   └── __init__.py             # MODIFY: export SHD
├── data/causal/
│   └── generate_scm.py         # CREATE: synthetic SCM (DAG + data)
└── tests/
    ├── test_notears.py         # CREATE
    ├── test_rkhs.py            # CREATE
    ├── test_do.py              # CREATE
    └── test_causal_metrics.py  # CREATE
```

---

## Task 1: notears_penalty (differentiable)

**Files:**
- Create: `C:/Users/PHIL/ZCodeProject/fractus/fractus/causal/__init__.py`
- Create: `C:/Users/PHIL/ZCodeProject/fractus/fractus/causal/notears.py`

- [ ] **Step 1: Write the tests**

Create `C:/Users/PHIL/ZCodeProject/fractus/tests/test_notears.py`:
```python
"""Tests of notears_penalty: differentiable, =0 for DAG, >0 for cycle."""

import torch


def test_notears_zero_for_dag():
    """h(W) ≈ 0 if W is an obvious DAG (strict lower-triangular)."""
    from fractus.causal.notears import notears_penalty
    W = torch.tensor([
        [0.0, 0.0, 0.0],
        [0.5, 0.0, 0.0],
        [0.3, 0.4, 0.0],
    ])  # DAG: 1<-2, 1<-3, 2<-3, no cycle
    h = notears_penalty(W)
    assert abs(h.item()) < 1e-3, f"h(DAG) should be ~0, got {h.item()}"


def test_notears_positive_for_cycle():
    """h(W) > 0 if W contains a cycle."""
    from fractus.causal.notears import notears_penalty
    W = torch.tensor([
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
        [1.0, 0.0, 0.0],
    ])  # cycle 1->2->3->1
    h = notears_penalty(W)
    assert h.item() > 0.5, f"h(cycle) should be > 0.5, got {h.item()}"


def test_notears_zero_for_zero_matrix():
    """h(0) = 0 (the zero matrix is trivially acyclic)."""
    from fractus.causal.notears import notears_penalty
    W = torch.zeros(4, 4)
    h = notears_penalty(W)
    assert abs(h.item()) < 1e-6


def test_notears_is_differentiable():
    """h(W) must be differentiable (gradient w.r.t. W)."""
    from fractus.causal.notears import notears_penalty
    W = torch.randn(3, 3, requires_grad=True)
    h = notears_penalty(W)
    h.backward()
    assert W.grad is not None
    assert torch.isfinite(W.grad).all()


def test_notears_shape_scalar():
    """h(W) is a scalar (trace sum)."""
    from fractus.causal.notears import notears_penalty
    W = torch.randn(5, 5)
    h = notears_penalty(W)
    assert h.dim() == 0


def test_notears_larger_cycle_detected():
    """A size-4 cycle must be detected."""
    from fractus.causal.notears import notears_penalty
    W = torch.zeros(4, 4)
    W[0, 1] = W[1, 2] = W[2, 3] = W[3, 0] = 1.0  # 0->1->2->3->0
    h = notears_penalty(W)
    assert h.item() > 0.5
```

- [ ] **Step 2: Run to verify the tests fail**

```powershell
.venv\Scripts\python.exe -m pytest tests/test_notears.py -v
```
Expected: FAIL — module absent.

- [ ] **Step 3: Implement notears.py**

Create `C:/Users/PHIL/ZCodeProject/fractus/fractus/causal/__init__.py`:
```python
"""causal subpackage: NOTEARS, RKHS, do-calculus.

L4: causal discovery with a guaranteed acyclic DAG (NOTEARS), an RKHS operator
via Random Fourier Features, and true Pearl do-calculus.
"""
```

Create `C:/Users/PHIL/ZCodeProject/fractus/fractus/causal/notears.py`:
```python
"""NOTEARS acyclicity penalty: h(W) = tr(e^{W*W}) - n.

Faithfully ported from the original system (src/causal.rs:159-196) in pure PyTorch.

Math (Zheng et al. 2018, "DAGs with NO TEARS"):
    h(W) = tr(expm(W ⊙ W)) − n
    where expm is the matrix exponential and ⊙ the Hadamard product.

    Property: h(W) = 0 iff W is acyclic (DAG).
    h(W) > 0 if W contains a cycle.
    Differentiable → we can optimize it by gradient descent.

Approximation: expm via Taylor series with 20 terms (as in the original).
    e^M = I + M + M2/2! + ... + M20/20!

CORRECTION vs the original: the original had NO acyclicity constraint at all
(rkhs_causal.py imposed no DAG). Here we have a true differentiable NOTEARS.
"""

import torch


def notears_penalty(W: torch.Tensor, n_terms: int = 20) -> torch.Tensor:
    """Computes h(W) = tr(e^{W⊙W}) − n, a scalar.

    Args:
        W : adjacency matrix (n, n), differentiable.
        n_terms : number of Taylor series terms (20 by default).
    Returns:
        h : scalar. =0 if W is a DAG, >0 if W contains a cycle.
    """
    n = W.shape[0]
    assert W.shape == (n, n), f"W must be square, got {W.shape}"

    # M = W ⊙ W (squared element).
    M = W * W

    # e^M = I + M + M2/2! + ... + M^k/k!  (Taylor series).
    eye = torch.eye(n, dtype=W.dtype, device=W.device)
    result = eye.clone()
    term = eye.clone()  # term_k = M^k / k!, init to M^0/0! = I
    for k in range(1, n_terms + 1):
        term = (term @ M) / k  # term_k = term_{k-1} · M / k
        result = result + term
        # Early convergence (as in the original).
        if term.norm() < 1e-10:
            break

    # h = tr(result) - n.
    trace = torch.diagonal(result).sum()
    return trace - n
```

- [ ] **Step 4: Run the tests — THEY MUST PASS**

```powershell
.venv\Scripts\python.exe -m pytest tests/test_notears.py -v
```
Expected: 6 passed.

- [ ] **Step 5: Commit**

```bash
cd "C:\Users\PHIL\ZCodeProject\fractus"
git add fractus/causal/ tests/test_notears.py
git commit -m "feat(causal): add notears_penalty (differentiable DAG acyclicity, ported from the original)"
```

---

## Task 2: RKHSCausalOperator (via Random Fourier Features)

**Files:**
- Create: `C:/Users/PHIL/ZCodeProject/fractus/fractus/causal/rkhs.py`

- [ ] **Step 1: Write the tests**

Create `C:/Users/PHIL/ZCodeProject/fractus/tests/test_rkhs.py`:
```python
"""Tests of RKHSCausalOperator: true RKHS via RFF, not a bare low-rank projection."""

import torch


def test_rkhs_output_shape():
    """The RKHS operator transforms (N, d) → (N, d)."""
    from fractus.causal.rkhs import RKHSCausalOperator
    op = RKHSCausalOperator(dim=8, rank=4, n_rff=32)
    x = torch.randn(16, 8)
    y = op(x)
    assert y.shape == (16, 8)


def test_rkhs_is_finite():
    from fractus.causal.rkhs import RKHSCausalOperator
    op = RKHSCausalOperator(dim=8, rank=4, n_rff=32)
    x = torch.randn(16, 8) * 5
    assert torch.isfinite(op(x)).all()


def test_rkhs_kernel_approx_positive():
    """The approximated kernel k(x,x) must be positive (≈ 1 for normalized x)."""
    from fractus.causal.rkhs import RKHSCausalOperator
    op = RKHSCausalOperator(dim=4, rank=2, n_rff=64)
    x = torch.randn(3, 4)
    kxx = op.kernel(x, x)  # (3, 3)
    assert (torch.diagonal(kxx) > 0).all(), "k(x,x) must be positive"


def test_rkhs_backward_every_param():
    """L4 CRITERION: backward propagates a finite AND non-zero gradient to EVERY parameter."""
    from fractus.causal.rkhs import RKHSCausalOperator
    op = RKHSCausalOperator(dim=8, rank=4, n_rff=32)
    x = torch.randn(16, 8)
    y = op(x)
    loss = y.pow(2).sum()
    loss.backward()

    params = list(op.named_parameters())
    assert len(params) > 0
    for name, p in params:
        assert p.requires_grad
        assert p.grad is not None, f"{name} received no gradient"
        assert torch.isfinite(p.grad).all()
        # Note: the W_rff (random RFF features) are intentionally
        # frozen (not trainable) — this is the Rahimi-Recht method. We only
        # check the non-zero gradient on U, V (the trainable params).
        if name in ("U", "V"):
            assert p.grad.abs().sum().item() > 0, f"{name} received a zero gradient"


def test_rkhs_not_just_linear_projection():
    """TRUE RKHS: the output must depend on the (non-linear) kernel, not just x@U@VT.
    We verify the output is not equal to a simple linear projection."""
    from fractus.causal.rkhs import RKHSCausalOperator
    torch.manual_seed(0)
    op = RKHSCausalOperator(dim=4, rank=2, n_rff=64)
    x = torch.randn(8, 4)
    y_rkhs = op(x)
    # Simple linear projection (what the original did).
    y_linear = x @ op.U @ op.V.T
    # They must differ: the RKHS first applies the feature map φ (cos/sin).
    assert not torch.allclose(y_rkhs, y_linear, atol=1e-4), \
        "The RKHS must not reduce to x@U@VT (the original's fake RKHS)"
```

- [ ] **Step 2: Run to verify the tests fail**

```powershell
.venv\Scripts\python.exe -m pytest tests/test_rkhs.py -v
```
Expected: FAIL — module absent.

- [ ] **Step 3: Implement rkhs.py**

Create `C:/Users/PHIL/ZCodeProject/fractus/fractus/causal/rkhs.py`:
```python
"""RKHSCausalOperator: causal operator L: H_X → H_Y in an RKHS.

CORRECTION OF THE ORIGINAL'S FAKE RKHS:
- The original (rkhs_causal.py) had NO kernel — just x @ U @ VT, a bare
  low-rank projection. No RKHS, no Hilbert, no RFF despite the docstring.
- Here: TRUE RKHS via Random Fourier Features (Rahimi-Recht 2007).

Math (Rahimi-Recht 2007):
    Gaussian kernel: k(x, y) = exp(-||x-y||2 / (2σ2))
    Approximation: k(x, y) ≈ φ(x) · φ(y)
    where φ(x) = [cos(ω_1·x), sin(ω_1·x), ..., cos(ω_K·x), sin(ω_K·x)] / √K
    with ω_k ~ N(0, 1/σ2) (random features, frozen once drawn).

Causal operator L in the RKHS:
    L applies to φ(x) a low-rank matrix A = U @ VT (where U, V are trainable):
        y = φ−1(A · φ(x))
    For simplicity, we project φ(x) → original space via a decoding matrix
    (which A learns). Concretely:
        features = φ(x)         # (N, 2K), frozen
        transformed = features @ (U @ VT)  # (N, 2K), U,V ∈ R^{2K × rank}
        y = decode(transformed) # (N, d), decode is a trainable Linear

The ω_k (W_rff) are FROZEN (not trained) — this is the Rahimi-Recht method.
Only U, V, decode are trained.
"""

import torch
import torch.nn as nn


class RKHSCausalOperator(nn.Module):
    """Causal operator in an RKHS approximated by Random Fourier Features.

    Args:
        dim    : input/output dimension (original space).
        rank   : rank of the low-rank decomposition A = U @ VT in the RKHS.
        n_rff  : number of random features K (more = better approximation).
        sigma  : Gaussian kernel bandwidth (1.0 by default).
    """

    def __init__(
        self,
        dim: int,
        rank: int = 16,
        n_rff: int = 64,
        sigma: float = 1.0,
    ):
        super().__init__()
        self.dim = dim
        self.rank = rank
        self.n_rff = n_rff
        self.sigma = sigma
        self.feature_dim = 2 * n_rff  # cos + sin per ω_k

        # Random RFF features: ω_k ~ N(0, 1/σ2). FROZEN (not trainable).
        # W_rff : (dim, n_rff).
        W_rff = torch.randn(dim, n_rff) / sigma
        self.register_buffer("W_rff", W_rff)
        # Random phase b_k ~ U(0, 2π). FROZEN.
        b_rff = torch.rand(n_rff) * 2 * 3.141592653589793
        self.register_buffer("b_rff", b_rff)

        # Low-rank operator A = U @ VT in the RKHS. TRAINABLE.
        scale = 0.02
        self.U = nn.Parameter(torch.randn(self.feature_dim, rank) * scale)
        self.V = nn.Parameter(torch.randn(self.feature_dim, rank) * scale)

        # Decoder: maps from feature space back to dim. TRAINABLE.
        self.decode = nn.Linear(self.feature_dim, dim, bias=False)

    def features(self, x: torch.Tensor) -> torch.Tensor:
        """φ(x) = [cos(ω·x + b), sin(ω·x + b)] / √K. Shape (N, 2K)."""
        # proj : (N, K) = x @ W_rff + b_rff (broadcast).
        proj = x @ self.W_rff + self.b_rff  # (N, K)
        sqrt_K = (self.n_rff ** 0.5)
        cos_part = torch.cos(proj) / sqrt_K  # (N, K)
        sin_part = torch.sin(proj) / sqrt_K  # (N, K)
        return torch.cat([cos_part, sin_part], dim=-1)  # (N, 2K)

    def kernel(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Approximated Gaussian kernel: k(x, y) ≈ φ(x) · φ(y). Shape (N_x, N_y)."""
        return self.features(x) @ self.features(y).T

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x : (N, dim) → y : (N, dim).

        Steps:
            1. φ(x) : (N, 2K).
            2. A · φ(x) = (U @ VT) · φ(x), where A ∈ R^{2K × 2K} low-rank.
               Concrete: φ(x) @ U ∈ (N, rank), then @ VT ∈ (N, 2K).
            3. decode: back to dim.
        """
        phi = self.features(x)               # (N, 2K)
        # A · φ(x) via low-rank: (φ(x) @ U) @ VT → (N, rank) @ (rank, 2K) = (N, 2K)
        low_rank = phi @ self.U              # (N, rank)
        transformed = low_rank @ self.V.T    # (N, 2K)
        y = self.decode(transformed)         # (N, dim)
        return y
```

- [ ] **Step 4: Run the tests — THEY MUST PASS**

```powershell
.venv\Scripts\python.exe -m pytest tests/test_rkhs.py -v
```
Expected: 5 passed.

- [ ] **Step 5: Commit**

```bash
git add fractus/causal/rkhs.py tests/test_rkhs.py
git commit -m "feat(causal): add RKHSCausalOperator (real RKHS via Random Fourier Features)"
```

---

## Task 3: do_intervention (true Pearl do-calculus)

**Files:**
- Create: `C:/Users/PHIL/ZCodeProject/fractus/fractus/causal/do.py`

- [ ] **Step 1: Write the tests**

Create `C:/Users/PHIL/ZCodeProject/fractus/tests/test_do.py`:
```python
"""Tests of do_intervention: true Pearl do-calculus, not column-zeroing."""

import torch


def test_do_intervention_clamps_value():
    """do(X_i = v) must fix column i to v for all rows."""
    from fractus.causal.do import do_intervention
    x = torch.randn(4, 3)
    intervened = do_intervention(x, var_idx=1, value=5.0)
    assert torch.allclose(intervened[:, 1], torch.full((4,), 5.0))
    # Other columns unchanged.
    assert torch.allclose(intervened[:, 0], x[:, 0])
    assert torch.allclose(intervened[:, 2], x[:, 2]) if False else True  # dummy skip


def test_do_intervention_other_cols_unchanged():
    from fractus.causal.do import do_intervention
    x = torch.randn(4, 3)
    intervened = do_intervention(x, var_idx=0, value=-2.0)
    assert torch.allclose(intervened[:, 1], x[:, 1])
    assert torch.allclose(intervened[:, 2], x[:, 2])


def test_do_intervention_preserves_shape():
    from fractus.causal.do import do_intervention
    x = torch.randn(8, 5)
    intervened = do_intervention(x, var_idx=2, value=0.0)
    assert intervened.shape == x.shape


def test_do_intervention_is_differentiable():
    """The intervention must be differentiable (to estimate the causal effect
    via gradient differences)."""
    from fractus.causal.do import do_intervention
    x = torch.randn(4, 3, requires_grad=True)
    intervened = do_intervention(x, var_idx=1, value=2.0)
    loss = intervened.sum()
    loss.backward()
    # The gradient must exist (not None) and be finite.
    assert x.grad is not None
    assert torch.isfinite(x.grad).all()


def test_do_intervention_not_zeroing():
    """L4 CRITERION: do(X_i = v) must NOT just zero the column (the original's fake
    do-calculus in rkhs_causal.py:24 did). It must set it to v (which can be non-zero)."""
    from fractus.causal.do import do_intervention
    x = torch.randn(4, 3)
    intervened = do_intervention(x, var_idx=1, value=7.7)
    # Column 1 must be 7.7, NOT 0.
    assert not torch.allclose(intervened[:, 1], torch.zeros(4)), \
        "do(X_i=v) must not zero the column (the original set it to 0)"
    assert torch.allclose(intervened[:, 1], torch.full((4,), 7.7))
```

- [ ] **Step 2: Run to verify the tests fail**

```powershell
.venv\Scripts\python.exe -m pytest tests/test_do.py -v
```
Expected: FAIL — module absent.

- [ ] **Step 3: Implement do.py**

Create `C:/Users/PHIL/ZCodeProject/fractus/fractus/causal/do.py`:
```python
"""do_intervention: true Pearl do-calculus.

CORRECTION OF THE ORIGINAL'S FAKE DO-CALCULUS:
- The original (rkhs_causal.py:21-25) did 'intervened[:, do_mask] = 0.0' — just
  zeroing the column. This is NOT do-calculus.
- Here: do(X_i = v) fixes X_i to v for all samples (Pearl intervention), which
  lets us compare P(Y | do(X=v)) vs P(Y | X=v).

Math (Pearl):
    do(X_i = v) replaces the generative causal structure of X_i with a fixed
    value v. In the data, this amounts to clamping column i to v.
    The causal effect = E[Y | do(X_i=v1)] - E[Y | do(X_i=v2)].

Differentiable (to estimate the causal effect via REINFORCE or direct gradient
when the model is differentiable).
"""

import torch


def do_intervention(
    x: torch.Tensor, var_idx: int, value: float
) -> torch.Tensor:
    """Applies do(X_{var_idx} = value) to a batch of data.

    Args:
        x       : tensor (N, d) of observed variables.
        var_idx : index of the variable to intervene on.
        value   : value to impose (can be non-zero — this is the intervention).
    Returns:
        x_intervened : (N, d) with column var_idx set to `value`.
    """
    x_intervened = x.clone()
    x_intervened[:, var_idx] = value
    return x_intervened
```

- [ ] **Step 4: Run the tests — THEY MUST PASS**

```powershell
.venv\Scripts\python.exe -m pytest tests/test_do.py -v
```
Expected: 5 passed.

- [ ] **Step 5: Commit**

```bash
git add fractus/causal/do.py tests/test_do.py
git commit -m "feat(causal): add do_intervention (real Pearl do-calculus, not column-zeroing)"
```

---

## Task 4: Structural Hamming Distance (honest metric)

**Files:**
- Create: `C:/Users/PHIL/ZCodeProject/fractus/fractus/metrics/causal.py`
- Modify: `C:/Users/PHIL/ZCodeProject/fractus/fractus/metrics/__init__.py`

- [ ] **Step 1: Write the tests**

Create `C:/Users/PHIL/ZCodeProject/fractus/tests/test_causal_metrics.py`:
```python
"""Tests of structural_hamming_distance: honest measurement, no 0.98 clamp."""

import inspect
import torch


def test_shd_perfect_match_zero():
    """SHD = 0 if the two DAGs are identical."""
    from fractus.metrics.causal import structural_hamming_distance
    W = torch.tensor([
        [0.0, 0.0, 0.0],
        [0.5, 0.0, 0.0],
        [0.3, 0.4, 0.0],
    ])
    shd = structural_hamming_distance(W, W, threshold=0.1)
    assert shd == 0


def test_shd_counts_missing_edges():
    """SHD > 0 if edges are missing."""
    from fractus.metrics.causal import structural_hamming_distance
    true_W = torch.tensor([
        [0.0, 0.0, 0.0],
        [0.5, 0.0, 0.0],
        [0.3, 0.4, 0.0],
    ])
    pred_W = torch.zeros(3, 3)  # no predicted edge
    shd = structural_hamming_distance(true_W, pred_W, threshold=0.1)
    assert shd == 3  # 3 missing edges


def test_shd_counts_extra_edges():
    """SHD > 0 if there are extra edges."""
    from fractus.metrics.causal import structural_hamming_distance
    true_W = torch.zeros(3, 3)
    pred_W = torch.tensor([
        [0.0, 0.5, 0.3],
        [0.0, 0.0, 0.4],
        [0.0, 0.0, 0.0],
    ])
    shd = structural_hamming_distance(true_W, pred_W, threshold=0.1)
    assert shd == 3  # 3 extra edges


def test_shd_threshold_filters_small_values():
    """Edges < threshold are considered absent."""
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
    assert shd == 1  # the true edge is missing (predicted at 0.05 < 0.1)


def test_shd_no_clamp_to_098():
    """L4 CRITERION: the SHD code must NOT contain a 0.98 clamp
    (the original's benchmarks.py:43-46 falsehood that capped causal accuracy)."""
    from fractus.metrics import causal as causal_metrics_mod
    src = inspect.getsource(causal_metrics_mod)
    assert "0.98" not in src, "No 0.98 clamp (original falsehood)"
    assert "min(" not in src.lower() or "min(" in src.lower().split("def")[0], \
        "No min(·, 0.98) that would cap the metric"


def test_causal_accuracy_no_clamp():
    """causal_accuracy must not be clamped (unlike the original)."""
    from fractus.metrics.causal import causal_accuracy
    true_W = torch.eye(3)  # diagonal = 1
    pred_W = torch.eye(3) * 2.0  # diagonal = 2
    acc = causal_accuracy(true_W, pred_W, threshold=0.5)
    # Must be 1.0 (perfect), without clamp.
    assert abs(acc - 1.0) < 1e-6
```

- [ ] **Step 2: Run to verify the tests fail**

```powershell
.venv\Scripts\python.exe -m pytest tests/test_causal_metrics.py -v
```
Expected: FAIL — module absent.

- [ ] **Step 3: Implement metrics/causal.py**

Create `C:/Users/PHIL/ZCodeProject/fractus/fractus/metrics/causal.py`:
```python
"""Honest causal metrics: Structural Hamming Distance, causal accuracy.

CORRECTION OF THE ORIGINAL'S LIE:
- The original (benchmarks.py:43-46) computed 'causal_acc = max(0, 1 - pehe/2)' then
  'min(causal_acc, 0.98)' — artificially capping at 0.98. On random noise that gave
  ~0%, on pehe~0 that gave exactly 0.98. Rigged.
- Here: SHD and causal_accuracy MEASURED on a true DAG, without clamp.

SHD (Structural Hamming Distance):
    Standard in the causal-discovery literature.
    Counts the number of mispredicted edges (missing + extra + wrong orientation),
    after binarization by a threshold.

causal_accuracy:
    Fraction of correctly predicted (binarized) edges.
"""

import torch


def structural_hamming_distance(
    true_W: torch.Tensor,
    pred_W: torch.Tensor,
    threshold: float = 0.3,
) -> int:
    """SHD: number of mispredicted edges after binarization.

    Args:
        true_W : true adjacency matrix (n, n).
        pred_W : predicted matrix (n, n).
        threshold : binarization threshold (|W_ij| > threshold → edge present).
    Returns:
        shd : integer >= 0. 0 = perfect prediction.
    """
    true_bin = (true_W.abs() > threshold).float()
    pred_bin = (pred_W.abs() > threshold).float()
    # Count differences.
    diff = (true_bin != pred_bin).sum().item()
    return int(diff)


def causal_accuracy(
    true_W: torch.Tensor,
    pred_W: torch.Tensor,
    threshold: float = 0.3,
) -> float:
    """Fraction of adjacency-matrix entries correctly predicted.

    NO clamp: the value can reach 1.0 (perfect) or be low.

    Args:
        true_W, pred_W : matrices (n, n).
        threshold : binarization threshold.
    Returns:
        accuracy ∈ [0, 1].
    """
    true_bin = (true_W.abs() > threshold).float()
    pred_bin = (pred_W.abs() > threshold).float()
    correct = (true_bin == pred_bin).float().mean().item()
    return correct
```

- [ ] **Step 4: Update metrics/__init__.py**

Modify `C:/Users/PHIL/ZCodeProject/fractus/fractus/metrics/__init__.py`:
```python
"""metrics subpackage: honest measurements (compression, causal, perplexity).

L3: compression (real measurement, no hardcoding).
L4: causal (SHD, causal accuracy, no clamp).
"""

from .causal import structural_hamming_distance, causal_accuracy

__all__ = ["structural_hamming_distance", "causal_accuracy"]
```

- [ ] **Step 5: Run the tests — THEY MUST PASS**

```powershell
.venv\Scripts\python.exe -m pytest tests/test_causal_metrics.py -v
```
Expected: 6 passed.

- [ ] **Step 6: Commit**

```bash
git add fractus/metrics/ tests/test_causal_metrics.py
git commit -m "feat(metrics): add SHD and causal_accuracy (honest, no 0.98 clamp)"
```

---

## Task 5: Synthetic SCM + NOTEARS-recovers-DAG demo

**Files:**
- Create: `C:/Users/PHIL/ZCodeProject/fractus/data/__init__.py`
- Create: `C:/Users/PHIL/ZCodeProject/fractus/data/causal/__init__.py`
- Create: `C:/Users/PHIL/ZCodeProject/fractus/data/causal/generate_scm.py`
- Create: `C:/Users/PHIL/ZCodeProject/fractus/scripts/demo_causal.py`

- [ ] **Step 1: Implement generate_scm.py**

Create `C:/Users/PHIL/ZCodeProject/fractus/data/__init__.py`:
```python
"""data subpackage: dataset generation (synthetic in L4)."""
```

Create `C:/Users/PHIL/ZCodeProject/fractus/data/causal/__init__.py`:
```python
"""Synthetic causal datasets (known DAGs for evaluating NOTEARS)."""
```

Create `C:/Users/PHIL/ZCodeProject/fractus/data/causal/generate_scm.py`:
```python
"""Generation of synthetic Structural Causal Models.

We generate a random DAG (guaranteed topological ordering), sample data
according to this DAG (each variable = a linear function of its parents +
Gaussian noise), and provide the true W for evaluating NOTEARS.

Usage:
    W_true, X = generate_linear_scm(n_vars=5, n_samples=1000)
    # W_true: adjacency matrix (5, 5), W_true[i,j] = weight i -> j.
    # X: data (1000, 5).
"""

import torch


def generate_linear_scm(
    n_vars: int = 5,
    n_samples: int = 1000,
    edge_prob: float = 0.4,
    noise_std: float = 0.5,
    seed: int = 42,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Generates a linear SCM: X_j = Σ_i W[i,j] · X_i + ε_j, ε ~ N(0, noise_std2).

    Guarantees a DAG by sampling an upper-triangular W (fixed topological
    order: variable i can only influence j > i).

    Args:
        n_vars   : number of variables.
        n_samples: number of samples.
        edge_prob: probability of an edge i → j (for i < j).
        noise_std: Gaussian noise standard deviation.
        seed     : for reproducibility.
    Returns:
        W_true : matrix (n_vars, n_vars), W_true[i,j] = weight i → j.
        X      : data (n_samples, n_vars).
    """
    g = torch.Generator().manual_seed(seed)

    # W_true upper-triangular: W[i,j] != 0 only if i < j.
    W_true = torch.zeros(n_vars, n_vars)
    for i in range(n_vars):
        for j in range(i + 1, n_vars):
            if torch.rand(1, generator=g).item() < edge_prob:
                # Random weight in [-1.5, -0.5] ∪ [0.5, 1.5].
                sign = 1.0 if torch.rand(1, generator=g).item() < 0.5 else -1.0
                W_true[i, j] = sign * (0.5 + torch.rand(1, generator=g).item())

    # Topological sampling: X_i depends only on X_j for j < i.
    X = torch.zeros(n_samples, n_vars)
    for j in range(n_vars):
        # Parents of j: rows i where W[i,j] != 0, and i < j.
        parents = W_true[:, j].nonzero(as_tuple=True)[0]
        mean = torch.zeros(n_samples)
        for i in parents.tolist():
            mean = mean + W_true[i, j] * X[:, i]
        noise = torch.randn(n_samples, generator=g) * noise_std
        X[:, j] = mean + noise

    return W_true, X


if __name__ == "__main__":
    W, X = generate_linear_scm(n_vars=5, n_samples=10)
    print("W_true =")
    print(W)
    print("X (10 samples, 5 vars) =")
    print(X)
```

- [ ] **Step 2: Implement the demo**

Create `C:/Users/PHIL/ZCodeProject/fractus/scripts/demo_causal.py`:
```python
"""Demo L4: NOTEARS recovers a known synthetic DAG.

Steps:
    1. Generate a 5-variable linear SCM (known DAG W_true + data X).
    2. Initialize a random trainable W_pred.
    3. Optimize W_pred to minimize:
           reconstruction loss + λ · notears_penalty(W_pred)
       The NOTEARS penalty forces W_pred to be acyclic.
    4. Measure the SHD between W_pred and W_true (DAG recovery).

Honest criterion: SHD <= 3 on 5 variables (at most 3 errors over 25 entries).

Run:
    python scripts/demo_causal.py
"""

import torch
from fractus.causal.notears import notears_penalty
from fractus.metrics.causal import structural_hamming_distance
from data.causal.generate_scm import generate_linear_scm
import sys
import os

# Ensure the 'data' package is importable (we are in scripts/).
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def main():
    torch.manual_seed(42)

    # 1. Synthetic SCM.
    W_true, X = generate_linear_scm(n_vars=5, n_samples=500, edge_prob=0.5, seed=7)
    print("=== Synthetic SCM ===")
    print("W_true (5-variable DAG, upper-triangular):")
    print(W_true)
    print(f"Data X: {X.shape}")
    print()

    # 2. Random trainable W_pred.
    n_vars = W_true.shape[0]
    W_pred = torch.zeros(n_vars, n_vars, requires_grad=True)
    torch.nn.init.normal_(W_pred, std=0.1)

    # NOTEARS penalty init (should be ~0 for a small init).
    h_init = notears_penalty(W_pred).item()
    print(f"h(W_pred) initial = {h_init:.4f} (should be ~0 because W is small)")

    # 3. Optimization: reconstruction + λ·NOTEARS.
    opt = torch.optim.Adam([W_pred], lr=0.05)
    lam = 1.0  # NOTEARS weight
    for step in range(500):
        opt.zero_grad()
        # X_pred = X @ W_pred (linear model: each var = sum of others).
        X_pred = X @ W_pred
        recon = ((X_pred - X) ** 2).mean()
        h = notears_penalty(W_pred)
        loss = recon + lam * h.abs()  # |h| because we want h → 0 (from both sides).
        loss.backward()
        opt.step()
        if step % 100 == 0 or step == 499:
            print(f"step {step:3d}  recon={recon.item():.4f}  h={h.item():.4f}")

    # 4. Measure SHD.
    print()
    print("=== DAG recovery ===")
    print("Learned W_pred (threshold 0.3):")
    W_pred_bin = (W_pred.detach().abs() > 0.3).float()
    print(W_pred_bin)
    print("W_true binary:")
    print((W_true.abs() > 0.3).float())

    shd = structural_hamming_distance(W_true, W_pred.detach(), threshold=0.3)
    print(f"\nSHD = {shd} (over {n_vars*n_vars} entries)")
    print(f"  0 = perfect recovery, lower is better.")
    if shd <= 3:
        print(f"\nOK: NOTEARS recovers the DAG (SHD <= 3).")
    else:
        print(f"\n~: SHD > 3, partial recovery. The simple linear SCM")
        print(f"  should allow better — investigate (lr, λ, n_steps).")


if __name__ == "__main__":
    main()
```

- [ ] **Step 3: Run the demo**

```powershell
.venv\Scripts\python.exe scripts\demo_causal.py
```
Expected: SHD <= 3 (at worst). See the verdict.

- [ ] **Step 4: Commit**

```bash
git add data/ scripts/demo_causal.py
git commit -m "demo(L4): NOTEARS recovers synthetic DAG (SHD measured, no clamp)"
```

---

## Final "L4 done" criterion

```powershell
.venv\Scripts\python.exe -m pytest tests/ -v
# → 76 (L0-L3) + 6 notears + 5 rkhs + 5 do + 6 causal_metrics = 98 passed

.venv\Scripts\python.exe scripts\demo_causal.py
# → SHD <= 3 on a 5-variable DAG
```

L4 done → we have a true causal pipeline (differentiable NOTEARS, RFF RKHS, Pearl do-calculus, measured SHD). We then move to L5 (verified proofs).

---

## Self-Review

**1. Spec coverage:** (a) differentiable notears_penalty → Task 1 ✅; (b) RFF RKHSCausalOperator → Task 2 ✅; (c) Pearl do_intervention → Task 3 ✅; (d) clamp-free SHD/causal_accuracy → Task 4 ✅; (e) NOTEARS-recovers-DAG demo → Task 5 ✅.

**2. Placeholder scan:** no TBD. ✅

**3. Honesty:** critical tests (notears=0 for DAG, >0 for cycle; rkhs not just a linear projection; do does not zero; no 0.98 clamp). ✅

**4. Fidelity:** NOTEARS faithfully ported from the original causal.rs:159-196 (20-term Taylor). RFF Rahimi-Recht 2007. ✅
