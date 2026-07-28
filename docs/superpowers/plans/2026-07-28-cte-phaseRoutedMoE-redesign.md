# CTE → PhaseRoutedMoE Redesign Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the `ContinuousThoughtEngine` route its MoE through the differentiable `PhaseRoutedMoE` component (extended to support low-rank experts), so expert weights receive gradients during Phase 3 — fixing the frozen-expert defect that confounded the EDT AB-test. Then re-run the AB-test for a valid A-vs-B verdict.

**Architecture:** Extend `PhaseRoutedMoE` with an `expert_rank` flag (dense weights `(E,D,F)` when None, LoRA-style low-rank factors `U/V/scale` when set). Refactor `ContinuousThoughtEngine` to instantiate `self.moe = PhaseRoutedMoE(...)` instead of its ad-hoc `experts_w1`/`experts_w2` ModuleList + hand-rolled routing. Adapt `ablib.py` Phase-1 helpers to train individual experts on the shared low-rank parameters via gradient-row masking. Re-run all three AB arms.

**Tech Stack:** PyTorch 2.9 (CPU), pytest, Windows Store CPython 3.13 (`$PY`).

**Reference spec:** `docs/superpowers/specs/2026-07-28-cte-phaseRoutedMoE-redesign.md`. Every design decision lives there; this plan only encodes it.

**API facts (verified — do not re-derive):**
- `PhaseRoutedMoE.__init__(d_model, n_experts, top_k, kappa=4.0, temperature=1.0, d_ff=64)` in `fractus/nn/moe.py`. `forward(h, phases) -> (output (B,L,D), lb_loss scalar)`. Dense weights `w1 (E,D,F)`, `b1 (E,F)`, `w2 (E,F,D)`, `b2 (E,D)`. Routing: von Mises gate on Farey phases, top-k, load-balance loss. Dense path used when `n_experts ≤ 2·top_k`.
- `ContinuousThoughtEngine.__init__` signature: `vocab_size, d_model, n_heads, d_head, n_levels, n_oscillators, coupling_rank, n_experts, top_k, expert_d_ff, siren_rank`. 13M config: `d_model=128, n_heads=2, d_head=64, n_levels=2, n_oscillators=8, coupling_rank=4, n_experts=4, top_k=2, expert_d_ff=128, siren_rank=32`.
- `tick(observation) -> (logits, confidence)` and `tick_chunk(observations) -> logits` are the two forward entry points; both currently hand-roll the MoE routing at continuous_engine.py:206-231 (tick) and 303-329 (tick_chunk), reading `e._cached_W` directly (the detached buffer).
- `OnlineTrainer` (fractus/train/online.py) uses pure cross-entropy; does not reference `lb_loss`, `force_refresh`, or expert params.
- 13M uses dense MoE path (`E=4 ≤ 2·K=4`), so the low-rank dense-only decision (spec D2) covers the AB test.

---

## File structure

| Path | Responsibility |
|---|---|
| `fractus/nn/moe.py` | Extend `PhaseRoutedMoE` with low-rank expert mode |
| `fractus/continuous_engine.py` | Replace ad-hoc MoE with `self.moe = PhaseRoutedMoE(...)` |
| `experiments/edt_ab/ablib.py` | Adapt Phase-1 helpers to shared low-rank params via gradient masking |
| `tests/test_moe.py` | Add low-rank tests (shape, backward-every-param, dense non-regression) |
| `tests/test_continuous_engine_grad.py` | New — regression test for the defect (experts receive gradient) |
| `tests/test_ablib.py` | Verify Phase-1/arm tests still pass (no exact-value assertions) |

---

## Task 1: Low-rank expert weights in `PhaseRoutedMoE`

**Files:**
- Modify: `fractus/nn/moe.py:46-77` (constructor) and `135-146` (`_dense_expert_forward`)
- Test: `tests/test_moe.py`

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_moe.py`:
```python
def test_moe_lowrank_output_shape():
    """Low-rank mode: same output shape as dense."""
    from fractus.nn.moe import PhaseRoutedMoE
    moe = PhaseRoutedMoE(d_model=16, n_experts=4, top_k=2, kappa=4.0, d_ff=32, expert_rank=8)
    h = torch.randn(2, 8, 16)
    phases = torch.rand(2, 8, 4) * 2 * math.pi
    out, lb_loss = moe(h, phases)
    assert out.shape == (2, 8, 16)
    assert lb_loss.dim() == 0


def test_moe_lowrank_backward_every_param():
    """Low-rank mode: gradient reaches U1, V1, U2, V2, scale1, scale2, b1, b2."""
    from fractus.nn.moe import PhaseRoutedMoE
    moe = PhaseRoutedMoE(d_model=16, n_experts=4, top_k=2, kappa=4.0, d_ff=32, expert_rank=8)
    h = torch.randn(2, 8, 16)
    phases = torch.rand(2, 8, 4) * 2 * math.pi
    out, lb_loss = moe(h, phases)
    loss = out.pow(2).mean() + 0.1 * lb_loss
    loss.backward()
    for name, p in moe.named_parameters():
        assert p.requires_grad, f"{name} should requires_grad=True"
        assert p.grad is not None, f"{name} received no gradient"
        assert torch.isfinite(p.grad).all(), f"{name} has non-finite grad"
        assert p.grad.abs().sum().item() > 0, f"{name} received zero gradient"


def test_moe_lowrank_has_expected_params():
    """Low-rank mode exposes U1/V1/U2/V2/scale factors, not w1/w2."""
    from fractus.nn.moe import PhaseRoutedMoE
    moe = PhaseRoutedMoE(d_model=16, n_experts=4, top_k=2, d_ff=32, expert_rank=8)
    names = {n for n, _ in moe.named_parameters()}
    assert "U1" in names and "V1" in names and "U2" in names and "V2" in names
    assert "scale1" in names and "scale2" in names
    assert "w1" not in names and "w2" not in names
    # Shapes: U1 (E, F, r), V1 (E, D, r), U2 (E, D, r), V2 (E, F, r).
    assert moe.U1.shape == (4, 32, 8)
    assert moe.V1.shape == (4, 16, 8)
    assert moe.U2.shape == (4, 16, 8)
    assert moe.V2.shape == (4, 32, 8)


def test_moe_dense_still_has_dense_params():
    """Dense mode (expert_rank=None) unchanged: w1/w2 present, no U/V."""
    from fractus.nn.moe import PhaseRoutedMoE
    moe = PhaseRoutedMoE(d_model=16, n_experts=4, top_k=2, d_ff=32)  # no expert_rank
    names = {n for n, _ in moe.named_parameters()}
    assert "w1" in names and "w2" in names
    assert "U1" not in names
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `$PY -m pytest tests/test_moe.py -k "lowrank or dense_still" -v`
Expected: FAIL — `TypeError: __init__() got an unexpected keyword argument 'expert_rank'`.

- [ ] **Step 3: Extend `PhaseRoutedMoE.__init__`**

In `fractus/nn/moe.py`, edit the constructor (currently lines 46-77). Add `expert_rank: int | None = None` to the signature (after `d_ff`), store `self.expert_rank = expert_rank`, and replace the weight-construction block (lines 71-77) with:

```python
        self.d_ff = d_ff
        self.expert_rank = expert_rank

        # Expert phases (Farey precomputation, off-graph).
        phases = expert_phases(n_experts)
        self.register_buffer("expert_phases", torch.tensor(phases, dtype=torch.float32))

        if expert_rank is None:
            # Dense experts: W1 (E,D,F), W2 (E,F,D). Xavier uniform.
            scale1 = math.sqrt(2.0 / d_model)
            scale2 = math.sqrt(2.0 / d_ff)
            self.w1 = nn.Parameter(torch.empty(n_experts, d_model, d_ff).uniform_(-scale1, scale1))
            self.b1 = nn.Parameter(torch.zeros(n_experts, d_ff))
            self.w2 = nn.Parameter(torch.empty(n_experts, d_ff, d_model).uniform_(-scale2, scale2))
            self.b2 = nn.Parameter(torch.zeros(n_experts, d_model))
        else:
            # Low-rank (LoRA-style): W1 ≈ scale1 · U1 @ V1ᵀ, W2 ≈ scale2 · U2 @ V2ᵀ.
            # U1 (E, F, r), V1 (E, D, r); U2 (E, D, r), V2 (E, F, r).
            r = expert_rank
            su1 = math.sqrt(2.0 / (d_ff + r))
            sv1 = math.sqrt(2.0 / (d_model + r))
            su2 = math.sqrt(2.0 / (d_model + r))
            sv2 = math.sqrt(2.0 / (d_ff + r))
            self.U1 = nn.Parameter(torch.empty(n_experts, d_ff, r).uniform_(-su1, su1))
            self.V1 = nn.Parameter(torch.empty(n_experts, d_model, r).uniform_(-sv1, sv1))
            self.U2 = nn.Parameter(torch.empty(n_experts, d_model, r).uniform_(-su2, su2))
            self.V2 = nn.Parameter(torch.empty(n_experts, d_ff, r).uniform_(-sv2, sv2))
            self.scale1 = nn.Parameter(torch.ones(n_experts, 1, 1))
            self.scale2 = nn.Parameter(torch.ones(n_experts, 1, 1))
            self.b1 = nn.Parameter(torch.zeros(n_experts, d_ff))
            self.b2 = nn.Parameter(torch.zeros(n_experts, d_model))
```

- [ ] **Step 4: Extend `_dense_expert_forward` for low-rank**

Replace `_dense_expert_forward` (currently lines 135-146) with a version that branches on `self.expert_rank`:

```python
    def _dense_expert_forward(self, h: torch.Tensor) -> torch.Tensor:
        """DENSE forward: compute ALL E experts. Used when n_experts <= 2*top_k.

        h: (B, L, d_model) → outputs of all experts (B, L, E, d_model).
        """
        B, L, D = h.shape
        if self.expert_rank is None:
            h1 = torch.einsum("bld,edf->blef", h, self.w1) + self.b1.view(1, 1, self.n_experts, self.d_ff)
            h1_act = _gelu(h1)
            out = torch.einsum("blef,efd->bled", h1_act, self.w2) + self.b2.view(1, 1, self.n_experts, self.d_model)
            return out
        # Low-rank: W1 ≈ scale1 · U1 @ V1ᵀ  →  h1 = scale1 · (h @ V1) @ U1ᵀ.
        # h: (B,L,D); V1: (E,D,r) → hV1: (B,L,E,r); U1ᵀ: (E,r,F) → h1: (B,L,E,F).
        hV1 = torch.einsum("bld,edr->bler", h, self.V1)            # (B,L,E,r)
        h1 = self.scale1 * torch.einsum("bler,erf->blef", hV1, self.U1) + self.b1.view(1, 1, self.n_experts, self.d_ff)
        h1_act = _gelu(h1)
        # W2 ≈ scale2 · U2 @ V2ᵀ  →  out = scale2 · (h1_act @ V2) @ U2ᵀ.
        hV2 = torch.einsum("blef,efr->bler", h1_act, self.V2)      # (B,L,E,r)
        out = self.scale2 * torch.einsum("bler,erd->bled", hV2, self.U2) + self.b2.view(1, 1, self.n_experts, self.d_model)
        return out
```

- [ ] **Step 5: Make sparse path warn-and-fallback for low-rank**

In `forward` (around the existing `if self.n_experts > 2 * self.top_k:` branch), wrap so that low-rank mode always uses dense. Edit the dispatch to:

```python
        # Adaptive: dense when small E (einsum wins on CPU), sparse when large E.
        # Low-rank mode is dense-only in this iteration (spec D2).
        use_sparse = (self.n_experts > 2 * self.top_k) and (self.expert_rank is None)
        if self.expert_rank is not None and self.n_experts > 2 * self.top_k:
            import warnings
            warnings.warn(
                "PhaseRoutedMoE: low-rank sparse dispatch not implemented; "
                "falling back to dense forward.",
                stacklevel=2,
            )
        if use_sparse:
            topk_out = self._sparse_expert_forward(h, topk_idx)
        else:
            all_out = self._dense_expert_forward(h)
            idx_exp = topk_idx.unsqueeze(-1).expand(-1, -1, -1, self.d_model)
            topk_out = torch.gather(all_out, dim=2, index=idx_exp)
```

- [ ] **Step 6: Run the tests to verify they pass**

Run: `$PY -m pytest tests/test_moe.py -v`
Expected: PASS — all existing dense tests (6) plus the 4 new low-rank tests.

- [ ] **Step 7: Commit**

```bash
git add fractus/nn/moe.py tests/test_moe.py
git commit -m "feat(nn): PhaseRoutedMoE low-rank expert mode (LoRA-style)"
```

---

## Task 2: New `ContinuousThoughtEngine` MoE wiring

**Files:**
- Modify: `fractus/continuous_engine.py:94-112` (constructor MoE block), `206-231` (tick MoE section), `303-329` (tick_chunk MoE section), top imports
- Test: `tests/test_continuous_engine_grad.py` (new)

- [ ] **Step 1: Write the failing regression test**

Create `tests/test_continuous_engine_grad.py`:
```python
"""Regression test for the frozen-expert Phase 3 defect.

Before the fix, tick_chunk read the experts' detached _cached_W buffer
directly, so expert params received no gradient. This test fails on the
old code and must pass on the refactored engine.
"""
import torch
from fractus.continuous_engine import ContinuousThoughtEngine


def _build_13m():
    return ContinuousThoughtEngine(
        vocab_size=50257, d_model=128, n_heads=2, d_head=64,
        n_levels=2, n_oscillators=8, coupling_rank=4,
        n_experts=4, top_k=2, expert_d_ff=128, siren_rank=32,
    )


def test_cte_experts_receive_gradient():
    """Every expert parameter gets a finite, non-zero gradient after tick_chunk + backward."""
    eng = _build_13m()
    eng.reset_thought(batch_size=1)
    tokens = torch.randint(0, eng.vocab_size, (1, 16))
    logits = eng.tick_chunk(tokens)
    loss = logits.pow(2).mean()
    loss.backward()

    # The MoE is at eng.moe. Collect its parameters.
    moe_params = list(eng.moe.named_parameters())
    assert len(moe_params) > 0, "engine.moe has no parameters"
    for name, p in moe_params:
        assert p.requires_grad, f"{name} should requires_grad=True"
        assert p.grad is not None, f"{name} received NO gradient (frozen-expert bug)"
        assert torch.isfinite(p.grad).all(), f"{name} has non-finite grad"
        assert p.grad.abs().sum().item() > 0, f"{name} received zero gradient"


def test_cte_has_moe_attribute():
    """The CTE must use a PhaseRoutedMoE (not the old experts_w1/experts_w2)."""
    from fractus.nn.moe import PhaseRoutedMoE
    eng = _build_13m()
    assert isinstance(eng.moe, PhaseRoutedMoE)
    assert not hasattr(eng, "experts_w1"), "old ad-hoc MoE should be removed"
    assert not hasattr(eng, "experts_w2")
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `$PY -m pytest tests/test_continuous_engine_grad.py -v`
Expected: FAIL — `AttributeError: 'ContinuousThoughtEngine' object has no attribute 'moe'` (and `experts_w1` still exists).

- [ ] **Step 3: Replace the constructor MoE block**

In `fractus/continuous_engine.py`, replace the block at lines 94-112 (from the `# 3. MoE` comment through `self.norm_moe = nn.LayerNorm(d_model)`) with:

```python
        # 3. MoE (transforms the thought, routed by Kuramoto phases).
        #    Uses the documented PhaseRoutedMoE (L2b): von Mises gate on Farey
        #    phases, top-k, load-balance loss, end-to-end differentiable.
        #    Low-rank experts (LoRA-style) when siren_rank > 0, dense otherwise.
        self.n_experts = n_experts
        self.top_k = top_k
        self.expert_d_ff = expert_d_ff
        from .nn.moe import PhaseRoutedMoE
        self.moe = PhaseRoutedMoE(
            d_model=d_model, n_experts=n_experts, top_k=top_k,
            kappa=4.0, d_ff=expert_d_ff,
            expert_rank=(siren_rank if siren_rank else None),
        )
        self.norm_moe = nn.LayerNorm(d_model)
        self.register_buffer("last_lb_loss", torch.tensor(0.0))
```

Also remove the now-unused imports at the top of the file: `from .nn.farey import expert_phases` and `from .nn.cached_siren import CachedStructuredSirenLinear` (if present). Leave `FractalLinearAttention`, `KuramotoLayer`, `elu_plus_one` imports.

- [ ] **Step 4: Rewrite the MoE section of `tick`**

In `tick` (currently around lines 206-231 — the block starting with `# 3. MoE: transform the thought, routed by phases.` through `h = h + moe_out.unsqueeze(1)`), replace with:

```python
        # 3. MoE: transform the thought, routed by Kuramoto phases.
        h_flat = h[:, 0, :]                              # (B, d_model)
        h_moe = self.norm_moe(h_flat).unsqueeze(1)       # (B, 1, d_model)
        phases_in = theta[:, 0:1, :]                     # (B, 1, n_oscillators)
        moe_out, lb_loss = self.moe(h_moe, phases_in)    # moe_out: (B, 1, d_model)
        self.last_lb_loss = lb_loss.detach()
        h = h + moe_out
```

`theta` here is the Kuramoto phase tensor already computed earlier in `tick` (the existing `self.kuramoto_phases = theta.detach()` line). Keep that computation intact; only the MoE-application block changes.

- [ ] **Step 5: Rewrite the MoE section of `tick_chunk`**

In `tick_chunk` (currently lines 303-329 — from `# 3. MoE` / `h_moe = self.norm_moe(h)` through `h = h + moe_out`), replace with:

```python
        # 3. MoE: transform the chunk, routed by the last-position Kuramoto phase.
        h_moe = self.norm_moe(h)                         # (B, C, d_model)
        # Use the last position's phase to route the whole chunk (as the original did).
        phases_last = theta[:, -1:, :]                   # (B, 1, n_oscillators)
        phases_in = phases_last.expand(-1, h_moe.shape[1], -1)  # (B, C, n_oscillators)
        moe_out, lb_loss = self.moe(h_moe, phases_in)    # (B, C, d_model)
        self.last_lb_loss = lb_loss.detach()
        h = h + moe_out
```

Note: `theta` is the Kuramoto phase tensor computed earlier in `tick_chunk` (around line 297-303 in the current code) — keep that computation intact; only the MoE-application block changes.

- [ ] **Step 6: Run the regression test to verify it passes**

Run: `$PY -m pytest tests/test_continuous_engine_grad.py -v`
Expected: PASS — both `test_cte_experts_receive_gradient` and `test_cte_has_moe_attribute`.

- [ ] **Step 7: Run the full moe + engine test surface to check for regressions**

Run: `$PY -m pytest tests/test_moe.py tests/test_continuous_engine_grad.py -v`
Expected: PASS (all moe tests + the 2 new engine tests).

- [ ] **Step 8: Commit**

```bash
git add fractus/continuous_engine.py tests/test_continuous_engine_grad.py
git commit -m "refactor(cte): route MoE through PhaseRoutedMoE (experts now trainable)"
```

---

## Task 3: Adapt `ablib.py` Phase-1 helpers to shared low-rank params

**Files:**
- Modify: `experiments/edt_ab/ablib.py:78-132` (`_expert_forward`, `_eval_expert_mse`, `_train_one_expert`), `230-238` (phase3_joint force_refresh loop)
- Test: `tests/test_ablib.py` (existing Phase-1 tests must still pass)

- [ ] **Step 1: Verify the existing Phase-1 tests fail against the refactored engine**

Run: `$PY -m pytest tests/test_ablib.py::test_phase1_shared_reduces_mse_on_bank tests/test_ablib.py::test_phase1_partitioned_trains_each_expert_on_own_bank -v`
Expected: FAIL — `AttributeError: 'ContinuousThoughtEngine' object has no attribute 'experts_w1'` (the old ablib code references the removed attributes).

- [ ] **Step 2: Rewrite `_expert_forward` for the new MoE**

In `experiments/edt_ab/ablib.py`, replace `_expert_forward` (around lines 78-86) with:

```python
def _expert_forward(engine: ContinuousThoughtEngine, idx: int,
                    h: torch.Tensor) -> torch.Tensor:
    """Output of expert idx on h: gelu(W1(h)) → W2.

    Works for both dense and low-rank PhaseRoutedMoE. h: (..., d_model).
    """
    moe = engine.moe
    if moe.expert_rank is None:
        # Dense: slice the shared weight parameter at expert idx.
        w1 = moe.w1[idx]            # (d_model, d_ff)
        b1 = moe.b1[idx]            # (d_ff,)
        w2 = moe.w2[idx]            # (d_ff, d_model)
        b2 = moe.b2[idx]            # (d_model,)
        h1 = h @ w1 + b1
        h1_act = torch.nn.functional.gelu(h1)
        return h1_act @ w2 + b2
    # Low-rank: W1 ≈ scale1 · U1 @ V1ᵀ ; W2 ≈ scale2 · U2 @ V2ᵀ.
    # h: (..., D) @ V1[idx]: (D, r) → (..., r) @ U1[idx]ᵀ: (r, F) → (..., F).
    hV1 = h @ moe.V1[idx]                                   # (..., r)
    h1 = moe.scale1[idx] * (hV1 @ moe.U1[idx].T) + moe.b1[idx]   # (..., F)
    h1_act = torch.nn.functional.gelu(h1)
    hV2 = h1_act @ moe.V2[idx]                              # (..., r)
    return moe.scale2[idx] * (hV2 @ moe.U2[idx].T) + moe.b2[idx]  # (..., D)
```

- [ ] **Step 3: Rewrite `_eval_expert_mse` and `_train_one_expert`**

Replace `_eval_expert_mse` (around lines 88-93) — it stays almost the same but now calls the new `_expert_forward` (no `force_refresh`):

```python
def _eval_expert_mse(engine: ContinuousThoughtEngine, idx: int, bank: dict) -> float:
    """Mean MSE of expert idx on a bank (no grad)."""
    engine.eval()
    with torch.no_grad():
        out = _expert_forward(engine, idx, bank["h_in"])
        return torch.nn.functional.mse_loss(out, bank["h_target"]).item()
```

Replace `_train_one_expert` (around lines 96-132) with the gradient-row-masking version (spec D5):

```python
def _train_one_expert(engine: ContinuousThoughtEngine, idx: int, bank: dict,
                      steps: int, lr: float, batch_size: int, seed: int) -> list:
    """Train expert idx in isolation on bank (MSE). Returns per-step losses.

    The MoE weights are shared nn.Parameters of shape (E, ...). To train only
    expert idx, we zero the gradient rows j != idx of every shared parameter
    before each optimizer step (spec D5).
    """
    engine.train()
    moe = engine.moe
    # Shared parameters we optimize (the MoE owns them as (E, ...) tensors).
    shared = [moe.b1, moe.b2]
    if moe.expert_rank is None:
        shared += [moe.w1, moe.w2]
    else:
        shared += [moe.U1, moe.V1, moe.U2, moe.V2, moe.scale1, moe.scale2]
    opt = torch.optim.AdamW(shared, lr=lr, weight_decay=0.01)

    g = torch.Generator().manual_seed(seed)
    n = bank["h_in"].size(0)
    losses = []
    E = moe.n_experts
    for _ in range(steps):
        idx_b = torch.randint(0, n, (batch_size,), generator=g)
        h_in = bank["h_in"][idx_b]
        h_tgt = bank["h_target"][idx_b]
        opt.zero_grad()
        out = _expert_forward(engine, idx, h_in)
        loss = torch.nn.functional.mse_loss(out, h_tgt)
        loss.backward()
        # Mask: keep only row idx of each shared (E, ...) gradient.
        with torch.no_grad():
            for p in shared:
                if p.grad is None or p.dim() < 1:
                    continue
                grad = p.grad
                mask = torch.zeros_like(grad)
                mask[idx] = 1.0
                grad.mul_(mask)
        torch.nn.utils.clip_grad_norm_(shared, 1.0)
        opt.step()
        losses.append(loss.item())
    return losses
```

- [ ] **Step 4: Remove `force_refresh` calls from `phase3_joint`**

In `phase3_joint` (around lines 230-238), delete the loop:
```python
    # Force-refresh all expert caches so Phase 3 starts from trained matrices.
    for i in range(engine.n_experts):
        engine.experts_w1[i].force_refresh()
        engine.experts_w2[i].force_refresh()
```
(There is no cache to refresh anymore; the OnlineTrainer will now actually update the experts.)

- [ ] **Step 5: Run the Phase-1 tests to verify they pass**

Run: `$PY -m pytest tests/test_ablib.py::test_phase1_shared_reduces_mse_on_bank tests/test_ablib.py::test_phase1_partitioned_trains_each_expert_on_own_bank -v`
Expected: PASS (both). The MSE-decrease property must still hold — the experts train on the bank via the new shared-param path.

If the MSE does not decrease, the gradient masking is likely wrong (e.g. zeroing the wrong rows, or `mul_` not in-place). Debug the mask before weakening the test.

- [ ] **Step 6: Run the full ablib test suite**

Run: `$PY -m pytest tests/test_ablib.py -k "not arm_" -v`
Expected: PASS (all fast tests). The arm tests are deselected (they take ~7 min; verified separately in Task 5).

- [ ] **Step 7: Commit**

```bash
git add experiments/edt_ab/ablib.py
git commit -m "refactor(edt-ab): Phase 1 on shared low-rank MoE params via gradient masking"
```

---

## Task 4: Update REPORT.md provenance + smoke-run the corrected CLI

**Files:**
- Modify: `experiments/edt_ab/REPORT.md` (provenance note about the engine change)
- Run: `scripts/ab_edt_13m.py --budget 1600` (tiny smoke run, not committed)

- [ ] **Step 1: Add a provenance note to REPORT.md**

In `experiments/edt_ab/REPORT.md`, in the **Provenance** section, append a bullet:

```markdown
- **Engine correction (2026-07-28):** the numbers in this report are from the PRE-correction run, where the CTE's ad-hoc MoE read detached `_cached_W` buffers and Phase 3 could not fine-tune the experts (see Limitations §1). The engine has since been refactored to use `PhaseRoutedMoE` (differentiable, low-rank-capable). A re-run with the corrected engine follows in Task 5; this report's headline numbers will be overwritten then.
```

- [ ] **Step 2: Commit the provenance note**

```bash
git add experiments/edt_ab/REPORT.md
git commit -m "docs(edt-ab): note pending engine-correction re-run in provenance"
```

- [ ] **Step 3: Smoke-run the corrected CLI (tiny budget)**

Run: `$PY scripts/ab_edt_13m.py --budget 1600 --n-holdout 200 --chunk-len 16`
Expected: prints three arm sections with ppl/div/time, writes `experiments/edt_ab/results.json`, no traceback. This validates the full corrected pipeline end-to-end (Phase 1 via shared params → Phase 3 → eval) before the expensive re-run.

If a traceback occurs, capture it and fix before proceeding — do NOT proceed to the full re-run with a broken pipeline.

- [ ] **Step 4: Sanity-check the tiny-run output**

Run: `cat experiments/edt_ab/results.json`
Expected: three arm keys with finite ppl/diversity/wall_clock_s. Diversity values should still show the B-vs-C ordering signal (C more diverse than B), since the specialization mechanism is unchanged.

No commit (smoke-run artifacts are gitignored).

---

## Task 5: Full re-run with the corrected engine + report update

**Files:**
- Modify: `experiments/edt_ab/REPORT.md` (overwrite numbers + verdicts)
- Run: `scripts/ab_edt_13m.py --budget 400000` (full run, detached)

- [ ] **Step 1: Launch the full re-run detached**

From `C:\Users\PHIL\ZCodeProject\fractus-test`, launch (Git Bash, nohup so the harness timeout does not kill it):
```bash
MSYS_NO_PATHCONV=1 nohup "$PY" scripts/ab_edt_13m.py --budget 400000 --n-holdout 30000 --chunk-len 16 > experiments/edt_ab/run_v2.log 2>&1 &
echo "PID=$!"
```
where `$PY=/c/Users/PHIL/AppData/Local/Microsoft/WindowsApps/PythonSoftwareFoundation.Python.3.13_qbz5n2kfra8p0/python.exe`.

Expected runtime: ~2.5-3h (Phase 1 is faster without SIREN reconstruction; Phase 3 is similar or slightly faster with low-rank matmuls). Monitor `experiments/edt_ab/run_v2.log` until it prints `Wrote ... results.json`.

- [ ] **Step 2: Read the new numbers**

Run: `cat experiments/edt_ab/results.json`
Record ppl, accuracy, diversity, phase3_final_loss, wall_clock_s for all three arms.

- [ ] **Step 3: Re-apply the pre-registered decision rule (spec §9 of the EDT spec)**

Compute by hand from the new numbers:
- `accel_ok = ppl_B ≤ 0.95 · ppl_A`
- `spec_ppl_ok = ppl_C ≤ 0.95 · ppl_B`
- `spec_div_ok = diversity_C ≤ diversity_B − 0.05`
- `spec_ok = spec_ppl_ok AND spec_div_ok`

- [ ] **Step 4: Overwrite REPORT.md headline + verdicts**

Replace the **Headline numbers** table and the **Pre-registered verdicts** section with the new numbers and the recomputed TRUE/FALSE breakdowns. Update the **Interpretation** paragraphs to reflect the new verdict (the §1 Limitation about frozen experts is now FIXED — remove it; the A-vs-B verdict is now a clean test of EDT acceleration).

If the new A-vs-B verdict differs from the old one (e.g. EDT now beats from-scratch), state this plainly and note that the earlier "NOT SUPPORTED" was confounded by the engine defect.

- [ ] **Step 5: Commit the updated report**

```bash
git add experiments/edt_ab/REPORT.md
git commit -m "docs(edt-ab): re-run with corrected engine — updated verdicts"
```

- [ ] **Step 6: Update the EDT.md "First Measurement" section (fractus repo)**

If the verdict changed materially, update the "First Measurement" subsection in `C:\Users\PHIL\ZCodeProject\fractus\docs\EDT.md` to reflect the corrected numbers. If the verdict is unchanged, add a one-line note: "Re-run with a differentiable MoE (2026-07-28) confirmed the verdict."

Commit in the `fractus` repo:
```bash
cd C:\Users\PHIL\ZCodeProject\fractus
git add docs/EDT.md
git commit -m "docs(edt): note corrected-engine re-run result"
```

---

## Task 6: Wrap-up — final review of the whole change

**Files:** read-only review across the branch.

- [ ] **Step 1: Run the entire test suite**

Run: `$PY -m pytest tests/test_moe.py tests/test_continuous_engine_grad.py tests/test_ablib.py -k "not arm_" -v`
Expected: all pass. Then optionally run the arm tests (slow, ~7 min) to confirm the end-to-end pipeline.

- [ ] **Step 2: Dispatch a final holistic code review**

Dispatch a code-reviewer subagent over the full diff `git diff <base>..<head>` for the redesign branch. Focus: (a) does the regression test actually catch the defect (would it fail on the old code), (b) is the gradient masking in `_train_one_expert` correct (only row idx updated), (c) any leftover `force_refresh`/`experts_w1` references, (d) is the new A-vs-B verdict trustworthy now.

- [ ] **Step 3: Finish the branch**

Use `superpowers:finishing-a-development-branch` to merge / PR / cleanup per the user's choice.

---

## Self-review notes (done by the planner, not a task)

- **Spec coverage:** §3 D1 (expert_rank flag, two weight layouts) → Task 1. §3 D2 (low-rank dense-only, warn fallback) → Task 1 Step 5. §3 D3/D4 (CTE drops ad-hoc MoE, uses PhaseRoutedMoE, siren_rank preserved) → Task 2. §3 D5 (gradient masking for Phase-1 isolation) → Task 3. §4.4 tests → Task 1 (moe), Task 2 (engine regression). §4.5 re-run + report → Tasks 4-5. §7 risks handled (sparse low-rank fallback in Task 1 Step 5; checkpoint incompatibility accepted and noted in Task 4). §9 success criteria: regression test (Task 2), dense non-regression (Task 1), ablib tests (Task 3), re-run verdict (Task 5), Limitation §1 removed (Task 5 Step 4).
- **Placeholder scan:** no TBD/TODO; every code step shows the full code. The earlier draft had a placeholder/concrete pair in Task 2 Step 4 — corrected to a single concrete block.
- **Type consistency:** `PhaseRoutedMoE(expert_rank=...)`, `engine.moe`, `moe.U1/V1/U2/V2/scale1/scale2/b1/b2` (low-rank) and `moe.w1/w2/b1/b2` (dense) — names match across Tasks 1-3. `_expert_forward(engine, idx, h)` signature consistent. `_train_one_expert` returns a losses list, consumed unchanged by `phase1_experts_shared/partitioned` (which already `extend` into a flat list — verified in the prior EDT plan Task 9).
```
