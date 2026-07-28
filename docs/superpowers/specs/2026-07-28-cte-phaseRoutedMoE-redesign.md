# CTE → PhaseRoutedMoE Redesign — Design spec

**Date:** 2026-07-28
**Author:** Philippe-Antoine Robert (with ZCode)
**Status:** Design — pending user review
**Scope:** Core Fractus change. Modifies `fractus/nn/moe.py` and `fractus/continuous_engine.py`, adapts `experiments/edt_ab/ablib.py`. Followed by a full re-run of the EDT AB-test.

---

## 1. Motivation

The EDT AB-test (`experiments/edt_ab/REPORT.md`, 2026-07-27) concluded that EDT acceleration is "NOT SUPPORTED" at the 13M scale, but the final holistic review discovered the verdict is **confounded**: during Phase 3, the experts do not receive any gradient, so the headline EDT claim ("pre-train experts then jointly fine-tune") was never actually exercised.

**Root cause** (re-read from the documentation, not re-derived): the `ContinuousThoughtEngine` does **not** use the documented, tested, end-to-end-differentiable `PhaseRoutedMoE` component from L2b (`docs/2026-06-19-fractus-L2b-kuramoto-moe.md`, `fractus/nn/moe.py`). Instead it reimplements its own MoE ad hoc using `CachedStructuredSirenLinear` experts and reads the experts' detached `_cached_W` buffer directly during the forward (`fractus/continuous_engine.py:315-316`):

```python
w1_stack = torch.stack([e._cached_W for e in self.experts_w1])  # detached buffer
w2_stack = torch.stack([e._cached_W for e in self.experts_w2])
```

Because `_cached_W` is `.detach()`'d at creation (`fractus/nn/cached_siren.py:82`), no gradient reaches the expert parameters `U`/`V`/`residual_siren` during `tick`/`tick_chunk`. The final review verified empirically that **0 of 36 expert parameters move during Phase 3**, while 16 of 18 non-expert parameter groups do update.

This is **not** a property of EDT, the AB-test code, or the spec — it is a defect in the CTE's MoE wiring. The documented architecture (`LN → attn → Kuramoto → PhaseRoutedMoE`, L2b §Task 4) already provides the correct component. The CTE simply does not use it.

The project already contains a second relevant component: `LazyStructuredSirenLinear` (`fractus/nn/lazy_siren.py`), a fully-differentiable LoRA-style low-rank layer (`W = scale·U@Vᵀ`, forward via two cheap matmuls, no detached buffer) that was built precisely to fix the 1B's memory blowup. Its own docstring notes: *"The MoE code uses `_cached_W` for the stack — we override the MoE to use U,V directly instead."* — the author already flagged this exact defect as a known follow-up.

## 2. Goal

Make the `ContinuousThoughtEngine` route its thought state through a `PhaseRoutedMoE`, so that expert weights receive a gradient during joint training. Extend `PhaseRoutedMoE` to optionally use low-rank experts (LoRA-style, compatible with the 1B's compression) so the same component serves the 13M (AB test) and the 1B. Then re-run the EDT AB-test to obtain a scientifically valid A-vs-B verdict.

**Non-goal:** changing the CTE's attention, Kuramoto, or thought-state mechanics. **Non-goal:** loading existing 13M checkpoints (the forward output of the MoE will change numerically; we retrain from scratch). **Non-goal:** adding the load-balance loss to the Phase-3 objective (the `OnlineTrainer` uses pure cross-entropy; exposing `lb_loss` is enough for now, using it is future work).

## 3. Architecture decisions

**D1 — One `PhaseRoutedMoE` class, two expert modes.** Add a constructor arg `expert_rank: int | None = None`:
- `expert_rank=None` → dense experts, weights `w1 (E, D, F)`, `w2 (E, F, D)` (current behaviour, unchanged — `test_moe.py` must still pass).
- `expert_rank=r` → low-rank experts, weights `U1 (E, F, r)`, `V1 (E, D, r)`, `U2 (E, D, r)`, `V2 (E, F, r)`, `scale1 (E,1,1)`, `scale2 (E,1,1)`, `b1 (E,F)`, `b2 (E,D)`. Expert forward: `gelu(scale1 · (h @ V1) @ U1ᵀ + b1) → scale2 · (_ @ V2) @ U2ᵀ + b2`.

Rationale: one class matches the documentation (L2b), avoids a parallel hierarchy, and the flag is the minimal extension. Both modes share the routing (von Mises gate, top-k, load-balance loss) unchanged.

**D2 — Dense vs sparse dispatch.** `PhaseRoutedMoE.forward` already picks dense (einsum over all E) when `n_experts ≤ 2·top_k` and sparse (gather-first) otherwise. In this iteration the **low-rank mode supports the dense path only** (the 13M uses dense — E=4 ≤ 2·K=4 — so this covers the AB test). If `expert_rank is not None` and the sparse path would be selected (`n_experts > 2·top_k`), the dense path is used as a fallback and a one-line `warnings.warn` is emitted. Sparse low-rank is deferred to future work (see §8) — it is not needed at 13M and the gather-over-low-rank-factors indexing is fiddly enough to deserve its own task.

**D3 — CTE drops its ad-hoc MoE.** Remove from `ContinuousThoughtEngine`: `experts_w1`, `experts_w2` (ModuleList), the `expert_phases`/`kappa` buffers, and the hand-rolled routing in `tick` (continuous_engine.py:206-231) and `tick_chunk` (315-329). Replace with `self.moe = PhaseRoutedMoE(d_model, n_experts, top_k, kappa=4.0, d_ff=expert_d_ff, expert_rank=(siren_rank if siren_rank else None))`. In both `tick` and `tick_chunk`: `moe_out, lb_loss = self.moe(self.norm_moe(h_flat), kuramoto_phases); self.last_lb_loss = lb_loss.detach()`. The `lb_loss` is exposed (stored, detached) but not added to the Phase-3 loss in this iteration.

**D4 — `siren_rank` semantics preserved in the CTE constructor.** The existing `__init__(..., siren_rank=32)` is reused: when `siren_rank > 0`, the CTE builds a low-rank `PhaseRoutedMoE(expert_rank=siren_rank)`; when `siren_rank == 0` (or `None`), dense. This keeps every existing CTE caller working without signature changes. The `CachedStructuredSirenLinear` import is removed from `continuous_engine.py`.

**D5 — `ablib.py` accesses expert weights via helpers.** `_expert_forward`, `_eval_expert_mse`, `_train_one_expert` currently call `engine.experts_w1[i]`/`experts_w2[i]`. Introduce two helpers:
- `_expert_forward(engine, i, h) -> Tensor`: returns the output of expert `i` on `h`, reconstructing W1/W2 on the fly (dense: direct matmul; low-rank: `scale·(h@V)@Uᵀ`). Pure function, no grad trickery.
- `_train_one_expert(engine, i, bank, ...)`: trains **only** expert `i` in isolation. Because the low-rank weights are shared `nn.Parameter`s of shape `(E, D, r)` (one row per expert), training expert `i` must not touch rows `j≠i`. Implementation: compute the loss using `_expert_forward(engine, i, ...)` (which slices to row `i`), backward, then **zero the gradient rows `j≠i` before `opt.step()`** (mask the `.grad` of each shared parameter to row `i` only). The optimizer is built over the shared parameters (`moe.U1`, `moe.V1`, `moe.U2`, `moe.V2`, `moe.b1`, `moe.b2`, and `moe.scale1`/`moe.scale2` if present). This guarantees Phase 1 trains expert `i` without contaminating the other experts — the same isolation property the old per-expert `ModuleList` gave for free.

The cache-invalidation `force_refresh()` calls become unnecessary (no cache in `PhaseRoutedMoE`/low-rank) and are removed. The Phase 1 freeze of `residual_siren` (Task 5 deviation) disappears — there is no `residual_siren` in the new MoE.

## 4. Components

### 4.1 `fractus/nn/moe.py` — extend `PhaseRoutedMoE`

Add `expert_rank` arg and a low-rank weight layout. Implement `_expert_forward_dense`, `_expert_forward_lowrank` (or branch inside the existing `_dense_expert_forward` / `_sparse_expert_forward`). Public surface unchanged: `forward(h, phases) -> (output, lb_loss)`.

### 4.2 `fractus/continuous_engine.py` — refactor MoE wiring

- Remove: `experts_w1`, `experts_w2`, `expert_phases` buffer, `kappa`, all hand-rolled routing in `tick` and `tick_chunk`.
- Add: `self.moe = PhaseRoutedMoE(...)` per D3; `self.last_lb_loss` buffer initialised to 0.
- Rewrite the MoE section of `tick` (≈ lines 206-231) and `tick_chunk` (≈ lines 303-329) to call `self.moe`.
- Remove the `CachedStructuredSirenLinear` import.

### 4.3 `experiments/edt_ab/ablib.py` — adapt expert access

- Rewrite `_expert_forward(engine, i, h)`: dense → `moe.w2[i](gelu(moe.w1[i](h)))` style (using the per-expert slice of the shared parameter); low-rank → reconstruct W1/W2 from row `i` of U/V/scale and apply two matmuls. No `force_refresh`.
- Rewrite `_train_one_expert(engine, i, bank, ...)`: optimizer over the shared MoE parameters; after each `loss.backward()`, **zero the gradient rows `j≠i`** of every shared parameter (mask) before `opt.step()`, so only expert `i` is updated. This preserves Phase-1 per-expert isolation under the new shared-parameter layout (D5).
- `_eval_expert_mse`, `expert_diversity`: use the new `_expert_forward`.
- Drop all `force_refresh()` calls in `ablib.py` (no cache to invalidate).
- Drop the `residual_siren` freeze filter from Task 5 (no residual_siren anymore).
- `phase3_joint`: drop the `force_refresh` loop at start. The `OnlineTrainer` now actually fine-tunes the experts (the fix).
- Everything else (`build_engine`, `load_corpus`, `make_hidden_bank`, `phase2b_embedding`, `evaluate_ppl`, `greedy_sample`, the three arms, the CLI) unchanged.

### 4.4 Tests

- **Extend `tests/test_moe.py`:** `test_moe_lowrank_output_shape`; `test_moe_lowrank_backward_every_param` (gradient to U1, V1, U2, V2, scale1, scale2, b1, b2 — finite and non-zero); `test_moe_lowrank_matches_dense_shape`; keep all existing dense tests passing unchanged.
- **New `tests/test_continuous_engine_grad.py`:** `test_cte_experts_receive_gradient` — build the 13M CTE, run one `tick_chunk`, compute a dummy loss, backward, assert every expert parameter (low-rank factors, scales, biases) has a finite, non-zero gradient. This is the **regression test for the defect** — it would have failed before the fix and must pass after.
- **Update `tests/test_ablib.py`:** the Phase 1 tests (`test_phase1_shared_reduces_mse_on_bank`, `test_phase1_partitioned_trains_each_expert_on_own_bank`) must still pass with the new `_get_expert_params` path. The `build_engine` determinism test still applies. The arm tests run end-to-end.

### 4.5 Re-run + report update

- Re-run all three arms at `--budget 400000 --n-holdout 30000 --chunk-len 16` with the corrected engine.
- Replace the numbers and verdicts in `experiments/edt_ab/REPORT.md`. Remove the §1 Limitation (frozen-expert Phase 3) — it is fixed. Re-apply the pre-registered §9 decision rule to the new numbers.
- Update `docs/EDT.md` (in the `fractus` repo) "First Measurement" section if the verdict changes.

## 5. Data flow (Phase 3, corrected)

```
tick_chunk(tokens)
  → embed + attn + kuramoto (unchanged)
  → moe_out, lb_loss = self.moe(norm_moe(h), kuramoto_phases)
       └── von Mises gates on Farey phases, top-k gather, expert forward
           (dense: einsum over w1/w2; low-rank: two matmuls per expert)
           → gradient flows to expert U/V/scale/bias ✓
  → h = h + moe_out
  → output_head → logits
loss = CE(logits, targets); loss.backward()
  → experts update ✓ (the fix)
```

## 6. Budget / cost

- Code change: ~half a day of implementation (moe.py extension + CTE refactor + ablib adaptation + tests).
- Re-run: ~3h CPU (same as the first run: A ≈ 50 min, B ≈ 47 min, C ≈ 63 min, plus Phase 1 is now faster without SIREN reconstruction).
- No new dependencies. No GPU.

## 7. Risks

| Risk | Mitigation |
|---|---|
| Low-rank expert forward is numerically different from the old Cached+SIREN path → existing 13M checkpoints won't load | Accepted. We retrain from scratch; the AB test rebuilds engines from seed 42 anyway. Documented in `REPORT.md` provenance. |
| `_sparse_expert_forward` index_select on low-rank factors is fiddly (gather over the E dim of 4D tensors) | Mitigated by D2: low-rank mode uses the dense path only (with a `warnings.warn` fallback). The 13M AB test uses dense anyway (`E=4 ≤ 2·K=4`). Sparse low-rank is explicitly out of scope (§8). |
| The CTE forward output changes → `test_ablib.py` arm tests (which check shapes/keys, not exact values) still pass, but any test asserting old numbers would break | The AB-test tests assert shapes, key presence, and "loss goes down" — none assert exact old values. Confirmed by reading `tests/test_ablib.py`. |
| Removing `force_refresh` calls but leaving a stale reference | Grep for `force_refresh` in `ablib.py` and `continuous_engine.py` after the refactor; the only valid remaining callers are `LazyStructuredSirenLinear.force_refresh` (a no-op) and `CachedStructuredSirenLinear.force_refresh` (still used elsewhere if at all) — remove only the ones in the edited paths. |

## 8. Out of scope

- Adding `lb_loss` to the Phase-3 training objective (exposed only; future work).
- Sparse low-rank dispatch (dense path is correct and used at 13M scale; sparse low-rank deferred).
- The `CachedStructuredSirenLinear` class itself (left in place; only the CTE stops using it).
- Re-evaluating the 1B design (the low-rank `PhaseRoutedMoE` is *available* for the 1B after this change, but wiring the 1B to use it is a separate task).

## 9. Success criteria

1. New test `test_cte_experts_receive_gradient` passes (every expert param gets a finite, non-zero grad after one `tick_chunk` + backward).
2. All existing `test_moe.py` dense tests still pass unchanged.
3. All `test_ablib.py` tests pass (Phase 1 still reduces MSE; arm tests still produce the required keys).
4. Re-run produces a valid `results.json`; REPORT.md verdicts are re-derived from the new numbers using the pre-registered §9 rule.
5. The "Limitations §1 (frozen experts)" section is removed from REPORT.md because the defect is fixed.
