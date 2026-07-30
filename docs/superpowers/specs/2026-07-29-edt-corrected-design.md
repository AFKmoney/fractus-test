# EDT Corrected (routing filter + objective search) — Design spec

**Date:** 2026-07-29
**Author:** Philippe-Antoine Robert (with ZCode)
**Status:** Design — pending user review
**Scope:** Modifies the 13M EDT library (`experiments/edt_ab/ablib.py`) and CLI; validates at 13M scale, then ports the winning variant to 1B if successful. No changes to the Fractus core (Kuramoto, MoE routing mechanics).

---

## 1. Motivation

Two post-hoc diagnostics (`scripts/diagnose_hypothesis_b*.py`, `scripts/diagnose_routing.py`, committed in `4e24b8e`) identified **structural defects** that explain why EDT underperformed in the 13M AB-test (ppl 1321 vs 1150, ~15% worse than from-scratch):

**Defect 1 — Phase-1 objective misaligned with Phase-3.** Phase 1 trains each expert to *predict the next hidden state* (MSE regression `expert(h_t) ≈ h_{t+1}`); Phase 3 uses the expert as a *nonlinear brick in a residual stack* (next-token CE, with `h_out = h + expert(h)`). These objectives are not aligned: across 12 (seed, block, expert) configs, the Pearson correlation between Phase-1 MSE and full-model hold-out PPL was **never positive** (3/12 strongly negative, 9/12 ≈ 0). An expert that minimizes Phase-1 MSE is at best neutral and at worst harmful to the final CE.

**Defect 2 — Kuramoto routing concentrates, so most Phase-1 training is wasted.** The router's von Mises gate selects top-k experts per token based on the **Kuramoto phases**. A measurement showed that on real CTE phases, only **2 of 4 experts are ever routed** (each on 100% of tokens); the other 2 are *never* selected. (Notably, with random uniform phases all 4 experts are used — so the concentration comes from the Kuramoto dynamics producing near-constant phases, not from kappa/top_k.) Phase 1 trains all 4 experts, but Phase 3's router ignores half of them — their training is invisible to the final loss. This explains the 9/12 zero-correlation configs in the robustness check.

The two defects compound: EDT pre-trains experts the router then ignores (Defect 2), and for the few experts that *are* routed, the pre-training objective is misaligned (Defect 1).

**This spec defines the correction.** Both defects are addressable without touching the Fractus core: (a) Phase 1 trains only the experts the router actually uses, and (b) Phase 1's objective is replaced — empirically selected from three candidates aligned with the brick role. Validation at 13M scale (CPU-feasible); port to 1B follows if a corrected variant beats from-scratch.

## 2. Claims to adjudicate

| Comparison | Claim under test |
|---|---|
| EDT-corrected (best objective) vs from-scratch | Correcting the two defects makes EDT beat from-scratch at equal token budget. |
| next_hidden vs denoise vs identity vs residual (Phase-1 objective search) | Which Phase-1 objective is best aligned with Phase-3 CE? |

The pre-registered decision rule (spec §9 of `2026-07-27-edt-ab-test-design.md`) is reused: a corrected variant "beats from-scratch" iff its hold-out PPL ≤ 0.95 × from-scratch PPL.

## 3. Correction A — Phase 1 trains only routed experts (Defect 2 fix)

New function `detect_routed_experts(model, probe_tokens, *, min_fraction=0.0) -> dict[int, list[int]]`:
- Runs a forward probe of the model over `probe_tokens` (chunked), instruments the MoE routing at each block, counts per-expert top-k hits.
- Returns `{block_idx: [expert indices selected on at least one token]}` (or with a `min_fraction` threshold to drop near-dead experts).
- Implemented generically so it works for both the 13M CTE (single MoE) and 1B (16 blocks). For the CTE, `blocks` is replaced by `[model.moe]` and `block_idx=0`.

`phase1_experts_shared` / `phase1_experts_partitioned` gain a `routed_experts: dict[int, list[int]] | None` parameter:
- If `None` → train all experts (current behavior, the reference).
- If provided → train **only** the experts listed per block. Unrouted experts stay at init; Phase 3 can still adjust them if the routing evolves (their `requires_grad` stays True).

Both arms B and C receive the same `routed_experts` filter → fair.

## 4. Correction B — Phase-1 objective search (Defect 1 fix)

`_train_one_expert` and the phase1 entry points gain a `phase1_objective: str` parameter. The hidden bank (`make_hidden_bank`) already produces `(h_t, h_{t+1})` pairs; the four objectives derive their input/target from these:

| `phase1_objective` | input | target | rationale |
|---|---|---|---|
| `"next_hidden"` (reference, the old EDT) | `h_t` | `h_{t+1}` | predict the next hidden state (the misaligned original) |
| `"denoise"` | `h_t + 0.1·noise` | `h_t` | reconstruct clean h from noisy — aligned with the brick/denoising role, same self-supervised objective as Phase 2a |
| `"identity"` | `h_t` | `h_t` | learn a near-identity transformation; Phase 3 refines from a neutral starting point |
| `"residual"` | `h_t` | `h_{t+1} − h_t` | predict the *residual* that the block would add; aligned with the residual-stack form `h_out = h + expert(h)` |

MSE loss in all cases. The objective is a pure swap of the target tensor — no architectural change.

**Empirical selection at 13M:** run Arm B with each of the four objectives (next_hidden as reference) at equal budget, plus the from-scratch Arm A. Compare hold-out PPL. The objective with the lowest PPL becomes the default for the 1B port. If multiple objectives are within 1% of each other, prefer `denoise` (aligned with Phase 2a, most principled).

## 5. Validation experiment (13M, CPU)

Five "variants" (each is a full arm at 400k tokens, identical seed/shuffle/holdout):

| Variant | Phase-1 objective | Routing filter |
|---|---|---|
| **A** from-scratch | (no Phase 1) | n/a |
| **B-nh** EDT vanilla (reference) | `next_hidden` | all experts |
| **B-denoise** | `denoise` | routed only |
| **B-identity** | `identity` | routed only |
| **B-residual** | `residual` | routed only |

(The reference B-nh keeps the *original* objective AND all experts, so it reproduces the prior negative result and serves as the baseline-to-beat for the corrected variants. The three corrected variants apply BOTH corrections — new objective AND routing filter — so they test the combined fix directly. **We are NOT isolating the two corrections here** (e.g. there is no "next_hidden + routing filter" variant): the prior diagnostics already established each defect is harmful on its own, so the question is whether fixing *both together* produces a working EDT, not which single fix contributes more. The goal is to find a combined configuration that works.)

Budget per variant: 400k tokens (~1h on CPU). Total ~5h. Single seed (same caveat as before; the §9 5% threshold triggers multi-seed only on near-ties).

**Decision rule:** if the best corrected variant has PPL ≤ 0.95 × A's PPL → EDT-corrected works at 13M, port to 1B. Otherwise → EDT is fundamentally incompatible with Fractus at this scale; document and stop.

## 6. Port to 1B (if a corrected variant wins)

The 1B EDT library (`experiments/edt_1b_ab/ablib_1b.py`) already exists. Ports required:
- `detect_routed_experts_1b(model, probe_tokens) -> dict[int, list[int]]` (probe forward, instrument `SparseStructuredMoE` routing per block).
- `phase1_objective` flag threaded through `_train_one_expert_1b`, `phase1_experts_1b_shared_1b`, `phase1_experts_1b_partitioned`.
- `routed_experts` filter parameter on the two phase1 entry points.
- Arm B/C accept `phase1_objective` (default = the 13M winner) and `routed_experts` (computed inside the arm from a probe).
- Phase 2a/2b/3 unchanged.

Then the existing 1B CLI + RUNBOOK apply unchanged (just swap `phase1_objective` default).

## 7. Components

| Path | Responsibility |
|---|---|
| `experiments/edt_ab/ablib.py` | + `detect_routed_experts`, `phase1_objective` flag, `routed_experts` param |
| `scripts/ab_edt_corrected.py` | CLI: runs the 5 variants, writes results.json |
| `tests/test_ablib.py` | + tests for `detect_routed_experts` and the 4 objectives |
| `experiments/edt_1b_ab/ablib_1b.py` (port, if 13M wins) | + same two additions |

## 8. Risks

| Risk | Mitigation |
|---|---|
| Routing filter based on an init-time probe may not match Phase-3 routing (weights move) | Measure post-Phase-3 routing too; report overlap. If the overlap is low, the filter was based on a stale snapshot — note as a limitation, don't silently expand. |
| `identity` objective may collapse the expert to a true identity (no specialization signal) | That's actually fine — Phase 3 then trains a fresh-but-aligned expert. The 13M experiment will show whether identity beats the others. |
| `residual` target `h_{t+1} − h_t` may have small magnitude (near-zero) → MSE insensitive | Report the target's variance; if it's degenerate, `residual` will underperform and `denoise` wins. Self-correcting. |
| Testing 4 objectives multiplies CPU time (~5h) | Acceptable at 13M; only the winner ports to 1B (no 4-way test at GPU cost). |
| A corrected variant wins by luck (single seed) | The §9 rule triggers a 3-seed re-run if the gap to from-scratch is < 5%; only near-ties get re-run. A clear win (>5%) on single seed is sufficient evidence to port. |

## 9. Out of scope

- Changing the Kuramoto router or its phases (the routing concentration is a Fractus property; we adapt EDT to it, not the reverse).
- Testing objectives at 1B scale (only the 13M winner ports).
- Semantic (code/prose) domain partitioning — position-based only, same as all prior experiments.
- Multi-seed sweeps beyond the §9 re-run trigger.

## 10. Success criteria

1. `detect_routed_experts` correctly identifies the 2/4 active experts on the 13M (matches the diagnostic measurement in `diagnose_routing.py`).
2. All four Phase-1 objectives run and reduce their respective loss (MSE/CE) during Phase 1.
3. The 5-variant 13M run produces a `results.json` with comparable PPL across variants.
4. A clear winner emerges (best corrected variant) OR a clear negative (no variant beats A by 5%) — either outcome is documented honestly.
5. If a variant wins: the 1B port compiles, its tests pass, and the RUNBOOK is updated to use the winning objective + routing filter.
