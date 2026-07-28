# EDT AB-test — Results (13M model, 2026-07-27)

**Spec:** docs/superpowers/specs/2026-07-27-edt-ab-test-design.md
**Plan:** docs/superpowers/plans/2026-07-27-edt-ab-test.md
**Budget:** 400 000 tokens/arm, 30 000-token hold-out, chunk_len=16, lr=3e-4.

## Headline numbers

| Arm | Hold-out PPL | Token acc | Inter-expert cosine | Wall-clock |
|---|---|---|---|---|
| A from-scratch    | 1310.25 | 0.124 | -0.066 | 3025s |
| B EDT vanilla     | 1504.80 | 0.119 | 0.944  | 2803s |
| C EDT + spec      | 1539.07 | 0.117 | 0.428  | 3771s |

(Token accuracy for all three arms comes from `experiments/edt_ab/results.json`; Arm C's value is corroborated by `results_c.json`. Accuracy is rounded to 3 decimals here; full precision is in `results.json`. The `run.log` text log only prints ppl/div/time per arm, so the ppl/div numbers below are sourced from `run.log` for A and B and from `run.log`/`results.json` for C, with `results_c.json` corroborating C. The wall-clock for C is the dedicated C run time from `results_c.json`/`run_c.log`; `run.log` and `results.json` record a slightly larger value (3829s) that includes extra wait time before the C process exited — both are reported here for honesty.)

## Pre-registered verdicts (spec §9)

- **EDT accelerates (A vs B): NOT SUPPORTED under this regime (see Limitations §1).**
  Rule: ppl_B ≤ 0.95·ppl_A → 1504.80 ≤ 0.95·1310.25 = 1244.74 → FALSE.
  EDT vanilla is ~15 % worse than from-scratch on hold-out perplexity (1504.80 vs 1310.25), and the accuracy ordering agrees (A 0.124 ≥ B 0.119). **However the engine's `tick_chunk` reads the experts' detached `_cached_W` directly, so Phase 3 cannot fine-tune the experts EDT pre-trained — the headline "pre-train then jointly fine-tune" claim was never actually exercised.** This verdict should be read as "EDT acceleration unproven at this scale," not as evidence that EDT is harmful.

- **Specialization matters (B vs C): NOT SUPPORTED on ppl, but the diversity sub-condition IS met.**
  Rule: ppl_C ≤ 0.95·ppl_B AND div_C ≤ div_B − 0.05.
  → ppl: 1539.07 ≤ 0.95·1504.80 = 1429.56 → FALSE.
  → div: 0.428 ≤ 0.944 − 0.05 = 0.894 → TRUE.
  The conjunctive rule fails. Partitioning experts onto disjoint data DID produce markedly more distinct experts (cosine 0.428 vs 0.944 — a 0.52 reduction, i.e. the specialization mechanism works as designed), but this did not translate into better final perplexity. C is ~2 % worse than B (1539 vs 1504) and ~17 % worse than A.

## Interpretation

**A-vs-B (EDT acceleration): NOT SUPPORTED under this regime — but the regime does not give EDT a clean test (see Limitations §1 below).** At equal token budget, EDT vanilla (Arm B) reached worse hold-out perplexity than plain online training (Arm A): 1504.80 vs 1310.25, a ~15 % gap in the wrong direction, with a matching small regression in token accuracy (0.119 vs 0.124). However this comparison is confounded by an engine property the spec did not account for: during Phase 3 the experts are effectively frozen (the engine's `tick_chunk` reads the detached `_cached_W` buffer directly, so expert U/V/residual_siren params receive no gradient — verified: 0/36 expert params move during Phase 3). So A-vs-B actually measures "network adapting to MSE-pretrained fixed experts vs network adapting to random fixed experts," not "EDT pre-training + joint fine-tuning vs from-scratch joint fine-tuning." The 15 % gap is consistent with the hypothesis that MSE-pretrained experts are a *worse* fixed routing target than random ones — an artifact of the frozen-expert online path, not necessarily a property of EDT. The verdict is therefore best read as: **EDT acceleration is unproven at this scale; this experiment does not cleanly test it.**

**B-vs-C (specialization): trustworthy.** The only variable between B and C is the Phase 1 call (shared bank vs disjoint per-expert banks); Phase 2b and Phase 3 are byte-identical, and the frozen-expert property applies symmetrically. Arm C produced experts with much lower inter-expert cosine (0.428 vs B's 0.944), confirming that disjoint per-expert data does force divergence — exactly the design gap the experiment was designed to test (the original `edt_full.py` fed all experts identical data and they collapsed toward a shared expert). However C's final perplexity did not improve; on a 4-expert single-MoE model with position-based (not semantic) partitioning, the extra diversity did not help next-token prediction. C is ~2 % worse than B on ppl and ~0.002 worse on accuracy. The ppl gap is above the 5 % threshold but in the wrong direction, so the conjunctive §9 rule fails on the ppl limb regardless of the diversity limb being met; no multi-seed re-run is triggered.

## Limitations

1. **Phase 3 cannot fine-tune the experts (engine property, symmetric across arms).** `fractus/continuous_engine.py:315-316` reads `e._cached_W` — a detached buffer — directly, bypassing `CachedStructuredSirenLinear.forward()` where the every-8-calls refresh-with-grad happens. Empirically, 0 of 36 expert parameters move during `phase3_joint` (verified), while 16 of 18 non-expert parameter groups do update. This is the stock engine behavior (the built-in `OnlineTrainer` has the same property), not a bug introduced by the AB code, and it is symmetric so it does not bias B-vs-C. But it materially affects A-vs-B (see above) and means **the headline EDT claim — pre-train then jointly fine-tune — was never actually exercised** at this scale, because the "joint fine-tune" step cannot touch the experts EDT pre-trained. A fair test of EDT acceleration would require either (a) an engine forward path that routes through `expert.forward()` so the cache refreshes with grad, or (b) a Phase 3 that force-refreshes expert caches periodically with grad enabled.

2. **Token-budget overlap (spec-compliant, but disadvantages EDT in A-vs-B).** Arm A trains over `train[:400000]` (400k distinct source tokens). Arms B/C draw Phase 1 from `train[:60000]`, Phase 2b from `train[:120000]`, Phase 3 from `train[:220000]` — all nested prefixes. So B/C touch ~220k distinct source tokens vs A's 400k (~1.8×). The spec §5 defines budget as "tokens consumed" (step counts summing to N=400k), so this is spec-compliant, but Arm A's win could partly come from seeing more distinct data. Symmetric for B-vs-C (both use the same nested slices), so it does not threaten that comparison.

3. **Diversity metric implementation note.** The spec §7 wording suggests a 4×4 mean-cosine over per-sample cosines; the implementation flattens each expert's (256, d_model) output batch into one vector and takes one cosine per pair. Applied identically to all arms, so no bias; the qualitative ordering (A≈0 < C=0.43 < B=0.94) is robust to either definition.

4. **The 13M is a deliberate test of principle** (spec §8 caveat 1): 4 experts and a single MoE (no 16-layer stack) is a very different regime from Fractus-1B (128 experts × 16 layers). A negative result here does NOT refute EDT-1B. Single seed per arm. Specialization is by-position, not semantic (mono-domain corpus). Phase 2a omitted at this scale (no layer stack); `residual_siren` frozen during Phase 1 for stability, applied identically to B and C.

> **Update (2026-07-28):** Limitation §1 above (frozen-expert Phase 3) has since been **fixed** by refactoring the CTE to route its MoE through the differentiable `PhaseRoutedMoE` (low-rank-capable, spec `docs/superpowers/specs/2026-07-28-cte-phaseRoutedMoE-redesign.md`). The numbers in this report are from the PRE-correction run. A re-run with the corrected engine follows; this report's headline numbers and the A-vs-B verdict will be overwritten then. Limitation §4's `residual_siren` freeze is also obsolete (the new MoE has no residual_siren).

## Provenance

- All numbers from `experiments/edt_ab/results.json` (the canonical aggregate; gitignored as a generated artifact). Arm C's numbers are corroborated to 15+ significant figures by an independent redundant run (`experiments/edt_ab/run_arm_c.py` → `results_c.json`), confirming bit-identical reproducibility.
- Samples: `experiments/edt_ab/samples/{A_from_scratch,B_edt_vanilla,C_edt_spec}.txt` (degenerate/repetitive — expected for a lightly-trained 13M; all three arms produced near-identical "Theinal."-style output).
- Code: `experiments/edt_ab/ablib.py`, `scripts/ab_edt_13m.py`, `experiments/edt_ab/run_arm_c.py` (branch `edt-ab-test`).
- All three arms ran in a single detached process (`scripts/ab_edt_13m.py`), each building a fresh engine from seed 42 and drawing from the same fixed corpus shuffle (seed 42), so the arms are directly comparable.
