# EDT AB-test — Results (13M model, 2026-07-27)

**Spec:** docs/superpowers/specs/2026-07-27-edt-ab-test-design.md (§9 decision rule)
**Redesign:** docs/superpowers/specs/2026-07-28-cte-phaseRoutedMoE-redesign.md (§4.5)
**Plan:** docs/superpowers/plans/2026-07-27-edt-ab-test.md, Task 5 of `2026-07-28-cte-phaseRoutedMoE-redesign.md`
**Budget:** 400 000 tokens/arm, 30 000-token hold-out, chunk_len=16, lr=3e-4.

> **This report is the corrected-engine re-run.** The first run (2026-07-27) was confounded by a frozen-expert defect: the CTE's ad-hoc MoE read a detached `_cached_W` buffer, so Phase 3 could not fine-tune the experts EDT had pre-trained. The engine was refactored to route the MoE through the differentiable `PhaseRoutedMoE` (redesign Tasks 1-3), and the experiment was re-run with experts now trainable in Phase 3. **The corrected re-run confirms the original negative verdict.** Pre-correction numbers are preserved in git history (commit `4b259fe` and earlier) for comparison.

## Headline numbers

| Arm | Hold-out PPL | Token acc | Inter-expert cosine | Wall-clock | phase3_final_loss |
|---|---|---|---|---|---|
| A from-scratch    | 1149.79 | 0.1254 | 0.002 | 3545s | 7.857 |
| B EDT vanilla     | 1321.22 | 0.1251 | 0.216 | 3263s | 6.542 |
| C EDT + spec      | 1325.42 | 0.1246 | 0.095 | 3472s | 6.475 |

(Full precision from `experiments/edt_ab/results.json`; the text log `run_v2.log` prints ppl/div/time per arm at the same precision shown here. `phase3_final_loss` is the final Phase-3 training-stream cross-entropy recorded in `results.json`. All three arms ran in a single detached process, each building a fresh engine from seed 42 and drawing from the same fixed corpus shuffle (seed 42), so the arms are directly comparable. Single seed per arm.)

## Pre-registered verdicts (spec §9, applied verbatim)

- **EDT accelerates (A vs B): NOT SUPPORTED.**
  Rule: `ppl_B ≤ 0.95 · ppl_A`.
  thr_AB = 0.95 × 1149.79 = **1092.30**. Is 1321.22 ≤ 1092.30? → **FALSE**.
  EDT vanilla is ~15 % worse than from-scratch on hold-out perplexity (1321.22 vs 1149.79), and the accuracy ordering agrees (A 0.1254 ≥ B 0.1251).

- **Specialization matters (B vs C): NOT SUPPORTED on ppl, but the diversity sub-condition IS met.**
  Rule: `ppl_C ≤ 0.95 · ppl_B` AND `diversity_C ≤ diversity_B − 0.05`.
  → ppl limb: thr_BC = 0.95 × 1321.22 = **1255.16**. Is 1325.42 ≤ 1255.16? → **FALSE** (ppl limb fails).
  → div limb: 0.216 − 0.05 = 0.166. Is 0.095 ≤ 0.166? → **TRUE** (diversity limb met).
  Conjunctive rule → **FALSE (NOT SUPPORTED)**. Partitioning experts onto disjoint data DID produce markedly more distinct experts (cosine 0.095 vs 0.216), so the specialization mechanism works as designed, but it did not improve final perplexity (C is ~0.3 % worse than B).

## Interpretation

**A-vs-B (EDT acceleration): NOT SUPPORTED. The corrected-engine re-run confirms the original negative verdict — this is not an artifact of the frozen-expert defect.** At equal token budget, EDT vanilla (Arm B) reached worse hold-out perplexity than plain online training (Arm A): 1321.22 vs 1149.79, a ~15 % gap in the wrong direction, with a matching small regression in token accuracy (0.1251 vs 0.1254). The pre-correction run showed essentially the same gap (1504.80 vs 1310.25, also ~15 %), so the frozen-expert defect was **not** the cause of EDT underperforming: even with experts now fine-tunable in Phase 3, EDT does not accelerate learning at this scale. This is a robust negative result, not a hedge — the regime is now fair, and the verdict holds. No multi-seed re-run is triggered: the A-vs-B gap (15 %) is well outside the 5 % re-run threshold and points in the wrong direction.

**B-vs-C (specialization): trustworthy, and still informative.** The only variable between B and C is the Phase 1 call (shared bank vs disjoint per-expert banks); Phase 2b and Phase 3 are byte-identical. Arm C produced experts with much lower inter-expert cosine (0.095 vs B's 0.216), confirming that disjoint per-expert data does force divergence — the design gap the experiment was meant to test (the original `edt_full.py` fed all experts identical data and they collapsed toward a shared expert). However C's final perplexity did not improve; on a 4-expert single-MoE model with position-based (not semantic) partitioning, the extra diversity did not help next-token prediction. C is ~0.3 % worse than B on ppl. The ppl gap is tiny and well under the 5 % re-run threshold, and the diversity gap is large (0.121) — but the conjunctive §9 rule fails on the ppl limb regardless, so no multi-seed re-run is triggered for B-vs-C.

**Train-loss vs hold-out perplexity (suggestive, partly confounded).** Arm A's `phase3_final_loss` (7.857) is *higher* than B's (6.542) and C's (6.475), yet A has *lower* hold-out perplexity (1149.79 vs 1321/1325). The hold-out ordering is a clean signal (it is evaluated on a fixed unseen set, identical across arms). The train-loss direction is **only suggestive**, because `phase3_final_loss` is the cross-entropy of the *final* Phase-3 chunk, and the arms are at different trajectory positions: A trains over 400k tokens (final chunk ≈ 25000), B/C over 220k tokens (final chunk ≈ 13750) — so the comparison conflates trajectory position with fitting intensity. Taken cautiously, the pattern is consistent with EDT pre-training pushing the model toward a configuration that Phase 3 then refines on the training stream at the cost of generalization — but this would need matched-trajectory-point measurements (e.g. all arms evaluated at the same chunk index) to state firmly. A single seed and a 13M model make this suggestive rather than conclusive.

## Limitations

1. **Token-budget overlap (spec-compliant, but disadvantages EDT in A-vs-B).** Arm A trains over `train[:400000]` (400k distinct source tokens). Arms B/C draw Phase 1 from `train[:60000]`, Phase 2b from `train[:120000]`, Phase 3 from `train[:220000]` — all nested prefixes. So B/C touch ~220k distinct source tokens vs A's 400k (~1.8×). The spec §5 defines budget as "tokens consumed" (step counts summing to N=400k), so this is spec-compliant, but Arm A's win could partly come from seeing more distinct data. Symmetric for B-vs-C (both use the same nested slices), so it does not threaten that comparison. The corrected re-run did not change this design; the A-vs-B gap (~15 %) is large enough that overlap alone is an unlikely full explanation, but it remains a confound.

2. **Diversity metric implementation note.** The spec §7 wording suggests a 4×4 mean-cosine over per-sample cosines; the implementation flattens each expert's (256, d_model) output batch into one vector and takes one cosine per pair. Applied identically to all arms, so no bias; the qualitative ordering (A≈0 < C=0.095 < B=0.216) is robust to either definition.

3. **The 13M is a deliberate test of principle** (spec §8 caveat 1): 4 experts and a single MoE (no 16-layer stack) is a very different regime from Fractus-1B (128 experts × 16 layers). A negative result here does NOT refute EDT-1B. Single seed per arm. Specialization is by-position, not semantic (mono-domain corpus). Phase 2a omitted at this scale (no layer stack).

## Root-cause analysis (post-hoc, 2026-07-29)

After this report, a diagnostic investigation (`scripts/diagnose_hypothesis_b.py`, `diagnose_hypothesis_b_robust.py`, `diagnose_routing.py`) identified **two structural defects** that explain why EDT underperformed. These are properties of the EDT-Fractus pairing, not of this experiment's wiring.

**Defect 1 — Phase-1 objective is misaligned with Phase-3.** Phase 1 trains each expert to *predict the next hidden state* (MSE regression `h_t → h_{t+1}`); Phase 3 uses the expert as a *nonlinear transformation in a stack* (next-token CE). These objectives are not aligned. Measured across 12 (seed, block, expert) configs at reduced scale: the Pearson correlation between Phase-1 MSE and full-model hold-out PPL was **never positive** — 3/12 strongly negative (−0.99, −0.71, −0.69), 9/12 ≈ 0. An expert that minimizes Phase-1 MSE is at best neutral and at worst harmful to the final CE. This is consistent with the train-loss-vs-holdout pattern observed in the main experiment (EDT fit training tokens better, generalized worse).

**Defect 2 — the Kuramoto router ignores most experts, so most Phase-1 training is wasted.** At the 13M config (4 experts, top_k=2), measuring which experts the router actually selects on hold-out tokens: only **2 of 4 experts are ever routed** (each on 100% of tokens); the other 2 are *never* selected. Phase 1 trains all 4, but Phase 3's router ignores half of them — their training is invisible to the final loss. This explains the 9/12 zero-correlation configs: the trained expert simply wasn't routed. The router concentrates on the experts whose Farey phases are closest to the mean token phase; the rest are structurally dead weight in this regime.

**Implication.** The two defects compound: EDT pre-trains experts that the router then ignores (Defect 2), and for the few experts that *are* routed, the pre-training objective is misaligned (Defect 1). Both are addressable — see the follow-up design for "EDT-routé" (pre-train only routed experts) + routing relaxation. The 13M negative verdict stands and is now *explained*, not merely observed.

4. **`phase3_final_loss` is not directly comparable across arms.** It is the cross-entropy of the final Phase-3 chunk, sampled at different trajectory positions (chunk ≈ 25000 for A, ≈ 13750 for B/C) over different (nested) data prefixes. It is reported for completeness only; the headline verdicts rest on hold-out perplexity, which is a clean apples-to-apples metric.

> **Note on the prior Limitation §1 (frozen-expert Phase 3): REMOVED — the defect is fixed.** The original run reported that `fractus/continuous_engine.py` read the experts' detached `_cached_W` buffer directly, freezing 0/36 expert params during Phase 3. This was fixed by refactoring the CTE to route its MoE through the differentiable `PhaseRoutedMoE` (low-rank-capable; redesign spec `2026-07-28-cte-phaseRoutedMoE-redesign.md`, commit `92f5d35`). The numbers in *this* report come from the corrected engine, where Phase 3 *does* fine-tune the experts EDT pre-trained — and the headline A-vs-B verdict is unchanged. The `residual_siren` freeze mentioned in the old §4 is also obsolete: the new MoE has no `residual_siren`.

## Provenance

- All numbers from the corrected-engine run: `experiments/edt_ab/results.json` (the canonical aggregate; gitignored as a generated artifact) and `experiments/edt_ab/run_v2.log` (the text log printing ppl/div/time per arm). The `phase3_final_loss` column is taken from `results.json`. Wall-clock values are from `results.json` (the `run_v2.log` times agree to the second).
- **Pre-correction numbers are preserved in git history** (commit `4b259fe` and earlier versions of this file) for direct comparison: pre-correction A=1310.25 / B=1504.80 / C=1539.07; corrected A=1149.79 / B=1321.22 / C=1325.42. The absolute perplexities improved (the corrected engine trains better overall), but the A-vs-B *gap* stayed at ~15 % — the key robustness check.
- Samples: `experiments/edt_ab/samples/{A_from_scratch,B_edt_vanilla,C_edt_spec}.txt` (degenerate/repetitive — expected for a lightly-trained 13M; all three arms produced near-identical "Theinal."-style output).
- Code: `experiments/edt_ab/ablib.py`, `scripts/ab_edt_13m.py` (branch `edt-ab-test`), running against the refactored `PhaseRoutedMoE` engine.
- Single seed per arm. The A-vs-B gap (~15 %) is well outside the 5 % re-run trigger and in the wrong direction, so no multi-seed re-run is triggered for that pair. The B-vs-C ppl gap is tiny (~0.3 %) and the diversity gap is large (0.121), but the ppl limb of the conjunctive rule fails regardless, so no multi-seed re-run is triggered for B-vs-C either.
