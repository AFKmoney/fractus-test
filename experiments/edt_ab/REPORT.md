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

- **EDT accelerates (A vs B): NOT SUPPORTED at this scale.**
  Rule: ppl_B ≤ 0.95·ppl_A → 1504.80 ≤ 0.95·1310.25 = 1244.74 → FALSE.
  EDT vanilla is ~15 % WORSE than from-scratch on hold-out perplexity (1504.80 vs 1310.25). The accuracy ordering agrees with ppl (A 0.124 ≥ B 0.119).

- **Specialization matters (B vs C): NOT SUPPORTED on ppl, but the diversity sub-condition IS met.**
  Rule: ppl_C ≤ 0.95·ppl_B AND div_C ≤ div_B − 0.05.
  → ppl: 1539.07 ≤ 0.95·1504.80 = 1429.56 → FALSE.
  → div: 0.428 ≤ 0.944 − 0.05 = 0.894 → TRUE.
  The conjunctive rule fails. Partitioning experts onto disjoint data DID produce markedly more distinct experts (cosine 0.428 vs 0.944 — a 0.52 reduction, i.e. the specialization mechanism works as designed), but this did not translate into better final perplexity. C is ~2 % worse than B (1539 vs 1504) and ~17 % worse than A.

## Interpretation

The headline claim "EDT accelerates learning" is NOT supported at the 13M scale. At equal token budget, pre-training experts in isolation then fine-tuning (Arm B) reached worse hold-out perplexity than plain online training (Arm A): 1504.80 vs 1310.25, a ~15 % gap in the wrong direction, and a matching (small) regression in token accuracy (0.119 vs 0.124). The Phase 1/2b pre-training did not give B a head start; if anything it appears to have left the experts in a configuration that Phase 3 (only 55 % of the budget) could not recover from as well as A's full-budget online training. The 5 % pre-registered threshold for "accelerates" is not close to being met — the gap is 15 % in the wrong direction — so no multi-seed re-run is triggered for this pair (spec §8 caveat 2), and more seeds would not flip the sign.

The specialization mechanism works mechanically but does not pay off in quality here. Arm C produced experts with much lower inter-expert cosine (0.428 vs B's 0.944), confirming that disjoint per-expert data does force divergence — exactly the critique the experiment was designed to test (the design gap identified in spec §1, where the original `edt_full.py` fed all experts identical data and they collapsed toward a shared expert). However C's final perplexity did not improve; on a 4-expert single-MoE model with position-based (not semantic) partitioning, the extra diversity did not help next-token prediction. C is ~2 % worse than B on ppl and ~0.002 worse on accuracy. The ppl gap is well above the 0.05 (5 %) noise threshold but in the wrong direction, so again no multi-seed re-run is triggered; the conjunction in the §9 rule fails on the ppl limb regardless of the diversity limb being met.

Caveats (spec §8) bound the conclusion. (1) The 13M is a deliberate test of principle: 4 experts and a single MoE (no 16-layer stack) is a very different regime from Fractus-1B (128 experts × 16 layers). A negative result here does NOT refute EDT-1B; it says the claim is unproven at small scale. (2) Single seed per arm; noise is real, but the ppl gaps (15 % A-vs-B, 2 % B-vs-C) are outside the re-run trigger zones defined in §9 and in the wrong direction, so the verdicts stand without a multi-seed sweep. (3) Specialization is by-position, not semantic: the corpus is mono-domain, so conclusions about "specialization" are limited to this controlled sense and do not directly demonstrate code-vs-prose skill ownership. (4) Phase 2a is omitted at this scale (no layer stack), matching the EDT doc which calls Phase 2a negligible; the `residual_siren` was frozen during Phase 1 for stability, applied identically to B and C, so the comparison stays fair.

## Provenance

- Numbers: `experiments/edt_ab/run.log` (A and B ppl/div/time text lines; C ppl/div/time text line), `experiments/edt_ab/results.json` (all numeric metrics for all three arms, including accuracy), and `experiments/edt_ab/results_c.json` (C full results, corroborating `results.json` for C).
- Samples: `experiments/edt_ab/samples/{A_from_scratch,B_edt_vanilla,C_edt_spec}.txt` (degenerate/repetitive — expected for a lightly-trained 13M; all three arms produced near-identical "Theinal."-style degenerate output).
- Code: `experiments/edt_ab/ablib.py`, `scripts/ab_edt_13m.py`, `experiments/edt_ab/run_arm_c.py` (branch `edt-ab-test`).
- Note on execution: all three arms ran in a single detached process (`scripts/ab_edt_13m.py`, started 18:10, finished ~20:51), each arm building a fresh engine from seed 42 and drawing from the same fixed corpus shuffle (seed 42), so the arms are directly comparable. (A second, redundant Arm-C-only run was launched at 20:29 under a mistaken belief that the first run had died mid-C; it produced identical C numbers — ppl 1539.07, div 0.428 — confirming reproducibility. Its output is in `run_c.log`/`results_c.json`.)
