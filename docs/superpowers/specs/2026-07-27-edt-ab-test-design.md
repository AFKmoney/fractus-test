# EDT AB-test on the 13M model — Design spec

**Date:** 2026-07-27
**Author:** Philippe-Antoine Robert (with ZCode)
**Status:** Design — pending user review
**Scope:** Single experiment, fully contained in `fractus-test`. No changes to the Fractus core.

---

## 1. Motivation

The Fractus project makes a strong claim in `docs/EDT.md`:

> **EDT reduces training time for a 1-billion-parameter sparse MoE model from 358 days to ~2 days on a single consumer GPU.**

The white paper (`Fractus_White_Paper.pdf`, §14) and the EDT doc both frame EDT as
a universal MoE accelerator that applies to *any* sparse-MoE model, transformer or
not. The codebase currently carries **two inconsistent implementations**:

- `scripts/edt_pipeline.py` + `scripts/edt_full.py` train Phase 1 experts on
  **Gaussian noise** (`h = randn`, `target = shift(h) + 0.1·noise`). A
  position-wise FFN cannot learn a positional shift, and all experts receive
  identical data, so they converge toward a single shared expert.
- `docs/EDT.md` corrects part of this: Phase 1 feeds experts **real hidden
  states from the embedding**. But all experts still see the **same** hidden
  bank with the **same** objective, which provides no mechanism for the
  specialization that §11 of the white paper and `fractus/specialization.py`
  require (orthogonal, domain-owning experts).

The project's stated discipline is *"measure, do not claim."* The 189× figure
and the universality claim are currently **estimates and arguments**, not
measurements. This experiment is the first controlled measurement of whether
EDT actually accelerates learning at all, and whether the missing
specialization mechanism matters.

**The 13M model is a test of principle, not a refutation at 1B scale.** A
negative result here does not refute EDT-1B (4 experts, no layer stacking is a
very different regime from 128×16). But a positive result *supports* the
claim, and a positive specialization result directly validates the design gap
identified above.

## 2. Claims to adjudicate

| Comparison | Claim under test |
|---|---|
| EDT-vanilla  vs  from-scratch | EDT accelerates learning: at equal token budget, EDT reaches lower hold-out perplexity. |
| EDT-spec     vs  EDT-vanilla  | Pre-training experts without a specialization mechanism wastes them; routing each expert to a disjoint data partition improves final quality and lowers inter-expert cosine. |
| EDT-spec     vs  from-scratch | The corrected EDT beats the from-scratch baseline. |

## 3. Architecture under test

The small model from `scripts/train_13m_rag.py`:

```python
ContinuousThoughtEngine(
    vocab_size=50257, d_model=128, n_heads=2, d_head=64,
    n_levels=2, n_oscillators=8, coupling_rank=4,
    n_experts=4, top_k=2, expert_d_ff=128, siren_rank=32,
)
# ~13M parameters.
```

Relevant internal structure (from `fractus/continuous_engine.py`):

- `engine.observe` — input embedding (`nn.Embedding(vocab, d_model)`). This is
  the "separable embedding" EDT requires.
- `engine.experts_w1[i]`, `engine.experts_w2[i]` for `i in range(4)` — the
  MoE experts (`CachedStructuredSirenLinear`), independently trainable.
- `engine.attn`, `engine.norm_attn`, `engine.kuramoto`, `engine.norm_kur`,
  `engine.norm_moe` — the rest of the engine.
- `engine.output_head` — `nn.Linear(d_model, vocab, bias=False)`, **not** tied
  to `observe`. Used as the LM head for cross-entropy.
- `engine.tick(observation)` — single-token forward; the unit of the online
  trainer.
- `OnlineTrainer` (`fractus/train/online.py`) — online SGD with two methods:
  `train_on_stream` (1 backward/token) and `train_on_stream_minibatch`
  (gradient accumulation, faster).

Note the CTE is **not** a stack of 16 blocks: it is a single attention + single
MoE. Phase 2a (per-attention-layer pre-training) therefore does not apply at
this scale and is **omitted** from all arms. Only Phase 1 (experts), Phase 2b
(embedding), and Phase 3 (joint) are exercised. This is consistent with the
EDT doc, which describes Phase 2a as "<1 second, negligible" — its omission
changes nothing.

## 4. The three arms

All arms use the **same engine construction**, the **same RNG seed (42)** for
weight init, the **same corpus shuffle**, and the **same total token budget N**.

### Arm A — From-scratch (baseline)

`OnlineTrainer.train_on_stream_minibatch` directly on N tokens from the train
split. No pre-training phases. This is the honest baseline the EDT doc compares
against.

### Arm B — EDT vanilla (faithful to `docs/EDT.md`)

1. **Phase 1 — experts (shared bank).** Generate `hidden_bank` as pairs
   `(h_t, h_{t+1})` by running `engine.observe(tokens)` (no_grad) on
   **consecutive** positions of each sampled chunk — i.e. for a chunk
   `tokens[c:c+L]`, emit `(observe(tokens[c+t]), observe(tokens[c+t+1]))` for
   each `t`. All 4 experts are trained **independently** with
   MSE(`expert_w2(gelu(expert_w1(h_t)))`, `h_{t+1}`), all sampling from the
   **same shared** bank. The pairs are positionally aligned within a chunk;
   we do **not** cross pairs across independently sampled chunks (that is the
   non-learnable target the original `edt_full.py` inadvertently created).
2. **Phase 2b — embedding + readout.** Freeze every parameter except
   `engine.observe` **and** `engine.output_head`; train both with
   cross-entropy next-token. Rationale: on the CTE these two are **not
   weight-tied** (unlike the 1B `Fractus1B`, where `lm_head` shares the
   embedding table). Training `observe` against a frozen random `output_head`
   would teach the embedding to invert an arbitrary projection, which is not
   what Phase 2b is meant to do. Training the minimal two-table LM
   (`observe` + `output_head`, no layers) is the faithful CTE analogue of the
   doc's "train the embedding alone."
3. **Phase 3 — joint.** Unfreeze everything; `OnlineTrainer` on the remaining
   train tokens, same regime as Arm A.

Total tokens consumed across all three phases = N.

### Arm C — EDT + specialization (the corrected variant)

Identical to Arm B **except Phase 1**: the train tokens are partitioned into 4
**disjoint** contiguous slices, one per expert. Each expert's hidden bank is
drawn **only** from its own slice. This is the **only** variable that differs
from Arm B, so the B-vs-C comparison isolates the specialization mechanism.

Rationale for contiguous-by-position partition rather than semantic (code vs
prose): the source corpus is essentially mono-domain, so a semantic split is
not available without an external labeling step that would introduce its own
confound. Position-based partition is the cleanest control — the only change
is "shared data vs disjoint data."

## 5. Budget and accounting

- **N = 400 000 tokens per arm** (≈ 1 h/arm at the measured ~117 tok/s;
  total ≈ 3–4 h, matching the user's "Moyen" budget choice).
- **Source corpus:** `data/communication_corpus.pt` (18 645 351 tokens,
  int32). A single fixed shuffle (seed 42) is applied once; the first N tokens
  of the shuffle are the train budget shared across all arms, the next 30 000
  are the hold-out.
- **Phase split for arms B and C** (by tokens consumed from the budget):
  - Phase 1 (experts): **15 %** = 60 000 tokens
  - Phase 2b (embedding): **30 %** = 120 000 tokens
  - Phase 3 (joint): **55 %** = 220 000 tokens

  Rationale: the EDT doc assigns the bulk of cost to Phase 3 (41 h / 45 h
  total). Phase 1 is light because experts on the 13M are tiny.

  Arm A consumes all N = 400 000 tokens in a single online pass (its analogue
  of "Phase 3").

- **Normalization unit:** tokens consumed from the corpus (the input signal).
  This is what the EDT doc counts (500 M / 100 M). Wall-clock is recorded for
  information but is not the basis of comparison, because different phases
  have very different tok/s and time-normalization would mix variables.

## 6. Data

- **Train:** 400 000 tokens per arm, drawn from the same fixed shuffle.
- **Hold-out:** 30 000 tokens, never seen during training, identical across
  arms. Used for the primary perplexity comparison.
- **Arm C domain partition:** the 60 000 Phase-1 tokens are split into 4
  contiguous 15 000-token slices; expert `i` sees only slice `i`. The Phase 3
  (joint) tokens are **not** partitioned — all experts see all of Phase 3
  through the live Kuramoto routing, exactly as in Arm B.

## 7. Metrics

| Metric | When | Role |
|---|---|---|
| **Hold-out perplexity** | end of each arm | **primary** — the headline comparison |
| Hold-out token accuracy | end of each arm | secondary |
| Phase 3 loss curve | every minibatch of Phase 3 | convergence-speed visualization |
| **Inter-expert cosine diversity** | end of each arm | diagnostic for specialization. Probe: feed a **fixed, shared** batch of 256 hidden vectors — `engine.observe(tokens)` over a held-out probe slice of 256 tokens, identical for all arms — through each expert's FFN (`expert_w2(gelu(expert_w1(h)))`), compute the 4×4 mean-cosine matrix of the 4 expert output batches, report the mean of the off-diagonal entries. Spec (Arm C) expected to be lower than vanilla (Arm B). |
| Generative sample | end of each arm | qualitative; fixed prompt, greedy decode, 80 tokens |
| Wall-clock per phase | recorded throughout | informational only |

Final output: `experiments/edt_ab/results.json` (all numeric metrics) plus
`experiments/edt_ab/loss_curves.png` and `experiments/edt_ab/diversity.png`.

## 8. Interpretation caveats (to appear verbatim in the report)

1. **The 13M under-tests EDT.** 4 experts and a single MoE (no 16-layer
   stacking) is a very different regime from Fractus-1B (128 experts × 16
   layers). A negative result here does **not** refute EDT-1B; a positive
   result supports it. The 13M is a test of principle.
2. **Single seed per arm** given the "Moyen" budget. Noise is real. If the
   gap between any two arms is smaller than a pre-registered threshold (see
   §9), that specific **pair** of arms is re-run with 3 seeds each (six
   additional runs) before any conclusion is drawn for that pair; the other
   pair is unaffected.
3. **Specialization is by-position, not semantic.** The corpus is
   mono-domain. Conclusions about "specialization" are limited to this
   controlled sense and do not directly demonstrate code-vs-prose skill
   ownership.
4. **Phase 2a is omitted** at this scale (no layer stack); this matches the
   EDT doc, which calls Phase 2a negligible.

## 9. Pre-registered decision rule

To avoid post-hoc rationalization, the interpretation is fixed in advance:

- **EDT accelerates** (A vs B) is *supported* iff B's hold-out perplexity is
  at least **5 % lower** than A's. Below 5 %, the comparison is declared
  *inconclusive at this scale* and re-run with 3 seeds.
- **Specialization matters** (B vs C) is *supported* iff C's hold-out
  perplexity is at least 5 % lower than B's **and** C's inter-expert cosine
  is at least 0.05 lower than B's. Both conditions required.
- Otherwise the corresponding claim is *not supported at this scale* (with
  caveat 1 applying).

5 % is chosen because it is well above run-to-run noise typically observed on
small LM fine-tunes and is the smallest gap that would be practically
meaningful.

## 10. Code plan

New, isolated code only — no edits to the Fractus core. All under
`fractus-test/`:

```
scripts/ab_edt_13m.py            # entrypoint; orchestrates the 3 arms
experiments/edt_ab/              # outputs (gitignored)
  results.json
  loss_curves.png
  diversity.png
  samples/
    arm_a.txt arm_b.txt arm_c.txt
```

Functions (each independently testable):

```
build_engine(seed=42) -> ContinuousThoughtEngine     # fresh 13M, fixed init
load_corpus(path) -> dict                            # train (400k), holdout (30k),
                                                     #   train_phase1 (60k),
                                                     #   domain_split (4×15k)
phase1_experts_shared(engine, hidden_bank, steps)    # Arm B Phase 1
phase1_experts_partitioned(engine, banks, steps)     # Arm C Phase 1
phase2b_embedding(engine, tokens, steps)             # shared by B and C
phase3_joint(engine, tokens, steps)                  # shared by B, C, and A
                                                     #   (A calls it with all N)
arm_from_scratch(engine, train, holdout, budget)
arm_edt_vanilla(engine, train, holdout, budget)
arm_edt_spec(engine, train, holdout, budget, domain_split)
evaluate_ppl(engine, holdout) -> float
expert_diversity(engine, probe_batch) -> float       # mean off-diagonal cosine
greedy_sample(engine, prompt, n_tokens) -> str
run_all()                                            # writes results.json + plots
```

Reuses `OnlineTrainer`, `ContinuousThoughtEngine`, `FractusTokenizer`
unchanged. The Phase 1 / Phase 2b helpers operate directly on
`engine.observe`, `engine.experts_w1/w2`, `engine.output_head` — the same
attributes the existing EDT scripts already touch — so no core changes are
needed.

## 11. Out of scope

- No 1B run, no GPU run, no multi-seed sweep beyond the §9 re-run trigger.
- No changes to the Fractus model code, the existing EDT scripts, or the
  published checkpoints.
- No semantic corpus labeling (would require an external classifier and
  introduces its own confound).
- No Phase 2a (per-attention-layer pre-training); N/A on a single-MoE model.

## 12. Risks

| Risk | Mitigation |
|---|---|
| Phase 1 MSE on `next_hidden` is unstable (embedding output near-zero at init) | Generate the hidden bank **after** Phase 2b would order-violate; instead initialize `observe` with the same scheme the online trainer expects and clip Phase 1 loss. If still unstable, fall back to denoising target `h + 0.1·noise` (the `edt_full.py` target) but document it. |
| 400k tokens is too few for the from-scratch arm to move off init loss | Pre-check: run 20k tokens from-scratch first; if loss does not drop materially, raise N for all arms together (budget permitting) rather than biasing one arm. |
| Hold-out perplexity is infinite (model assigns zero prob somewhere) | Report per-token NLL capped at 20, and accuracy, alongside raw PPL. |
| `CachedStructuredSirenLinear` keeps an internal cache that breaks when weights are edited by an external optimizer | Call the layer's cache-invalidation hook (or rebuild cache) before each Phase 1 expert's forward; covered by a unit test before the full run. |
