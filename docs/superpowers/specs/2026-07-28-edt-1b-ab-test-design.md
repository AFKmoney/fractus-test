# EDT AB-test on the 1B model — Design spec

**Date:** 2026-07-28
**Author:** Philippe-Antoine Robert (with ZCode)
**Status:** Design — pending user review
**Scope:** New code only. Implements a faithful EDT pipeline for `Fractus1B`, with a 3-arm AB test mirroring the 13M experiment. The full run requires GPU compute that is **not yet available**; this spec delivers code that is correct, unit-tested, smoke-validated on CPU, and ready to launch on GPU.

---

## 1. Motivation

The 13M AB-test (`experiments/edt_ab/REPORT.md`) concluded EDT does **not** accelerate learning at the 4-expert single-MoE scale, even after the CTE was fixed to route its MoE through the differentiable `PhaseRoutedMoE`. The report's §3 limitation is explicit: *"4 experts and a single MoE (no 16-layer stack) is a very different regime from Fractus-1B (128 experts × 16 layers). A negative result here does NOT refute EDT-1B."*

EDT is fundamentally a large-scale method. The white paper's headline claim (189× speedup) and the EDT doc (`docs/EDT.md`) are both framed around **128 experts × 16 layers = 2048 experts**, pre-trained independently. Testing EDT on 4 experts × 1 layer is like testing a distributed algorithm on one node.

**This spec prepares the decisive test.** The compute requirement (1.6 tok/s measured on the dev CPU; Chinchilla-scale = ~1.76B tokens for 88M trainable params) makes a full CPU run infeasible (~3 weeks for an under-budget, contestable result). The decision, taken with the user: **build the code now, validate it on CPU at smoke scale, run it on GPU when compute is available.**

## 2. Claims to adjudicate

Same structure as the 13M experiment, at 1B scale:

| Comparison | Claim under test |
|---|---|
| EDT-vanilla  vs  from-scratch | At equal token budget and Chinchilla-scale data, EDT reaches lower hold-out perplexity. |
| EDT-spec     vs  EDT-vanilla  | Per-expert disjoint data partitioning (specialization) improves final quality and lowers inter-expert cosine. |
| EDT-spec     vs  from-scratch | Corrected EDT beats from-scratch at the scale EDT was designed for. |

The pre-registered decision rule (spec §9 of `2026-07-27-edt-ab-test-design.md`) is reused verbatim: 5% ppl thresholds, 0.05 diversity threshold, conjunctive rule for B-vs-C.

## 3. Architecture under test

`Fractus1B` (`fractus1B/model_1b.py`), config "K" (0.956B params per the docstring; 1.049B measured):

```python
Fractus1B(
    vocab_size=50257, d_model=1280, n_layers=16, n_heads=20, d_head=64,
    n_levels=2, n_experts=128, top_k=2, expert_d_ff=2048, siren_rank=64,
    max_seq_len=512,
)
```

Relevant internal structure (verified against the codebase):

- **16 `FractalBlockSparse` blocks** stacked. Each block: `LN → FractalLinearAttention → +x → LN → KuramotoLayer → phases → LN → SparseStructuredMoE(hidden, phases) → +x`. Returns `(x, lb_loss)`.
- **`SparseStructuredMoE`** (`fractus1B/model_1b.py:45`): n_experts `LazyStructuredSirenLinear` experts in a `ModuleList`, vectorized gather-first sparse dispatch over the low-rank factors. **End-to-end differentiable** — `torch.stack`/`index_select`/`bmm` preserve the graph to each expert's `U/V/scale/bias`. **Unlike the old CTE, the 1B has NO frozen-expert defect.** Confirmed empirically: each expert is a separate module with its own parameters; no shared `(E,...)` tensor, no detached buffer.
- **`BPEEmbedding`** (`tok_embed` + `pos_embed` + LayerNorm). `lm_head` is `nn.Linear(d_model, vocab, bias=False)` with **`lm_head.weight = embed.tok_embed.weight`** (tied). This differs from the 13M CTE where the head was untied — Phase 2b must account for tying.
- **`forward(ids) -> (logits, aux_loss)`**: `aux_loss` is the summed MoE load-balance loss across all 16 blocks. `_return_hidden=True` flag returns `(hidden, aux_loss)` for chunked-CE (skip `lm_head` in the body, compute it per-chunk in the trainer).
- **`PGSU`** (`fractus1B/pgsu.py`): `PGSU(model, n_active=4)` rotates which 4 of 16 blocks receive gradients per step. Over a cycle, each block is active `n_active` times. Reduces per-step cost (4/16 of the backward) at the price of slower convergence per step.

**API facts (verified):**
- Each expert: `moe.experts_w1[e]` and `moe.experts_w2[e]` are independent `LazyStructuredSirenLinear` modules. `moe.experts_w1[e].parameters()` returns that expert's `[U, V, scale, bias]` only — natural per-expert isolation, **no gradient masking needed** (unlike the 13M redesign).
- A `LazyStructuredSirenLinear` forward is `scale·(x@V)@Uᵀ + bias` — two cheap matmuls, fully differentiable.
- Block access: `model.blocks[l].moe`, `model.blocks[l].attn`, `model.blocks[l].norm1`, `model.blocks[l].norm_kur`, `model.blocks[l].norm_moe`.

## 4. The three arms

All arms use the same `Fractus1B` construction, the same RNG seed for weight init, the same corpus shuffle, and the same total token budget N. Phases consume disjoint or nested slices of the train split per the §5 accounting.

### Arm A — From-scratch (baseline)

Full-model joint training on N tokens. No pre-training. Uses the chunked-CE Phase-3 trainer with PGSU and AMP (bf16 on GPU, no-op on CPU).

### Arm B — EDT vanilla (faithful to `docs/EDT.md`)

1. **Phase 1 — experts (per-layer, shared hidden bank).** For each block `l` in 0..15: run a partial forward of the model up to the *input* of block `l` over a slice of train tokens (no_grad), cache the resulting hidden states as `(h_t, h_{t+1})` aligned pairs (same construction as the 13M `make_hidden_bank`, generalized to per-layer inputs). Then train each of the 128 experts in `model.blocks[l].moe` independently on that shared bank with MSE(`expert_out`, `h_{t+1}`). All 2048 experts see the **same per-layer bank**.
2. **Phase 2a — attention layers (standalone).** For each block `l`: train `blocks[l].attn` + `blocks[l].norm1` standalone on a denoising target (`h + 0.1·noise`, as in the original `edt_full.py` Phase 2a and the EDT doc). Independent per layer.
3. **Phase 2b — embedding + lm_head (tied).** Train `embed.tok_embed` (and the tied `lm_head.weight`) on next-token CE. Freeze everything else, as in the 13M Phase 2b — but because `lm_head` is tied to `tok_embed`, training the embedding *is* training the head.
4. **Phase 3 — joint.** Unfreeze everything; chunked-CE trainer with PGSU + AMP. **`aux_loss` (load-balance) is added to the CE loss** at 1B scale (coefficient 0.001, clamped) — unlike the 13M where it was exposed but unused. The 128-expert routing needs load balancing to avoid collapse.

### Arm C — EDT + specialization (the corrected variant)

Identical to Arm B **except Phase 1**: the Phase-1 train slice is partitioned into **128 contiguous domain slices** (one per expert). Each expert `e` in `blocks[l].moe` is trained only on the hidden bank derived from slice `e`. This is the only variable that differs from B, so B-vs-C isolates the specialization mechanism. At 128 experts (vs 4 at 13M), the specialization has room to matter.

## 5. Budget and accounting

The budget is **parameterizable**; defaults assume GPU Chinchilla-scale, smoke tests use tiny CPU budgets.

- **N (total tokens per arm)**: default `2_000_000_000` (2B, ~Chinchilla for 88M trainable × 20). Smoke CPU test uses `N = 50_000`.
- **Phase split (by tokens consumed)** for arms B and C:
  - Phase 1 (experts): **10 %** = 0.2B. The dominant cost in absolute terms but each of the 2048 experts is tiny (0.43M params, ~2000 steps). Per-layer forward-partial cache makes this tractable.
  - Phase 2a (attentions): **15 %** = 0.3B. Each of 16 attentions is 6.6M params, standalone.
  - Phase 2b (embedding): **30 %** = 0.6B. Tied head.
  - Phase 3 (joint): **45 %** = 0.9B. The "brief alignment" fine-tune.
- **Arm A** consumes all N in a single Phase-3-style joint run.
- **Normalization**: tokens consumed from the corpus (the input signal), matching the EDT doc's accounting and the 13M experiment.

**Smoke-scale caveat:** at the CPU smoke budget (N=50k), Phase 1 = 5k tokens split 128 ways = ~39 tokens/expert — too small to train meaningfully. The smoke tests therefore run with a **reduced config** (`n_layers=2, n_experts=4` via `config_overrides`) so the code paths are exercised without requiring meaningful training. The real GPU run uses the full config and Chinchilla-scale budget. The *code* is identical; only the config differs.

## 6. Data

- **Source corpus for the real run**: TBD at run time. `data/communication_corpus.pt` (18.6M tokens) and `data/ultimate_corpus.pt` (25M tokens) are far below Chinchilla-scale; the real GPU run will need a ~21B-token multi-domain corpus (the EDT doc's `data/fractus_1b_corpus.pt`, to be built via `scripts/build_fractus_1b_corpus.py`). **The code must not hard-code the corpus path** — it takes `--corpus` as a CLI arg, as in the 13M CLI.
- **Hold-out**: 100k tokens reserved, identical across arms. Used for the primary perplexity comparison.
- **Arm C domain partition**: the Phase-1 slice is split into 128 contiguous equal slices, one per expert. (As at 13M, position-based, not semantic — semantic partitioning needs a labeled multi-domain corpus and is out of scope for this spec.)

## 7. Metrics

Same as the 13M experiment (spec §7 of `2026-07-27-edt-ab-test-design.md`):

| Metric | When | Role |
|---|---|---|
| Hold-out perplexity | end of each arm | **primary** |
| Hold-out token accuracy | end | secondary |
| Phase-3 loss curve | during Phase 3 | convergence-speed visualization |
| Inter-expert cosine diversity | end of each arm | diagnostic for specialization. Probe: fixed 256-token batch through each of the 128 experts of a *single representative block* (e.g. block 8, the middle), mean off-diagonal cosine. (Computing it for all 16×128 experts is expensive and unnecessary — the middle block is representative.) |
| Generative sample | end | qualitative, fixed prompt, greedy decode |
| Wall-clock per phase | recorded | informational |

Output: `experiments/edt_1b_ab/results.json`, `loss_curves.png`, `samples/`.

## 8. Components (new code, no core changes)

All under `fractus-test/`, mirroring the 13M layout:

```
experiments/edt_1b_ab/
  __init__.py
  ablib_1b.py            # the 1B EDT helper library (pure functions, unit-testable)
  .gitignore             # results.json, *.png, samples/, *.pt
scripts/
  ab_edt_1b.py           # CLI: orchestrates the 3 arms, writes results.json + plots
tests/
  test_ablib_1b.py       # unit tests for every helper (CPU-fast)
```

**Why a separate `ablib_1b.py` and not extending `ablib.py`:** the 1B has a fundamentally different structure (16-block stack, tied head, `aux_loss`, PGSU, per-layer Phase 1) and a different model class (`Fractus1B` vs `ContinuousThoughtEngine`). Forcing them into one file would create a leaky abstraction. Two focused libraries are cleaner than one branching on model type. Shared constants (decision-rule thresholds, phase fractions) live in `ablib_1b.py` as needed.

### `ablib_1b.py` functions (each independently testable on CPU)

```
build_engine_1b(seed=42, **config_overrides) -> Fractus1B          # deterministic 1B
load_corpus_1b(path, *, n_train, n_holdout, n_phase1, seed=42) -> dict
                                                                    # same split logic as 13M,
                                                                    # but n_phase1 // 128 slices
make_hidden_bank_1b(model, tokens, *, after_block, chunk_len, n_chunks, seed) -> dict
                                                                    # NEW: partial forward up to
                                                                    # the input of block `after_block`
                                                                    # (= output of block after_block-1,
                                                                    # or the embedding if after_block==0),
                                                                    # cache (h_t, h_{t+1}) aligned pairs
phase1_experts_1b_shared(model, bank_per_block, *, steps, lr, seed) -> list
phase1_experts_1b_partitioned(model, domain_split_per_block, *, steps, lr, seed) -> list
                                                                    # each expert trained on its
                                                                    # own per-layer bank; natural
                                                                    # isolation (separate modules,
                                                                    # no gradient masking)
phase2a_attention_1b(model, tokens, *, steps, lr, seed) -> list    # 16 attentions standalone
phase2b_embedding_1b(model, tokens, *, steps, lr, seed) -> list    # tied embedding+head
phase3_joint_1b(model, tokens, *, steps, lr, seed) -> dict         # chunked-CE + PGSU + AMP,
                                                                    # aux_loss added to CE
evaluate_ppl_1b(model, holdout) -> float
expert_diversity_1b(model, probe_tokens, block_idx=8) -> float
greedy_sample_1b(model, prompt, n_tokens) -> str
arm_from_scratch_1b(model, *, train, holdout, budget, ...) -> dict
arm_edt_vanilla_1b(model, *, train, holdout, budget, ...) -> dict
arm_edt_spec_1b(model, *, train, holdout, budget, domain_split_per_block, ...) -> dict
run_all_1b(...)                                                     # used by the CLI
```

### `scripts/ab_edt_1b.py`

Mirrors `scripts/ab_edt_13m.py`. Args: `--budget` (default 2e9), `--n-holdout` (1e5), `--chunk-len` (64, the 1B's max_seq_len), `--lr`, `--corpus`, `--seed`. Auto-detects CUDA for AMP; falls back to CPU. Writes results + plots. On CPU without GPU, prints a clear warning that the run will be extremely slow and exits unless `--force-cpu` is passed (prevents accidental multi-week CPU runs).

## 9. Testing strategy (all CPU-feasible)

Because the full run waits for GPU, **the tests are the primary validation**. Every helper is unit-tested with tiny inputs.

- `test_build_engine_1b_deterministic`: same seed → bit-identical; param count ~1.05B; runs on CPU (4GB RAM) — verify memory headroom.
- `test_make_hidden_bank_1b_per_layer`: for `after_block=0`, bank shape `(P, d_model)`; for `after_block=5`, the bank differs (hidden states have passed through 6 blocks). Confirm partial forward caches the right layer's input.
- `test_phase1_experts_1b_reduces_mse`: train 1 expert (block 0, expert 0) on a tiny bank, MSE decreases. No gradient masking needed — verify isolation is natural (training expert 0 doesn't move expert 1's params, since they're separate modules).
- `test_phase2a_attention_1b_reduces_loss`: 1 attention layer, standalone denoising, loss decreases.
- `test_phase2b_embedding_1b_tied`: training `embed.tok_embed` also updates `lm_head.weight` (assert they remain the same tensor object); frozen params untouched.
- `test_phase3_joint_1b_runs_with_aux_loss`: small Phase 3 run, loss curve returned, `aux_loss` contributes (loss > CE alone).
- `test_evaluate_ppl_1b_finite`, `test_expert_diversity_1b_range`, `test_greedy_sample_1b_string`.
- `test_arm_*_1b_smoke`: each arm runs end-to-end at micro-budget (e.g. 500 tokens), returns the required dict keys. **Slow (~minutes each on CPU) but feasible.**
- **Smoke CLI run**: `python scripts/ab_edt_1b.py --budget 1000 --force-cpu` completes without traceback, writes results.json. Validates the full pipeline.

**Explicitly NOT tested on CPU**: a budget anywhere near Chinchilla-scale. The smoke tests prove the *wiring*; the *result* requires GPU.

## 10. Risks

| Risk | Mitigation |
|---|---|
| Partial forward for Phase 1 is expensive (16 forwards up to layer l, for l=0..15) | Cache per-layer: for each block l, do ONE partial forward over the Phase-1 slice (no_grad), cache the hidden bank; reuse for all 128 experts of that block. Total Phase-1 forward cost ≈ 16 partial forwards, not 2048. |
| 1B model doesn't fit in CPU RAM for unit tests | `Fractus1B` docstring says ~4GB; dev machine has 16GB. Verify with `test_build_engine_1b_deterministic`. If OOM, reduce config in tests via `config_overrides` (e.g. `n_layers=2, n_experts=4`) — the *code paths* are what's tested, not the full scale. |
| `aux_loss` destabilizes Phase 3 if coefficient wrong | Use 0.001 with `clamp(max=1.0)` (same as `edt_pipeline.py`'s Phase 3). Smoke test confirms finiteness. |
| GPU run produces NaNs (bf16, large model) | AMP is auto-detected and opt-out (`--no-amp`); the code should fall back to fp32 gracefully. Document in the CLI help. |
| Hold-out perplexity is infinite (model assigns 0 prob) | Same NLL cap (20) as the 13M `evaluate_ppl`. Report accuracy alongside. |
| The 1B's PGSU rotates which blocks get gradients — does this interact badly with Phase 3 starting from EDT-pretrained weights? | PGSU is symmetric across arms (A, B, C all use it), so it doesn't bias the comparison. It slows convergence uniformly. |
| Tied `lm_head` means Phase 2b updates the embedding AND the output projection simultaneously — different semantics than 13M | Acknowledged in §4 Arm B Phase 2b. The test `test_phase2b_embedding_1b_tied` pins the tying behavior. |

## 11. Out of scope

- Building the 21B-token corpus (separate task; `scripts/build_fractus_1b_corpus.py` exists as a starting point).
- The actual GPU run (awaits compute).
- Semantic (code/prose/maths) domain partitioning — position-based only, same controlled sense as 13M.
- Tuning PGSU `n_active` or AMP precision — defaults (4/16, bf16) are used.
- Multi-seed sweeps (single seed, same as 13M; the §9 rule triggers a re-run only if a pairwise gap is < 5%).

## 12. Success criteria

1. All `tests/test_ablib_1b.py` pass on CPU (including the 3 arm smoke tests at micro-budget).
2. `python scripts/ab_edt_1b.py --budget 1000 --force-cpu` completes, writes a valid `results.json` with the 3 arm keys.
3. The code is structured so that swapping `--budget 2000000000 --corpus data/fractus_1b_corpus.pt` on a GPU machine is the *only* change needed to launch the real run.
4. A `RUNBOOK.md` (in `experiments/edt_1b_ab/`) documents the exact GPU launch command, expected runtime estimate, and how to interpret results against the §9 rule — so the run can be executed later without re-deriving the design.

## 13. Relationship to the 13M result

This experiment does **not** supersede the 13M result — it tests a different regime. If the 1B verdict is also negative, EDT is refuted at every scale tested and the project should stop claiming it works. If the 1B verdict is positive (EDT beats from-scratch), it confirms EDT at the decisive scale *and* explains the 13M negative as a scale artifact. Either outcome is publishable and honest.
