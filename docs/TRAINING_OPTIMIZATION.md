# Fractus CTE — Training Optimization Analysis

**Date:** 2026-08-04
**Measured on:** Ryzen 5 5500U, 12 threads, d_model=128 (palier 0 config)

## Profile: where does the time go?

### Per-iteration breakdown (chunk_len=32, fwd+bwd+step)

| Phase | Time (ms) | % of total |
|---|---|---|
| Forward | 67 | 23% |
| Backward | 122 | 41% |
| Optimizer (AdamW) | 94 | 31% |
| **Total per iteration** | **299** | 100% |

### Forward sub-component breakdown

| Component | Time (ms) | % of forward |
|---|---|---|
| Embedding (observe) | 0.0 | 0% |
| QKV projections | 8.2 | 14% |
| Attention core (causal linear, vectorized) | 13.6 | 23% |
| Kuramoto (encode_from_hidden + RK4 derivative) | 8.5 | 14% |
| MoE (4 experts, low-rank dense) | 16.6 | 28% |
| Output head (1 position, tied) | 12.6 | 21% |

### Key insight

The backward (41%) and optimizer (31%) together are **72%** of training cost.
The forward alone runs at 475 tok/s — the CTE architecture is fast.
Training overhead (autograd + AdamW) is 4x the forward.

## Optimizations applied

### 1. Tied head ✅ (committed)

`output_head.weight = observe.weight`. Halves vocab params (6.4M shared instead of 2×6.4M).
Doesn't reduce the head matmul FLOPs but reduces optimizer state (one less param group).

### 2. Head-partial (`tick_chunk_train`) ✅ (committed)

Head computed on 1 position (last) instead of all C positions.
At C=32: head FLOPs ÷ 32. Head forward: 12.6ms → ~0.4ms.
Head backward: proportionally reduced.

### 3. Gradient accumulation (accumulation_steps=8) ✅ (committed)

Backward every chunk, optimizer step every 8 chunks.
Optimizer steps: 16x fewer (1249→78 on 20k tokens).
Measured: 1.37x tok/s improvement (4→6 tok/s on 20k).

### 4. Chunk_len=32 (committed)

Larger chunk amortizes Python overhead across more tokens.
Attention core benefits (causal vectorized has O(C²) cost but constant per-head).

## Optimizations explored (Fractus-specific)

### 5. Detach Kuramoto from autograd graph (PROPOSED)

**Finding:** Kuramoto takes 8.5ms forward (14% of fwd). The backward through
Kuramoto's `_encode_from_hidden` and `_derivative` adds proportional cost.

**Insight:** The Kuramoto oscillator is a **clock** — it routes tokens via phases.
It should not need gradient to "tell time." The phases are deterministic functions
of the hidden state, and the routing quality depends on the MoE experts, not the
phase computation.

**Proposed change:** In `tick_chunk_train`, wrap the Kuramoto computation in
`torch.no_grad()`. The phases feed the MoE gates (which ARE learned), but the
phase computation itself is frozen. This removes the backward through Kuramoto.

**Expected gain:** ~8.5ms forward + ~8.5ms backward = ~17ms saved per iteration.
At 299ms/iter, that's ~6% speedup. Modest but free.

### 6. Sparse MoE path for low-rank mode (PROPOSED)

**Finding:** MoE is 16.6ms (28% of forward) — the **most expensive component**.
At 4 experts with top-2, the dense path runs all 4 experts. At higher paliers
(16-128 experts), this becomes the dominant cost.

**Current issue:** The sparse path (`_sparse_expert_forward`) falls back to dense
for low-rank mode because `index_select` on low-rank factors wasn't implemented.
The warning `PhaseRoutedMoE: low-rank sparse dispatch not implemented; falling
back to dense forward` appears in every log.

**Proposed change:** Implement the sparse low-rank path: gather only the top-k
experts' U/V factors, compute only those K experts. At top-k=2, this is 2/4=50%
of the matmul work at 4 experts, and 2/128=1.5% at 128 experts.

**Expected gain at 4 experts:** ~8ms saved (dense computes 4 experts, sparse
computes 2). At 128 experts (palier 4): ~98% of MoE work eliminated.

### 7. Attention state carry optimization (EXPLORED)

**Finding:** The (S, z) attention state is detached between ticks (online
paradigm — no BPTT through history). Within a chunk, the attention computes QKV
projections WITH grad.

**Current:** QKV projections (8.2ms) are in the autograd graph. The attention
core (13.6ms) uses the causal vectorized linear attention.

**Possible:** Freeze QKV during progressive growth phases where attention is
already "good enough" (lower paliers). The attention state (S, z) still
accumulates, but QKV params don't update. This removes 8.2ms from backward.

**Risk:** Attention must adapt to the new d_model at each growth palier.
Freezing QKV would prevent adaptation. Only viable if growth preserves the
attention's learned projections (which zero-padding does, for the old dims).

### 8. Vocabulary reduction (PROPOSED, biggest lever)

**Finding:** The head (tied with embedding) is `Linear(d, 50257)`. At d=128:
6.4M params. This is 50% of the 13M model. At d=1280 (1B): 64M params.

**Proposed change:** Train with a reduced vocabulary (8k-16k SentencePiece
instead of 50k GPT-2 BPE). The corpus uses far fewer distinct tokens.

**Expected gain:** Head params ÷ 3-6x. Head FLOPs ÷ 3-6x. At d=128, head
drops from 12.6ms to ~2.5-4ms. At d=1280 (1B), head drops from dominating
to negligible. This is the single largest optimization available.

**Trade-off:** Reduced vocab means worse tokenization for rare words. Acceptable
for training throughput; full vocab can be used for inference via projection.

### 9. torch.compile (AVAILABLE, untested on CPU)

**Status:** Flag `--compile` added to `train_progressive.py`. Calls
`torch.compile(engine.tick_chunk_train, mode="reduce-overhead")`.

**Expected:** On CPU, `torch.compile` can fuse small ops and reduce Python
overhead. The CTE's forward has many small einsums and element-wise ops that
could benefit from kernel fusion.

**Risk:** `torch.compile` has a long warmup (first call recompiles). May not
work with the dynamic shapes in the CTE (thought_state changes during growth).

### 10. Optimizer choice (EXPLORED)

**Current:** AdamW (2 state tensors per param: momentum + variance). At 6.6M
params, optimizer state = 13.2M tensors. The optimizer step (94ms, 31%) reads
and writes all of them.

**Alternative:** SGD with momentum (1 state tensor per param). Halves optimizer
memory and step cost. Risk: SGD converges slower than AdamW for the MoE
low-rank factors.

**Alternative:** 8-bit Adam (bitsandbytes). Halves optimizer state memory.
Only useful if memory is the bottleneck (it's not on CPU — compute is).

## Summary: what gives the most speedup?

| Optimization | Status | Expected tok/s gain | Complexity |
|---|---|---|---|
| Gradient accumulation (accum=8) | ✅ Applied | 1.37x | Done |
| Head-partial (tick_chunk_train) | ✅ Applied | ~1.5x | Done |
| Tied head | ✅ Applied | ~1.1x | Done |
| Chunk_len=32 | ✅ Applied | ~1.1x | Done |
| Detach Kuramoto | Proposed | ~1.06x | Easy |
| Sparse MoE low-rank | Proposed | ~1.1x (4 experts), ~50x (128 experts) | Medium |
| Vocabulary reduction | Proposed | ~3-6x on head | Medium |
| torch.compile | Available | Unknown on CPU | Easy (flag exists) |
| Combined estimate | — | **~5-8x total** | — |

From 77 tok/s baseline → estimated **~400-600 tok/s** with all optimizations
on CPU at d=128. At d=768 (palier 3), proportionally less but still significant.

On GPU: all of these compound with bf16 AMP + CUDA parallelism.
Estimated GPU speedup over CPU: ~50-100x (typical for this workload size).
