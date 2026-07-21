# Fractus — Unified rebuild of the original systems

**Date:** 2026-06-19
**Original author:** the original system's authors (the two prior architectures)
**Status:** Spec validated by brainstorming, awaiting an implementation plan

---

## 1. Context and motivation

The user has two pre-existing systems, both designed by the original author:

- **The first system** (~830 lines, Python + Rust via PyO3) — a 33-page white paper.
  Thesis: "Lightweight AGI via fractal operators on non-Archimedean manifolds."
  Promises: 20.4× compression, O(n) causal reasoning, CPU-only, Lean 4 + ZK-SNARK verification.

- **The second system** (~10,000 lines, pure Rust, 265+ tests) — "A Fractal Neural Network:
  A Unified Architecture for AGI". A fractal transformer with Kuramoto oscillators,
  Farey/von Mises MoE, NOTEARS causality, a proof generator/verifier, and a self-development loop.

An in-depth analysis (code read line by line) revealed that **neither system works
as advertised**, despite genuine mathematical culture and several correctly coded modules.

### Critical errors found (verified in the source code)

**The second system:**
- **No autodiff.** `training.rs:399-426` updates the weights with
  `scale * rand::random::<f64>() * 0.01` (noise) instead of a gradient. The comment
  in `training.rs:156` admits it: "Since we don't have autodiff, we apply a simple loss-scaling signal."
  → The 11-term `AGILoss` is computed but never truly minimized.
- **Fictitious perplexity.** `model.rs:537-546`: `perplexity()` returns a proxy based on
  the embedding norm, not a true perplexity.
- **Empty benchmarks.** `benches/fnn_bench.rs` contains only `fn bench_stub(_c) {}`.
- **Shallow tests.** `tests.rs:1840`: "AGINFNModel is currently a stub with only new()."
  The ~272 tests mostly check tensor shapes (`.dim()`) and finiteness.

**The first system:**
- **The Rust does not compile.** `rust/src/lib.rs:3-4` declares `pub mod causal; pub mod shield;`
  but the files `causal.rs` and `shield.rs` do not exist. The author's TODO confirms this.
- **Fake SIREN.** `torus_siren.py:15-17` uses `nn.SiLU`, not `sin(ω0·)`.
- **The 2-adic vortex is orphaned.** No Python file imports the Rust vortex module.
- **W decompressed then discarded.** `training_loop.py:30-37` computes W, corrects it, then
  the causal operator runs on the raw input (W ignored).
- **Hardcoded numbers.** `training_loop.py:52`: `"compression_ratio": 20.4` (literal).
  `benchmarks.py:43-46`: `min(causal_acc, 0.98)` (clamped exactly at the target).

### Modules that are actually correct (to keep)

- **The second system**: low-rank Kuramoto RK4 (`phase_ode.rs`), von Mises/Farey MoE
  (`moe.rs` + `farey.rs`), causal linear attention (`attention.rs`), NOTEARS (`causal.rs`),
  exact proof verification (`proof.rs`), truncated SVD via power iteration (`svd.rs`),
  Gamma sampling via Marsaglia-Tsang.
- **The first system**: 2-adic arithmetic (`vortex.rs`) — valuation v2, ultrametric distance
  `2^v2(a⊕b)`, norm `2^{-v2}`. The only mathematically correct and non-trivial module.

## 2. Objective

**Demonstrable prototype**: a system that runs on real data, whose loss genuinely
decreases, with a convincing demo (generated text, proven valid theorems, causal queries).

**Explicit non-goals (future work):** publishable paper, commercial product, Lean 4,
ZK-SNARK, scaling to large models, "AGI" in the strong sense.

## 3. Tech stack

**Hybrid Rust + Python**, with a strict separation:

- **Rust (`fractus-core`)**: pure computation, outside the autodiff graph. Exact
  mathematics, proof verification, precomputation, exogenous metrics. No I/O, no dataset.
- **Python (`fractus`, PyTorch)**: trainable model, forward/backward, native autodiff,
  datasets, training loops, logging.
- **Bridge**: PyO3/maturin. Numpy tensors in/out. Rust does not participate in the autodiff
  graph (writing a custom `torch.autograd.Function` for each function would be costly to
  maintain and would lead to fictitious backwards — the trap we want to avoid).

**Honest naming decisions:** "Mandelbrot frequencies" → "Mandelbrot-decayed Fourier basis";
"RKHS Causal Operator" → we implement a true RKHS (with a kernel via RFF in L4),
therefore we keep the name but the substance follows; "Bose-Einstein condensate" (the
original `condensate.rs`) → we do not integrate this module into fractus (incremental SVD
alone does not justify the name); "Lyapunov Shield" → "Lyapunov monitor of the Kuramoto
subsystem"; "Collatz ergodic flow" → "Collatz hash" (the ergodicity of Collatz is unproven,
an open problem).

## 4. Hardware target

User's machine (diagnosed):
- CPU: AMD Ryzen 5 5500U, 6 cores / 12 threads @ 2.1 GHz
- RAM: ~12 GB
- GPU: integrated AMD Radeon (APU), ~4 GB **shared** → ROCm does not support integrated
  AMD APUs under Windows → **effective CPU-only training**.

Consequence: small model (< 1M parameters), tiny dataset (tinyshakespeare ~1 MB),
training in a few hours. Coherent with the original "CPU-only deployment" thesis.

## 5. Repository layout

```
fractus/
├── crate/fractus-core/        # Rust: pure mathematical core
│   └── src/                   #   2-adic vortex, SIREN (ref.), NOTEARS (ref.),
│                              #   Kuramoto/Farey (precomputation), proof verification
├── crate/fractus-py/          # Rust: PyO3/maturin bindings
│   └── src/lib.rs             #   exposes fractus_core to Python
├── fractus/                   # Python: the trainable model (PyTorch)
│   ├── nn/                    #   embedding, blocks, attention, MoE, decoder, siren
│   ├── causal/                #   NOTEARS layer, RKHS, do-calculus
│   ├── reasoning/             #   proofs (GRU), conjectures, ACT
│   ├── stability/             #   Lyapunov (Kuramoto subsystem)
│   ├── metrics/               #   compression, causal (SHD), honest perplexity
│   ├── train/                 #   loops, datasets, losses
│   └── viz/                   #   interactive demos (optional)
├── tests/                     # Rust↔Python integration tests
├── data/                      # tinyshakespeare, math/causal datasets
├── scripts/                   # train.py, demo.py, benchmark.py, serve.py
└── docs/                      # spec, honest revised white paper, results
```

## 6. The 7 implementation layers

Each layer = a design → code → test → standalone demo cycle. We only move to the next one
when the previous one is verified. We can stop at any point with something that works.

### L0 — Technical foundation

**Fixed:** the original does not compile; the Python↔Rust bridge was never functional.

**Components:**
1. `pyproject.toml` with pinned versions (CPU-only torch, maturin, numpy, pytest).
2. `fractus-core` crate: `lib.rs` declares ONLY modules that have a file. Port of the
   original's `vortex.rs` (the 2-adic part, already correct) with corrections: the tautological
   test `assert!(d1 <= d2.max(d1))` → a true ultrametric test `d(x,z) ≤ max(d(x,y), d(y,z))`;
   the unused `HashMap` import removed.
3. `fractus-py` crate: standard maturin configuration (`extension-module`), not the
   misconfigured `[features] python = ["pyo3"]` of the original.
4. A smoke test that crosses everything: `tests/test_smoke.py` — `add_in_rust(2,3)==5` + `torch` available.

**"Done" criterion:** these 4 commands succeed:
`cargo build --release`; `maturin develop --release`;
`python -c "import torch; import fractus"`; `pytest tests/test_smoke.py`.

### L1 — Fractal embedding + 2-adic vortex wired in

**Fixed:** orphaned vortex; misnamed "Mandelbrot frequencies".

**Components:**
1. `fractus/nn/embedding.py`: fractal codepoint embedding (PyTorch). Fourier base with
   Mandelbrot decay `(φ2)−k` (renamed honestly), + 16 morphological features
   (case, digit, punctuation). Trainable parameter via the final `nn.Linear`.
2. `fractus-core/src/vortex.rs`: the 2-adic core ported from the original.
3. **Bridge (validated option B):** the 2-adic Collatz hash is computed in Rust (off-graph,
   exact) and **conditions a trainable MLP** (in the graph) that produces the embedding
   phases. The vortex influences learning without pretending to be differentiable.

**"Done" criterion:** `test_fractal_embedding_shape` (output `[N, d_model]` finite) +
`test_vortex_distance_is_ultrametric` (strong ultrametric inequality on 1000 random triplets).

### L2 — Fractal transformer block (split into L2a + L2b)

**Fixed:** the original does not learn (noise instead of gradients).

**Split (post-brainstorming decision):** L2 is the biggest layer
(~600 lines PyTorch + ~30 tests). We cut it into two independently validable halves.
At the end of L2a we already have a functional fractal transformer
(without Kuramoto/MoE) capable of learning text — the first demonstrable milestone.

**Components (all in pure PyTorch for autodiff):**
1. `fractus/nn/attention.py`: causal linear attention (Katharopoulos `S_t += k_t⊗v_t`),
   feature map `elu(x + ω_k) + 1`. A true `nn.Module` with trainable parameters.
2. `fractus/nn/phase_ode.py`: low-rank Kuramoto RK4 `K = UΛUT`. In pure PyTorch to stay
   in the graph.
3. `fractus/nn/moe.py`: MoE with von Mises routing on Farey phases. Experts = GeLU MLP,
   standard load-balance auxiliary loss.
4. `fractus/nn/block.py`: assembly `LayerNorm → FractalLinearAttention → PhaseSoliton →
   PhaseRoutedMoE`, with KuramotoODE advancing the phases by one step per block.

**L2a (fast demonstrable milestone):**
- `fractus/nn/stats.py`: utilities (`elu_plus_one`, stable softmax, layer_norm).
- `fractus/nn/attention.py`: `FractalLinearAttention` (causal recurrence
  `S_t += φ(k_t) ⊗ v_t`, `y_t = φ(q_t)TS_t / φ(q_t)Tz_t`, feature map
  `elu_plus_one(x + ω_level)`, offsets ω_level = (φ2)^{-level}).
- `fractus/nn/block.py`: minimal `FractalBlock` = LayerNorm → attention → residual.
- Demo: overfit a sequence of toy tokens — the loss must drop.
- **"L2a done" criterion:** `test_block_forward_backward` proves that backward
  propagates a finite AND non-zero gradient to EVERY block parameter.

**L2b (Kuramoto + MoE graft):**
- `fractus/nn/farey.py`: Farey sequence + `expert_phases` (off-graph precomputation).
- `fractus/nn/phase_ode.py`: `KuramotoODE` (low-rank RK4, `encode_from_hidden`,
  `decode_to_bias`, `phase_loss`).
- `fractus/nn/moe.py`: `PhaseRoutedMoE` (von Mises gate, top-k, load-balance loss).
- `fractus/nn/block.py` extended: integrates Kuramoto + MoE into the block.

Rust keeps the pure functions (Farey, `bessel_i0`, `von_mises_pdf`) for precomputation and
off-graph metrics (Kuramoto order parameter).

**"Done" criterion:** `test_block_forward_backward` — `backward()` works, all parameters
receive finite gradients. This is exactly what the original was missing.

### L3 — True SIREN compression + honest measurement

**Fixed:** fake SIREN (SiLU); W decompressed then discarded; 20.4× hardcoded.

**Components:**
1. `fractus/nn/siren.py`: TRUE SIREN on the torus T2. Non-linearity `sin(ω0·(Wx+b))` with
   `ω0 = 30.0` (Sitzmann 2020 empirical value, NOT 56). Evaluate the SIREN on the
   `h×w` grid to regenerate the matrix.
2. Integration: **the attention projections** (`q_proj`, `k_proj`, `v_proj` — the ones that
   are the largest and most compressible) are replaced by `SirenLinear`. The SIREN
   **IS** the matrix, it is in the graph, its parameters are trained. Small matrices
   (LayerNorm, biases) stay dense. The exact ratio is measured (L3.3), not assumed.
3. `fractus/metrics/compression.py`: `measure_compression_ratio(model)` genuinely
   measures the ratio (equivalent dense size / SIREN params). No hardcoded literal.

`fractus-core/src/siren.rs`: non-trained reference implementation for cross-validation
(PyTorch and Rust must give the same output for the same weights).

**"Done" criterion:** `test_siren_produces_real_sinus` (`torch.sin` present, `SiLU` absent) +
`test_siren_is_in_autograd_graph` (SIREN weights receive gradients) +
`test_compression_ratio_is_measured_not_hardcoded` (no `'20.4'` in the source).

### L4 — NOTEARS causality + RKHS on real data

**Fixed:** the "RKHS Causal" that was only a low-rank projection; the trivial do-calculus
(column-zeroing); causal accuracy clamped at 0.98.

**Components:**
1. `fractus/causal/notears.py`: NOTEARS acyclicity penalty `h(W) = tr(e^{W⊙W}) − n` via
   a 20-term Taylor expansion, differentiable, integrated as a loss term.
2. `fractus/causal/rkhs.py`: TRUE RKHS via Random Fourier Features (Rahimi-Recht 2007) —
   approximate Gaussian kernel, operator `L = U @ VT` in feature space.
3. `fractus/causal/do.py`: true Pearl do-calculus (post-intervention sampling),
   not just column-zeroing.
4. Synthetic datasets: `data/causal/generate_scm.py` (known Structural Causal Models),
   `data/causal/lucas.py` (LUCAS, standard).
5. `fractus/metrics/causal.py`: Structural Hamming Distance (SHD), measured causal accuracy
   (no clamp).

`fractus-core/src/causal.rs` (finally created): NOTEARS penalty in pure Rust for cross-validation.
`fractus-core/src/rkhs.rs`: RFF and Gaussian kernel in Rust for precomputation/metrics.

**"Done" criterion:** `test_notears_penalty_is_zero_for_dag` (h(W)≈0 for an obvious DAG) +
`test_notears_penalty_is_positive_for_cycle` (h(W)>0.5 for a cycle) +
`test_causal_recovery_on_known_dag` (SHD ≤ 3 on a 5-variable SCM after 50 steps).

### L5 — Reasoning (verified proofs + conjectures + ACT)

**Fixed:** (the original's proof pipeline was already the most defensible module; we make it
functional).

**Components:**
1. `fractus/reasoning/proof.py`: ProofGenerator GRU trained by **REINFORCE** (policy
   gradient, since verification is non-differentiable). Reward
   `0.6·correctness + 0.3·brevity + 0.1·novelty`.
2. `fractus-core/src/proof.rs`: EXACT verification in Rust (soundness guaranteed). 20 inference
   rules, Fermat/Wilson/GCD specializations. Stays off-graph as a reward oracle.
3. `fractus/reasoning/conjecture.py`: discoverer of falsifiable conjectures (Popperian) —
   10 templates, 6 falsification strategies.
4. `fractus/reasoning/act.py`: Adaptive Computation Time (Graves 2016).

**"Done" criterion:** `test_verify_accepts_valid_proof` + `test_verify_rejects_invalid_proof`
(Rust accepts/rejects correctly) + `test_proof_generator_can_learn_simple_theorem`
(success > 50% on "even+even=even" after 500 REINFORCE steps — ambitious criterion).

### L6 — Lyapunov stability + honest metrics

**Fixed:** fake Lyapunov (tracking `‖y‖2` without a dynamical system); clamped metrics.

**Components:**
1. `fractus/stability/lyapunov.py`: Lyapunov function of the **Kuramoto subsystem** (the
   only true modeled dynamical system). `V(θ) = 1⁄2 Σ (θi − θ*)2`, `dV/dt = ∇V · f(θ) ≤ 0`.
2. `fractus-core/src/lyapunov.rs`: numerical verification in Rust for cross-validation.
3. `fractus/metrics/honest.py`: `honest_perplexity` (true perplexity `exp(val_loss)`, not
   a proxy), `honest_compression`, `honest_causal`.

**Explicit non-claims:** Lyapunov guaranteed only on Kuramoto (not on the whole network);
Lean 4 and ZK-SNARK omitted (absent from the code, noted as future work).

**"Done" criterion:** `test_lyapunov_decreases_on_sync` (V decreases and is monotone on
the synchronization trajectory) + `test_perplexity_is_real` (between 1 and 1000 for a
~100 vocabulary, computed on a real dataset).

### L7 — Demo (final objective)

Three demonstrable demos:

1. **Text generation:** `scripts/train.py --task text --dataset tinyshakespeare --epochs 5`
   then `scripts/generate.py`. Validation loss traced, dense vs SIREN comparison.
2. **Mathematical reasoning:** `scripts/train.py --task proofs` then `scripts/prove.py
   --theorem even_plus_even`. "% valid proofs" vs steps curve.
3. **Causal inference:** `scripts/train.py --task causal --dataset lucas` then
   `scripts/causal.py --query`. SHD reported, counterfactual vs observational answer.

CPU-only deployment: `scripts/serve.py --cpu-only` → local HTTP API on the Ryzen 5.

## 7. Correction summary

| Layer | Fixed | "Done" criterion |
|---|---|---|
| L0 Foundation | The original does not compile | `pytest test_smoke` crosses Python→Rust |
| L1 Embedding+Vortex | Orphaned vortex | Vortex conditions an MLP + ultrametric tested |
| L2 Transformer block | The original does not learn | `backward()` propagates finite gradients everywhere |
| L3 SIREN | Fake SIREN, 20.4× hardcoded | True `sin(ω0·)`, W used, ratio measured |
| L4 Causal | Fake RKHS, fake do-calculus | NOTEARS recovers a synthetic DAG (SHD test) |
| L5 Reasoning | (already well coded) | Generator succeeds >50% after 500 REINFORCE steps |
| L6 Stability | Fake Lyapunov, clamped metrics | V decreases on Kuramoto, real perplexity |
| L7 Demo | (did not exist) | 3 demos run + loss curves |

## 8. Key decisions

- Rust stays **outside the autodiff graph** (exact computation, verification, metrics, precomputation).
- The forward/backward is **pure PyTorch** (native autodiff, not noise).
- The 2-adic vortex **conditions** a trainable MLP (option B).
- `ω0 = 30` (justified by the SIREN paper), not 56.
- Honest naming everywhere: we keep the exact terms, we rename those that oversold.

## 9. Suggested attack order

L0 → L1 → L2 first (fast text demo), then L3 (compression), then L4 (causal), L5
(reasoning), L6 (stability), L7 (integrated demos). Each layer is independently deliverable.

**The implementation plan (next process step) will be split by layer.** Each
layer will get its own sub-plan with granular tasks. We will NOT write a
single monolithic plan for the 7 layers — that would be unmanageable. Concretely: we
will start with the L0 plan, then execute it, then move to the L1 plan, etc.

## 10. Future work (honesty)

Lean 4 formal proofs, ZK-SNARK attestation, K3 automorphic compression, Groth16 timing,
scaling to large models, evaluation on standardized benchmarks like MMLU/HellaSwag.
