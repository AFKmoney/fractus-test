# Fractus — A Complete Walkthrough

> **Read this first.** This document explains, from A to Z, what fractus is, why it exists, how every piece works, and how to use it. It is the single entry point for a new contributor or reviewer.

**Fractus** is a trainable fractal transformer with SIREN weight compression, NOTEARS causal discovery, and verified mathematical reasoning — all running on a CPU laptop. It is an honest rebuild of two prior systems that corrected their errors while preserving every one of their concepts.

The distinguishing discipline of this project: **measure, don't claim.** Every number in the repo is measured on the target hardware (AMD Ryzen 5 5500U, CPU-only), every optimization has an equivalence test, and every "known limitation" is written down rather than hidden.

---

## Table of contents

1. [Why fractus exists](#1-why-fractus-exists)
2. [The core idea: a fractal transformer](#2-the-core-idea-a-fractal-transformer)
3. [Architecture at a glance](#3-architecture-at-a-glance)
4. [The 2-adic Vortex (Rust core)](#4-the-2-adic-vortex-rust-core)
5. [The fractal embedding (L1)](#5-the-fractal-embedding-l1)
6. [Causal linear attention (L2a)](#6-causal-linear-attention-l2a)
7. [Kuramoto oscillators + von Mises MoE (L2b)](#7-kuramoto-oscillators--von-mises-moe-l2b)
8. [SIREN weight compression (L3)](#8-siren-weight-compression-l3)
9. [Causal discovery: NOTEARS + RKHS + do-calculus (L4)](#9-causal-discovery-notears--rkhs--do-calculus-l4)
10. [Mathematical reasoning: proofs, conjectures, ACT (L5)](#10-mathematical-reasoning-proofs-conjectures-act-l5)
11. [Stability: Lyapunov on Kuramoto (L6)](#11-stability-lyapunov-on-kuramoto-l6)
12. [Lightweight training (L8)](#12-lightweight-training-l8)
13. [The method: measure, don't claim](#13-the-method-measure-dont-claim)
14. [How to run everything](#14-how-to-run-everything)
15. [Honest limitations](#15-honest-limitations)

---

## 1. Why fractus exists

Two prior systems (by the same original author) proposed a "fractal neural network for AGI" with bold claims: 20.4× compression, O(n) causal reasoning, CPU-only deployment, formal verification. A line-by-line code review revealed that **neither system actually worked as advertised**, despite genuine mathematical culture and several correctly-coded modules.

The critical errors found (with `file:line` evidence):

- **No autodiff.** `training.rs:399` updated weights with `rand::random()*0.01` (noise) instead of a gradient. The model never learned.
- **Fake SIREN.** `torus_siren.py:15` used `nn.SiLU` instead of the `sin(ω0·)` nonlinearity that defines a SIREN.
- **Hardcoded numbers.** `training_loop.py:52` literally returned `"compression_ratio": 20.4`. The real figure (measured) is ~1.5×.
- **Orphaned math.** The 2-adic Rust core was never imported by any Python file.
- **Tautological tests.** `tests.rs:1840` admitted "AGINFNModel is currently a stub with only new()."
- **Fake RKHS, trivial do-calculus, clamped metrics.** And more (9 falsehoods total).

**Fractus is the honest rebuild.** Same concepts, same mathematical ambitions — but with real autodiff, faithful math, honest tests, and measured numbers. The project is built layer by layer (L0 through L8), each layer independently verifiable, each with a demo that proves it works.

---

## 2. The core idea: a fractal transformer

At its heart, fractus is a transformer with a distinctive design philosophy:

- **Linear attention** (Katharopoulos 2020) instead of softmax — O(L·d²) instead of O(L²·d), and crucially it carries a **running state** `(S, z)` that lets us process long sequences in chunks (the Mamba/RWKV trick).
- **Multi-level "Mandelbrot-decayed" offsets** — each attention level uses a feature map `elu_plus_one(x + ω_level)` where `ω_level = (φ²)^{-level}` and φ is the golden ratio. This gives a geometric scale separation across levels. (Honest naming: the original called these "Mandelbrot frequencies," but there is no Mandelbrot iteration — just a geometric sequence. We say "Mandelbrot-decayed Fourier basis.")
- **Coupled Kuramoto oscillators** integrated by RK4 — a genuine dynamical system whose phases route tokens to experts.
- **Phase-routed mixture-of-experts** — a von Mises gate on Farey-distributed expert phases selects which experts process each token.

All of this is **pure PyTorch, fully differentiable end-to-end**. The Rust core handles only exact, off-graph computation (2-adic arithmetic, verification, precomputation).

---

## 3. Architecture at a glance

```
fractus/
├── crate/fractus-core/     Rust: 2-adic vortex (pure math, testable alone)
│   └── src/{lib.rs, vortex.rs}
├── crate/fractus-py/       Rust: PyO3 bindings (no logic, just wrappers)
├── fractus/
│   ├── nn/                 the neural network (PyTorch)
│   │   ├── embedding.py        fractal codepoint embedding (char + Fourier + vortex)
│   │   ├── attention.py        multi-level causal linear attention
│   │   ├── phase_ode.py        low-rank Kuramoto RK4 oscillators
│   │   ├── moe.py              von Mises / Farey-routed MoE
│   │   ├── farey.py            Farey sequence + expert phase selection
│   │   ├── siren.py            true sin(ω0·) SIREN for weight compression
│   │   ├── siren_linear.py     nn.Linear whose W comes from a SIREN
│   │   └── block.py            FractalBlock + FractalBlockFull (assembly)
│   ├── causal/             NOTEARS, RKHS, Pearl do-calculus
│   ├── reasoning/          proofs, conjectures, prime generation, ACT
│   ├── stability/          Lyapunov on the Kuramoto subsystem
│   ├── metrics/            honest measurements (compression, SHD, perplexity)
│   ├── math/               primes, Fibonacci, stats utilities
│   └── train/              lightweight trainers (L8: state-carry, AMP, surprise-gating)
├── data/                   tinyshakespeare + synthetic causal SCMs
├── tests/                  27 test files, 166 tests, all pass
├── scripts/                demos, train_hf.py, bench_train.py
└── docs/                   this file, SPEC, layer plans L0–L8
```

**The golden rule:** Rust stays **outside** the autodiff graph. The forward/backward pass is pure PyTorch. Rust does exact computation, verification, and precomputation — it never pretends to be differentiable.

---

## 4. The 2-adic Vortex (Rust core)

**Files:** `crate/fractus-core/src/vortex.rs`, exposed to Python via `crate/fractus-py/src/lib.rs`.

The only mathematically correct and non-trivial module inherited from the original system. It implements 2-adic arithmetic:

- **Valuation** `v₂(x)` = the largest k such that 2^k divides x.
- **Ultrametric distance** `d(a,b) = 2^{-v₂(a⊕b)}` — satisfies the strong triangle inequality `d(x,z) ≤ max(d(x,y), d(y,z))`, stricter than a normal metric.
- **2-adic norm** `‖x‖₂ = 2^{-v₂(x)}`.

**The bug we fixed:** the original computed `2^{+v₂}` (the inverse of the canonical p-adic norm). We corrected it to `2^{-v₂}`. The original's test was tautological (`assert!(d1 <= d2.max(d1))` — always true); ours is a real ultrametric test on random triplets including the discriminating case (7, 56, 13).

**How it's used:** the 2-adic Collatz hash of a token id (computed exactly in Rust, off-graph) **conditions** a trainable MLP (in the PyTorch graph) that produces embedding phases. The vortex influences learning without pretending to be differentiable. This is "option B" from the spec — exact conditioning, not fake autodiff.

---

## 5. The fractal embedding (L1)

**File:** `fractus/nn/embedding.py`

Each token id is embedded by combining three feature sources:

1. **16 morphological features** (`char_features.py`) — is_vowel, is_digit, case, punctuation, etc. Deterministic, no parameters.
2. **Mandelbrot-decayed Fourier basis** (`fourier.py`) — for each frequency `ω_k = (φ²)^{-k}`, the pair `(sin(ω_k·t), cos(ω_k·t))`. Deterministic, no parameters.
3. **Vortex conditioning** — the Collatz hash (from Rust) feeds a trainable MLP that produces phase offsets.

These are concatenated and projected to `d_model` by a trainable `nn.Linear`. **The whole forward is differentiable end-to-end** — `test_fractal_embedding_backward_propagates` checks that `backward()` propagates a finite, non-zero gradient to EVERY parameter. This is the test the original failed (it used noise instead of gradients).

---

## 6. Causal linear attention (L2a)

**File:** `fractus/nn/attention.py`

Katharopoulos linear attention: instead of the O(L²) softmax, we maintain a running state and update it causally:

```
S_t = Σ_{i≤t} φ(k_i) ⊗ v_i       (a d×d matrix, accumulated)
z_t = Σ_{i≤t} φ(k_i)               (a d-vector, accumulated)
y_t = (φ(q_t)·S_t) / (φ(q_t)·z_t)
```

where `φ(x; level) = elu_plus_one(x + ω_level)` is a strictly-positive feature map (positivity keeps the denominator well-defined).

**Multi-level:** we run this for `n_levels` levels (each with a different Mandelbrot offset `ω_level`) and aggregate with softmax-weighted sums.

**L8 optimization (the big win):** the original looped over levels AND heads (n_levels × n_heads separate Python calls). Profiling showed this was the real bottleneck — NOT Kuramoto as the README claimed. We now batch all heads × levels into ONE vectorized call by flattening `(B, n_levels, n_heads)` into a single batch dimension. **Measured: 17.3 ms → 6.6 ms (×2.6 faster), output identical** (causality + equivalence tests pass).

---

## 7. Kuramoto oscillators + von Mises MoE (L2b)

**Files:** `fractus/nn/phase_ode.py`, `fractus/nn/moe.py`, `fractus/nn/farey.py`

### Kuramoto (phase_ode.py)
Low-rank coupled oscillators `dθ_i/dt = ω_i - damping·θ_i + Σ_j K_ij sin(θ_j - θ_i)`, with the coupling `K = UΛUᵀ` (rank-r, so O(N·r) not O(N²)). Integrated by **RK4** (4 sub-steps), stateless (recomputed from hidden states each forward). The phases it produces route tokens to experts.

**L8 optimization:** the n_steps RK4 loop is unrolled (killing Python overhead), with per-step mod-wrap preserved for exact equivalence (`atol=1e-6` vs the looped reference).

### MoE (moe.py)
Expert phases are drawn from the **Farey sequence** `F_{2E}` — a deterministic, dense, non-collapsing distribution. A **von Mises gate** `g_e = exp(κ·cos(θ - θ_e))` routes each token to its top-k experts. A load-balance auxiliary loss keeps experts evenly used.

**L8 optimization:** adaptive dispatch. When `n_experts > 2·top_k` (big waste), we gather-first — index_select the top-k experts' weights and compute only those. When `n_experts` is small, the dense einsum is faster on CPU (measured: sparse was ×1.5 slower for E=4,K=2), so we keep dense. The sparse path is proven bit-identical to dense by `test_moe_sparse_matches_reference`.

---

## 8. SIREN weight compression (L3)

**Files:** `fractus/nn/siren.py`, `fractus/nn/siren_linear.py`

A **SIREN** (Sitzmann et al. 2020) represents a weight matrix as a scalar field over the torus `T² = [0,1)²`, regenerated by evaluating `sin(ω₀·(Wx+b))` on a grid. The nonlinearity is genuinely `sin` (not SiLU), `ω₀ = 30` (the paper's empirical value, not the original's unjustified 56), and the init follows Sitzmann section 3.2.

`SirenLinear` is an `nn.Linear`-like layer whose weight matrix **is** the SIREN output — it's in the autodiff graph, trained normally, and the decompressed matrix is never discarded (the original computed it then threw it away).

**Honest result:** the compression ratio is **measured, not hardcoded** (`metrics/compression.py`). On dense network weights the real ratio is ~1.5–5×, NOT 20.4×. This is the truth — SIREN compresses smooth functions well, and trained weights are essentially dense structured noise. The L3 demo documents this frankly.

---

## 9. Causal discovery: NOTEARS + RKHS + do-calculus (L4)

**Files:** `fractus/causal/{notears.py, rkhs.py, do.py}`, `fractus/metrics/causal.py`

### NOTEARS (notears.py)
The acyclicity penalty `h(W) = tr(e^{W⊙W}) - n`, computed via a 20-term Taylor expansion. `h(W) = 0` iff `W` is a DAG (acyclic); `h(W) > 0` if it contains a cycle. Differentiable, so we can optimize `W` to be both a good fit AND acyclic. `test_notears_zero_for_dag` and `test_notears_positive_for_cycle` validate the math.

### RKHS (rkhs.py)
A **true** RKHS via Random Fourier Features (Rahimi-Recht 2007): the Gaussian kernel is approximated by `φ(x)·φ(y)` where `φ` uses random `cos/sin` features. The original's "RKHS" was just `x@U@Vᵀ` — a bare low-rank projection with no kernel. Ours is a real Hilbert-space operator. `test_rkhs_not_just_linear_projection` proves the output goes through the non-linear feature map.

### do-calculus (do.py)
Pearl's `do(X_i = v)` fixes a variable to a value (intervention), enabling counterfactual queries. The original just zeroed the column; ours sets it to `v` (which can be non-zero). `test_do_intervention_not_zeroing` catches the difference.

### Metric: SHD (causal.py)
Structural Hamming Distance — the standard count of mispredicted edges. **No 0.98 clamp** (the original capped causal accuracy at exactly 0.98 to rig benchmarks). The demo shows NOTEARS recovering a synthetic DAG with low SHD.

---

## 10. Mathematical reasoning: proofs, conjectures, ACT (L5)

**Files:** `fractus/reasoning/{proof.py, proof_trainer.py, conjecture.py, prime_generator.py, act.py, self_consistency.py}`

### Proofs (proof.py, proof_trainer.py)
A GRU generates proof steps; an **exact verifier** (soundness guaranteed) checks the numerical conclusion. The generator is trained by REINFORCE with a curriculum (easy targets first, progressively harder).

**Honest verdict:** the original "converge to 1e-3" task is unattainable with this architecture — we diagnosed and documented this rather than hiding it. The redefined task (produce prime numbers) works at ~100% validity.

### Conjectures (conjecture.py)
Popperian falsification: a neural proposer suggests conjectures from 10 templates (Fermat, Wilson, GCD, Fibonacci...); a tester runs random trials; survivors enter a memory with novelty detection. This genuinely discovers true identities.

### Prime generation (prime_generator.py)
A simple MLP learns to output prime numbers, verified by an exact sieve. After ~150 REINFORCE steps it reaches >50% validity (vs 25% random chance), and every accepted n is mathematically prime (soundness).

### ACT (act.py)
Adaptive Computation Time (Graves 2016) — the model "thinks" for a variable number of steps per position, with a halting probability.

---

## 11. Stability: Lyapunov on Kuramoto (L6)

**File:** `fractus/stability/lyapunov.py`

A true Lyapunov function `V(θ) = ½·Σ(θ_i - θ*)²` on the **Kuramoto subsystem** (the only genuine dynamical system in the model). `V > 0` away from synchronization, `V = 0` at the target, and along a stable trajectory `V` decreases monotonically. The original's "Lyapunov Shield" just tracked `‖y‖²` on a non-dynamical output — meaningless. Ours is a real stability certificate on a real dynamical system.

---

## 12. Lightweight training (L8)

**Files:** `fractus/train/{trainer.py, surprise_gate.py}`, `scripts/bench_train.py`, `docs/2026-06-26-fractus-L8-lightweight-training.md`

The most recent layer. Goal: train fractus on a CPU laptop using minimal time and energy, by exploiting what makes fractus structurally unique.

**The key discovery:** profiling overturned the README's assumption. The README said "Kuramoto is the bottleneck." Profiling showed the real cost was the **Python loop over heads × levels in attention** (17.3 ms) — Kuramoto was secondary (22.8 ms for the full RK4, but only 1.65 ms per derivative call). This is why "measure, don't claim" matters.

Six optimizations, each measured and equivalence-tested:

1. **Attention batched heads × levels** — ×2.6 faster (17.3→6.6 ms).
2. **MoE adaptive dense/sparse dispatch** — sparse when it helps, dense when it's faster on CPU.
3. **Kuramoto RK4 unrolled** — Python overhead removed, exact equivalence.
4. **State-carrying attention** — carry `(S,z)` across chunks: O(chunk_len) memory instead of O(seq_len). Proven equivalent to full-sequence processing.
5. **LightweightTrainer** — bf16 autocast + fused AdamW + cosine LR scheduler.
6. **SurpriseGatedTrainer** (new method) — backpropagate only on high-loss "surprising" tokens. Energy-proportional: spend compute where the gradient is non-trivial, skip where it's ~0.

---

## 13. The method: measure, don't claim

This is the project's defining discipline, and it's why L8 succeeded:

- **Profile before optimizing.** We didn't trust the README's bottleneck claim — we measured, and found it was wrong.
- **Equivalence-test every optimization.** No "it should be the same" — we prove it (`atol` bounds against a reference implementation).
- **Measure before/after on the real hardware.** Component-level numbers (reproducible) are the trustworthy evidence; full-model throughput on a shared CPU is too variable alone.
- **Document limitations honestly.** State-carry is proven at the attention level but not yet wired through the full block stack. Surprise-gating is a biased estimator. These are written down, not hidden.

---

## 14. How to run everything

```bash
# Setup (one-time)
py -m venv .venv && .venv\Scripts\Activate.ps1
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
maturin develop --release

# Verify everything works (166 tests)
pytest tests/ -q

# Demos
python scripts/demo_transformer.py         # fractal transformer learns "hello world"
python scripts/demo_prime_reinforce.py     # prime-number generation
python scripts/demo_causal.py              # NOTEARS recovers a DAG
python scripts/demo_shakespeare.py         # tinyshakespeare perplexity
python scripts/demo_siren_compression.py   # measured SIREN ratio
python scripts/demo_full.py                # all three tasks integrated

# Train on a dataset
python scripts/train_hf.py --preset cpu-small --dataset tinyshakespeare

# Benchmark (measure your own throughput)
python scripts/bench_train.py --tag baseline --runs 3
python scripts/bench_train.py --tag optimized --runs 3
python scripts/bench_train.py --compare
```

---

## 15. Honest limitations

1. **State-carry is attention-level only.** Carrying `(S,z)` through the full block stack (embedding + N blocks + head) is documented future work.
2. **Surprise-gating is biased.** Useful in the low-gradient regime, not a universal free lunch.
3. **Weak SIREN compression.** ~1.5–5× on dense weights (SIREN compresses smooth functions; trained weights are noisy).
4. **Linear NOTEARS.** Robust to moderate non-linearity (tanh), not to strongly non-linear relations.
5. **Full-model bench is variance-dominated on a shared CPU.** Trust the isolated component numbers; run `bench_train.py` on an idle machine for a clean figure.
6. **No Lean 4 / ZK-SNARK.** The original systems claimed formal verification; this is honestly absent and listed as future work.

---

## Further reading

- `docs/SPEC.md` — the complete specification (the 7-layer plan).
- `docs/2026-06-19-fractus-L0-socle.md` through `L4-causal.md` — per-layer implementation plans.
- `docs/2026-06-26-fractus-L8-lightweight-training.md` — the L8 lightweight training deep-dive.

Every claim in this document is backed by a test or a measurement in the repo. If you find one that isn't, that's a bug.
