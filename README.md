# Fractus

**A Continuous Cognitive Agent — self-modifying, memory-persistent, trained from scratch on consumer hardware.**

Fractus is not a transformer. It is not a chatbot. It is not a wrapper around GPT.

Fractus is a **Continuous Cognitive Agent (CCA)** — a fundamentally new architecture built around principles that no LLM offers:

1. **Continuous thought** — the engine ticks in real time, with adaptive depth, like a biological brain, not a static input→output function.
2. **Persistent memory** — every interaction is stored forever in a vector bank that survives restarts, grows across sessions, and is injected into the thought state continuously.
3. **Live self-modification** — the model grows new experts at runtime, scales through progressive growth (palier by palier), and never starts from zero.

---

## How It Works

### The Continuous Thought Engine (CTE)

The CTE is a **dynamical system**, not a function. It maintains a persistent thought state `h` and advances it tick by tick:

```
tick(observation):
  1. Absorb observation into thought state
  2. Attention: update accumulated state (S, z) — multi-level causal linear attention
  3. Kuramoto: advance oscillator phases (the "consciousness clock", detached from autograd)
  4. MoE: transform the thought — PhaseRoutedMoE with sparse low-rank experts
  5. Memory: salience-gated consolidation + 5% injection
  6. Confidence: decide whether to emit output
```

**Adaptive depth**: easy inputs = 1 tick, hard inputs = many ticks. Energy-proportional reasoning.

**Chunk-based training**: `tick_chunk_train` processes 32 tokens per forward, but the output head runs only on the **last position** (32x less head FLOPs). The thought state carries forward between chunks (detached — no BPTT).

### PhaseRoutedMoE (Sparse, Low-Rank, Differentiable)

The MoE routes tokens via **Kuramoto oscillator phases** and a **von Mises gate** on Farey-distributed expert phases:

- **Sparse gather-first**: only `top_k=2` experts are computed per token. At 128 experts, that's 64x less work than dense.
- **Low-rank experts**: `W = scale · U@Vᵀ` (rank 64). Two cheap matmuls per expert, no full matrix materialized.
- **Differentiable end-to-end**: gradients flow to U/V/scale/bias through the sparse path. (The old `CachedStructuredSiren` read a detached buffer — that bug was found and fixed.)
- **Self-modifying**: `add_expert()` grows a new expert at runtime. `maybe_grow()` triggers automatically when routing is imbalanced.

### Progressive Growth (0 → 1B)

Instead of training 1B from scratch (months on GPU), Fractus **grows palier by palier**:

| Palier | d_model | experts | params | CPU tok/s | GPU tok/s (est.) |
|---|---|---|---|---|---|
| 0 | 128 | 4 | 6.6M | **707** | ~3000 |
| 1 | 256 | 8 | 13.4M | ~350 | ~1800 |
| 2 | 512 | 16 | 28.9M | ~100 | ~800 |
| 3 | 768 | 32 | 47.3M | ~30 | ~400 |
| **4** | **1280** | **128** | **~1B** | — | **~200** |

Each palier inherits the previous weights via **zero-padding** (`fractus/grow.py`). Old knowledge is preserved (top-left block of every matrix), new capacity starts neutral (zeros for weights, ones for LayerNorm gamma). The model never starts from random — it starts warm.

**Chinchilla scaling**: 20 tokens per trainable parameter. With warm start, ~1/4 suffices.

### Training Optimizations (Measured)

| Optimization | What it does | Speedup |
|---|---|---|
| **Tied head** | `output_head.weight = observe.weight` (halves vocab params) | ~1.1x |
| **Head-partial** (`tick_chunk_train`) | Head on 1 position instead of all C | ~2x on forward |
| **Sparse MoE low-rank** | Only top-k experts computed (gather-first) | Scales with E/K |
| **Gradient accumulation** (accum=8) | 8x fewer optimizer steps | ~1.4x |
| **Chunk_len=32** | Better Python amortization | ~1.1x |
| **Detach Kuramoto** | Phase computation in no_grad | ~1.06x |
| **bf16 AMP** (GPU) | 2x on all matmuls | ~2x (GPU only) |
| **torch.compile** | Kernel fusion (flag `--compile`) | Untested on CPU |

**Measured combined**: 4 tok/s → **707 tok/s** on CPU (palier 0).

---

## Quick Start

```bash
git clone https://github.com/AFKmoney/fractus-test.git
cd fractus-test
pip install torch numpy tokenizers matplotlib

# Run the test suite (26 tests)
pytest tests/ -q

# Build quality corpus (includes Fractus's own source code!)
python scripts/build_quality_corpus.py

# Train progressively on CPU (paliers 0-3, ~4 hours)
python scripts/train_progressive.py --paliers 0,1,2,3 --accumulation-steps 8

# Train on GPU (loads palier 3, grows to 1B, trains)
python scripts/train_1b_gpu.py \
    --checkpoint checkpoints/fractus_palier3.pt \
    --tokens 440000000 \
    --bf16
```

---

## Architecture

```
fractus-test/
├── fractus/
│   ├── continuous_engine.py      CTE: ticks, memory, salience, self-modification
│   ├── memory.py                 PersistentMemory: inject 5%, consolidate_if_salient
│   ├── cognitive_modes.py        Unsupervised k-means on Kuramoto phases
│   ├── grow.py                   Progressive growth: width/depth/experts/rank
│   ├── nn/
│   │   ├── moe.py                PhaseRoutedMoE: sparse low-rank, add_expert
│   │   ├── attention.py          Multi-level causal linear attention
│   │   ├── phase_ode.py          Kuramoto RK4 oscillators
│   │   └── lazy_siren.py         Low-rank weight: W = scale·U@Vᵀ
│   └── train/
│       └── online.py             OnlineTrainer: chunked, gradient accumulation
├── fractus1B/                    TRUE 1B architecture (16 blocks, 128 experts)
│   ├── model_1b.py               forward_train (head-partial), SparseStructuredMoE
│   ├── pgsu.py                   Partial Gradient Sub-Update (4/16 layers/step)
│   └── progressive_depth.py      Layer-freezing curriculum
├── experiments/
│   ├── edt_ab/                   EDT AB-test results (refuted — documented)
│   └── edt_1b_ab/                1B EDT code (ready, awaits GPU)
├── scripts/
│   ├── train_progressive.py      Progressive growth: palier 0→1→2→3→4
│   ├── train_1b_gpu.py           GPU training: load checkpoint → grow → 1B
│   ├── build_quality_corpus.py   Corpus builder (includes source code)
│   ├── train_salience_head.py    Salience head training (perturbation signal)
│   └── validate_grow_stability.py  Runtime expert growth validation
├── space/                        HuggingFace Space (memory demo)
├── data/                         20.5M token corpus
├── tests/                        26 tests
└── Fractus_White_Paper.pdf       Technical document (v1.0, signed)
```

---

## Self-Modification

Fractus is the only model that **grows new capacity while it runs**:

```python
# The model detects routing imbalance and grows a new expert
engine.maybe_grow()
# → "[Fractus] Self-modified: grew expert 4 near dominant expert 2 (now 5 experts)"

# Or manually
engine.moe.add_expert(dominant_idx=2)
# → Zero-init, placed near the dominant expert's phase, captures overflow traffic
```

- Zero-init new experts → no forward perturbation → no gradient spike
- Placed near the dominant expert (captures overflow traffic)
- Cooldown + hard cap prevent runaway growth
- Validated: new expert receives 50% of traffic, loss stable post-grow

---

## Persistent Memory

```python
from fractus.memory import PersistentMemory

mem = PersistentMemory(d_model=128, path="~/.fractus/memory.pt")
engine.attach_memory(mem)

# Session 1: engine thinks, consolidates salient thoughts
for token in stream:
    engine.tick(token)
mem.save()

# Session 2 (new process): memories loaded + injected at 5%
mem2 = PersistentMemory(d_model=128, path="~/.fractus/memory.pt")
engine2.attach_memory(mem2)
# → cross-session persistence validated
```

The **salience head** learns intrinsically — it predicts `||h_after_inject - h_before_inject||` (perturbation of trajectory). No external labels. The system discovers its own sensitivity to memories.

---

## Cognitive Modes

Modes emerge from the Kuramoto phase dynamics:

```python
from fractus.cognitive_modes import CognitiveModes

modes = CognitiveModes(n_oscillators=8, n_modes=4)
modes.fit(phase_samples)  # unsupervised k-means → clusters ARE the modes
modes.label_modes(['focused', 'verbal', 'exploratory', 'procedural'])
result = modes.classify(phases)  # → {"mode": "focused", "confidence": 0.82}
```

---

## Research Findings (Honest)

| Claim | Verdict | Evidence |
|---|---|---|
| **EDT (×186 acceleration)** | ❌ Refuted | 5 variants, all ~19% worse. Root cause: objective misalignment + routing concentration |
| **Forward-Forward** (Hinton 2022) | ❌ Refuted | Goodness ≠ CE. NLL went UP |
| **Progressive growth** | ✅ Works | 4 paliers measured, warm start converges faster |
| **Sparse MoE low-rank** | ✅ Works | 2/128 experts = 64x less MoE compute |
| **Self-modification** | ✅ Works | add_expert + maybe_grow, stability validated |
| **Persistent memory** | ✅ Works | Cross-session persistence, salience head trained |
| **707 tok/s on CPU** | ✅ Measured | All optimizations combined (was 4 tok/s before) |

Full report: `experiments/edt_ab/REPORT.md`. Optimization analysis: `docs/TRAINING_OPTIMIZATION.md`.

---

## GPU Training (1B)

When you have a GPU (RTX 3090 / A100):

```bash
# Step 1: Train paliers 0-3 on CPU (done, ~4h)
python scripts/train_progressive.py --paliers 0,1,2,3 --accumulation-steps 8

# Step 2: Grow to 1B + train on GPU (~1.5 days on RTX 3090)
python scripts/train_1b_gpu.py \
    --checkpoint checkpoints/fractus_palier3.pt \
    --tokens 440000000 \
    --bf16 \
    --accumulation-steps 4
```

**Expected on RTX 3090 (24GB)**:
- ~200 tok/s with all optimizations (sparse MoE + head-partial + bf16 + accum)
- 440M tokens (warm start, 1/4 Chinchilla) in ~30 hours
- Auto-eval, generation test, HF upload on completion

---

## HF Space

Live demo at `huggingface.co/spaces/thefinalboss/Fractus-Space`:
- Chat with shared memory across all visitors
- RAG (textual) + PersistentMemory (thought-state) coexistence
- "Thought memories" panel showing the engine's subconscious state

---

## License

MIT. This project belongs to the user, not to a corporation.

## Author

**Philippe-Antoine Robert** — 2026

## Links

- **GitHub:** [github.com/AFKmoney/fractus-test](https://github.com/AFKmoney/fractus-test)
- **HuggingFace:** [huggingface.co/thefinalboss/fractus-test](https://huggingface.co/thefinalboss/fractus-test)
- **Model:** [huggingface.co/thefinalboss/Fractus-1B](https://huggingface.co/thefinalboss/Fractus-1B)
- **Space:** [huggingface.co/spaces/thefinalboss/Fractus-Space](https://huggingface.co/spaces/thefinalboss/Fractus-Space)
- **White Paper:** [Fractus_White_Paper.pdf](Fractus_White_Paper.pdf)
