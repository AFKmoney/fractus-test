# Fractus

**A Continuous Cognitive Agent — self-modifying, memory-persistent, trained from scratch on consumer hardware.**

Fractus is not a transformer. It is not a chatbot. It is not a wrapper around GPT.

Fractus is a **Continuous Cognitive Agent (CCA)** — a fundamentally new architecture built around principles that no LLM offers:

1. **Continuous thought** — the engine ticks in real time, with adaptive depth, like a biological brain, not a static input→output function.
2. **Persistent memory** — every interaction is stored forever in a vector bank that survives restarts, grows across sessions, and is injected into the thought state continuously.
3. **Live self-modification** — the model grows new experts at runtime, scales through progressive growth (palier by palier), and never starts from zero.

---

## What's New (2026-08-04)

This repository (`fractus-test`) is the experimental branch where the architecture was pushed beyond the original white paper. Everything here is **measured, not claimed**.

### Validated
| Feature | Status | How |
|---|---|---|
| **PhaseRoutedMoE** (differentiable) | ✅ | Replaced the ad-hoc MoE that froze experts during training |
| **Persistent memory** (wired to CTE) | ✅ | Continuous 5% injection + salience-gated consolidation |
| **Salience head** (trained) | ✅ | Learns by perturbation of trajectory (intrinsic signal) |
| **Cognitive modes** (unsupervised) | ✅ | K-means on Kuramoto phase features — modes emerge from data |
| **Runtime self-modification** | ✅ | `add_expert()` + `maybe_grow()` — the model grows while it thinks |
| **Tied head + head-partial training** | ✅ | 2x faster training (117→236 tok/s), half the params |
| **Progressive growth** | ✅ | 0→47M params in 4 paliers on CPU, each inheriting previous weights |
| **HF Space** (memory demo) | ✅ | RAG + PersistentMemory coexistence, live thought-memories panel |

### Refuted (honestly)
| Claim | Verdict | Evidence |
|---|---|---|
| **EDT (×186 acceleration)** | ❌ Refuted at 13M | 5 variants tested, all ~19% worse than from-scratch. Root cause: objective misalignment + Kuramoto routing concentration |
| **Forward-Forward** (Hinton 2022) | ❌ Refuted for CTE | Goodness signal ≠ CE. NLL went UP. Same misalignment as EDT |

### In progress
| Feature | Status |
|---|---|
| **Progressive growth → 1B** | Paliers 0-3 validated on CPU (~47M). Palier 4 (1B) needs GPU. |
| **Corpus** | 20.5M tokens including Fractus's own source code (palimpseste principle) |

---

## Architecture

```
Fractus/
├── fractus/
│   ├── continuous_engine.py      The Continuous Thought Engine (ticks, memory, salience)
│   ├── memory.py                 PersistentMemory (inject, consolidate, salience-gated)
│   ├── cognitive_modes.py        Unsupervised k-means modes from Kuramoto phases
│   ├── grow.py                   Progressive growth operator (width/depth/experts/rank)
│   ├── nn/
│   │   ├── moe.py                PhaseRoutedMoE (low-rank, differentiable, add_expert)
│   │   ├── attention.py          Multi-level causal linear attention (batched)
│   │   ├── phase_ode.py          Kuramoto RK4 oscillators
│   │   ├── lazy_siren.py         Low-rank weight storage (W = scale·U@Vᵀ)
│   │   └── ...                   13 neural modules total
│   ├── train/                    Online, chunked, forward-forward (refuted)
│   └── rag.py                    RAG + KnowledgeBase + Plugins + MetaCognition
├── fractus1B/                    TRUE 1B architecture (FractalBlockSparse × 16)
│   ├── model_1b.py               forward_train (head-partial), SparseStructuredMoE
│   ├── pgsu.py                   Partial Gradient Sub-Update (4/16 layers per step)
│   └── progressive_depth.py      Layer-freezing curriculum
├── experiments/
│   ├── edt_ab/                   EDT AB-test results + root-cause analysis
│   └── edt_1b_ab/                1B EDT code (ready, awaits GPU)
├── scripts/
│   ├── train_progressive.py      Progressive growth: palier 0→1→2→3→4
│   ├── build_quality_corpus.py   Corpus with Fractus source code
│   ├── train_salience_head.py    Salience head training (perturbation signal)
│   ├── validate_grow_stability.py  Runtime expert growth stability test
│   └── diagnose_routing.py       Kuramoto routing analysis
├── space/                        HuggingFace Space (FastAPI + memory demo)
├── data/                         Corpora (20.5M tokens quality_corpus.pt)
├── tests/                        40+ tests across 8 test files
└── Fractus_White_Paper.pdf       Technical document (v1.0, signed)
```

---

## The Continuous Thought Engine

The CTE is a **dynamical system**, not a function. It maintains a persistent thought state `h` and advances it tick by tick:

```
tick(observation):
  1. Absorb observation into thought state
  2. Attention: update accumulated state (S, z)
  3. Kuramoto: advance oscillator phases (the "consciousness clock")
  4. MoE: transform the thought, routed by Kuramoto phases
  5. Memory: salience-gated consolidation + 5% injection
  6. Confidence: decide whether to emit output
```

**Adaptive depth**: easy inputs = 1 tick, hard inputs = many ticks. Energy-proportional reasoning.

**Chunk-based processing**: `tick_chunk` processes 16 tokens per forward (4.7x speedup). `tick_chunk_train` computes the head on the last position only (2x additional speedup for training).

---

## Progressive Growth

Instead of training 1B from scratch (impossible on CPU), Fractus grows palier by palier:

| Palier | d_model | experts | params | tokens | tok/s CPU | time |
|---|---|---|---|---|---|---|
| 0 | 128 | 4 | 6.6M | 1M | 77 | ~3.5h |
| 1 | 256 | 8 | 13.4M | 1M | ~40 | ~7h |
| 2 | 512 | 16 | 28.9M | 500k | ~15 | ~9h |
| 3 | 768 | 32 | 47.3M | 500k | ~7 | ~20h |
| **4** | **1280** | **128** | **~1B** | **GPU** | **GPU** | **~2-3 weeks** |

Each palier inherits the previous weights via zero-padding (`fractus/grow.py`). The model never starts from random — it starts warm.

**Chinchilla scaling**: 20 tokens per trainable parameter.
- Palier 0: 6.6M × 20 = 132M tokens (~19 days CPU, ~18h GPU)
- 1B: 88M × 20 = 1.76B tokens (~136 days GPU full, ~2-3 weeks with warm start)

---

## Self-Modification

Fractus is the only model that **grows new capacity while it runs**:

```python
# The model detects routing imbalance and grows a new expert
engine.maybe_grow()  # → "[Fractus] Self-modified: grew expert 4 (now 5 experts)"
```

- `PhaseRoutedMoE.add_expert()`: zero-init new expert near the dominant one
- `maybe_grow()`: triggers when one expert dominates routing (> threshold)
- Cooldown + hard cap prevent runaway growth
- Validated: new expert receives 50% of traffic, loss stable post-grow

---

## Persistent Memory

Fractus remembers across sessions. No context window. No forgetting.

```python
from fractus.memory import PersistentMemory

mem = PersistentMemory(d_model=128, path="~/.fractus/memory.pt")
engine.attach_memory(mem)

# Session 1: the engine thinks, consolidates salient thoughts
for token in stream:
    engine.tick(token)
mem.save()

# Session 2 (new process, new engine): memories are loaded and injected
mem2 = PersistentMemory(d_model=128, path="~/.fractus/memory.pt")
engine2.attach_memory(mem2)  # memories influence the thought state at 5%
```

The **salience head** learns intrinsically: it predicts how much a memory injection will perturb the thought state (`||h_after - h_before||`). No external labels — the system discovers its own sensitivity to memories.

---

## Cognitive Modes

Modes emerge from the Kuramoto phase dynamics, not from labels:

```python
from fractus.cognitive_modes import CognitiveModes

modes = CognitiveModes(n_oscillators=8, n_modes=4)
modes.fit(phase_samples)  # unsupervised k-means → clusters ARE the modes
modes.label_modes(['focused', 'verbal', 'exploratory', 'procedural'])

result = modes.classify(current_phases)
# → {"mode": "focused", "confidence": 0.82}
```

---

## HF Space

Live demo at `huggingface.co/spaces/thefinalboss/Fractus-Space`:

- Chat interface with shared memory across all visitors
- RAG (textual memory) + PersistentMemory (thought-state memory) coexistence
- "Thought memories" panel showing the engine's subconscious state
- `/pmemories` endpoint for transparency

---

## Quick Start

```bash
git clone https://github.com/AFKmoney/fractus-test.git
cd fractus-test
pip install torch numpy tokenizers matplotlib fastapi uvicorn pydantic

# Run the test suite (40+ tests)
pytest tests/ -q

# Train progressively (paliers 0-1 on CPU)
python scripts/train_progressive.py --paliers 0,1

# Build a quality corpus (includes Fractus's own source code)
python scripts/build_quality_corpus.py
```

---

## Research Findings (honest)

This repository is a laboratory. Not everything worked. The findings are documented for the community:

1. **EDT does not work on Fractus** — pre-training experts independently produces worse models than from-scratch. The objectives (MSE on hidden states) are misaligned with the final CE. ([Full report](experiments/edt_ab/REPORT.md))

2. **Forward-Forward does not work on the CTE** — local goodness signals don't align with next-token prediction. The only signal that works is global CE backprop.

3. **The output head is 94% of training cost** — tied head + head-partial (`tick_chunk_train`) gives 2x speedup. Vocabulary reduction would give 3-6x more.

4. **Kuramoto routing concentrates** — at small scale, only 2/4 experts receive traffic. This is a property of the phase dynamics, not the router parameters.

5. **Progressive growth works** — the model grows palier by palier, inheriting weights via zero-padding. Each palier converges faster than from-scratch.

---

## License

MIT. This project belongs to the user, not to a corporation.

## Author

**Philippe-Antoine Robert** — 2026

## Links

- **GitHub (main):** [github.com/AFKmoney/fractus](https://github.com/AFKmoney/fractus)
- **GitHub (test/experimental):** [github.com/AFKmoney/fractus-test](https://github.com/AFKmoney/fractus-test)
- **HuggingFace (model):** [huggingface.co/thefinalboss/Fractus-1B](https://huggingface.co/thefinalboss/Fractus-1B)
- **HuggingFace (Space):** [huggingface.co/spaces/thefinalboss/Fractus-Space](https://huggingface.co/spaces/thefinalboss/Fractus-Space)
- **White Paper:** [Fractus_White_Paper.pdf](Fractus_White_Paper.pdf) (v1.0, signed)
