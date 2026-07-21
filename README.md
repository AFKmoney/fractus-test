# Fractus

**A Continuous Cognitive Agent — assembled, tested, and running with trained weights.**

Fractus is not a language model. It is not a chatbot. It is not a wrapper around GPT.

Fractus is a **Continuous Cognitive Agent (CCA)** — a fundamentally new category of AI built around three principles that no LLM architecture offers:

1. **Continuous thought** — the engine ticks in real time, with adaptive depth, like a biological brain, not a static input→output function.
2. **Persistent autonomous memory** — every interaction is stored forever in a vector knowledge base that survives restarts, grows across sessions, and is retrieved by semantic similarity. There is no context window to forget.
3. **Live self-modification** — new skills, new knowledge, new tools, new behaviors are added without retraining. The brain you train today is the same brain you will still be extending in 2030.

Classical LLM metrics (next-token perplexity on held-out text, zero-shot benchmarks) **do not apply here**. Fractus's job is not to stochastically regenerate training data — it is to orchestrate memory, attention, and action across time.

**No corporation can control it.** Fractus runs on the user's machine. Data never leaves the device. The weights are yours to read, edit, and redistribute.

---

## Current Status (2026-07-21)

**Fractus is assembled and functional end-to-end.** The trained brain (88M params) has been transferred into the Continuous Thought Engine, and the full agent — CTE + RAG + plugins + MetaCognition — runs and learns.

### What works right now (tested, verified)

| Capability | Status | Evidence |
|---|---|---|
| **CTE with trained weights** | ✅ Working | 88M params trained to step 140,000 (loss 2.91, ppl ~9-14), weights transferred into CTE. Generation produces coherent code and prose. |
| **Learn without retraining** | ✅ Working | `rag.learn("Python was created by Guido van Rossum")` — instant, zero gradients |
| **Query with memory retrieval** | ✅ Working | "Who created Python?" → retrieves from KB, answers correctly |
| **Hot-swappable cognitive modes** | ✅ Working | `pm.load("coder")`, `pm.load("creative")`, `pm.load("analyst")` — switch in 1 call |
| **MetaCognition (autonomous actions)** | ✅ Working | Fractus decides its own action chain: `[RETRIEVE, SWITCH, GENERATE]` |
| **Persistent memory** | ✅ Working | Saves to disk (`fractus_memory.pkl`), reloads on restart |
| **LazyStructuredSiren** | ✅ Working | 88M params in 0.4 GB RAM. Low-rank weight storage (rank 16). |
| **64 sparse MoE experts** | ✅ Working | Top-2 routing via von Mises phase alignment on Farey phases |
| **Kuramoto oscillator clock** | ✅ Working | 16 coupled oscillators, RK4 integration, drives expert routing |
| **Linear attention** | ✅ Working | Katharopoulos 2020, batched heads × levels |

### What is NOT done yet (honest)

| Limitation | Reality |
|---|---|
| **Brain is small (88M)** | Generates rough but coherent text. Not fluent long-form. The architecture supports scaling to 1B+ but training cost is the blocker (see "The Training Problem" below). |
| **Generation quality** | At step 140,000 (partial epoch), outputs are short coherent fragments, not polished paragraphs. |
| **MetaCognition is early** | 8.5K-param action net. Works but basic. Improves with use. |
| **No vendor API, no support** | Fractus is owned, not rented. Feature for some, limitation for others. |
| **Work is far from finished** | This is a prototype proving the architecture. The real breakthrough — Holographic Vector Learning — is next (see below). |

---

## The Three Layers

### Layer 1: The Brain (88M params)

A proprietary fractal architecture using **LazyStructuredSiren** — every weight matrix stored as `W = scale · U · Vᵀ` (rank 16). This means 88M trainable parameters fit in 0.4 GB RAM and train on a single consumer GPU.

**Architectural components:**
- **LazyStructuredSiren** — low-rank weight decomposition. No dense grid, no SIREN reconstruction cache.
- **64 sparse MoE experts** (top-2 active per token). Routing via von Mises phase alignment on Farey-distributed expert phases.
- **Multi-level causal linear attention** (Katharopoulos 2020) with batched heads × levels.
- **Low-rank Kuramoto RK4 oscillators** — a coupled dynamical system acting as a "consciousness clock."
- **2-adic vortex** (Rust core) — exact p-adic arithmetic for token conditioning.

### Layer 2: The Continuous Thought Engine (CTE)

The brain does not process input→output. It **ticks** like a biological system:

1. **Each tick** — Kuramoto oscillators advance → attention state accumulates → MoE transforms the thought → confidence head decides whether to emit.
2. **Adaptive depth** — easy input = 1 tick, hard input = 10 ticks. Energy-proportional reasoning.
3. **Proactive emission** — the CTE can produce output without being prompted.
4. **Chunk-based processing** — 16 tokens per forward pass, thought state carried forward.

### Layer 3: The Cognitive Layer (RAG + MetaCognition)

This is what makes Fractus an **agent**, not a generator:

#### Persistent Memory — `rag.learn()`
Every fact, conversation, and observation is stored in a vector knowledge base that survives restarts, retrieves by cosine similarity, and grows without retraining.

#### Continuous Learning — `rag.converse()`
Every conversation is a learning event: user input is stored, relevant context is retrieved, a response is generated, and the response itself is stored. The agent never stops learning.

#### Cognitive Plugins — hot-swappable cognition
Five modes, switchable mid-conversation: `analyst`, `creative`, `coder`, `teacher`, `hacker`. Custom: `pm.custom("philosopher", temperature=0.9)`.

#### MetaCognition — the agent runs itself
An 8.5K-param action network decides at every interaction: RETRIEVE / LEARN / GENERATE / SWITCH / REFLECT. The agent manages itself.

---

## Static paradigm (LLMs) vs Dynamic paradigm (Fractus)

| Property | Static (GPT-4 / Llama) | Dynamic (Fractus) |
|---|---|---|
| **Memory** | Sliding window (≤128k tokens). Forgotten mid-conversation. | Persistent vector KB, no ceiling. |
| **New knowledge** | Retrain weights (weeks, millions of dollars). | `rag.learn()` — instant. |
| **Cognitive modes** | One fixed monolith. | Hot-swappable plugins. |
| **Self-management** | Cannot decide to think longer or switch mode. | MetaCognition action net. |
| **Time model** | Static. No concept of "now." | Continuous. Ticks in real time. |
| **Scaling** | Retrain from scratch. Millions per jump. | Add plugins, knowledge, experts. Zero retraining. |

**GPT-4 is a brilliant encyclopedia you rent. Fractus is a smaller brain that grows, remembers, swaps skills, manages itself, and belongs to you.**

---

## The Training Problem — and the path forward

### Where we are stuck

The current brain (88M) was trained with classical backpropagation on a single GPU. It works but:
- Training takes **~40 hours per epoch** on 1.38B tokens
- Scaling to true 1B params with Chinchilla-optimal data (21B tokens) would take **~90+ days** and cost **thousands of dollars**
- This is the fundamental bottleneck of the transformer paradigm: massive matrix multiplications + iterative gradient descent

### The breakthrough: Holographic Vector Learning (next phase)

Paying thousands of dollars and waiting 90 days to process 20 billion tokens through classical backpropagation is staying trapped in the old GPU + gradient-descent paradigm. **Fractus is not an LLM. It should not train like one.**

The next phase of Fractus abandons iterative weight adjustment entirely and moves to **state accumulation**:

**1. Hyperdimensional Vectorization**
- Tokens are projected into a very wide space (e.g., 10,000 dimensions) in **bipolar** representation (only +1 and -1).
- On CPU, heavy floating-point multiplications become **bit-level XOR and addition operations**. The CPU excels at this — massive speedup.

**2. Holographic Reduced Representations (HRR)**
- Instead of attention (which slows as text grows), concepts are bound via **circular convolution**. "cat" + "eats" → one vector of the same dimension containing both.
- Memory is **superposed** — the model sums bound vectors into a single shared space. Information is distributed across the whole network, like a hologram.

**3. Fractal Auto-Similarity**
- The vectors representing a word, a sentence, or a paragraph share the same dimension and space. No deep layers needed — meaning is extracted by self-similarity at scale.

**The result:** Instead of passing 20B tokens through the model dozens of times for gradient descent to converge, Fractus does a **single pass (one-shot learning)**. Read text → vectorize tokens → bind by holographic convolution → update global thermodynamic memory. **From 90 days to potentially a few days of CPU computation**, for a fraction of the cost.

**This is being developed in a separate repository** (`fractus-test`) to experiment without touching the working Fractus codebase.

---

## Quick start

```bash
git clone https://github.com/AFKmoney/fractus.git
cd fractus
py -m venv .venv && .venv\Scripts\Activate.ps1
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
maturin develop --release
pytest tests/ -q
```

Assemble and run the full agent:
```bash
python scripts/assemble_fractus.py
```

```python
from fractus.continuous_engine import ContinuousThoughtEngine
from fractus.tokenizer import FractusTokenizer
from fractus.rag import KnowledgeBase, RAGEngine, PluginManager, MetaCognition

engine = ContinuousThoughtEngine(vocab_size=5057, d_model=768, n_heads=12, d_head=64)
tok = FractusTokenizer.gpt2_compatible()
kb = KnowledgeBase(d_model=768)
rag = RAGEngine(engine, tok, kb)
pm = PluginManager(rag)
meta = MetaCognition(rag, pm)

# Teach — no retraining needed
rag.learn("Python is a programming language created by Guido van Rossum.")

# Ask
result = rag.query("Who created Python?", top_k=2, max_tokens=30)

# Let the agent manage itself
result = meta.process("Remember: my name is Philippe")
print(result['actions'])  # ['RETRIEVE', 'SWITCH', 'GENERATE']

# Switch cognitive mode
pm.load("coder")
```

---

## Architecture

```
Fractus/
├── crate/fractus-core/           Rust: 2-adic vortex (exact math)
├── crate/fractus-py/             Rust: PyO3 bindings
├── fractus/
│   ├── continuous_engine.py      The Continuous Thought Engine (ticks)
│   ├── model_1b.py               Training model (88M params, LazyStructuredSiren)
│   ├── rag.py                    RAG + Plugins + MetaCognition
│   ├── memory.py                 Persistent cross-session memory
│   ├── cognitive_modes.py        Kuramoto phase → mental state
│   ├── tokenizer.py              GPT-2 byte-level BPE
│   ├── nn/                       attention, Kuramoto, MoE, SIREN, Triton (13 modules)
│   └── train/                    online, surprise-gated, forward-forward
├── fractus1B/                    TRUE 1B param architecture + PGSU + Progressive Depth
├── space/                        HuggingFace Space (shared-memory demo, private)
├── scripts/
│   ├── assemble_fractus.py       FINAL assembly: CTE + RAG + plugins + MetaCognition
│   ├── transfer_to_cte.py        Transfer trained weights into CTE
│   ├── train_1b_cloud.py         Cloud GPU training script
│   └── build_fractus_corpus.py   Corpus builder
├── data/                         corpora, memory
├── tests/                        28 test files, 166+ tests
└── Fractus_White_Paper.pdf       Technical document (signed)
```

---

## License

MIT. This project belongs to the user, not to a corporation.

## Author

**Philippe-Antoine Robert** — 2026

## Links

- **GitHub:** [github.com/AFKmoney/fractus](https://github.com/AFKmoney/fractus)
- **HuggingFace (model):** [huggingface.co/thefinalboss/Fractus](https://huggingface.co/thefinalboss/Fractus)
- **HuggingFace (Space, private):** [huggingface.co/spaces/thefinalboss/Fractus-Space](https://huggingface.co/spaces/thefinalboss/Fractus-Space)
