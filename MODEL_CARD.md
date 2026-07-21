---
language: ["en", "fr", "code"]
tags:
  - continuous-thought-engine
  - kuramoto
  - mixture-of-experts
  - siren
  - cpu-trained
  - decentralized-ai
license: mit
---

# Fractus-1B — Continuous Thought Engine

**The world's first real-time reasoning AI trained entirely on a CPU laptop.**

Not a static LLM (input→output like GPT/Claude). A **dynamical system** that thinks continuously, remembers across sessions, has cognitive modes, and plans before generating.

## Training details

- **Hardware**: AMD Ryzen 5 5500U (6 cores, 12 threads, CPU-only)
- **Architecture**: ContinuousThoughtEngine (Kuramoto + linear attention + sparse MoE + StructuredSiren)
- **Parameters**: 13M trainable (effective capacity scales with MoE)
- **Tokenizer**: GPT-2 byte-level BPE (vocab 50257)
- **Training data**: tinyshakespeare (30k tokens) + Python code + math text
- **Training method**: chunk-based online learning (16 tokens/chunk, 117 tok/s on CPU)
- **Final metrics**: loss=1.00, perplexity=2.7, accuracy=77% (next-token prediction)

## What makes this different from GPT/Claude

| Property | GPT/Claude | Fractus |
|---|---|---|
| Processing | Static (1 forward pass) | Continuous (tick-by-tick) |
| Memory | Context window only | Persistent memory bank |
| Skills | Generic | Specialized MoE experts |
| Mental state | Stateless | Cognitive modes (Kuramoto) |
| Generation | Token-by-token | Generative planning |
| Training | Datacenter GPU | CPU laptop |

## Files

- `fractus_continuous.onnx` — ONNX export (tick-based inference)
- `checkpoints/checkpoint_871410.pt` — full PyTorch checkpoint (53.7 MB)
- Training log shows 30 epochs of convergence

## Usage

```python
from fractus.continuous_engine import ContinuousThoughtEngine
from fractus.tokenizer import FractusTokenizer

# Load the engine.
ckpt = torch.load("checkpoints/checkpoint_871410.pt", weights_only=False)
engine = ContinuousThoughtEngine(vocab_size=50257, d_model=128, ...)
engine.load_state_dict(ckpt["model_state"])

# Think continuously.
tok = FractusTokenizer.gpt2_compatible()
engine.reset_thought(1)
for tid in tok.encode("Hello"):
    engine.tick(torch.tensor([tid]))
logits, confidence = engine.tick()  # one thought tick
```

## License

MIT. This model belongs to the user, not a corporation.

## Reproducing

```bash
git clone https://github.com/AFKmoney/fractus.git
cd fractus
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
maturin develop --release
python scripts/train_continuous.py --epochs 30 --d-model 128
```

Full source: https://github.com/AFKmoney/fractus
