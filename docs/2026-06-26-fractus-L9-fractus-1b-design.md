# Fractus L9 — Fractus-1B: A 1B-Capacity Model Trained on CPU

**Date:** 2026-06-26
**Goal:** Build and train a 1B-capacity language model on a consumer CPU (AMD Ryzen 5 5500U), by combining three innovations that exploit fractus's unique structure. Then export to ONNX and publish to HuggingFace Hub as `Fractus-1B`.

---

## The Claim (and why it's not dogma)

Every source says "1B models require datacenter GPUs." That claim assumes dense fp32 training where every parameter is active at every token and backpropagated at every step. Fractus's architecture breaks all three assumptions:

1. **Structured SIREN** — weight matrices are decomposed as `W ≈ U·Vᵀ + SIREN(residual)`, so 1B of *effective matrix capacity* is stored and trained as ~15-30M *actual parameters*.
2. **Massive sparse MoE** — 64 experts, only top-2 active per token → compute per token is ~3% of total capacity.
3. **Surprise-gated training** — backpropagate only on high-loss tokens → gradient work is proportional to what the model still needs to learn, not to total token count.

Combined, the math: ~20M trainable params × 64-expert capacity × sparse activation = a model whose *forward sees* ~1B of matrix capacity but whose *training cost* is that of a ~20M model. This is what makes CPU training of a "1B" model structurally possible.

---

## Architecture

### StructuredSirenLinear (the core innovation)
Each large weight matrix is reconstructed from two low-parameter components:

```
W[out, in] ≈ U[out, r] · V[in, r]ᵀ  +  SIREN_spectral(R)
```

- **Low-rank core** `U·Vᵀ` (rank r=64-128): captures the dominant singular directions. Storage: `(out+in)·r` instead of `out·in`. For 4096×4096 with r=128: ~1M params instead of 16M.
- **Spectral residual SIREN**: learns only `R = W - U·Vᵀ`. The residual has exploitable spectral structure (the high-energy singular values are already captured by U·Vᵀ), so the SIREN compresses it far better than it would compress W directly. Target: 10-30× over storing W dense.

The forward pass reconstructs W on-the-fly (in-graph, differentiable), then does `y = x @ W + b`. Both U, V, and the SIREN params receive gradients.

### Model configuration (Fractus-1B)
```
vocab_size:          50257 (GPT-2 BPE, byte-level)
d_model:             1024
n_layers:            12
n_heads:             16
d_head:              64
n_levels:            4 (attention Mandelbrot levels)
n_experts:           64 (sparse MoE)
top_k:               2 (experts active per token)
expert_d_ff:         1024
siren_rank:          128 (StructuredSiren low-rank)
siren_residual_hidden: 256
max_seq_len:         512
```

**Capacity math:**
- Each expert's StructuredSirenLinear: effective 1024×1024 = 1M matrix, trained as ~50k params (rank-128 + small SIREN residual).
- 12 layers × 64 experts × 1M = ~768M expert capacity.
- Plus attention, embedding, head: ~250M effective.
- **Total effective capacity: ~1B.** Total trainable params: ~15-25M.

### Byte-level BPE tokenizer
GPT-2-compatible byte-level BPE (vocab=50257). 4× sequence compression vs char-level, essential for code/maths/multi-language. The fractal embedding adapts: char-features are replaced by byte-pair features, the Fourier basis stays.

### Training pipeline
- **Datasets:** 20-25 quality HF datasets (code, maths, cybersec, science, FR+EN), streamed and filtered.
- **Checkpointing:** every 1000 steps → upload to HF Hub → delete local copy (disk management).
- **Benchmarking:** every 1000 steps → perplexity on held-out validation set → log to HF.
- **Async prefetch:** data loading + Kuramoto phase precomputation on separate threads.

### Export & publication
- **ONNX export:** decompress all SIREN matrices to dense, export as a standard transformer ONNX.
- **HuggingFace Hub:** `Fractus-1B` repo with model.onnx, tokenizer, config, model card, paper-style README.
- **GitHub:** full training code, reproducibility instructions, benchmark results.

---

## Measured success criteria (not claims)
1. The model trains on the Ryzen 5 without OOM (memory fits in ~8GB).
2. Perplexity on the validation set decreases monotonically (it learns).
3. Generation produces coherent multi-language text/code after training.
4. ONNX export runs in `onnxruntime` and matches PyTorch output (atol=1e-4).
5. Checkpoints land on HF Hub every 1000 steps.

## Honest risks
1. The SIREN residual compression ratio on real trained weights is the research unknown — measured at 1.5× for smooth fields, could be 5-15× for spectral residuals. If it's too low, we fall back to rank-256 or pure low-rank (LoRA-style).
2. Training time: even at 20M params, multi-day on CPU. The async pipeline + surprise-gating mitigate but don't eliminate this.
3. Model quality at 1B-capacity-from-20M-params: it won't match a true dense 1B. But it will be the first *demonstrable* 1B-capacity model trained from scratch on a laptop CPU.
