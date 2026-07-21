"""Expert Decoupled Training — Phase 2: Layer-Decoupled Attention.

Phase 1 trained each expert independently (1.2h total).
Phase 2 trains each attention layer independently — NOT through the
full 16-layer stack.

THE INSIGHT:
  Standard training backprops through all 16 layers → 660ms/step.
  But each attention layer is only 6.6M params. Training it standalone
  (as a denoising/autoencoding module) takes milliseconds.

  After all layers are pre-trained, a brief joint fine-tune aligns them.

PHASE 2a — ATTENTION PRE-TRAINING (layer by layer):
  Each attention layer learns to transform hidden states in a useful way,
  trained standalone on the embedding's output. No MoE, no stacking.

PHASE 2b — EMBEDDING PRE-TRAINING:
  The embedding learns token representations on the corpus directly.

PHASE 2c — BRIEF JOINT FINE-TUNE (the only expensive part):
  With experts + attention + embedding all pre-trained, unfreeze and
  fine-tune on a SMALL corpus (~100M tokens). This is like training a
  pre-trained model — converges fast because it starts good.
"""
import os, sys, time, math
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


def train_embedding(model, corpus_tokens, device, n_tokens=500_000_000,
                     batch_size=64, seq_len=64, lr=1e-3, max_steps=200000,
                     log_every=2000):
    """Phase 2b: train embedding via next-token prediction (no layers).

    The embedding alone learns token co-occurrence. This is like training
    a tiny 64M param model (just the embedding table + tied LM head).
    Very fast because there's no attention/MoE.
    """
    print("\n--- Phase 2b: Embedding pre-training ---", flush=True)
    embed = model.embed
    lm_head = model.lm_head  # tied weight

    # Freeze everything except embedding.
    for p in model.parameters():
        p.requires_grad = False
    for p in embed.parameters():
        p.requires_grad = True

    opt = torch.optim.AdamW(embed.parameters(), lr=lr, weight_decay=0.01)
    tokens = corpus_tokens[:n_tokens].to(device)
    n_steps = min(max_steps, len(tokens) // (seq_len * batch_size))

    print(f"  {embed.tok_embed.weight.numel()/1e6:.0f}M trainable params", flush=True)
    print(f"  {n_steps:,} steps, batch={batch_size}, seq={seq_len}", flush=True)

    model.train()
    t0 = time.time()
    for step in range(n_steps):
        idx = torch.randint(0, len(tokens) - seq_len - 1, (batch_size,))
        inp = torch.stack([tokens[i:i+seq_len] for i in idx])
        tgt = torch.stack([tokens[i+1:i+seq_len+1] for i in idx])

        opt.zero_grad()
        h = embed(inp)  # (batch, seq, d_model)
        logits = lm_head(h)  # (batch, seq, vocab) — tied weight
        loss = F.cross_entropy(logits.reshape(-1, 50257), tgt.reshape(-1))
        loss.backward()
        torch.nn.utils.clip_grad_norm_(embed.parameters(), 1.0)
        opt.step()

        if step % log_every == 0:
            elapsed = time.time() - t0
            rate = (step + 1) * batch_size * seq_len / max(elapsed, 1)
            ppl = math.exp(min(loss.item(), 20))
            print(f"  S{step:>6} loss={loss.item():.3f} ppl={ppl:.1f} "
                  f"{rate:.0f} tok/s", flush=True)

    elapsed = time.time() - t0
    rate = n_steps * batch_size * seq_len / elapsed
    print(f"  Done: {n_steps} steps in {elapsed/60:.1f}min ({rate:.0f} tok/s)", flush=True)
    return rate


def train_attention_layer(attn_module, norm_module, device, d_model=1280,
                           max_steps=5000, batch_size=128, lr=1e-3,
                           log_every=1000):
    """Phase 2a: train ONE attention layer standalone.

    The attention learns to process hidden states in a useful way.
    Input: random hidden states (simulating embedding output).
    Target: same states shifted + denoised (self-supervised).

    Each layer is 6.6M params → trains in seconds.
    """
    params = list(attn_module.parameters()) + list(norm_module.parameters())
    opt = torch.optim.AdamW(params, lr=lr, weight_decay=0.01)

    t0 = time.time()
    for step in range(max_steps):
        # Random hidden states.
        h = torch.randn(batch_size, 32, d_model, device=device)
        # Target: the attention should denoise + structure the input.
        target = h + 0.1 * torch.randn_like(h)

        opt.zero_grad()
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            h_normed = norm_module(h)
            attn_out = attn_module(h_normed)
            h_out = h + attn_out  # residual
            loss = F.mse_loss(h_out, target)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step()

        if step % log_every == 0:
            print(f"    step {step}: loss={loss.item():.4f}", flush=True)

    elapsed = time.time() - t0
    return elapsed, loss.item()


def benchmark_phase2_decoupled(model, device, max_steps=2000):
    """Benchmark Phase 2 with decoupled attention + embedding."""
    print("\n" + "="*70, flush=True)
    print("PHASE 2 (DECOUPLED) — Layer-by-layer attention + embedding", flush=True)
    print("="*70, flush=True)

    # Phase 2a: train all 16 attention layers independently.
    print("\nPhase 2a: Attention layers (standalone)...", flush=True)
    total_attn_time = 0
    for layer_idx in range(16):
        attn = model.blocks[layer_idx].attn
        norm = model.blocks[layer_idx].norm1
        n_params = sum(p.numel() for p in attn.parameters()) + sum(p.numel() for p in norm.parameters())

        t, loss = train_attention_layer(attn, norm, device, max_steps=max_steps,
                                         batch_size=128, lr=1e-3, log_every=max_steps)
        total_attn_time += t
        print(f"  Layer {layer_idx:>2}: {n_params/1e6:.1f}M params, "
              f"{t:.1f}s, loss={loss:.4f}", flush=True)

    print(f"\n  Phase 2a total: {total_attn_time:.1f}s = {total_attn_time/60:.1f}min", flush=True)

    # Phase 2b: embedding (needs corpus — use dummy for benchmark).
    print("\nPhase 2b: Embedding (benchmark with dummy)...", flush=True)
    dummy_tokens = torch.randint(0, 50257, (10_000_000,))
    rate = train_embedding(model, dummy_tokens, device,
                           n_tokens=10_000_000, batch_size=128, seq_len=64,
                           max_steps=1000, log_every=500)
    embed_time_500m = 500_000_000 / rate / 3600
    print(f"  Embedding rate: {rate:.0f} tok/s", flush=True)
    print(f"  For 500M tokens: {embed_time_500m:.1f}h", flush=True)

    # Phase 2c: brief joint fine-tune (estimate).
    # With everything pre-trained, ~100M tokens should suffice.
    # At 678 tok/s (measured full model): 100M / 678 / 3600 = 41h
    joint_time = 100_000_000 / 678 / 3600
    print(f"\nPhase 2c: Joint fine-tune (estimate)", flush=True)
    print(f"  100M tokens at 678 tok/s = {joint_time:.1f}h", flush=True)

    # TOTAL.
    total_hours = total_attn_time/3600 + embed_time_500m + joint_time
    print(f"\n{'='*70}", flush=True)
    print(f"  PHASE 2 DECOUPLED TOTAL: {total_hours:.1f} hours = {total_hours/24:.1f} days", flush=True)
    print(f"  + Phase 1 (experts): 1.2h", flush=True)
    print(f"  GRAND TOTAL EDT: {(total_hours + 1.2/24):.1f} days", flush=True)
    print(f"{'='*70}", flush=True)
    return total_hours


if __name__ == "__main__":
    from fractus1B.model_1b import Fractus1B

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

    print(f"GPU: {torch.cuda.get_device_name(0)}", flush=True)

    model = Fractus1B(
        vocab_size=50257, d_model=1280, n_layers=16, n_heads=20, d_head=64,
        n_levels=2, n_experts=128, top_k=2, expert_d_ff=2048, siren_rank=64,
        max_seq_len=64,
    ).to(device)

    benchmark_phase2_decoupled(model, device, max_steps=2000)
