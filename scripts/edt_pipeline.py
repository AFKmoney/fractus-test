#!/usr/bin/env python
"""Expert Decoupled Training — complete pipeline for Fractus 1B.

Runs all 3 phases in sequence on the GPU pod:
  Phase 1: Train 2048 experts independently (1.2h)
  Phase 2a: Train 16 attention layers independently (<1s)
  Phase 2b: Train embedding on 500M tokens (3.2h)
  Phase 3: Brief joint fine-tune on 100M tokens (41h)

Total: ~2 days for a true 1B model on 21B token corpus.

Usage:
    HF_TOKEN=xxx python scripts/edt_pipeline.py \
        --corpus data/fractus_1b_corpus.pt \
        --phase1-steps 2000 \
        --embed-tokens 500000000 \
        --joint-tokens 100000000 \
        --save-every 5000
"""
import argparse, os, sys, time, math, json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

HF_TOKEN = os.environ.get("HF_TOKEN", "")
HF_REPO = "thefinalboss/fractus-test"


def upload_hf(path, repo_path):
    if not HF_TOKEN:
        return
    try:
        from huggingface_hub import HfApi
        api = HfApi(token=HF_TOKEN)
        api.upload_file(path_or_fileobj=path, path_in_repo=repo_path,
                       repo_id=HF_REPO, repo_type="model")
        print(f"  [HF] Uploaded {repo_path}", flush=True)
    except Exception as e:
        print(f"  [HF] Failed: {e}", flush=True)


def save_checkpoint(model, opt, step, phase, loss, path, keep_last=2):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    torch.save({
        "model_state": model.state_dict(),
        "optimizer_state": opt.state_dict() if hasattr(opt, 'state_dict') else None,
        "step": step, "phase": phase, "loss": loss,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }, path)
    size_mb = os.path.getsize(path) / 1e6
    print(f"  [ckpt] {path} ({size_mb:.0f}MB)", flush=True)
    upload_hf(path, f"checkpoints/edt_{phase}_step{step}.pt")
    upload_hf(path, "checkpoints/edt_latest.pt")


# =============================================================================
# PHASE 1: EXPERT PRE-TRAINING
# =============================================================================

def phase1_experts(model, device, steps_per_expert=2000, batch_size=256,
                    seq_len=32, lr=1e-3):
    """Train all 2048 experts independently."""
    print("\n" + "="*70, flush=True)
    print("PHASE 1: EXPERT PRE-TRAINING (2048 experts, independent)", flush=True)
    print("="*70, flush=True)

    t0 = time.time()
    total_experts = 0

    for layer_idx in range(len(model.blocks)):
        moe = model.blocks[layer_idx].moe
        n_experts = moe.n_experts

        for expert_idx in range(n_experts):
            expert_w1 = moe.experts_w1[expert_idx]
            expert_w2 = moe.experts_w2[expert_idx]
            params = list(expert_w1.parameters()) + list(expert_w2.parameters())

            opt = torch.optim.AdamW(params, lr=lr, weight_decay=0.01)
            d_model = expert_w1.in_features

            for step in range(steps_per_expert):
                # Random hidden states (simulating attention output).
                h_in = torch.randn(batch_size, seq_len, d_model, device=device)
                h_target = h_in.roll(-1, dims=1) + 0.1 * torch.randn_like(h_in)

                opt.zero_grad()
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    h1 = expert_w1(h_in)
                    h1_act = F.gelu(h1)
                    h_out = expert_w2(h1_act)
                    loss = F.mse_loss(h_out, h_target)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(params, 1.0)
                opt.step()

            total_experts += 1

        elapsed = time.time() - t0
        est_total = elapsed / (layer_idx + 1) * len(model.blocks)
        print(f"  Layer {layer_idx+1}/{len(model.blocks)} done "
              f"({n_experts} experts) — {elapsed:.0f}s elapsed, "
              f"ETA {est_total:.0f}s", flush=True)

    elapsed = time.time() - t0
    print(f"\nPhase 1 complete: {total_experts} experts in {elapsed/60:.1f}min", flush=True)
    return elapsed


# =============================================================================
# PHASE 2a: ATTENTION PRE-TRAINING
# =============================================================================

def phase2a_attention(model, device, steps_per_layer=5000, batch_size=16,
                       seq_len=8, lr=1e-3):
    """Train all 16 attention layers independently."""
    print("\n" + "="*70, flush=True)
    print("PHASE 2a: ATTENTION PRE-TRAINING (16 layers, independent)", flush=True)
    print("="*70, flush=True)

    t0 = time.time()
    d_model = model.d_model

    for layer_idx in range(len(model.blocks)):
        attn = model.blocks[layer_idx].attn
        norm = model.blocks[layer_idx].norm1
        params = list(attn.parameters()) + list(norm.parameters())
        opt = torch.optim.AdamW(params, lr=lr, weight_decay=0.01)

        for step in range(steps_per_layer):
            h = torch.randn(batch_size, seq_len, d_model, device=device)
            target = h + 0.1 * torch.randn_like(h)

            opt.zero_grad()
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                h_normed = norm(h)
                attn_out = attn(h_normed)
                h_out = h + attn_out
                loss = F.mse_loss(h_out, target)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            opt.step()

        print(f"  Layer {layer_idx+1}/16 done ({sum(p.numel() for p in params)/1e6:.1f}M params)", flush=True)

    elapsed = time.time() - t0
    print(f"\nPhase 2a complete: 16 layers in {elapsed:.1f}s", flush=True)
    return elapsed


# =============================================================================
# PHASE 2b: EMBEDDING PRE-TRAINING
# =============================================================================

def phase2b_embedding(model, corpus_tokens, device, n_tokens=500_000_000,
                       batch_size=128, seq_len=64, lr=1e-3,
                       log_every=2000, save_every=50000):
    """Train embedding on next-token prediction (no layers)."""
    print("\n" + "="*70, flush=True)
    print("PHASE 2b: EMBEDDING PRE-TRAINING", flush=True)
    print("="*70, flush=True)

    embed = model.embed
    lm_head = model.lm_head

    # Freeze everything except embedding.
    for p in model.parameters():
        p.requires_grad = False
    for p in embed.parameters():
        p.requires_grad = True

    opt = torch.optim.AdamW(embed.parameters(), lr=lr, weight_decay=0.01)

    tokens = corpus_tokens[:n_tokens].to(device)
    n_steps = n_tokens // (batch_size * seq_len)
    print(f"  {n_tokens/1e6:.0f}M tokens, {n_steps:,} steps", flush=True)

    t0 = time.time()
    for step in range(n_steps):
        idx = torch.randint(0, len(tokens) - seq_len - 1, (batch_size,))
        inp = torch.stack([tokens[i:i+seq_len] for i in idx])
        tgt = torch.stack([tokens[i+1:i+seq_len+1] for i in idx])

        opt.zero_grad()
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            h = embed(inp)
            logits = lm_head(h)
            loss = F.cross_entropy(logits.reshape(-1, 50257), tgt.reshape(-1))
        loss.backward()
        torch.nn.utils.clip_grad_norm_(embed.parameters(), 1.0)
        opt.step()

        if step % log_every == 0:
            elapsed = time.time() - t0
            rate = (step + 1) * batch_size * seq_len / max(elapsed, 1)
            ppl = math.exp(min(loss.item(), 20))
            print(f"  S{step:>7} loss={loss.item():.3f} ppl={ppl:.1f} "
                  f"{rate:.0f} tok/s", flush=True)

        if save_every > 0 and step > 0 and step % save_every == 0:
            save_checkpoint(model, opt, step, "embed", loss.item(),
                           f"checkpoints/edt_embed_step{step}.pt")

    elapsed = time.time() - t0
    print(f"\nPhase 2b complete: {n_tokens/1e6:.0f}M tokens in {elapsed/3600:.1f}h", flush=True)
    return elapsed


# =============================================================================
# PHASE 3: JOINT FINE-TUNE
# =============================================================================

def phase3_joint(model, corpus_tokens, device, n_tokens=100_000_000,
                  batch_size=8, seq_len=32, lr=3e-4,
                  log_every=500, save_every=5000):
    """Brief joint fine-tune: all components pre-trained, align them."""
    from fractus1B.pgsu import PGSU

    print("\n" + "="*70, flush=True)
    print("PHASE 3: JOINT FINE-TUNE (all components, brief alignment)", flush=True)
    print("="*70, flush=True)

    # Unfreeze everything.
    for p in model.parameters():
        p.requires_grad = True

    try:
        import bitsandbytes as bnb
        opt = bnb.optim.AdamW8bit(model.parameters(), lr=lr, weight_decay=0.01)
        print("  Optimizer: AdamW8bit", flush=True)
    except ImportError:
        opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)

    pgsu = PGSU(model, n_active=4)
    tokens = corpus_tokens[:n_tokens].to(device)
    n_steps = n_tokens // (batch_size * seq_len)
    use_amp = device.type == "cuda"

    print(f"  {n_tokens/1e6:.0f}M tokens, {n_steps:,} steps, PGSU=4/16", flush=True)

    t0 = time.time()
    initial_loss = None
    step = 0

    model.train()
    while step < n_steps:
        batch_start = torch.randint(0, len(tokens) - batch_size * seq_len - 1, (1,)).item()
        inp = torch.stack([tokens[batch_start + b*seq_len : batch_start + (b+1)*seq_len]
                          for b in range(batch_size)]).to(device)
        tgt = torch.stack([tokens[batch_start + b*seq_len + 1 : batch_start + (b+1)*seq_len + 1]
                          for b in range(batch_size)]).to(device)

        pgsu.step_begin()
        opt.zero_grad()

        if use_amp:
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                logits, aux = model(inp)
                ce = F.cross_entropy(logits.reshape(-1, 50257), tgt.reshape(-1))
                loss = ce + 0.001 * torch.clamp(aux, max=1.0)
            loss.backward()
        else:
            logits, aux = model(inp)
            ce = F.cross_entropy(logits.reshape(-1, 50257), tgt.reshape(-1))
            loss = ce + 0.001 * torch.clamp(aux, max=1.0)
            loss.backward()

        if not torch.isfinite(loss):
            opt.zero_grad()
            pgsu.step_end()
            continue

        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        pgsu.step_end()
        step += 1

        if initial_loss is None:
            initial_loss = loss.item()

        if step % log_every == 0:
            elapsed = time.time() - t0
            rate = step * batch_size * seq_len / max(elapsed, 1)
            ppl = math.exp(min(loss.item(), 20))
            print(f"  S{step:>6} loss={loss.item():.4f} ppl={ppl:.1f} "
                  f"{rate:.0f} tok/s", flush=True)

        if save_every > 0 and step > 0 and step % save_every == 0:
            save_checkpoint(model, opt, step, "joint", loss.item(),
                           f"checkpoints/edt_joint_step{step}.pt")

    # Final save.
    elapsed = time.time() - t0
    avg_loss = loss.item()
    save_checkpoint(model, opt, step, "final", avg_loss,
                   "checkpoints/fractus1b_edt_final.pt")

    print(f"\nPhase 3 complete: {step} steps in {elapsed/3600:.1f}h", flush=True)
    print(f"  Loss: {initial_loss:.4f} → {avg_loss:.4f}", flush=True)
    return elapsed


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Expert Decoupled Training — Fractus 1B")
    parser.add_argument("--corpus", type=str, required=True)
    parser.add_argument("--phase1-steps", type=int, default=2000)
    parser.add_argument("--phase2a-steps", type=int, default=5000)
    parser.add_argument("--embed-tokens", type=int, default=500_000_000)
    parser.add_argument("--joint-tokens", type=int, default=100_000_000)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--seq-len", type=int, default=32)
    parser.add_argument("--save-every", type=int, default=5000)
    parser.add_argument("--skip-phase1", action="store_true")
    parser.add_argument("--skip-phase2", action="store_true")
    parser.add_argument("--resume", type=str, default=None)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

    print(f"Device: {device}", flush=True)
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}", flush=True)
        print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory/1e9:.1f} GB", flush=True)

    from fractus1B.model_1b import Fractus1B
    model = Fractus1B(
        vocab_size=50257, d_model=1280, n_layers=16, n_heads=20, d_head=64,
        n_levels=2, n_experts=128, top_k=2, expert_d_ff=2048, siren_rank=64,
        max_seq_len=64,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model: {n_params:,} params ({n_params/1e9:.3f}B)", flush=True)

    if args.resume and os.path.exists(args.resume):
        print(f"Resuming from {args.resume}", flush=True)
        ck = torch.load(args.resume, weights_only=False, map_location=device)
        model.load_state_dict(ck["model_state"], strict=False)
        print(f"  Loaded (phase={ck.get('phase')}, step={ck.get('step')})", flush=True)

    # Load corpus.
    print(f"Loading corpus: {args.corpus}", flush=True)
    corpus_tokens = torch.load(args.corpus, weights_only=False).long()
    print(f"Corpus: {len(corpus_tokens):,} tokens ({len(corpus_tokens)/1e9:.2f}B)", flush=True)

    grand_t0 = time.time()

    # Phase 1.
    if not args.skip_phase1:
        phase1_experts(model, device, steps_per_expert=args.phase1_steps)
        save_checkpoint(model, None, 0, "phase1_done", 0.0,
                       "checkpoints/edt_after_phase1.pt")

    # Phase 2.
    if not args.skip_phase2:
        phase2a_attention(model, device, steps_per_layer=args.phase2a_steps)
        phase2b_embedding(model, corpus_tokens, device,
                         n_tokens=args.embed_tokens,
                         save_every=args.save_every)
        save_checkpoint(model, None, 0, "phase2_done", 0.0,
                       "checkpoints/edt_after_phase2.pt")

    # Phase 3.
    phase3_joint(model, corpus_tokens, device,
                n_tokens=args.joint_tokens,
                batch_size=args.batch_size, seq_len=args.seq_len,
                save_every=args.save_every)

    grand_elapsed = time.time() - grand_t0
    print(f"\n{'='*70}", flush=True)
    print(f"EDT COMPLETE — Fractus 1B trained in {grand_elapsed/3600:.1f}h "
          f"({grand_elapsed/3600/24:.1f} days)", flush=True)
    print(f"{'='*70}", flush=True)


if __name__ == "__main__":
    main()
