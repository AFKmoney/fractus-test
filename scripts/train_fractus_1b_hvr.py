#!/usr/bin/env python
"""Fractus 1B HVR Training — encode 21B tokens holographically + fine-tune CTE.

TWO-PHASE TRAINING (the fast paradigm):

Phase 1 — HVR ENCODING (one pass, no backprop):
    Read the 21B token corpus, encode all token transitions into the
    HolographicMemoryGPU. This takes ~1-2 days on GPU instead of 90 days
    of backprop. Auto-saves to HF every 1B tokens.

Phase 2 — CTE FINE-TUNE (short, with HVR bias):
    The 1B CTE is fine-tuned on a SMALL corpus (~2B tokens) with the HVR
    providing knowledge biases during generation. The CTE learns to reason
    + query HVR, not to memorize facts. ~1-3 days on GPU.

Both phases auto-upload checkpoints to HuggingFace.

Usage:
    HF_TOKEN=xxx python scripts/train_fractus_1b_hvr.py \
        --corpus data/fractus_1b_corpus.pt \
        --phase both \
        --hvr-save-every 1000000000 \
        --cte-save-every 5000
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


def upload_hvr_memory(hvr, path, repo_path):
    """Save + upload the HVR memory to HF."""
    hvr.save(path)
    size_mb = os.path.getsize(path) / 1e6
    print(f"  [HVR] Saved {path} ({size_mb:.0f}MB)", flush=True)
    upload_hf(path, repo_path)


# =============================================================================
# PHASE 1: HVR ENCODING
# =============================================================================

def phase_hvr_encode(corpus_path, device, save_every=1_000_000_000):
    """Encode the entire corpus into HVR memory in one pass."""
    from fractus.hvr_gpu import HolographicMemoryGPU

    print("="*70, flush=True)
    print("PHASE 1: HVR ENCODING (one pass, no gradient descent)", flush=True)
    print("="*70, flush=True)

    # Load corpus.
    print(f"Loading corpus: {corpus_path}", flush=True)
    tokens = torch.load(corpus_path, weights_only=False).long()
    n_tokens = len(tokens)
    print(f"Corpus: {n_tokens:,} tokens ({n_tokens/1e9:.2f}B)", flush=True)

    # Build HVR.
    print(f"Building HVR memory (dim=10000, vocab=50257) on {device}...", flush=True)
    hvr = HolographicMemoryGPU(dim=10_000, vocab_size=50257, seed=42, device=str(device))

    # Encode in chunks.
    chunk_size = 500_000  # 500k tokens per chunk
    t0 = time.time()
    tokens_processed = 0

    for start in range(0, n_tokens - 1, chunk_size):
        end = min(start + chunk_size + 1, n_tokens)
        chunk = tokens[start:end].to(device)
        hvr.learn_sequence(chunk)

        tokens_processed = end
        elapsed = time.time() - t0
        rate = tokens_processed / max(elapsed, 1)
        pct = tokens_processed / n_tokens * 100
        eta = (n_tokens - tokens_processed) / max(rate, 1) / 3600

        print(f"  {tokens_processed/1e9:.2f}B / {n_tokens/1e9:.2f}B tokens "
              f"({pct:.1f}%), {rate:.0f} tok/s, ETA {eta:.1f}h", flush=True)

        # Auto-save + upload to HF every save_every tokens.
        if save_every > 0 and tokens_processed % save_every < chunk_size:
            ckpt_path = f"hvr_memory_{tokens_processed//1_000_000}M.pt"
            upload_hvr_memory(hvr, ckpt_path,
                              f"hvr/hvr_memory_checkpoint.pt")

    # Final save.
    final_path = "hvr_memory_final.pt"
    upload_hvr_memory(hvr, final_path, "hvr/hvr_memory_final.pt")

    elapsed = time.time() - t0
    print(f"\nHVR encoding complete: {n_tokens:,} tokens in {elapsed/3600:.1f}h", flush=True)
    print(f"  Average speed: {n_tokens/elapsed:.0f} tok/s", flush=True)
    print(f"  Memory norm: {hvr.memory.norm().item():.1f}", flush=True)

    return hvr


# =============================================================================
# PHASE 2: CTE FINE-TUNE WITH HVR BIAS
# =============================================================================

def phase_cte_finetune(corpus_path, hvr, device, epochs=1, batch_size=4,
                        seq_len=32, lr=3e-4, save_every=5000, log_every=100,
                        cte_checkpoint=None):
    """Fine-tune the CTE to reason + query HVR.

    The CTE learns on a SMALL corpus (it doesn't need to memorize facts —
    HVR handles that). It learns attention patterns + MoE routing + how
    to use HVR biases for generation.
    """
    from fractus.model_1b import Fractus1B
    from fractus.hvr_cte_bridge import HVRContextProvider

    print("\n" + "="*70, flush=True)
    print("PHASE 2: CTE FINE-TUNE (with HVR knowledge bias)", flush=True)
    print("="*70, flush=True)

    # Build the 1B model.
    print("Building Fractus1B...", flush=True)
    model = Fractus1B(
        vocab_size=50257, d_model=1280, n_layers=16, n_heads=20, d_head=64,
        n_levels=2, n_experts=128, top_k=2,
        expert_d_ff=2048, siren_rank=64,
        max_seq_len=seq_len,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Params: {n_params:,} ({n_params/1e9:.3f}B)", flush=True)

    # Load checkpoint if provided (rank-expanded from 88M).
    if cte_checkpoint and os.path.exists(cte_checkpoint):
        print(f"Loading checkpoint: {cte_checkpoint}", flush=True)
        ck = torch.load(cte_checkpoint, weights_only=False, map_location=device)
        model.load_state_dict(ck.get("model_state", ck), strict=False)
        print(f"  Loaded (step={ck.get('source_step', '?')})", flush=True)

    # Build HVR context provider.
    provider = HVRContextProvider(hvr, top_k=10, vocab_size=50257, bias_strength=2.0)

    # Optimizer (8-bit if available).
    use_8bit = device.type == "cuda"
    if use_8bit:
        try:
            import bitsandbytes as bnb
            opt = bnb.optim.AdamW8bit(model.parameters(), lr=lr, weight_decay=0.01)
            print("  Optimizer: AdamW8bit", flush=True)
        except ImportError:
            opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
            print("  Optimizer: AdamW 32-bit (bnb not installed)", flush=True)
    else:
        opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)

    # Load corpus (use only first 2B tokens for fine-tune — HVR has the rest).
    print(f"Loading corpus for fine-tune: {corpus_path}", flush=True)
    tokens = torch.load(corpus_path, weights_only=False).long()
    # Cap at 2B tokens for the fine-tune phase.
    max_finetune_tokens = 2_000_000_000
    if len(tokens) > max_finetune_tokens:
        print(f"  Capping fine-tune corpus to {max_finetune_tokens/1e9:.1f}B tokens", flush=True)
        tokens = tokens[:max_finetune_tokens]

    n_steps = len(tokens) // (seq_len * batch_size)
    use_amp = device.type == "cuda"
    print(f"  {n_steps:,} steps/epoch, batch={batch_size}, seq={seq_len}", flush=True)

    step = 0
    initial_loss = None

    model.train()
    for epoch in range(epochs):
        t0 = time.time()
        ep_loss = 0.0
        ep_n = 0

        for batch_start in range(0, len(tokens) - seq_len * batch_size - 1, seq_len * batch_size):
            # Build batch.
            inp = torch.stack([tokens[batch_start + b*seq_len : batch_start + (b+1)*seq_len]
                              for b in range(batch_size)]).to(device)
            tgt = torch.stack([tokens[batch_start + b*seq_len + 1 : batch_start + (b+1)*seq_len + 1]
                              for b in range(batch_size)]).to(device)

            opt.zero_grad()

            if use_amp:
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    logits, aux = model(inp)
                    ce = F.cross_entropy(logits.reshape(-1, 50257), tgt.reshape(-1))

                    # HVR knowledge bias: add logit bias for each position.
                    with torch.no_grad():
                        for b in range(batch_size):
                            for s in range(seq_len):
                                tid = inp[b, s].item()
                                if tid < 50257:
                                    bias = provider.get_logit_bias(tid).to(device)
                                    logits[b, s] += bias

                    aux_clamped = torch.clamp(aux, max=1.0)
                    loss = ce + 0.001 * aux_clamped
                loss.backward()
            else:
                logits, aux = model(inp)
                ce = F.cross_entropy(logits.reshape(-1, 50257), tgt.reshape(-1))
                loss = ce + 0.001 * torch.clamp(aux, max=1.0)
                loss.backward()

            if not torch.isfinite(loss):
                opt.zero_grad()
                continue

            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            step += 1

            loss_val = loss.item()
            ep_loss += loss_val
            ep_n += 1
            if initial_loss is None:
                initial_loss = loss_val

            if step % log_every == 0:
                elapsed = time.time() - t0
                sps = step / max(elapsed, 1)
                ppl = math.exp(min(loss_val, 20))
                print(f"  E{epoch} S{step:>8}/{n_steps} loss={loss_val:.4f} "
                      f"ppl={ppl:.1f} {sps:.2f}step/s", flush=True)

            if save_every > 0 and step % save_every == 0:
                ckpt_path = f"fractus1b_hvr_step{step}.pt"
                torch.save({
                    "model_state": model.state_dict(),
                    "step": step,
                    "epoch": epoch,
                    "loss": loss_val,
                    "config": {"seq_len": seq_len, "batch_size": batch_size, "lr": lr},
                }, ckpt_path)
                size_mb = os.path.getsize(ckpt_path) / 1e6
                print(f"  [ckpt] step{step} ({size_mb:.0f}MB)", flush=True)
                upload_hf(ckpt_path, f"checkpoints/fractus1b_hvr_step{step}.pt")
                upload_hf(ckpt_path, "checkpoints/fractus1b_hvr_latest.pt")

        avg = ep_loss / max(ep_n, 1)
        print(f"\n[EPOCH {epoch}] avg_loss={avg:.4f} ppl={math.exp(min(avg,20)):.1f} "
              f"time={(time.time()-t0)/3600:.1f}h", flush=True)

    print(f"\nCTE fine-tune complete. loss={initial_loss:.4f} → {avg:.4f}", flush=True)
    return model


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Fractus 1B HVR Training")
    parser.add_argument("--corpus", type=str, required=True,
                       help="Path to the 21B token corpus .pt file")
    parser.add_argument("--phase", type=str, default="both",
                       choices=["hvr", "cte", "both"],
                       help="Which phase to run: hvr=encode only, cte=fine-tune only, both")
    parser.add_argument("--cte-checkpoint", type=str, default=None,
                       help="Checkpoint for CTE init (rank-expanded from 88M)")
    parser.add_argument("--hvr-checkpoint", type=str, default=None,
                       help="Pre-built HVR memory (skip encoding)")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--seq-len", type=int, default=32)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--hvr-save-every", type=int, default=1_000_000_000,
                       help="Save+upload HVR memory every N tokens")
    parser.add_argument("--cte-save-every", type=int, default=5000)
    parser.add_argument("--log-every", type=int, default=100)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}", flush=True)
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}", flush=True)
        print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory/1e9:.1f} GB", flush=True)

    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

    hvr = None

    # Phase 1: HVR encoding.
    if args.phase in ("hvr", "both"):
        if args.hvr_checkpoint and os.path.exists(args.hvr_checkpoint):
            print(f"Loading pre-built HVR: {args.hvr_checkpoint}", flush=True)
            from fractus.hvr_gpu import HolographicMemoryGPU
            hvr = HolographicMemoryGPU(dim=10_000, vocab_size=50257, device=str(device))
            hvr.load(args.hvr_checkpoint)
        else:
            hvr = phase_hvr_encode(args.corpus, device, args.hvr_save_every)

    # Phase 2: CTE fine-tune.
    if args.phase in ("cte", "both"):
        if hvr is None and args.hvr_checkpoint:
            from fractus.hvr_gpu import HolographicMemoryGPU
            hvr = HolographicMemoryGPU(dim=10_000, vocab_size=50257, device=str(device))
            hvr.load(args.hvr_checkpoint)
        if hvr is None:
            print("ERROR: no HVR memory available for CTE phase", flush=True)
            sys.exit(1)
        phase_cte_finetune(args.corpus, hvr, device, args.epochs,
                           args.batch_size, args.seq_len, args.lr,
                           args.cte_save_every, args.log_every,
                           args.cte_checkpoint)

    print("\nDONE.", flush=True)


if __name__ == "__main__":
    main()
