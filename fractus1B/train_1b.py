#!/usr/bin/env python
"""Fractus1B training script — TRUE 1B params, GPU optimized.

Designed for 48GB+ GPUs (RTX 6000 Ada, A100, H100).
Uses gradient accumulation to fit 1B params in available VRAM.

Auto-uploads checkpoints to HuggingFace every N steps.

Usage on GPU pod:
    HF_TOKEN=xxx python -m fractus1B.train_1b \
        --corpus /workspace/fractus/data/fractus_corpus.pt \
        --epochs 1 --seq-len 32 --batch-size 8 --grad-accum 8 \
        --log-every 100 --save-every 5000
"""
import argparse, gc, math, os, sys, time

# Path setup so fractus1B package is importable.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
import torch.nn.functional as F

from fractus1B.model_1b import Fractus1B
from fractus1B.tokenizer import FractusTokenizer


HF_TOKEN = os.environ.get("HF_TOKEN", "")
HF_REPO = "thefinalboss/Fractus1B"


def upload_hf(path, repo_path):
    """Upload to HuggingFace. Never fails the training."""
    if not HF_TOKEN:
        print(f"  [HF] No token, skipping.", flush=True)
        return
    try:
        from huggingface_hub import HfApi
        api = HfApi(token=HF_TOKEN)
        api.upload_file(path_or_fileobj=path, path_in_repo=repo_path,
                       repo_id=HF_REPO, repo_type="model")
        print(f"  [HF] Uploaded {repo_path}", flush=True)
    except Exception as e:
        print(f"  [HF] Failed: {type(e).__name__} — training continues.", flush=True)


def save_checkpoint(model, optimizer, step, epoch, loss, config, ckpt_dir,
                     keep_last=2):
    """Save checkpoint + upload to HF. Keeps last N on disk."""
    os.makedirs(ckpt_dir, exist_ok=True)
    path = os.path.join(ckpt_dir, f"fractus1b_step{step}.pt")
    torch.save({
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "config": config,
        "step": step,
        "epoch": epoch,
        "loss": loss,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }, path)
    size_mb = os.path.getsize(path) / 1e6
    print(f"  [ckpt] step{step} ({size_mb:.0f}MB) loss={loss:.4f}", flush=True)
    upload_hf(path, f"checkpoints/fractus1b_step{step}.pt")
    upload_hf(path, "checkpoints/fractus1b_latest.pt")
    # GC old checkpoints (keep last N).
    import glob
    step_ckpts = sorted(
        glob.glob(os.path.join(ckpt_dir, "fractus1b_step*.pt")),
        key=lambda p: int(p.split("step")[-1].split(".")[0]),
    )
    for old in step_ckpts[:-keep_last]:
        os.remove(old)
        print(f"  [disk] Removed old step ckpt {os.path.basename(old)}", flush=True)


def main():
    parser = argparse.ArgumentParser(description="Fractus1B Training (true 1B params)")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--seq-len", type=int, default=32,
                       help="Sequence length (32 = chunk-based, Fractus-native)")
    parser.add_argument("--batch-size", type=int, default=8,
                       help="Physical batch size per forward pass")
    parser.add_argument("--grad-accum", type=int, default=8,
                       help="Gradient accumulation steps. Effective batch = batch_size * grad_accum.")
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--corpus", type=str, required=True,
                       help="Path to corpus .pt file")
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--save-every", type=int, default=5000,
                       help="Save+upload checkpoint every N steps")
    parser.add_argument("--warmup-steps", type=int, default=1000,
                       help="Linear LR warmup over this many steps")
    parser.add_argument("--pgsu", type=int, default=4,
                       help="Phase-Gated Sparse Update: number of layers active per step. "
                            "0=disabled (all layers trained every step). 4=default (4 of 16 "
                            "active → ~4x faster backward).")
    parser.add_argument("--progressive-depth", type=int, default=4,
                       help="Progressive Depth: number of phases to grow the model from shallow "
                            "to full depth. 0=disabled (all layers from step 0). 4=default "
                            "(4 phases: 4→8→12→16 layers). ~2x total speedup.")
    parser.add_argument("--optim-8bit", dest="optim_8bit", action="store_true",
                       default=True,
                       help="Use 8-bit AdamW (bitsandbytes) — saves ~6GB VRAM at 1B params. "
                            "Default ON. Auto-fallback to 32-bit if bnb not installed.")
    parser.add_argument("--no-8bit", dest="optim_8bit", action="store_false",
                       help="Disable 8-bit optimizer, use 32-bit AdamW.")
    args = parser.parse_args()

    # Device.
    if torch.cuda.is_available():
        device = torch.device("cuda")
        gpu_name = torch.cuda.get_device_name(0)
        vram = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"GPU: {gpu_name} ({vram:.1f} GB VRAM)", flush=True)
        torch.backends.cudnn.benchmark = True
    else:
        device = torch.device("cpu")
        print("WARNING: No GPU. 1B training on CPU will take weeks.", flush=True)

    torch.manual_seed(42)
    print(f"Threads: {os.cpu_count()}", flush=True)

    # Load corpus.
    print(f"Loading corpus: {args.corpus}", flush=True)
    tokens = torch.load(args.corpus, weights_only=False).long()
    print(f"Corpus: {len(tokens):,} tokens", flush=True)

    # Build model — TRUE 1B params.
    print("Building Fractus1B (true 1B params)...", flush=True)
    model = Fractus1B(
        vocab_size=50257,
        d_model=1280, n_layers=16, n_heads=20, d_head=64,
        n_levels=2, n_experts=128, top_k=2,
        expert_d_ff=2048, siren_rank=64,
        max_seq_len=args.seq_len,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Params: {n_params:,} ({n_params/1e9:.3f}B)", flush=True)
    print(f"  RAM: {n_params*4/1e9:.2f}GB", flush=True)
    print(f"  Config: d=1280 L=16 H=20 dh=64 E=128 K=2 ff=2048 rank=64", flush=True)

    # Resume.
    start_epoch = 0
    start_step = 0
    if args.resume:
        print(f"Resuming from: {args.resume}", flush=True)
        ckpt = torch.load(args.resume, weights_only=False, map_location=device)
        model.load_state_dict(ckpt["model_state"])
        start_epoch = ckpt.get("epoch", 0)
        start_step = ckpt.get("step", 0)
        print(f"  Resumed from epoch {start_epoch}, step {start_step}, loss={ckpt.get('loss','?')}", flush=True)

    # Optimizer with warmup.
    # Optimizer — 8-bit AdamW if bitsandbytes available (saves ~8GB VRAM at 1B params),
    # otherwise standard 32-bit AdamW. The 8-bit version stores optimizer state
    # (momentum, variance) in 8-bit instead of 32-bit, cutting optimizer memory
    # from 8GB → 2GB for a 1B model. Quality is preserved (Dettmers et al. 2022).
    use_8bit = args.optim_8bit and device.type == "cuda"
    if use_8bit:
        try:
            import bitsandbytes as bnb
            opt = bnb.optim.AdamW8bit(model.parameters(), lr=args.lr, weight_decay=0.01)
            print(f"  Optimizer: AdamW8bit (bitsandbytes) — ~6GB VRAM saved", flush=True)
        except ImportError:
            print(f"  bitsandbytes not installed, falling back to AdamW 32-bit", flush=True)
            opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
            use_8bit = False
        except Exception as e:
            print(f"  AdamW8bit init failed ({e}), falling back to AdamW 32-bit", flush=True)
            opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
            use_8bit = False
    else:
        opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
        print(f"  Optimizer: AdamW 32-bit", flush=True)

    if args.resume and "optimizer_state" in ckpt:
        try:
            opt.load_state_dict(ckpt["optimizer_state"])
            print(f"  Optimizer state restored", flush=True)
        except Exception as e:
            print(f"  Optimizer load failed: {e}", flush=True)

    # Cosine schedule with warmup.
    total_steps_estimate = (len(tokens) // (args.seq_len * args.batch_size * args.grad_accum)) * args.epochs
    def lr_lambda(step):
        if step < args.warmup_steps:
            return step / max(args.warmup_steps, 1)
        progress = (step - args.warmup_steps) / max(total_steps_estimate - args.warmup_steps, 1)
        return 0.5 * (1.0 + math.cos(math.pi * min(progress, 1.0)))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)
    # Fast-forward scheduler to resume point.
    for _ in range(start_step):
        sched.step()

    tok = FractusTokenizer.gpt2_compatible()
    seq = args.seq_len
    batch_size = args.batch_size
    grad_accum = args.grad_accum
    eff_batch = batch_size * grad_accum
    ckpt_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "checkpoints")
    n_steps = len(tokens) // (seq * eff_batch)

    use_amp = device.type == "cuda"

    # PGSU — Phase-Gated Sparse Update. When enabled, only N layers per step
    # receive gradients (the rest are frozen this step but rotate in next step).
    # This makes backward ~n_layers/N times faster while preserving convergence.
    pgsu = None
    if args.pgsu > 0 and args.pgsu < len(model.blocks):
        from fractus1B.pgsu import PGSU
        pgsu = PGSU(model, n_active=args.pgsu)
        # Fast-forward the PGSU step counter to match resume point.
        for _ in range(start_step):
            pgsu.step_count += 1

    # Progressive Depth — grow the model from shallow to full depth.
    # Compatible with PGSU: ProgressiveDepth freezes deep layers entirely
    # (they never activate), while PGSU rotates among the unfrozen ones.
    progressive = None
    if args.progressive_depth > 0:
        from fractus1B.progressive_depth import ProgressiveDepth
        progressive = ProgressiveDepth(
            model, total_steps=n_steps * args.epochs,
            n_phases=args.progressive_depth,
        )
        # Apply the initial phase immediately (freezes deep layers).
        progressive.update(start_step)

    print(f"\nTraining {args.epochs} epochs (true 1B params)", flush=True)
    print(f"  seq_len={seq}, batch_size={batch_size}, grad_accum={grad_accum}", flush=True)
    print(f"  effective batch = {eff_batch} (= {eff_batch*seq} tokens/step)", flush=True)
    print(f"  lr={args.lr}, warmup={args.warmup_steps} steps", flush=True)
    print(f"  {n_steps:,} optimizer steps/epoch", flush=True)
    print(f"  AMP (bf16): {'ON' if use_amp else 'OFF'}", flush=True)
    if pgsu:
        print(f"  PGSU: {pgsu.n_active}/{pgsu.n_layers} layers active per step "
              f"(expected ~{pgsu.n_layers/pgsu.n_active:.1f}× backward speedup)", flush=True)
    else:
        print(f"  PGSU: OFF (all layers trained every step)", flush=True)
    if progressive:
        print(f"  ProgressiveDepth: {progressive.n_phases} phases "
              f"(layers {progressive.phase_n_active[0]}→{progressive.phase_n_active[-1]}, "
              f"~2× total speedup)", flush=True)
    else:
        print(f"  ProgressiveDepth: OFF", flush=True)
    print(f"  HF upload: every {args.save_every} steps → {HF_REPO}", flush=True)
    print("=" * 70, flush=True)

    initial_loss = None
    step = start_step

    for epoch in range(start_epoch, args.epochs):
        model.train()
        t0 = time.perf_counter()
        ep_loss = 0.0
        ep_n = 0
        accum_count = 0

        # Iterate over the corpus in effective-batch chunks.
        tokens_per_step = seq * eff_batch
        for batch_start in range(0, len(tokens) - tokens_per_step - 1, tokens_per_step):
            # Progressive Depth: update which layers are unfrozen for this phase.
            if progressive:
                progressive.update(step)
            # PGSU: activate this step's layer subset before forward.
            if pgsu:
                pgsu.step_begin()
            # Inside one optimizer step: do grad_accum forward/backward passes.
            opt.zero_grad()
            accum_loss = 0.0
            for inner in range(grad_accum):
                inner_start = batch_start + inner * seq * batch_size
                inp_list, tgt_list = [], []
                for b in range(batch_size):
                    offset = inner_start + b * seq
                    inp_list.append(tokens[offset:offset + seq])
                    tgt_list.append(tokens[offset + 1:offset + seq + 1])
                inp = torch.stack(inp_list).to(device)
                tgt = torch.stack(tgt_list).to(device)

                if use_amp:
                    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                        logits, aux = model(inp)
                        ce = F.cross_entropy(logits.reshape(-1, 50257), tgt.reshape(-1))
                        # Clamp aux to prevent divergence (seen at step 149k on 88M run).
                        aux_clamped = torch.clamp(aux, max=1.0)
                        loss = (ce + 0.001 * aux_clamped) / grad_accum
                    loss.backward()
                else:
                    logits, aux = model(inp)
                    ce = F.cross_entropy(logits.reshape(-1, 50257), tgt.reshape(-1))
                    aux_clamped = torch.clamp(aux, max=1.0)
                    loss = (ce + 0.001 * aux_clamped) / grad_accum
                    loss.backward()
                accum_loss += loss.item() * grad_accum

            # Skip step on non-finite loss (defensive).
            total_loss = accum_loss / grad_accum
            if not math.isfinite(total_loss):
                print(f"  [warn] non-finite loss at step {step}, skipping", flush=True)
                opt.zero_grad()
                continue

            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            # PGSU: restore requires_grad for next step's forward.
            if pgsu:
                pgsu.step_end()
            step += 1

            ep_loss += total_loss
            ep_n += 1
            if initial_loss is None:
                initial_loss = total_loss

            if step % args.log_every == 0:
                elapsed = time.perf_counter() - t0
                sps = (step - (epoch * n_steps)) / max(elapsed, 1e-6)
                ppl = math.exp(min(total_loss, 20))
                cur_lr = sched.get_last_lr()[0]
                print(f"  E{epoch} S{step:>8}/{n_steps*(epoch+1):>8} "
                      f"loss={total_loss:.4f} ppl={ppl:.1f} aux={aux.item():.4f} "
                      f"lr={cur_lr:.2e} {sps:.2f}step/s", flush=True)

            if args.save_every > 0 and step % args.save_every == 0:
                save_checkpoint(
                    model, opt, step, epoch, total_loss,
                    {"seq_len": seq, "batch_size": batch_size,
                     "grad_accum": grad_accum, "lr": args.lr,
                     "corpus": args.corpus, "n_params": n_params},
                    ckpt_dir, keep_last=2,
                )

        # End of epoch.
        avg = ep_loss / max(ep_n, 1)
        elapsed = time.perf_counter() - t0
        print(f"\n[EPOCH {epoch}] avg_loss={avg:.4f} ppl={math.exp(min(avg,20)):.1f} "
              f"time={elapsed/3600:.1f}h", flush=True)
        # Save epoch checkpoint.
        save_checkpoint(
            model, opt, step, epoch, avg,
            {"seq_len": seq, "batch_size": batch_size, "grad_accum": grad_accum,
             "lr": args.lr, "corpus": args.corpus, "n_params": n_params, "final_epoch": True},
            ckpt_dir, keep_last=3,
        )

        # Sample generation.
        try:
            model.eval()
            with torch.no_grad():
                prompt = "def fibonacci"
                ids = tok.encode(prompt)[:seq]
                x = torch.tensor([ids], device=device)
                out = []
                for _ in range(40):
                    lg, _ = model(x)
                    nxt = lg[0, -1].argmax().unsqueeze(0).unsqueeze(0)
                    x = torch.cat([x, nxt], dim=1)[:, -seq:]
                    out.append(nxt.item())
                    if nxt.item() == 50256:
                        break
                print(f"[SAMPLE] {(prompt + tok.decode(out))[:250]}\n", flush=True)
            model.train()
        except Exception as e:
            print(f"  [sample] failed: {e}", flush=True)

    print("\n" + "=" * 70, flush=True)
    print(f"TRAINING COMPLETE. Final loss={avg:.4f}", flush=True)
    print(f"Loss reduction: {initial_loss:.4f} -> {avg:.4f}", flush=True)


if __name__ == "__main__":
    main()
