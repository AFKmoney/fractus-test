#!/usr/bin/env python
"""Fractus-1B cloud training script — GPU optimized, auto-push to HF.

Designed for RunPod/QuickPod RTX 4090 or A100.
Automatically uploads every checkpoint to HuggingFace.

Usage on cloud GPU:
    pip install torch datasets huggingface_hub
    python train_1b_cloud.py --epochs 20
    
Environment variables needed:
    HF_TOKEN=your_token_here
"""
import argparse, gc, math, os, sys, time
import torch, torch.nn as nn, torch.nn.functional as F
from fractus.model_1b import Fractus1B
from fractus.tokenizer import FractusTokenizer

# Try to import Triton kernels (will fail silently on CPU / no-triton).
try:
    from fractus.nn.triton_kernels import fused_linear_cross_entropy, TRITON_READY, self_test as triton_self_test
    _HAS_TRITON_IMPORT = True
except Exception:
    _HAS_TRITON_IMPORT = False
    TRITON_READY = False

HF_TOKEN = os.environ.get("HF_TOKEN", "")
HF_REPO = "thefinalboss/Fractus"


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


def save_and_upload(model, optimizer, epoch, loss, acc, config, ckpt_dir):
    """Save checkpoint locally + upload to HF."""
    os.makedirs(ckpt_dir, exist_ok=True)
    path = os.path.join(ckpt_dir, f"fractus_1b_epoch{epoch}.pt")
    torch.save({
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "config": config,
        "epoch": epoch,
        "loss": loss,
        "accuracy": acc,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }, path)
    size_mb = os.path.getsize(path) / 1e6
    print(f"  [ckpt] {path} ({size_mb:.0f}MB)", flush=True)
    upload_hf(path, f"checkpoints/fractus_1b_epoch{epoch}.pt")
    upload_hf(path, "checkpoints/fractus_1b_latest.pt")
    # Delete old checkpoint to save disk on cloud.
    if epoch > 1:
        old = os.path.join(ckpt_dir, f"fractus_1b_epoch{epoch-1}.pt")
        if os.path.exists(old):
            os.remove(old)
            print(f"  [disk] Removed old checkpoint {old}", flush=True)


def save_step_checkpoint(model, optimizer, step, epoch, loss, config, ckpt_dir,
                          keep_last=2):
    """Save a mid-epoch checkpoint every N steps + upload to HF.

    Named by global step (e.g. fractus_1b_step10000.pt). Uploads as both the
    step-named file AND 'fractus_1b_latest.pt' so resume always picks up the
    newest. Keeps only the last `keep_last` step checkpoints on disk to avoid
    filling the pod.

    This is the CRASH-RECOVERY path: if the pod dies mid-epoch, you resume
    from the latest step checkpoint and lose at most save_every steps of work.
    """
    os.makedirs(ckpt_dir, exist_ok=True)
    path = os.path.join(ckpt_dir, f"fractus_1b_step{step}.pt")
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
    upload_hf(path, f"checkpoints/fractus_1b_step{step}.pt")
    upload_hf(path, "checkpoints/fractus_1b_latest.pt")

    # Garbage-collect old step checkpoints (keep the last `keep_last`).
    import glob
    step_ckpts = sorted(glob.glob(os.path.join(ckpt_dir, "fractus_1b_step*.pt")),
                        key=lambda p: int(p.split("step")[-1].split(".")[0]))
    for old in step_ckpts[:-keep_last]:
        os.remove(old)
        print(f"  [disk] Removed old step ckpt {os.path.basename(old)}", flush=True)


def chunked_cross_entropy(model, hidden, target, vocab, chunk_positions):
    """Compute lm_head + cross-entropy by chunks of positions to avoid
    materializing the full (B, L, vocab) tensor.

    hidden:  (B, L, d_model) — the final hidden states (already through all blocks).
    target:  (B, L) — next-token ids.
    Returns scalar loss (averaged over all positions).

    Processes positions in chunks of `chunk_positions`. Each chunk materializes
    only (B, chunk, vocab) — keeping VRAM low so batch can grow 4-8x.
    """
    B, L, _ = hidden.shape
    total_loss = 0.0
    n = 0
    # Detach hidden from the chunk loop's graph accumulation — we sum losses
    # and backward once at the end. Each chunk's logits share the same hidden,
    # so the gradient flows correctly through hidden to all blocks.
    losses = []
    for s in range(0, L, chunk_positions):
        e = min(s + chunk_positions, L)
        h_chunk = hidden[:, s:e]                      # (B, C, D)
        logits_chunk = model.lm_head(h_chunk)         # (B, C, vocab)
        tgt_chunk = target[:, s:e]                     # (B, C)
        l = F.cross_entropy(
            logits_chunk.reshape(-1, vocab),
            tgt_chunk.reshape(-1),
            reduction="sum",
        )
        losses.append(l)
        n += (e - s) * B
    total = torch.stack(losses).sum() / n
    return total


def main():
    parser = argparse.ArgumentParser(description="Fractus-1B Cloud Training")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--seq-len", type=int, default=64,
                       help="Longer seq = better context (GPU can handle it)")
    parser.add_argument("--batch-size", type=int, default=8,
                       help="Batch size (GPU parallelism)")
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--corpus", type=str, default=None,
                       help="Path to corpus. If not found, builds it.")
    parser.add_argument("--resume", type=str, default=None,
                       help="Checkpoint to resume from")
    parser.add_argument("--upload-every", type=int, default=1,
                       help="Upload checkpoint every N epochs")
    parser.add_argument("--log-every", type=int, default=500,
                       help="Log every N steps")
    parser.add_argument("--compile", dest="compile", action="store_true",
                       default=True,
                       help="Enable torch.compile (default ON on GPU)")
    parser.add_argument("--no-compile", dest="compile", action="store_false",
                       help="Disable torch.compile, use eager")
    parser.add_argument("--chunk-ce", type=int, default=0,
                       help="Chunk positions for CE (0=disabled, 8=recommended). "
                            "Avoids materializing full (B,L,vocab) tensor → bigger batch.")
    parser.add_argument("--triton-ce", dest="triton_ce", action="store_true",
                       default=True,
                       help="Use Triton fused linear+CE kernel (default ON on GPU). "
                            "Auto self-test; falls back if unavailable.")
    parser.add_argument("--no-triton-ce", dest="triton_ce", action="store_false")
    parser.add_argument("--save-every", type=int, default=10000,
                       help="Save+upload checkpoint every N steps (crash recovery). "
                            "Default 10000 = ~6%% of a 1.76B-token epoch at batch 512.")
    args = parser.parse_args()

    # Detect device.
    if torch.cuda.is_available():
        device = torch.device("cuda")
        gpu_name = torch.cuda.get_device_name(0)
        vram = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"GPU: {gpu_name} ({vram:.1f} GB VRAM)", flush=True)
        torch.backends.cudnn.benchmark = True
    else:
        device = torch.device("cpu")
        print("WARNING: No GPU detected. Running on CPU.", flush=True)

    torch.manual_seed(42)
    num_threads = os.cpu_count() or 4
    torch.set_num_threads(num_threads)
    print(f"Threads: {num_threads}", flush=True)

    # Load or build corpus.
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_dir = os.path.dirname(script_dir)
    
    if args.corpus:
        corpus_path = args.corpus
    else:
        # Try communication corpus first, then ultimate, then mega.
        for name in ["communication_corpus.pt", "ultimate_corpus.pt", "mega_corpus.pt"]:
            p = os.path.join(project_dir, "data", name)
            if os.path.exists(p):
                corpus_path = p
                break
        else:
            # Build it.
            print("No corpus found. Building communication corpus...", flush=True)
            import subprocess
            subprocess.run([sys.executable, os.path.join(script_dir, "build_communication_corpus.py")], check=True)
            corpus_path = os.path.join(project_dir, "data", "communication_corpus.pt")

    print(f"Loading corpus: {corpus_path}", flush=True)
    tokens = torch.load(corpus_path, weights_only=False).long()
    print(f"Corpus: {len(tokens):,} tokens", flush=True)

    # Build model.
    print("Building Fractus-1B...", flush=True)
    model = Fractus1B(
        vocab_size=50257, d_model=768, n_layers=8, n_heads=12, d_head=64,
        n_levels=2, n_experts=64, top_k=2, expert_d_ff=1024, siren_rank=16,
        max_seq_len=args.seq_len,
    ).to(device)
    n = model.n_params()
    cap = model.n_effective_capacity()
    print(f"  Params: {n:,} ({n/1e6:.0f}M)", flush=True)
    print(f"  Capacity: {cap:,} ({cap/1e9:.2f}B)", flush=True)
    print(f"  RAM: {n*4/1e9:.1f}GB", flush=True)

    # torch.compile — now that the MoE is vectorized (no dynamic control flow),
    # compile can fuse kernels. Cache limit raised to handle the 64 expert guards.
    # GUARD: wrapped in try/except, falls back to eager on any failure.
    use_compiled = False
    if args.compile and device.type == "cuda":
        try:
            import torch._dynamo as dyn
            dyn.config.cache_size_limit = 256
            dyn.config.accumulated_cache_size_limit = 512
            model = torch.compile(model, mode="reduce-overhead", dynamic=False)
            use_compiled = True
            print("  torch.compile: ON (mode=reduce-overhead)", flush=True)
        except Exception as e:
            print(f"  torch.compile: FAILED ({type(e).__name__}), using eager", flush=True)
    else:
        print(f"  torch.compile: OFF ({'disabled by --no-compile' if not args.compile else 'CPU device'})", flush=True)

    # Triton fused kernel — runs self-test before use. Falls back to eager/chunk-ce.
    use_triton_ce = False
    if device.type == "cuda" and _HAS_TRITON_IMPORT and args.triton_ce:
        try:
            ok = triton_self_test()
            use_triton_ce = bool(ok)
            print(f"  triton fused-CE: {'ON' if use_triton_ce else 'OFF (self-test failed)'}", flush=True)
        except Exception as e:
            print(f"  triton fused-CE: FAILED ({type(e).__name__})", flush=True)
    else:
        print(f"  triton fused-CE: OFF (cuda={device.type=='cuda'}, import={_HAS_TRITON_IMPORT})", flush=True)

    # Resume if specified.
    start_epoch = 0
    start_step = 0
    if args.resume:
        print(f"Resuming from: {args.resume}", flush=True)
        ckpt = torch.load(args.resume, weights_only=False, map_location=device)
        model.load_state_dict(ckpt["model_state"])
        start_epoch = ckpt.get("epoch", 0)
        # Restore the global step counter so checkpoint names + logs use the
        # true global step, not a local counter that restarts at 0 on resume.
        # This was a bug: resuming from step 140000 reset the counter to 0,
        # causing new checkpoints to overwrite old ones on HF by name.
        start_step = ckpt.get("step", 0)
        # Restore optimizer state if present (so Adam moments are preserved).
        if "optimizer_state" in ckpt:
            try:
                opt.load_state_dict(ckpt["optimizer_state"])
                print(f"  Optimizer state restored", flush=True)
            except Exception as e:
                print(f"  Optimizer state load failed: {e}", flush=True)
        print(f"  Resumed from epoch {start_epoch}, step {start_step}, loss={ckpt.get('loss','?')}", flush=True)

    # Optimizer + scheduler.
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs, eta_min=1e-6)
    if start_epoch > 0:
        for _ in range(start_epoch):
            sched.step()

    # AMP for GPU.
    use_amp = device.type == "cuda"
    scaler = torch.cuda.amp.GradScaler() if use_amp else None

    tok = FractusTokenizer.gpt2_compatible()
    seq = args.seq_len
    batch_size = args.batch_size
    ckpt_dir = os.path.join(project_dir, "checkpoints")
    n_steps = len(tokens) // seq // batch_size

    print(f"\nTraining {args.epochs} epochs", flush=True)
    print(f"  seq_len={seq}, batch_size={batch_size}, lr={args.lr}", flush=True)
    print(f"  {n_steps:,} steps/epoch", flush=True)
    print(f"  AMP: {'ON' if use_amp else 'OFF'}", flush=True)
    print(f"  HF upload: every {args.upload_every} epochs → {HF_REPO}", flush=True)
    print("=" * 70, flush=True)

    initial_loss = None
    step = start_step  # global step counter (preserved across resumes)

    for epoch in range(start_epoch, args.epochs):
        model.train()
        t0 = time.perf_counter()
        ep_loss = 0.0
        ep_n = 0

        # Create batches.
        for batch_start in range(0, len(tokens) - seq * batch_size - 1, seq * batch_size):
            # Build batch.
            inp_list = []
            tgt_list = []
            for b in range(batch_size):
                offset = batch_start + b * seq
                inp_list.append(tokens[offset:offset + seq])
                tgt_list.append(tokens[offset + 1:offset + seq + 1])
            inp = torch.stack(inp_list).to(device)
            tgt = torch.stack(tgt_list).to(device)

            opt.zero_grad()
            if use_amp:
                with torch.cuda.amp.autocast(dtype=torch.bfloat16):
                    if use_triton_ce:
                        # Triton fused path: skip lm_head in model, kernel does it.
                        model._return_hidden = True
                        hidden, aux = model(inp)
                        model._return_hidden = False
                        ce = fused_linear_cross_entropy(hidden, model.lm_head.weight, tgt)
                    elif args.chunk_ce > 0:
                        model._return_hidden = True
                        hidden, aux = model(inp)
                        ce = chunked_cross_entropy(model, hidden, tgt, 50257, args.chunk_ce)
                        model._return_hidden = False
                    else:
                        logits, aux = model(inp)
                        ce = F.cross_entropy(logits.reshape(-1, 50257), tgt.reshape(-1))
                    # Clip aux (load-balance loss) to prevent the divergence seen at step 149000.
                    # When all tokens route to one expert, lb_loss can spike to 5-15 and kill the
                    # main CE gradient. Cap its contribution at 0.001.
                    aux_clamped = torch.clamp(aux, max=1.0)
                    loss = ce + 0.001 * aux_clamped
                loss.backward()
            else:
                if use_triton_ce:
                    model._return_hidden = True
                    hidden, aux = model(inp)
                    model._return_hidden = False
                    ce = fused_linear_cross_entropy(hidden, m.lm_head.weight, tgt)
                elif args.chunk_ce > 0:
                    model._return_hidden = True
                    hidden, aux = model(inp)
                    ce = chunked_cross_entropy(model, hidden, tgt, 50257, args.chunk_ce)
                    model._return_hidden = False
                else:
                    logits, aux = model(inp)
                    ce = F.cross_entropy(logits.reshape(-1, 50257), tgt.reshape(-1))
                aux_clamped = torch.clamp(aux, max=1.0)
                loss = ce + 0.001 * aux_clamped
                loss.backward()
            # Skip the step if loss became NaN/inf (defensive — should not happen post-clamp).
            if not torch.isfinite(loss):
                opt.zero_grad()
                continue
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

            loss_val = loss.item()
            ep_loss += loss_val
            ep_n += 1
            step += 1
            if initial_loss is None:
                initial_loss = loss_val
            if step % args.log_every == 0:
                elapsed = time.perf_counter() - t0
                sps = (step - (epoch * n_steps)) / max(elapsed, 1e-6)
                ppl = math.exp(min(loss_val, 20))
                print(f"  E{epoch} S{step:>7}/{n_steps*(epoch+1):>7} "
                      f"loss={loss_val:.4f} ppl={ppl:.1f} aux={aux.item():.4f} "
                      f"{sps:.1f}step/s", flush=True)
            # Mid-epoch checkpoint for crash recovery (every save_every steps).
            if args.save_every > 0 and step % args.save_every == 0:
                save_step_checkpoint(
                    model, opt, step, epoch, loss_val,
                    {"seq_len": seq, "batch_size": batch_size,
                     "lr": args.lr, "corpus": args.corpus},
                    ckpt_dir, keep_last=2,
                )

        # End of epoch.
        avg = ep_loss / max(ep_n, 1)
        elapsed = time.perf_counter() - t0
        print(f"\n[EPOCH {epoch}] avg_loss={avg:.4f} ppl={math.exp(min(avg,20)):.1f} "
              f"time={elapsed/60:.1f}min", flush=True)
        save_and_upload(model, opt, epoch, avg, initial_loss,
                        {"seq_len": seq, "batch_size": batch_size,
                         "lr": args.lr, "corpus": args.corpus}, ckpt_dir)
        sched.step()

        # Sample generation to monitor quality.
        try:
            model.eval()
            with torch.no_grad():
                prompt = "def fibonacci"
                ids = tok.encode(prompt)
                x = torch.tensor([ids], device=device)
                out = []
                for _ in range(60):
                    lg, _ = model(x)
                    nxt = lg[0, -1].argmax().unsqueeze(0).unsqueeze(0)
                    x = torch.cat([x, nxt], dim=1)
                    out.append(nxt.item())
                    if nxt.item() == 50256:
                        break
                print(f"[SAMPLE] {(prompt + tok.decode(out))[:300]}\n", flush=True)
            model.train()
        except Exception as e:
            print(f"  [sample] failed: {e}", flush=True)

    print("\n" + "=" * 70, flush=True)
    print(f"TRAINING COMPLETE. Final loss={avg:.4f}", flush=True)
    print(f"Loss reduction: {initial_loss:.4f} -> {avg:.4f} "
          f"({(initial_loss-avg)/initial_loss*100:.1f}% improvement)", flush=True)


if __name__ == "__main__":
    main()
