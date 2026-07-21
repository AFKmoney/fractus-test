#!/usr/bin/env python
"""Fractus-1B: the real 1B-capacity model, trained on CPU.

Config: d_model=768, 8 layers, 64 experts, top-2, StructuredSiren rank=16.
89M trainable params, 0.86B effective capacity, 0.4GB RAM.
~4.3s/step, ~2h per epoch on 50k tokens.
"""
import argparse, math, os, sys, time, json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch, torch.nn as nn
from fractus.model_1b import Fractus1B

HF_TOKEN = os.environ.get("HF_TOKEN", "")
HF_REPO = "thefinalboss/Fractus-1B"

def save_ckpt(model, step, loss, config, cap_b, train_m):
    d = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "checkpoints")
    os.makedirs(d, exist_ok=True)
    p = os.path.join(d, f"fractus_1b_step_{step}.pt")
    torch.save({"model_state": model.state_dict(), "config": config,
                "step": step, "loss": loss, "capacity_b": cap_b,
                "trainable_m": train_m}, p)
    print(f"  [ckpt] {p} ({os.path.getsize(p)/1e6:.0f}MB)", flush=True)
    if HF_TOKEN:
        try:
            from huggingface_hub import HfApi
            api = HfApi(token=HF_TOKEN)
            api.upload_file(path_or_fileobj=p, path_in_repo=f"checkpoints/fractus_1b_step_{step}.pt",
                          repo_id=HF_REPO, repo_type="model")
            print(f"  [HF] Uploaded step {step}", flush=True)
        except Exception as e:
            print(f"  [HF] Skipped: {type(e).__name__}", flush=True)

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--seq-len", type=int, default=32)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--max-tokens", type=int, default=50000)
    args = p.parse_args()
    torch.set_num_threads(os.cpu_count() or 6)
    torch.manual_seed(42)

    # Load corpus.
    cpath = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "fractus_corpus.pt")
    tokens = torch.load(cpath, weights_only=False).long()[:args.max_tokens]
    print(f"Corpus: {len(tokens):,} tokens", flush=True)

    # Build model.
    print("Building Fractus-1B...", flush=True)
    model = Fractus1B(
        vocab_size=50257, d_model=768, n_layers=8, n_heads=12, d_head=64,
        n_levels=2, n_experts=64, top_k=2, expert_d_ff=1024, siren_rank=16,
        max_seq_len=args.seq_len,
    )
    n = model.n_params()
    cap = model.n_effective_capacity()
    print(f"  Trainable: {n:,} ({n/1e6:.0f}M)  Capacity: {cap/1e9:.2f}B  RAM: {n*4/1e9:.1f}GB", flush=True)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs, eta_min=1e-6)

    seq = args.seq_len
    print(f"\nTraining {args.epochs} epochs, seq={seq}, lr={args.lr}", flush=True)
    print("=" * 70, flush=True)

    step = 0
    initial_loss = None
    for epoch in range(args.epochs):
        model.train()
        t0 = time.perf_counter()
        ep_loss = 0.0
        ep_n = 0
        for i in range(0, len(tokens) - seq - 1, seq):
            inp = tokens[i:i+seq].unsqueeze(0)
            tgt = tokens[i+1:i+seq+1].unsqueeze(0)
            opt.zero_grad()
            with torch.amp.autocast('cpu', dtype=torch.bfloat16):
                logits, aux = model(inp)
                ce = nn.functional.cross_entropy(logits.reshape(-1, 50257), tgt.reshape(-1))
                loss = ce + 0.001 * aux.float()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            step += 1
            ep_loss += ce.item()
            ep_n += 1
            if initial_loss is None:
                initial_loss = ce.item()
            if step % 25 == 0:
                el = time.perf_counter() - t0
                tps = (i + seq) / el
                print(f"  step {step:5d}  ce={ce.item():.3f}  "
                      f"{tps:.0f} tok/s  lr={opt.param_groups[0]['lr']:.6f}  "
                      f"({el/60:.1f}min)", flush=True)
            if step % 250 == 0:
                save_ckpt(model, step, ep_loss/max(ep_n,1), model.config, cap/1e9, n/1e6)

        el = time.perf_counter() - t0
        avg = ep_loss / max(ep_n, 1)
        print(f"\nEpoch {epoch+1}/{args.epochs}: loss={avg:.3f} ppl={math.exp(min(avg,20)):.1f} "
              f"{len(tokens)/el:.0f} tok/s {el/60:.1f}min\n", flush=True)
        save_ckpt(model, step, avg, model.config, cap/1e9, n/1e6)

    # ONNX export.
    print("Exporting ONNX...", flush=True)
    model.eval()
    onnx_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "fractus_1b.onnx")
    dummy = torch.randint(0, 50257, (1, args.seq_len))
    try:
        torch.onnx.export(model, dummy, onnx_path, opset_version=17,
                         input_names=["input_ids"], output_names=["logits"])
        print(f"ONNX: {os.path.getsize(onnx_path)/1e6:.0f}MB", flush=True)
        if HF_TOKEN:
            from huggingface_hub import HfApi
            api = HfApi(token=HF_TOKEN)
            api.upload_file(path_or_fileobj=onnx_path, path_in_repo="fractus_1b.onnx",
                          repo_id=HF_REPO, repo_type="model")
            print("[HF] ONNX uploaded", flush=True)
    except Exception as e:
        print(f"ONNX: {e}", flush=True)
    print(f"\nFractus-1B done. {cap/1e9:.2f}B capacity, {n/1e6:.0f}M params.", flush=True)

if __name__ == "__main__":
    main()
