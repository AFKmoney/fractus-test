#!/usr/bin/env python
"""Train the REAL Fractus-1B (1.27B capacity, 158M trainable) on CPU.

This is the actual 1B model. Uses the Fractus1B architecture with
StructuredSiren cached experts + chunk-based training.
"""
import argparse, math, os, sys, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
import torch.nn as nn
from fractus.model_1b import Fractus1B
from fractus.tokenizer import FractusTokenizer

HF_TOKEN = os.environ.get("HF_TOKEN", "")
HF_REPO = "thefinalboss/Fractus-1B"

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--seq-len", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--max-tokens", type=int, default=50000)
    args = parser.parse_args()

    torch.set_num_threads(os.cpu_count() or 6)
    torch.manual_seed(42)

    # Load corpus.
    corpus_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "fractus_corpus.pt")
    print(f"Loading corpus...", flush=True)
    tokens = torch.load(corpus_path, weights_only=False).long()[:args.max_tokens]
    print(f"Corpus: {len(tokens):,} tokens", flush=True)

    # Build the 1B model.
    print("Building Fractus-1B (1.27B capacity)...", flush=True)
    model = Fractus1B(
        vocab_size=50257, d_model=768, n_layers=12, n_heads=12, d_head=64,
        n_levels=2, n_experts=64, top_k=2, expert_d_ff=1024, siren_rank=32,
        max_seq_len=args.seq_len,
    )
    n = model.n_params()
    cap = model.n_effective_capacity()
    print(f"  Trainable: {n:,} ({n/1e6:.0f}M)", flush=True)
    print(f"  Capacity:  {cap:,} ({cap/1e9:.2f}B)", flush=True)
    print(f"  RAM:       {n*4/1e9:.1f}GB", flush=True)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs, eta_min=1e-6)

    # Training loop: chunk-based.
    seq = args.seq_len
    n_chunks = len(tokens) // seq
    print(f"\nTraining {args.epochs} epochs, {n_chunks} chunks/epoch, seq={seq}", flush=True)
    print("=" * 70, flush=True)

    step = 0
    for epoch in range(args.epochs):
        model.train()
        epoch_loss = 0.0
        n_steps = 0
        t0 = time.perf_counter()

        for i in range(0, len(tokens) - seq - 1, seq):
            inp = tokens[i:i+seq].unsqueeze(0)
            tgt = tokens[i+1:i+seq+1].unsqueeze(0)

            opt.zero_grad()
            with torch.amp.autocast('cpu', dtype=torch.bfloat16):
                logits, aux = model(inp)
                ce = nn.functional.cross_entropy(
                    logits.reshape(-1, 50257), tgt.reshape(-1)
                )
                loss = ce + 0.001 * aux.float()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            scheduler.step()
            step += 1
            epoch_loss += ce.item()
            n_steps += 1

            if step % 50 == 0:
                elapsed = time.perf_counter() - t0
                tps = (i + seq) / elapsed
                print(f"  step {step:5d}  ce={ce.item():.3f}  "
                      f"{tps:.0f} tok/s  lr={opt.param_groups[0]['lr']:.6f}", flush=True)

            # Checkpoint every 500 steps.
            if step % 500 == 0:
                ckpt_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "checkpoints")
                os.makedirs(ckpt_dir, exist_ok=True)
                ckpt_path = os.path.join(ckpt_dir, f"fractus_1b_step_{step}.pt")
                torch.save({
                    "model_state": model.state_dict(),
                    "config": model.config,
                    "step": step,
                    "loss": epoch_loss/max(n_steps,1),
                    "capacity_b": cap/1e9,
                    "trainable_m": n/1e6,
                }, ckpt_path)
                size_mb = os.path.getsize(ckpt_path)/1e6
                print(f"  [checkpoint] {ckpt_path} ({size_mb:.0f}MB)", flush=True)
                # Try HF.
                if HF_TOKEN:
                    try:
                        from huggingface_hub import HfApi
                        api = HfApi(token=HF_TOKEN)
                        api.upload_file(path_or_fileobj=ckpt_path,
                                       path_in_repo=f"checkpoints/fractus_1b_step_{step}.pt",
                                       repo_id=HF_REPO, repo_type="model")
                        print(f"  [HF] Uploaded step {step}", flush=True)
                    except Exception as e:
                        print(f"  [HF] Skipped: {type(e).__name__}", flush=True)

        elapsed = time.perf_counter() - t0
        avg_loss = epoch_loss / max(n_steps, 1)
        ppl = math.exp(min(avg_loss, 20))
        print(f"Epoch {epoch+1}/{args.epochs}: loss={avg_loss:.3f} ppl={ppl:.1f} "
              f"{len(tokens)/elapsed:.0f} tok/s {elapsed:.0f}s", flush=True)

    # Export ONNX.
    print("\nExporting ONNX...", flush=True)
    model.eval()
    onnx_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "fractus_1b.onnx")
    dummy = torch.randint(0, 50257, (1, args.seq_len))
    try:
        torch.onnx.export(model, dummy, onnx_path, opset_version=17,
                         input_names=["input_ids"], output_names=["logits"],
                         dynamic_axes={"input_ids":{0:"batch",1:"seq"}, "logits":{0:"batch",1:"seq"}})
        print(f"ONNX: {onnx_path} ({os.path.getsize(onnx_path)/1e6:.0f}MB)", flush=True)
    except Exception as e:
        print(f"ONNX export: {e}", flush=True)

    print("\nFractus-1B training complete.", flush=True)

if __name__ == "__main__":
    main()
