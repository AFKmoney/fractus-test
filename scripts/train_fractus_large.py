#!/usr/bin/env python
"""Train Fractus-1B on the large multi-domain corpus.

Scales up: d_model=256, 8 experts, more data (500k tokens), more epochs.
Uses the chunk-based ContinuousThoughtEngine for fast CPU training.
Checkpoints to local + HuggingFace Hub every 5 epochs.

Usage:
    python scripts/train_fractus_large.py
    python scripts/train_fractus_large.py --epochs 100 --d-model 256
"""

import argparse
import math
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from fractus.continuous_engine import ContinuousThoughtEngine
from fractus.train.online import OnlineTrainer
from fractus.tokenizer import FractusTokenizer

HF_TOKEN = os.environ.get("HF_TOKEN", "")
HF_REPO_ID = "thefinalboss/Fractus-1B"


def save_checkpoint(engine, epoch, loss, accuracy, output_dir="checkpoints"):
    """Save checkpoint locally + try HF upload."""
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, f"fractus_epoch_{epoch}.pt")
    torch.save({
        "model_state": engine.state_dict(),
        "config": {"d_model": engine.d_model, "vocab_size": engine.vocab_size},
        "epoch": epoch,
        "loss": loss,
        "accuracy": accuracy,
    }, path)
    size_mb = os.path.getsize(path) / 1e6
    print(f"  [checkpoint] Saved {path} ({size_mb:.1f} MB)", flush=True)

    # Try HF upload.
    if HF_TOKEN:
        try:
            from huggingface_hub import HfApi
            api = HfApi(token=HF_TOKEN)
            api.upload_file(
                path_or_fileobj=path,
                path_in_repo=f"checkpoints/fractus_epoch_{epoch}.pt",
                repo_id=HF_REPO_ID,
                repo_type="model",
            )
            print(f"  [HF] Uploaded epoch {epoch}", flush=True)
        except Exception as e:
            print(f"  [HF] Skipped: {type(e).__name__}", flush=True)


def export_onnx(engine, path):
    """Export to ONNX."""
    engine.eval()
    for e in engine.experts_w1:
        e.force_refresh()
    for e in engine.experts_w2:
        e.force_refresh()

    class TickWrapper(torch.nn.Module):
        def __init__(self, engine):
            super().__init__()
            self.engine = engine
        def forward(self, obs):
            logits, conf = self.engine.tick(obs)
            return logits, conf

    wrapper = TickWrapper(engine)
    dummy = torch.tensor([100], dtype=torch.long)
    try:
        torch.onnx.export(
            wrapper, dummy, path,
            input_names=["observation"],
            output_names=["logits", "confidence"],
            opset_version=17,
        )
        print(f"  ONNX: {path} ({os.path.getsize(path)/1e6:.1f} MB)", flush=True)
    except Exception as e:
        print(f"  ONNX failed: {e}", flush=True)


def generate_sample(engine, tok, prompt="The ", n=60):
    """Generate a text sample for monitoring."""
    engine.eval()
    engine.reset_thought(1)
    import torch.nn.functional as F
    ids = tok.encode(prompt)
    for tid in ids[:16]:
        engine.tick(torch.tensor([tid]))
    generated = list(ids[:16])
    for _ in range(n):
        logits, _ = engine.tick()
        l = logits[0] / 0.8
        tv, ti = l.topk(40)
        probs = F.softmax(tv, dim=-1)
        idx = torch.multinomial(probs, 1).item()
        generated.append(ti[idx].item())
    engine.train()
    return tok.decode(generated)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--d-model", type=int, default=256)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--chunk-len", type=int, default=16)
    parser.add_argument("--checkpoint-every", type=int, default=5)
    parser.add_argument("--corpus", type=str, default=None)
    args = parser.parse_args()

    torch.set_num_threads(os.cpu_count() or 6)
    torch.manual_seed(42)

    # Load corpus.
    corpus_path = args.corpus or os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "data", "fractus_corpus.pt",
    )
    print(f"Loading corpus: {corpus_path}", flush=True)
    tokens = torch.load(corpus_path, weights_only=False).long()
    print(f"Corpus: {len(tokens):,} tokens", flush=True)

    # Tokenizer.
    tok = FractusTokenizer.gpt2_compatible()

    # Build scaled engine.
    d = args.d_model
    d_head = 64
    n_heads = max(2, d // d_head)
    siren_rank = min(64, d // 4)
    engine = ContinuousThoughtEngine(
        vocab_size=tok.vocab_size, d_model=d, n_heads=n_heads, d_head=d_head,
        n_levels=2, n_oscillators=8, coupling_rank=4,
        n_experts=8, top_k=2, expert_d_ff=d, siren_rank=siren_rank,
    )
    n_params = sum(p.numel() for p in engine.parameters())
    print(f"Engine: {n_params:,} params ({n_params/1e6:.1f}M), d_model={d}, "
          f"experts=8, heads={n_heads}", flush=True)

    # Trainer.
    trainer = OnlineTrainer(engine, lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        trainer.optimizer, T_max=args.epochs, eta_min=1e-5,
    )

    # Training loop.
    initial_loss = None
    print(f"\nTraining {args.epochs} epochs on {len(tokens):,} tokens...", flush=True)
    print("=" * 70, flush=True)

    for epoch in range(args.epochs):
        t0 = time.perf_counter()
        m = trainer.train_on_stream_chunked(tokens, chunk_len=args.chunk_len)
        elapsed = time.perf_counter() - t0
        tps = len(tokens) / elapsed

        if initial_loss is None:
            initial_loss = m["avg_loss"]
        scheduler.step()

        ppl = math.exp(min(m["avg_loss"], 20))
        lr = trainer.optimizer.param_groups[0]["lr"]
        print(f"Epoch {epoch+1:3d}/{args.epochs}: loss={m['avg_loss']:.3f}  "
              f"ppl={ppl:.1f}  acc={m['accuracy']:.1%}  "
              f"{tps:.0f} tok/s  lr={lr:.5f}  {elapsed:.0f}s", flush=True)

        # Checkpoint + sample generation.
        if (epoch + 1) % args.checkpoint_every == 0 or epoch == args.epochs - 1:
            save_checkpoint(engine, epoch + 1, m["avg_loss"], m["accuracy"])
            sample = generate_sample(engine, tok, "def fibonacci", n=40)
            print(f"  Sample: {sample[:100]}", flush=True)

    # Final export.
    print("\n=== Final Export ===", flush=True)
    onnx_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "fractus_continuous.onnx",
    )
    export_onnx(engine, onnx_path)
    save_checkpoint(engine, args.epochs, m["avg_loss"], m["accuracy"])

    # Final generation.
    print("\n=== Final Generation ===", flush=True)
    for prompt in ["The ", "def fibonacci", "In cybersecurity,", "La France est"]:
        sample = generate_sample(engine, tok, prompt, n=60)
        print(f"  [{prompt}]: {sample[:120]}", flush=True)

    print("\nDone.", flush=True)


if __name__ == "__main__":
    main()
