#!/usr/bin/env python
"""Progressive growth training: build small → train → grow → train → ... → large.

Instead of training a large model from scratch (impossible on CPU), this script
grows the model palier by palier. Each palier inherits the previous weights via
zero-padding, then trains briefly. The model never starts from random.

Usage:
    # Default: 4 paliers on CPU, then stop (palier 5 = 1B needs GPU).
    python scripts/train_progressive.py

    # Custom: specify paliers to run.
    python scripts/train_progressive.py --paliers 0,1

    # Tiny smoke test.
    python scripts/train_progressive.py --paliers 0 --tokens-per-palier 5000
"""
import argparse, os, sys, time, math
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
import torch.nn.functional as F
from fractus.continuous_engine import ContinuousThoughtEngine
from fractus.train.online import OnlineTrainer
from fractus.grow import grow_cte, grow_summary

CORPUS = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                      "data", "communication_corpus.pt")

# The growth ladder. Each palier is bigger than the last.
# d_model grows ~2x, n_heads keeps d_head=64, experts grow, rank grows.
PALIERS = [
    # (d_model, n_heads, n_experts, siren_rank, expert_d_ff, tokens, lr)
    dict(d_model=128,  n_heads=2,  n_experts=4,   siren_rank=32, expert_d_ff=128,  tokens=500_000, lr=3e-4),
    dict(d_model=256,  n_heads=4,  n_experts=8,   siren_rank=32, expert_d_ff=256,  tokens=500_000, lr=3e-4),
    dict(d_model=512,  n_heads=8,  n_experts=16,  siren_rank=64, expert_d_ff=512,  tokens=300_000, lr=2e-4),
    dict(d_model=768,  n_heads=12, n_experts=32,  siren_rank=64, expert_d_ff=768,  tokens=200_000, lr=1e-4),
    dict(d_model=1280, n_heads=20, n_experts=128, siren_rank=64, expert_d_ff=2048, tokens=0,        lr=1e-4),  # GPU only
]


def train_palier(engine, tokens, n_tokens, lr, palier_name):
    """Train one palier using the fast OnlineTrainer (chunked, head-last)."""
    print(f"\n{'='*60}", flush=True)
    print(f"TRAINING {palier_name}", flush=True)
    print(f"  d_model={engine.d_model}, experts={engine.moe.n_experts}, "
          f"rank={engine.moe.expert_rank}, "
          f"params={sum(p.numel() for p in engine.parameters()):,}", flush=True)
    print(f"  tokens={n_tokens:,}, lr={lr}", flush=True)
    print(f"{'='*60}", flush=True)

    if n_tokens == 0:
        print("  (skipped — 0 tokens, GPU only)", flush=True)
        return engine

    torch.set_num_threads(os.cpu_count() or 6)
    trainer = OnlineTrainer(engine, lr=lr)
    chunk_len = 16

    t0 = time.time()
    total_loss = 0.0
    n_chunks = 0
    log_every = max(n_tokens // (chunk_len * 20), 1)  # ~20 log lines

    for start in range(0, min(n_tokens, len(tokens) - chunk_len - 1), chunk_len):
        chunk_tokens = tokens[start:start + n_tokens] if start == 0 else tokens
        chunk = tokens[start:start + chunk_len]

        result = trainer.train_on_stream_chunked(
            tokens[start:start + chunk_len + chunk_len], chunk_len=chunk_len)
        total_loss += result["avg_loss"]
        n_chunks += 1

        if n_chunks % log_every == 0:
            elapsed = time.time() - t0
            processed = n_chunks * chunk_len
            rate = processed / max(elapsed, 1)
            avg = total_loss / max(n_chunks, 1)
            ppl = math.exp(min(avg, 20))
            print(f"  chunk {n_chunks:>6} loss={avg:.3f} ppl={ppl:.1f} "
                  f"{rate:.0f} tok/s ({processed:,}/{n_tokens:,})", flush=True)

    elapsed = time.time() - t0
    avg_loss = total_loss / max(n_chunks, 1)
    ppl = math.exp(min(avg_loss, 20))
    print(f"\n  {palier_name} DONE: loss={avg_loss:.3f} ppl={ppl:.1f} "
          f"({n_chunks * chunk_len:,} tokens in {elapsed/60:.1f}min)", flush=True)

    return engine


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--paliers", type=str, default="0,1,2,3",
                    help="comma-separated palier indices to run (default: 0,1,2,3)")
    ap.add_argument("--tokens-per-palier", type=int, default=None,
                    help="override tokens per palier (for quick tests)")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    palier_indices = [int(x) for x in args.paliers.split(",")]
    torch.manual_seed(args.seed)

    print("=== Fractus Progressive Growth ===", flush=True)
    print(f"Paliers: {palier_indices}", flush=True)
    print(f"Seed: {args.seed}", flush=True)

    # Load corpus.
    tokens = torch.load(CORPUS, weights_only=False).to(torch.int64)
    print(f"Corpus: {len(tokens):,} tokens", flush=True)

    engine = None
    for idx in palier_indices:
        config = PALIERS[idx]
        palier_name = f"Palier {idx}"

        if engine is None:
            # Palier 0: build from scratch.
            print(f"\n--- Building {palier_name} from scratch ---", flush=True)
            engine = ContinuousThoughtEngine(
                vocab_size=50257, d_model=config["d_model"],
                n_heads=config["n_heads"], d_head=64, n_levels=2,
                n_oscillators=8, coupling_rank=4,
                n_experts=config["n_experts"], top_k=2,
                expert_d_ff=config["expert_d_ff"],
                siren_rank=config["siren_rank"])
        else:
            # Grow from previous palier.
            print(f"\n--- Growing to {palier_name} ---", flush=True)
            grow_config = dict(
                d_model=config["d_model"],
                n_heads=config["n_heads"],
                d_head=64,
                n_experts=config["n_experts"],
                siren_rank=config["siren_rank"],
                expert_d_ff=config["expert_d_ff"],
            )
            engine = grow_cte(engine, grow_config)
            for k, v in grow_summary(engine, engine).items():
                pass  # summary printed by grow itself
            print(f"  Grown: d={engine.d_model}, E={engine.moe.n_experts}, "
                  f"params={sum(p.numel() for p in engine.parameters()):,}", flush=True)

        # Train this palier.
        n_tokens = args.tokens_per_palier or config["tokens"]
        engine = train_palier(engine, tokens, n_tokens, config["lr"], palier_name)

        # Save checkpoint.
        ckpt_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "checkpoints", f"fractus_palier{idx}.pt")
        os.makedirs(os.path.dirname(ckpt_path), exist_ok=True)
        torch.save({
            "model_state": engine.state_dict(),
            "config": {
                "d_model": engine.d_model,
                "n_experts": engine.moe.n_experts,
                "siren_rank": engine.moe.expert_rank,
                "expert_d_ff": engine.moe.d_ff,
                "n_heads": engine.attn.n_heads,
                "d_head": engine.attn.d_head,
                "palier": idx,
            },
            "params": sum(p.numel() for p in engine.parameters()),
        }, ckpt_path)
        print(f"  Saved: {ckpt_path} ({os.path.getsize(ckpt_path)/1e6:.0f}MB)", flush=True)

    print(f"\n{'='*60}", flush=True)
    print(f"Progressive growth complete.", flush=True)
    print(f"Final model: d={engine.d_model}, E={engine.moe.n_experts}, "
          f"params={sum(p.numel() for p in engine.parameters()):,}", flush=True)
    if engine.d_model < 1280:
        print(f"Next: run palier {max(palier_indices)+1} to continue growing.", flush=True)
    else:
        print(f"Reached 1B target! Ready for GPU fine-tuning.", flush=True)


if __name__ == "__main__":
    main()
