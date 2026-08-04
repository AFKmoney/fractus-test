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
                      "data", "quality_corpus.pt")

# The growth ladder. Each palier is bigger than the last.
# d_model grows ~2x, n_heads keeps d_head=64, experts grow, rank grows.
PALIERS = [
    # (d_model, n_heads, n_experts, siren_rank, expert_d_ff, tokens, lr)
    dict(d_model=128,  n_heads=2,  n_experts=4,   siren_rank=32, expert_d_ff=128,  tokens=2_000_000, lr=1e-3),
    dict(d_model=256,  n_heads=4,  n_experts=8,   siren_rank=32, expert_d_ff=256,  tokens=1_500_000, lr=5e-4),
    dict(d_model=512,  n_heads=8,  n_experts=16,  siren_rank=64, expert_d_ff=512,  tokens=1_000_000, lr=3e-4),
    dict(d_model=768,  n_heads=12, n_experts=32,  siren_rank=64, expert_d_ff=768,  tokens=500_000,   lr=2e-4),
    dict(d_model=1280, n_heads=20, n_experts=128, siren_rank=64, expert_d_ff=2048, tokens=0,         lr=1e-4),  # GPU only
]


def train_palier(engine, tokens, n_tokens, lr, palier_name, accumulation_steps=8):
    """Train one palier using the fast OnlineTrainer (chunked, head-last).

    Trains in SEGMENTS with periodic logging so we see progress.
    """
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
    chunk_len = 32  # larger chunk = better amortization of Python overhead
    trainer = OnlineTrainer(engine, lr=lr, accumulation_steps=accumulation_steps)

    # Train in segments of 50k tokens each, logging after each.
    segment_size = min(50_000, n_tokens)
    t0 = time.time()
    total_processed = 0
    all_losses = []

    for seg_start in range(0, n_tokens, segment_size):
        seg_end = min(seg_start + segment_size, n_tokens)
        seg_tokens = tokens[seg_start:seg_end]
        if len(seg_tokens) < chunk_len + 2:
            break
        result = trainer.train_on_stream_chunked(seg_tokens, chunk_len=chunk_len)
        processed = result.get("steps", 0)
        seg_loss = result["avg_loss"]
        seg_acc = result.get("accuracy", 0)
        total_processed += processed
        all_losses.append(seg_loss)

        elapsed = time.time() - t0
        rate = total_processed / max(elapsed, 1)
        ppl = math.exp(min(seg_loss, 20))
        print(f"  {total_processed:>8,}/{n_tokens:,} loss={seg_loss:.3f} "
              f"ppl={ppl:.1f} acc={seg_acc:.3f} {rate:.0f} tok/s", flush=True)

    elapsed = time.time() - t0
    avg_loss = sum(all_losses) / max(len(all_losses), 1)
    ppl = math.exp(min(avg_loss, 20))
    print(f"\n  {palier_name} DONE: loss={avg_loss:.3f} ppl={ppl:.1f} "
          f"({total_processed:,} tokens in {elapsed/60:.1f}min)", flush=True)

    return engine


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--paliers", type=str, default="0,1,2,3",
                    help="comma-separated palier indices to run (default: 0,1,2,3)")
    ap.add_argument("--tokens-per-palier", type=int, default=None,
                    help="override tokens per palier (for quick tests)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--accumulation-steps", type=int, default=8,
                    help="gradient accumulation steps (fewer optimizer steps = faster)")
    ap.add_argument("--compile", action="store_true",
                    help="enable torch.compile on tick_chunk_train (reduce-overhead mode)")
    args = ap.parse_args()

    palier_indices = [int(x) for x in args.paliers.split(",")]
    torch.manual_seed(args.seed)

    print("=== Fractus Progressive Growth ===", flush=True)
    print(f"Paliers: {palier_indices}", flush=True)
    print(f"Seed: {args.seed}", flush=True)
    print(f"Accumulation steps: {args.accumulation_steps}", flush=True)
    print(f"Compile: {args.compile}", flush=True)

    # Load corpus.
    tokens = torch.load(CORPUS, weights_only=False).to(torch.int64)
    print(f"Corpus: {len(tokens):,} tokens", flush=True)

    engine = None
    for idx in palier_indices:
        config = PALIERS[idx]
        palier_name = f"Palier {idx}"

        # Check for a checkpoint from the PREVIOUS palier.
        prev_ckpt = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "checkpoints", f"fractus_palier{idx - 1}.pt")

        if engine is None and os.path.exists(prev_ckpt):
            # Resume from previous palier checkpoint.
            print(f"\n--- Loading Palier {idx - 1} checkpoint: {prev_ckpt} ---", flush=True)
            ckpt = torch.load(prev_ckpt, weights_only=False, map_location="cpu")
            prev_config = ckpt.get("config", {})
            engine = ContinuousThoughtEngine(
                vocab_size=50257,
                d_model=prev_config.get("d_model", 128),
                n_heads=prev_config.get("n_heads", 2),
                d_head=prev_config.get("d_head", 64),
                n_levels=2, n_oscillators=8, coupling_rank=4,
                n_experts=prev_config.get("n_experts", 4),
                top_k=2,
                expert_d_ff=prev_config.get("expert_d_ff", 128),
                siren_rank=prev_config.get("siren_rank", 32))
            # Load weights, ignoring buffer size mismatches (kuramoto_phases etc).
            model_sd = ckpt["model_state"]
            own_sd = engine.state_dict()
            for key, val in model_sd.items():
                if key in own_sd and own_sd[key].shape == val.shape:
                    own_sd[key] = val
            engine.load_state_dict(own_sd)
            print(f"  Loaded: d={engine.d_model}, E={engine.moe.n_experts}, "
                  f"params={sum(p.numel() for p in engine.parameters()):,}", flush=True)

        if engine is None:
            # Palier 0: build from scratch (no checkpoint found).
            print(f"\n--- Building {palier_name} from scratch ---", flush=True)
            engine = ContinuousThoughtEngine(
                vocab_size=50257, d_model=config["d_model"],
                n_heads=config["n_heads"], d_head=64, n_levels=2,
                n_oscillators=8, coupling_rank=4,
                n_experts=config["n_experts"], top_k=2,
                expert_d_ff=config["expert_d_ff"],
                siren_rank=config["siren_rank"])
        elif engine.d_model < config["d_model"] or engine.moe.n_experts < config["n_experts"]:
            # Grow from previous palier (in-memory or just-loaded checkpoint).
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
            print(f"  Grown: d={engine.d_model}, E={engine.moe.n_experts}, "
                  f"params={sum(p.numel() for p in engine.parameters()):,}", flush=True)
        else:
            print(f"\n--- {palier_name} already at target config, training in place ---", flush=True)
            print(f"  Grown: d={engine.d_model}, E={engine.moe.n_experts}, "
                  f"params={sum(p.numel() for p in engine.parameters()):,}", flush=True)

        # torch.compile (optional — reduces Python overhead on repeated calls).
        if args.compile:
            print(f"  Compiling tick_chunk_train (reduce-overhead)...", flush=True)
            engine.tick_chunk_train = torch.compile(
                engine.tick_chunk_train, mode="reduce-overhead")

        # Train this palier.
        n_tokens = args.tokens_per_palier or config["tokens"]
        engine = train_palier(engine, tokens, n_tokens, config["lr"], palier_name,
                              accumulation_steps=args.accumulation_steps)

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
