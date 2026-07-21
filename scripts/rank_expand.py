"""Rank Expansion: take the trained 88M (rank 16) and expand to a larger
config (rank 64, more experts, more layers) while preserving the learned
knowledge.

This is NOT a full 88M→1B jump (d_model changes too). It's a pragmatic
middle ground: keep d_model=768 (so existing weights fit), expand rank
16→64, add experts 64→128, add layers 8→16.

The first 8 layers + first 64 experts + first 16 rank dimensions are
COPIED from the trained 88M. The rest is initialized fresh. The model
starts already "knowing" what the 88M knew, and learns the new capacity.
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
import torch.nn as nn
import numpy as np

from fractus1B.model_1b import Fractus1B


def expand_checkpoint(ckpt_path: str, output_path: str,
                       target_rank: int = 64,
                       target_experts: int = 128,
                       target_layers: int = 16):
    """Expand a trained 88M checkpoint to a larger config.

    Copies existing weights where shapes match. New parameters (higher rank,
    extra experts, extra layers) are initialized:
      - New rank columns: zeros (so they contribute nothing initially)
      - New experts: random init (standard)
      - New layers: random init (standard)

    Args:
        ckpt_path:   path to the 88M checkpoint.
        output_path: where to save the expanded checkpoint.
        target_rank: new SIREN rank (64 = 4× the original 16).
        target_experts: new expert count (128 = 2× the original 64).
        target_layers: new layer count (16 = 2× the original 8).
    """
    print(f"Loading source checkpoint: {ckpt_path}", flush=True)
    ck = torch.load(ckpt_path, weights_only=False, map_location="cpu")
    src_sd = ck["model_state"]
    src_step = ck.get("step", 0)
    src_loss = ck.get("loss", "?")
    print(f"  Source: step={src_step}, loss={src_loss}", flush=True)

    # Build the target model (larger config).
    print(f"Building target model (rank={target_rank}, experts={target_experts}, "
          f"layers={target_layers})...", flush=True)
    target = Fractus1B(
        vocab_size=50257, d_model=768,  # KEEP d_model so existing weights fit
        n_layers=target_layers, n_heads=12, d_head=64,
        n_levels=2, n_experts=target_experts, top_k=2,
        expert_d_ff=1024, siren_rank=target_rank,
        max_seq_len=16,
    )
    tgt_sd = target.state_dict()
    n_tgt = sum(v.numel() for v in tgt_sd.values())
    print(f"  Target params: {n_tgt:,} ({n_tgt/1e6:.0f}M)", flush=True)

    # Transfer strategy: for each key in target, try to copy from source.
    transferred = 0
    expanded = 0
    fresh = 0

    for key in tgt_sd:
        tgt_shape = tgt_sd[key].shape

        if key in src_sd:
            src_tensor = src_sd[key]
            src_shape = src_tensor.shape

            if src_shape == tgt_shape:
                # Exact match — direct copy.
                tgt_sd[key] = src_tensor.clone()
                transferred += 1

            elif len(src_shape) == 2 and len(tgt_shape) == 2:
                # 2D shape mismatch — could be rank expansion (U, V).
                # U: (out, rank) → copy first src_rank columns, zero-pad rest.
                # V: (in, rank)  → same.
                if (src_shape[0] == tgt_shape[0] and src_shape[1] < tgt_shape[1]):
                    new_tensor = torch.zeros(tgt_shape, dtype=tgt_tensor.dtype if False else src_tensor.dtype)
                    new_tensor[:, :src_shape[1]] = src_tensor
                    tgt_sd[key] = new_tensor
                    expanded += 1
                else:
                    fresh += 1
            else:
                fresh += 1
        else:
            fresh += 1

    # Load the mixed state dict.
    target.load_state_dict(tgt_sd, strict=False)

    print(f"\nExpansion complete:", flush=True)
    print(f"  Direct copy: {transferred}", flush=True)
    print(f"  Rank-expanded (zero-padded): {expanded}", flush=True)
    print(f"  Fresh (random init): {fresh}", flush=True)

    # Quick forward test.
    target.eval()
    with torch.no_grad():
        x = torch.randint(0, 50257, (1, 4))
        logits, aux = target(x)
        print(f"  Forward test: logits {logits.shape}, finite={torch.isfinite(logits).all()}", flush=True)

    # Save.
    torch.save({
        "model_state": target.state_dict(),
        "config": {
            "vocab_size": 50257, "d_model": 768,
            "n_layers": target_layers, "n_heads": 12, "d_head": 64,
            "n_levels": 2, "n_experts": target_experts, "top_k": 2,
            "expert_d_ff": 1024, "siren_rank": target_rank,
        },
        "source_step": src_step,
        "source_loss": src_loss,
        "expansion": f"rank 16→{target_rank}, experts 64→{target_experts}, layers 8→{target_layers}",
        "timestamp": __import__("time").strftime("%Y-%m-%d %H:%M:%S"),
    }, output_path)
    size_mb = os.path.getsize(output_path) / 1e6
    print(f"\nSaved expanded checkpoint: {output_path} ({size_mb:.0f}MB)", flush=True)
    return target


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str,
                       default="checkpoints/fractus_1b_latest.pt",
                       help="Path to the trained 88M checkpoint")
    parser.add_argument("--output", type=str,
                       default="checkpoints/fractus_expanded.pt",
                       help="Output path for the expanded checkpoint")
    parser.add_argument("--rank", type=int, default=64)
    parser.add_argument("--experts", type=int, default=128)
    parser.add_argument("--layers", type=int, default=16)
    args = parser.parse_args()
    expand_checkpoint(args.checkpoint, args.output, args.rank, args.experts, args.layers)
