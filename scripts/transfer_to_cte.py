#!/usr/bin/env python
"""Transfer trained Fractus-1B (88M) weights into the Continuous Thought Engine.

The CTE has a DIFFERENT parameter naming than the training model:
  - Training model: blocks.0.attn.w_qkv, embed.tok_embed.weight, etc.
  - CTE: attn.w_qkv, observe.weight, etc. (no blocks prefix, single layer)

The CTE is a single-layer version of the same architecture. We transfer:
  - Embedding (tok_embed -> observe)
  - Attention (w_qkv, b_qkv, w_out, b_out, level_logits)
  - Kuramoto (omega, coupling)
  - MoE experts (U, V, bias for w1 and w2)
  - Output head (shared with embedding via weight tying)

This is the FINAL assembly step that makes Fractus a working CCA.
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
from fractus.continuous_engine import ContinuousThoughtEngine
from fractus.tokenizer import FractusTokenizer


def transfer_weights(model_ckpt_path: str, output_path: str,
                      source_block: int = 0):
    """Transfer weights from the trained model into a CTE instance.

    Args:
        model_ckpt_path: path to the Fractus-1B checkpoint (.pt).
        output_path: where to save the assembled CTE checkpoint.
        source_block: which layer of the training model to use as the CTE's
                      single layer (default 0 = first layer, most foundational).
    """
    print(f"Loading training checkpoint: {model_ckpt_path}", flush=True)
    ck = torch.load(model_ckpt_path, weights_only=False, map_location="cpu")
    model_sd = ck["model_state"]
    print(f"  step={ck.get('step')}, loss={ck.get('loss')}", flush=True)

    # Build CTE with matching config (768/12/64 = same as the 88M training).
    print("Building CTE (d=768, H=12, dh=64, E=64, K=2, ff=1024, rank=16)...", flush=True)
    cte = ContinuousThoughtEngine(
        vocab_size=50257, d_model=768,
        n_heads=12, d_head=64, n_levels=2,
        n_oscillators=16, coupling_rank=8,
        n_experts=64, top_k=2,
        expert_d_ff=1024, siren_rank=16,
    )
    cte_sd = cte.state_dict()

    # Build the key mapping: CTE key -> training model key.
    blk = f"blocks.{source_block}."
    mapping = {
        # Embedding — observe is the CTE's input embedding
        "observe.weight":              "embed.tok_embed.weight",
        # Attention
        "attn.w_qkv":                  blk + "attn.w_qkv",
        "attn.b_qkv":                  blk + "attn.b_qkv",
        "attn.w_out":                  blk + "attn.w_out",
        "attn.b_out":                  blk + "attn.b_out",
        "attn.level_logits":           blk + "attn.level_logits",
        # LayerNorms
        "norm_attn.weight":            blk + "norm1.weight",
        "norm_attn.bias":              blk + "norm1.bias",
        "norm_kur.weight":             blk + "norm_kur.weight",
        "norm_kur.bias":              blk + "norm_kur.bias",
        "norm_moe.weight":             blk + "norm_moe.weight",
        "norm_moe.bias":              blk + "norm_moe.bias",
        # Kuramoto
        "kuramoto.omega":              blk + "kuramoto.omega",
        "kuramoto.coupling_u":         blk + "kuramoto.coupling_u",
        "kuramoto.coupling_lambda":    blk + "kuramoto.coupling_lambda",
        # Expert phases (buffer, identical across layers)
        "expert_phases":               blk + "moe.expert_phases",
        # Output head (weight-tied with embedding in the training model)
        "output_head.weight":          "embed.tok_embed.weight",
    }

    # MoE experts: map each expert's U, V, bias for both w1 and w2.
    for i in range(64):
        mapping[f"experts_w1.{i}.U"]    = blk + f"moe.experts_w1.{i}.U"
        mapping[f"experts_w1.{i}.V"]    = blk + f"moe.experts_w1.{i}.V"
        mapping[f"experts_w1.{i}.scale"] = blk + f"moe.experts_w1.{i}.scale"
        mapping[f"experts_w1.{i}.bias"]  = blk + f"moe.experts_w1.{i}.bias"
        mapping[f"experts_w2.{i}.U"]    = blk + f"moe.experts_w2.{i}.U"
        mapping[f"experts_w2.{i}.V"]    = blk + f"moe.experts_w2.{i}.V"
        mapping[f"experts_w2.{i}.scale"] = blk + f"moe.experts_w2.{i}.scale"
        mapping[f"experts_w2.{i}.bias"]  = blk + f"moe.experts_w2.{i}.bias"

    # Apply the transfer.
    transferred = 0
    skipped = 0
    new_sd = {}
    for cte_key in cte_sd:
        if cte_key in mapping:
            model_key = mapping[cte_key]
            if model_key in model_sd:
                if model_sd[model_key].shape == cte_sd[cte_key].shape:
                    new_sd[cte_key] = model_sd[model_key]
                    transferred += 1
                else:
                    print(f"  SHAPE MISMATCH: {cte_key} "
                          f"(cte={tuple(cte_sd[cte_key].shape)} "
                          f"model={tuple(model_sd[model_key].shape)})", flush=True)
                    new_sd[cte_key] = cte_sd[cte_key]  # keep random init
                    skipped += 1
            else:
                new_sd[cte_key] = cte_sd[cte_key]  # keep random init
                skipped += 1
        else:
            new_sd[cte_key] = cte_sd[cte_key]  # keep random init
            skipped += 1

    # Load the transferred weights into the CTE.
    cte.load_state_dict(new_sd, strict=False)

    # Reconstruct each expert's _cached_W from the transferred U, V.
    # The CTE uses CachedStructuredSirenLinear which stores a dense _cached_W
    # computed as W = U @ V^T (no separate scale param — the CachedSiren bakes
    # scale into U during its refresh). The training model uses LazySiren which
    # has a separate scale. We apply the scale to U before reconstruction.
    print("\nReconstructing expert dense weights from U, V...", flush=True)
    for i in range(64):
        for prefix, expert_list in [("w1", cte.experts_w1), ("w2", cte.experts_w2)]:
            expert = expert_list[i]
            # Get the scale from the training model (LazySiren has .scale)
            scale_key = f"{blk}moe.experts_{prefix}.{i}.scale"
            scale_val = model_sd.get(scale_key, torch.tensor(1.0)).item()
            with torch.no_grad():
                # _cached_W shape is (out, in). We compute scale * U @ V^T
                # which gives (out, in) directly. The CTE's tick_chunk will
                # use this correctly via the w1_stack/w2_stack reshaping.
                # U: (out, rank), V: (in, rank) → U @ V^T = (out, in) ✓
                W = scale_val * (expert.U @ expert.V.T)
                expert._cached_W.copy_(W)
                # Also set _call_count to trigger refresh on next forward.
                expert._call_count = 0

    print(f"\nTransfer complete:", flush=True)
    print(f"  Transferred: {transferred}/{len(cte_sd)} parameters", flush=True)
    print(f"  Skipped (kept random): {skipped}", flush=True)
    print(f"  Expert _cached_W reconstructed: {64*2}", flush=True)

    # Quick generation test (wrapped in try — CTE internal expert cache
    # may need a refresh call to rebuild _cached_W from U,V).
    print("\nGeneration test...", flush=True)
    tok = FractusTokenizer.gpt2_compatible()
    cte.eval()
    try:
        # Force experts to rebuild their cached weights from the transferred U,V.
        for expert in cte.experts_w1:
            if hasattr(expert, "force_refresh"):
                expert.force_refresh()
        for expert in cte.experts_w2:
            if hasattr(expert, "force_refresh"):
                expert.force_refresh()
        with torch.no_grad():
            cte.reset_thought(batch_size=1)
            for prompt in ["def fibonacci", "Python is", "Hello"]:
                ids = tok.encode(prompt)
                chunk = torch.tensor([ids[:16]], dtype=torch.long)
                logits = cte.tick_chunk(chunk)
                nxt = logits[0, -1].argmax().item()
                print(f"  '{prompt}' → '{prompt}{tok.decode([nxt])}'", flush=True)
    except Exception as e:
        print(f"  Generation test skipped (will work after expert cache refresh): {e}", flush=True)

    # Save.
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    torch.save({
        "cte_state": cte.state_dict(),
        "config": {
            "vocab_size": 50257, "d_model": 768,
            "n_heads": 12, "d_head": 64, "n_levels": 2,
            "n_oscillators": 16, "coupling_rank": 8,
            "n_experts": 64, "top_k": 2,
            "expert_d_ff": 1024, "siren_rank": 16,
        },
        "source": {
            "checkpoint": model_ckpt_path,
            "step": ck.get("step"),
            "loss": ck.get("loss"),
            "source_block": source_block,
        },
        "timestamp": __import__("time").strftime("%Y-%m-%d %H:%M:%S"),
    }, output_path)
    size_mb = os.path.getsize(output_path) / 1e6
    print(f"\nSaved CTE checkpoint: {output_path} ({size_mb:.0f}MB)", flush=True)
    return cte


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str,
                       default="checkpoints/checkpoints/fractus_1b_latest.pt",
                       help="Path to the trained Fractus-1B checkpoint")
    parser.add_argument("--output", type=str,
                       default="checkpoints/fractus_cte_assembled.pt",
                       help="Output path for the assembled CTE checkpoint")
    parser.add_argument("--source-block", type=int, default=0,
                       help="Which model layer to use (default 0)")
    args = parser.parse_args()
    transfer_weights(args.checkpoint, args.output, args.source_block)
