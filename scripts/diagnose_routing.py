#!/usr/bin/env python
"""Routing visibility diagnostic — are Phase-1-trained experts actually used in Phase 3?

The robustness check showed many configs where training an expert (MSE 0.99->0.47)
left the full-model PPL COMPLETELY unchanged (52246 -> 52246, corr=0.000). That
strongly suggests the trained expert is NEVER selected by the Kuramoto top-k router
on the eval tokens — so Phase-1 training is invisible to Phase 3.

This script measures: for each expert of block 0, what fraction of hold-out tokens
actually route to it (top-k selection)? If an expert is trained in Phase 1 but
rarely/never routed, its training is wasted.

This would explain WHY EDT underperforms: Phase 1 optimizes experts that Phase 3's
router then ignores.
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
import experiments.edt_1b_ab.ablib_1b as ablib

REDUCED = dict(d_model=128, n_layers=2, n_heads=2, d_head=64, n_levels=2,
               n_experts=4, top_k=2, expert_d_ff=128, siren_rank=16, max_seq_len=64)


def main():
    torch.set_num_threads(8)
    torch.manual_seed(42)
    print("=== Routing visibility: which experts does Phase-3 actually select? ===\n", flush=True)
    eng = ablib.build_engine_1b(seed=42, **REDUCED)
    tokens = torch.arange(4000, dtype=torch.int64) % 50257

    # For block 0, run the model forward chunk by chunk, capture the routing decisions.
    # We need to instrument the MoE forward to record topk_idx. Simplest: re-run the
    # block's routing logic on the actual hidden states the block sees during a full forward.
    seq_len = 16
    n_chunks = 50
    expert_hits = torch.zeros(eng.blocks[0].moe.n_experts, dtype=torch.long)
    total_tokens = 0

    eng.eval()
    with torch.no_grad():
        for s in range(0, n_chunks * seq_len, seq_len):
            chunk = tokens[s:s+seq_len].unsqueeze(0)
            # Reproduce block 0's routing: embed -> norm_moe -> kuramoto phases -> gates -> topk.
            h = eng.embed(chunk)  # (1, L, D)
            # Block 0 input is `h` (after_block=0). Apply block 0's routing.
            block = eng.blocks[0]
            h_attn = h + block.attn(block.norm1(h))
            phases = block.kuramoto(block.norm_kur(h_attn))  # (1, L, n_osc)
            h_moe = block.norm_moe(h_attn)
            gates = block.moe._compute_gates(phases)  # (1, L, E)
            topk_idx = gates.topk(block.moe.top_k, dim=-1).indices  # (1, L, K)
            for e in range(block.moe.n_experts):
                expert_hits[e] += (topk_idx == e).sum().item()
            total_tokens += seq_len

    print(f"Block 0 routing over {total_tokens} tokens (top_k={eng.blocks[0].moe.top_k}, "
          f"{eng.blocks[0].moe.n_experts} experts):", flush=True)
    print(f"\n{'expert':>7} | {'hits':>6} | {'%tokens':>8} | {'used?':>6}", flush=True)
    print("-" * 38, flush=True)
    for e in range(eng.blocks[0].moe.n_experts):
        pct = 100 * expert_hits[e].item() / total_tokens
        used = "YES" if expert_hits[e] > 0 else "NEVER"
        print(f"{e:>7} | {expert_hits[e].item():>6} | {pct:>7.1f}% | {used:>6}", flush=True)

    never = sum(1 for e in range(eng.blocks[0].moe.n_experts) if expert_hits[e] == 0)
    print(f"\n{never}/{eng.blocks[0].moe.n_experts} experts are NEVER routed "
          f"({100*never/eng.blocks[0].moe.n_experts:.0f}% invisible to Phase 3).", flush=True)
    print("\nIMPLICATION FOR EDT:", flush=True)
    if never > 0:
        print(f"  Phase 1 trains ALL {eng.blocks[0].moe.n_experts} experts, but Phase 3's router "
              f"ignores {never} of them.", flush=True)
        print("  Training an unrouted expert is wasted compute AND can't help PPL.", flush=True)
        print("  This is a STRUCTURAL EDT- Fractus mismatch: EDT assumes all experts are used;", flush=True)
        print("  Fractus's Kuramoto routing concentrates on a subset.", flush=True)


if __name__ == "__main__":
    main()
