#!/usr/bin/env python
"""Robustness check for Hypothesis B diagnostic — multiple seeds + blocks.

The single-seed single-block diagnostic showed corr(MSE_phase1, PPL) = -0.39.
Before concluding Hypothesis B (objectives misaligned) and redesigning Phase 1,
confirm the negative correlation holds across:
  - multiple seeds (init + bank sampling)
  - multiple blocks (block 0 vs block 1)
  - multiple experts within a block

If the negative correlation is robust, Hypothesis B is confirmed and we redesign
Phase 1's objective. If it flips or vanishes, the single result was a fluke.
"""
import os, sys, math
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
import experiments.edt_1b_ab.ablib_1b as ablib

REDUCED = dict(d_model=128, n_layers=2, n_heads=2, d_head=64, n_levels=2,
               n_experts=4, top_k=2, expert_d_ff=128, siren_rank=16, max_seq_len=64)
STEPS_GRID = [0, 30, 100, 400]


def pearson(xs, ys):
    n = len(xs)
    mx, my = sum(xs)/n, sum(ys)/n
    cov = sum((x-mx)*(y-my) for x,y in zip(xs,ys))
    vx = sum((x-mx)**2 for x in xs)
    vy = sum((y-my)**2 for y in ys)
    d = math.sqrt(vx*vy)
    return cov/d if d > 0 else 0.0


def run_one(seed, block_idx, expert_idx):
    torch.manual_seed(seed)
    tokens = torch.arange(4000, dtype=torch.int64) % 50257
    holdout = tokens[3500:3900]
    # Bank from a fresh model (untouched) so the bank reflects the architecture, not training.
    bank_model = ablib.build_engine_1b(seed=seed, **REDUCED)
    bank = ablib.make_hidden_bank_1b(bank_model, tokens[:1000], after_block=block_idx,
                                     chunk_len=16, n_chunks=40, seed=seed)
    mses, ppls = [], []
    for steps in STEPS_GRID:
        eng = ablib.build_engine_1b(seed=seed, **REDUCED)
        if steps > 0:
            ablib._train_one_expert_1b(eng, block_idx, expert_idx, bank,
                                       steps=steps, lr=1e-2, seed=seed)
        mses.append(ablib._eval_expert_mse_1b(eng, block_idx, expert_idx, bank))
        ppls.append(ablib.evaluate_ppl_1b(eng, holdout, seq_len=16))
    return pearson(mses, ppls), mses, ppls


def main():
    torch.set_num_threads(8)
    print("=== Hypothesis B robustness: corr(MSE_phase1, PPL) across seeds/blocks/experts ===\n", flush=True)
    corrs = []
    configs = []
    for seed in [42, 7, 123]:
        for block in [0, 1]:
            for expert in [0, 1]:
                c, mses, ppls = run_one(seed, block, expert)
                corrs.append(c)
                configs.append((seed, block, expert))
                flag = "MISALIGNED" if c < -0.2 else ("aligned" if c > 0.2 else "weak")
                print(f"seed={seed:3} block={block} expert={expert}: corr={c:+.3f}  [{flag}]  "
                      f"(mse {mses[0]:.2f}->{mses[-1]:.2f}, ppl {ppls[0]:.0f}->{ppls[-1]:.0f})", flush=True)
    mean_c = sum(corrs)/len(corrs)
    neg = sum(1 for c in corrs if c < -0.2)
    pos = sum(1 for c in corrs if c > 0.2)
    print(f"\nMean correlation: {mean_c:+.3f}", flush=True)
    print(f"Negative (<-0.2, MISALIGNED): {neg}/{len(corrs)}", flush=True)
    print(f"Positive (>0.2, aligned): {pos}/{len(corrs)}", flush=True)
    print("\nVERDICT:", flush=True)
    if neg > pos and mean_c < -0.2:
        print("  ROBUST negative correlation. Hypothesis B CONFIRMED:", flush=True)
        print("  Phase-1 MSE and Phase-3 CE are MISALIGNED. EDT's objective must change.", flush=True)
    elif pos > neg and mean_c > 0.2:
        print("  ROBUST positive correlation. Hypothesis B REFUTED — single-seed was a fluke.", flush=True)
    else:
        print(f"  MIXED signal ({neg} neg, {pos} pos, mean {mean_c:+.3f}). Inconclusive.", flush=True)


if __name__ == "__main__":
    main()
