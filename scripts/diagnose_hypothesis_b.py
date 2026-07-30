#!/usr/bin/env python
"""Diagnostic for Hypothesis B: is Phase-1 MSE aligned with Phase-3 next-token CE?

HYPOTHESIS B (the suspected EDT flaw):
  Phase 1 trains each expert to PREDICT the next hidden state (MSE regression).
  Phase 3 uses the expert as a nonlinear brick in a stack (CE next-token).
  These two objectives may NOT be aligned: an expert that minimizes Phase-1 MSE
  perfectly may be a WORSE brick for the final CE. If so, EDT pre-training is
  actively harmful — it pushes experts into a configuration Phase 3 must undo.

TEST:
  Take N experts of block 0. Train each to a DIFFERENT number of Phase-1 steps
  (so they end at different MSE levels). After each, measure the full-model
  hold-out perplexity (all blocks, next-token CE). Plot MSE-vs-PPL.

  - If PPL tracks MSE (lower MSE → lower PPL): objectives ALIGNED → EDT helps.
  - If PPL is flat or anti-correlated: objectives MISALIGNED → Hypothesis B confirmed.

  Reduced config (CPU-feasible). The correlation sign is what matters, not absolute scale.
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
    print("=== Hypothesis B diagnostic: Phase-1 MSE vs Phase-3 PPL alignment ===", flush=True)

    # Tokens: synthetic but structured enough to show signal.
    tokens = torch.arange(4000, dtype=torch.int64) % 50257
    holdout = tokens[3500:3900]
    bank0 = ablib.make_hidden_bank_1b(
        ablib.build_engine_1b(seed=42, **REDUCED),  # fresh model just for the bank (untouched)
        tokens[:1000], after_block=0, chunk_len=16, n_chunks=40, seed=0)
    print(f"bank: {bank0['h_in'].shape[0]} pairs", flush=True)

    # Train expert (0,0) incrementally; measure MSE + full-model PPL at checkpoints.
    steps_grid = [0, 20, 50, 100, 200, 400, 800]
    print(f"\n{'steps':>6} | {'mse_p1':>8} | {'ppl_full':>10} | {'ce_train_chunk':>14}", flush=True)
    print("-" * 50, flush=True)

    eng = ablib.build_engine_1b(seed=42, **REDUCED)
    results = []
    for target_steps in steps_grid:
        # Train up to target_steps total (cumulative).
        # Easiest: rebuild expert region from a fresh model each time so checkpoints are independent.
        eng = ablib.build_engine_1b(seed=42, **REDUCED)
        if target_steps > 0:
            ablib._train_one_expert_1b(eng, 0, 0, bank0, steps=target_steps, lr=1e-2, seed=0)
        mse = ablib._eval_expert_mse_1b(eng, 0, 0, bank0)
        ppl = ablib.evaluate_ppl_1b(eng, holdout, seq_len=16)
        # Also CE on a training chunk (does the better-fit-brick help training tokens too?).
        with torch.no_grad():
            chunk = tokens[:16].unsqueeze(0)
            tgt = tokens[1:17].reshape(-1)
            logits, _ = eng(chunk)
            ce = torch.nn.functional.cross_entropy(
                logits.reshape(-1, eng.vocab_size), tgt).item()
        results.append((target_steps, mse, ppl, ce))
        print(f"{target_steps:>6} | {mse:>8.4f} | {ppl:>10.1f} | {ce:>14.4f}", flush=True)

    # Correlation analysis.
    mses = [r[1] for r in results]
    ppls = [r[2] for r in results]
    # Pearson correlation.
    n = len(mses)
    mean_m, mean_p = sum(mses)/n, sum(ppls)/n
    cov = sum((m-mean_m)*(p-mean_p) for m,p in zip(mses,ppls))
    var_m = sum((m-mean_m)**2 for m in mses)
    var_p = sum((p-mean_p)**2 for p in ppls)
    import math
    denom = math.sqrt(var_m * var_p) if var_m > 0 and var_p > 0 else 0.0
    corr = cov / denom if denom > 0 else 0.0
    print(f"\nPearson corr(MSE_phase1, PPL_full) = {corr:.3f}", flush=True)
    print("\nINTERPRETATION:", flush=True)
    if corr > 0.3:
        print("  POSITIVE correlation (lower MSE → lower PPL): objectives ALIGNED.")
        print("  Hypothesis B NOT supported — Phase-1 pre-training helps Phase-3. EDT should work.")
    elif corr < -0.3:
        print("  NEGATIVE correlation (lower MSE → HIGHER PPL): objectives MISALIGNED.")
        print("  Hypothesis B SUPPORTED — Phase-1 hurts Phase-3. EDT's objective is wrong.")
        print("  -> Consider: change Phase-1 target (e.g. denoising/reconstruction, not next-hidden prediction).")
    else:
        print("  WEAK/NO correlation: objectives unrelated at this scale.")
        print("  Inconclusive — the 1B run may show a different signal.")


if __name__ == "__main__":
    main()
