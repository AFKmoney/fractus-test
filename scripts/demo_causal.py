"""Demo L4: NOTEARS recovers a known synthetic DAG.

HONESTY DISCLAIMER: this demo uses a LINEAR and upper-triangular SCM (trivial
topological order). This is the ideal toy case for NOTEARS — it was designed
exactly for this setting. SHD=0 here proves that the PIPELINE runs (the modules
communicate, NOTEARS optimizes, the penalty works), NOT that NOTEARS is competent
on real data. A non-linear SCM with an unknown topological order would be
noticeably harder (future work).

Steps:
    1. Generate a 5-variable linear SCM (known DAG W_true + data X).
    2. Initialize a random trainable W_pred.
    3. Optimize W_pred to minimize:
           reconstruction loss + λ · |notears_penalty(W_pred)|
       The NOTEARS penalty forces W_pred to be acyclic.
    4. Measure the SHD between W_pred and W_true.

Criterion: SHD <= 3 on 5 variables (ideal toy case — must pass).

Run:
    python scripts/demo_causal.py
"""

import sys
import os
import torch

# Ensure the 'data' package is importable from scripts/.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fractus.causal.notears import notears_penalty
from fractus.metrics.causal import structural_hamming_distance
from data.causal.generate_scm import generate_linear_scm


def main():
    torch.manual_seed(42)

    # 1. Synthetic SCM.
    W_true, X = generate_linear_scm(n_vars=5, n_samples=500, edge_prob=0.5, seed=7)
    print("=== Synthetic SCM ===")
    print("W_true (5-variable DAG, upper-triangular):")
    print(W_true)
    print(f"Data X: {X.shape}")
    print()

    # 2. Random trainable W_pred.
    n_vars = W_true.shape[0]
    W_pred = torch.zeros(n_vars, n_vars, requires_grad=True)
    torch.nn.init.normal_(W_pred, std=0.1)

    h_init = notears_penalty(W_pred).item()
    print(f"h(W_pred) initial = {h_init:.4f} (should be ~0 because W is small)")

    # 3. Optimization: reconstruction + λ·|NOTEARS|.
    opt = torch.optim.Adam([W_pred], lr=0.05)
    lam = 1.0
    for step in range(500):
        opt.zero_grad()
        X_pred = X @ W_pred
        recon = ((X_pred - X) ** 2).mean()
        h = notears_penalty(W_pred)
        loss = recon + lam * h.abs()
        loss.backward()
        opt.step()
        if step % 100 == 0 or step == 499:
            print(f"step {step:3d}  recon={recon.item():.4f}  h={h.item():.4f}")

    # 4. Measure SHD.
    print()
    print("=== DAG recovery ===")
    print("Learned W_pred (threshold 0.3):")
    W_pred_bin = (W_pred.detach().abs() > 0.3).float()
    print(W_pred_bin)
    print("W_true binary:")
    print((W_true.abs() > 0.3).float())

    shd = structural_hamming_distance(W_true, W_pred.detach(), threshold=0.3)
    print(f"\nSHD = {shd} (over {n_vars*n_vars} entries)")
    print(f"  0 = perfect recovery, lower is better.")
    print(f"  (Note: ideal toy case — linear + upper-triangular SCM. Does NOT prove")
    print(f"   competence on real data, only that the pipeline runs.)")
    if shd <= 3:
        print(f"\nOK: the causal pipeline runs (SHD <= 3 on the toy case).")
    else:
        print(f"\n~: SHD > 3, the pipeline has an issue even on the toy case.")


if __name__ == "__main__":
    main()
