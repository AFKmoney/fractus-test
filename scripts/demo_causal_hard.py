"""Demo L4+: NOTEARS on a non-linear SCM with an UNKNOWN topological order.

SERIOUS VALIDATION beyond the L4 toy case (linear + upper-triangular).
If NOTEARS recovers the DAG here, we have a real proof of competence.

Run:
    python scripts/demo_causal_hard.py
"""

import sys
import os
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fractus.causal.notears import notears_penalty
from fractus.metrics.causal import structural_hamming_distance
from data.causal.generate_scm_hard import generate_nonlinear_scm


def main():
    torch.manual_seed(42)

    print("=== Non-linear SCM with UNKNOWN topological order ===")
    W_true, X = generate_nonlinear_scm(n_vars=5, n_samples=2000, edge_prob=0.5, seed=11)
    n_edges = int((W_true != 0).sum())
    print(f"Variables: {W_true.shape[0]}, samples: {X.shape[0]}")
    print(f"True edges: {n_edges}")
    print("W_true (NOT upper-triangular — hidden topological order):")
    print((W_true != 0).int())

    print()
    print("=== Linear NOTEARS on non-linear data ===")
    n_vars = W_true.shape[0]
    W_pred = torch.zeros(n_vars, n_vars, requires_grad=True)
    torch.nn.init.normal_(W_pred, std=0.1)
    opt = torch.optim.Adam([W_pred], lr=0.05)
    for step in range(1000):
        opt.zero_grad()
        X_pred = X @ W_pred
        recon = ((X_pred - X) ** 2).mean()
        h = notears_penalty(W_pred)
        loss = recon + 1.0 * h.abs()
        loss.backward()
        opt.step()
        if step % 200 == 0:
            print(f"  step {step:4d}  recon={recon.item():.4f}  h={h.item():.4f}")

    shd = structural_hamming_distance(W_true, W_pred.detach(), threshold=0.3)
    n_pred = int((W_pred.detach().abs() > 0.3).sum())
    n_correct = int(((W_true.abs() > 0.3) & (W_pred.detach().abs() > 0.3)).sum())

    print()
    print(f"SHD = {shd} (over {n_vars*n_vars} entries)")
    print(f"Edges: true={n_edges}, predicted={n_pred}, correct={n_correct}")
    print("Learned W_pred:")
    print((W_pred.detach().abs() > 0.3).int())

    print()
    if shd <= 2:
        print(f"OK: NOTEARS recovers the non-linear DAG with unknown order (SHD <= 2).")
        print(f"  Validation beyond the L4 toy case: NOTEARS is competent.")
        print(f"  Note: linear NOTEARS is robust to moderate non-linearity")
        print(f"  (tanh ≈ identity for small inputs). For strong non-linearity,")
        print(f"  a non-linear NOTEARS would be needed (future work).")
    else:
        print(f"~: SHD = {shd} > 2. Linear NOTEARS struggles on this data.")
        print(f"  Consider: more samples, or non-linear NOTEARS.")


if __name__ == "__main__":
    main()
