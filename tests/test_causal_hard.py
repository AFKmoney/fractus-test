"""Tests of the non-linear SCM + NOTEARS validation beyond the toy case."""

import torch


def test_nonlinear_scm_shape():
    """generate_nonlinear_scm returns W (n,n) and X (samples, n)."""
    from data.causal.generate_scm_hard import generate_nonlinear_scm
    W, X = generate_nonlinear_scm(n_vars=5, n_samples=100, seed=42)
    assert W.shape == (5, 5)
    assert X.shape == (100, 5)


def test_nonlinear_scm_is_dag():
    """W_true must be acyclic (it is a DAG by construction)."""
    from data.causal.generate_scm_hard import generate_nonlinear_scm
    from fractus.causal.notears import notears_penalty
    W, _ = generate_nonlinear_scm(n_vars=5, n_samples=100, seed=42)
    h = notears_penalty(W)
    assert abs(h.item()) < 1e-3, f"W_true should be a DAG (h≈0), got {h.item()}"


def test_nonlinear_scm_not_triangular():
    """W_true must NOT be triangular (hidden topological order)."""
    from data.causal.generate_scm_hard import generate_nonlinear_scm
    # Over several seeds, at least one W must be non-triangular.
    found_nontrian = False
    for seed in range(20):
        W, _ = generate_nonlinear_scm(n_vars=5, n_samples=10, seed=seed)
        # Strict upper-triangular: W[i,j] != 0 implies i < j.
        # Non-triangular: exists i > j with W[i,j] != 0.
        for i in range(5):
            for j in range(i):
                if W[i, j] != 0:
                    found_nontrian = True
                    break
    assert found_nontrian, "No non-triangular W found in 20 seeds"


def test_notears_recovers_nonlinear_dag():
    """L4+ CRITERION: NOTEARS recovers a non-linear DAG with unknown order (SHD <= 3).

    This is the validation beyond the L4 toy case (linear + upper-triangular).
    """
    from data.causal.generate_scm_hard import generate_nonlinear_scm
    from fractus.causal.notears import notears_penalty
    from fractus.metrics.causal import structural_hamming_distance
    torch.manual_seed(42)
    W_true, X = generate_nonlinear_scm(n_vars=5, n_samples=2000, edge_prob=0.5, seed=11)
    n_vars = W_true.shape[0]
    W_pred = torch.zeros(n_vars, n_vars, requires_grad=True)
    torch.nn.init.normal_(W_pred, std=0.1)
    opt = torch.optim.Adam([W_pred], lr=0.05)
    for _ in range(800):
        opt.zero_grad()
        X_pred = X @ W_pred
        recon = ((X_pred - X) ** 2).mean()
        h = notears_penalty(W_pred)
        loss = recon + h.abs()
        loss.backward()
        opt.step()
    shd = structural_hamming_distance(W_true, W_pred.detach(), threshold=0.3)
    assert shd <= 3, f"NOTEARS should recover the non-linear DAG (SHD<=3), got {shd}"
