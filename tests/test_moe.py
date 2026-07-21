"""Tests of PhaseRoutedMoE: von Mises gate, top-k, load-balance, backward."""

import math
import torch
import pytest


def test_moe_output_shape():
    """Output (B, L, d_model) + scalar auxiliary loss."""
    from fractus.nn.moe import PhaseRoutedMoE
    moe = PhaseRoutedMoE(d_model=16, n_experts=4, top_k=2, kappa=4.0)
    h = torch.randn(2, 8, 16)
    phases = torch.rand(2, 8, 4) * 2 * math.pi
    out, lb_loss = moe(h, phases)
    assert out.shape == (2, 8, 16)
    assert lb_loss.dim() == 0


def test_moe_is_finite():
    from fractus.nn.moe import PhaseRoutedMoE
    moe = PhaseRoutedMoE(d_model=16, n_experts=4, top_k=2, kappa=4.0)
    h = torch.randn(2, 8, 16) * 5
    phases = torch.rand(2, 8, 4) * 2 * math.pi
    out, lb_loss = moe(h, phases)
    assert torch.isfinite(out).all()
    assert torch.isfinite(lb_loss)


def test_moe_load_balance_nonneg():
    """Load-balance loss >= 0 (it is a weighted sum of squares)."""
    from fractus.nn.moe import PhaseRoutedMoE
    moe = PhaseRoutedMoE(d_model=16, n_experts=4, top_k=2, kappa=4.0)
    h = torch.randn(2, 8, 16)
    phases = torch.rand(2, 8, 4) * 2 * math.pi
    _, lb_loss = moe(h, phases)
    assert lb_loss.item() >= -1e-6


def test_moe_backward_every_param():
    """L2b CRITERION: backward propagates a finite AND non-zero gradient to EVERY parameter."""
    from fractus.nn.moe import PhaseRoutedMoE
    moe = PhaseRoutedMoE(d_model=16, n_experts=4, top_k=2, kappa=4.0)
    h = torch.randn(2, 8, 16)
    phases = torch.rand(2, 8, 4) * 2 * math.pi
    out, lb_loss = moe(h, phases)
    loss = out.pow(2).mean() + 0.1 * lb_loss
    loss.backward()

    params = list(moe.named_parameters())
    assert len(params) > 0
    for name, p in params:
        assert p.requires_grad, f"{name} should requires_grad=True"
        assert p.grad is not None, f"{name} received no gradient"
        assert torch.isfinite(p.grad).all(), f"{name} has a non-finite gradient"
        assert p.grad.abs().sum().item() > 0, f"{name} received a zero gradient"


def test_moe_top_k_at_most_n_experts():
    """top_k > n_experts must raise an error."""
    from fractus.nn.moe import PhaseRoutedMoE
    with pytest.raises(ValueError):
        PhaseRoutedMoE(d_model=16, n_experts=4, top_k=8, kappa=4.0)


def test_moe_with_uniform_phases_uses_all_experts():
    """If all phases are identical, the routing must not crash."""
    from fractus.nn.moe import PhaseRoutedMoE
    moe = PhaseRoutedMoE(d_model=16, n_experts=4, top_k=2, kappa=4.0)
    h = torch.randn(2, 8, 16)
    phases = torch.zeros(2, 8, 4)
    out, lb_loss = moe(h, phases)
    assert torch.isfinite(out).all()


def test_moe_sparse_matches_reference():
    """L8 CRITERION: the gather-first sparse dispatch must produce the SAME output
    as a hand-written dense reference (compute all E experts, gather top-k, sum).

    This proves the L8 optimization did not change the math — only the FLOPs.
    Uses E=8,K=2 to force the sparse path (n_experts > 2*top_k).
    """
    from fractus.nn.moe import PhaseRoutedMoE, _gelu
    torch.manual_seed(0)
    E, K, D, F = 8, 2, 16, 64
    moe = PhaseRoutedMoE(d_model=D, n_experts=E, top_k=K, kappa=4.0, d_ff=F)
    # Sanity: this config must take the sparse branch.
    assert E > 2 * K, "test config must trigger sparse dispatch"
    h = torch.randn(3, 7, D)
    phases = torch.rand(3, 7, 4) * 2 * math.pi

    # Sparse forward (the optimized path).
    out_sparse, lb_sparse = moe(h, phases)

    # Dense reference: compute ALL E experts by hand, then gather top-k.
    gates = moe._compute_gates(phases)  # (B, L, E)
    topk_vals, topk_idx = gates.topk(K, dim=-1)  # (B, L, K)
    topk_sum = topk_vals.sum(dim=-1, keepdim=True)
    topk_vals_norm = torch.where(
        topk_sum > 1e-10, topk_vals / topk_sum,
        torch.full_like(topk_vals, 1.0 / K),
    )
    # Dense expert outputs (the OLD wasteful path, reimplemented here as reference).
    B, L, _ = h.shape
    h1 = torch.einsum("bld,edf->blef", h, moe.w1) + moe.b1.view(1, 1, E, F)
    h1_act = _gelu(h1)
    all_out = torch.einsum("blef,efd->bled", h1_act, moe.w2) + moe.b2.view(1, 1, E, D)
    # Gather top-k.
    idx_exp = topk_idx.unsqueeze(-1).expand(-1, -1, -1, D)
    topk_out_ref = torch.gather(all_out, dim=2, index=idx_exp)  # (B, L, K, D)
    out_ref = (topk_vals_norm.unsqueeze(-1) * topk_out_ref).sum(dim=2)  # (B, L, D)

    assert torch.allclose(out_sparse, out_ref, atol=1e-5), \
        f"sparse dispatch output differs from dense reference: " \
        f"max diff {(out_sparse - out_ref).abs().max()}"

    # Gradients still flow (backward correctness).
    loss = out_sparse.pow(2).sum() + 0.1 * lb_sparse
    loss.backward()
    for name, p in moe.named_parameters():
        assert p.grad is not None and torch.isfinite(p.grad).all(), \
            f"{name} gradient broken"


def test_moe_dense_path_still_correct():
    """L8: small configs (n_experts <= 2*top_k) take the dense einsum path.
    Must still produce a correct, finite, correctly-shaped output."""
    from fractus.nn.moe import PhaseRoutedMoE
    torch.manual_seed(1)
    D = 16
    # E=4, K=2 → 4 > 4 is False → dense path.
    moe = PhaseRoutedMoE(d_model=D, n_experts=4, top_k=2, kappa=4.0)
    assert not (moe.n_experts > 2 * moe.top_k), "config must take dense path"
    h = torch.randn(2, 5, D)
    phases = torch.rand(2, 5, 4) * 2 * math.pi
    out, lb = moe(h, phases)
    assert out.shape == (2, 5, D)
    assert torch.isfinite(out).all() and torch.isfinite(lb)
