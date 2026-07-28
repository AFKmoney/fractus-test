"""Regression test for the frozen-expert Phase 3 defect.

Before the fix, tick_chunk read the experts' detached _cached_W buffer
directly, so expert params received no gradient. This test fails on the
old code and must pass on the refactored engine.
"""
import torch
from fractus.continuous_engine import ContinuousThoughtEngine


def _build_13m():
    return ContinuousThoughtEngine(
        vocab_size=50257, d_model=128, n_heads=2, d_head=64,
        n_levels=2, n_oscillators=8, coupling_rank=4,
        n_experts=4, top_k=2, expert_d_ff=128, siren_rank=32,
    )


def test_cte_experts_receive_gradient():
    """Every expert parameter gets a finite, non-zero gradient after tick_chunk + backward."""
    eng = _build_13m()
    eng.reset_thought(batch_size=1)
    tokens = torch.randint(0, eng.vocab_size, (1, 16))
    logits = eng.tick_chunk(tokens)
    loss = logits.pow(2).mean()
    loss.backward()

    # The MoE is at eng.moe. Collect its parameters.
    moe_params = list(eng.moe.named_parameters())
    assert len(moe_params) > 0, "engine.moe has no parameters"
    for name, p in moe_params:
        assert p.requires_grad, f"{name} should requires_grad=True"
        assert p.grad is not None, f"{name} received NO gradient (frozen-expert bug)"
        assert torch.isfinite(p.grad).all(), f"{name} has non-finite grad"
        assert p.grad.abs().sum().item() > 0, f"{name} received zero gradient"


def test_cte_has_moe_attribute():
    """The CTE must use a PhaseRoutedMoE (not the old experts_w1/experts_w2)."""
    from fractus.nn.moe import PhaseRoutedMoE
    eng = _build_13m()
    assert isinstance(eng.moe, PhaseRoutedMoE)
    assert not hasattr(eng, "experts_w1"), "old ad-hoc MoE should be removed"
    assert not hasattr(eng, "experts_w2")
