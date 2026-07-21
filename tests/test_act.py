"""Tests of RecursiveReasoner (ACT) and SelfConsistencyCheck."""

import torch


def test_act_output_shape():
    """ACT preserves the shape (B, L, d)."""
    from fractus.reasoning.act import RecursiveReasoner
    reasoner = RecursiveReasoner(d_model=16, max_steps=4)

    def block_fn(h):
        return h * 0.5  # trivial block

    h = torch.randn(2, 5, 16)
    out, ponder = reasoner(h, block_fn)
    assert out.shape == (2, 5, 16)


def test_act_is_finite():
    from fractus.reasoning.act import RecursiveReasoner
    reasoner = RecursiveReasoner(d_model=16, max_steps=4)
    h = torch.randn(2, 5, 16) * 3
    out, _ = reasoner(h, lambda x: x * 0.5)
    assert torch.isfinite(out).all()


def test_act_ponder_loss_nonneg():
    """ponder_loss >= 0 (it is a mean of step counts)."""
    from fractus.reasoning.act import RecursiveReasoner
    reasoner = RecursiveReasoner(d_model=16, max_steps=4)
    out, ponder = reasoner(torch.randn(2, 5, 16), lambda x: x * 0.5)
    assert ponder >= 0.0


def test_act_backward_every_param():
    """L5 CRITERION: backward propagates a finite AND non-zero gradient to EVERY parameter."""
    from fractus.reasoning.act import RecursiveReasoner
    reasoner = RecursiveReasoner(d_model=16, max_steps=4)

    def block_fn(h):
        return h * 0.5

    h = torch.randn(2, 5, 16)
    out, _ = reasoner(h, block_fn)
    loss = out.pow(2).sum()
    loss.backward()
    for name, p in reasoner.named_parameters():
        assert p.requires_grad, f"{name} should requires_grad=True"
        assert p.grad is not None, f"{name} received no gradient"
        assert torch.isfinite(p.grad).all()
        assert p.grad.abs().sum().item() > 0, f"{name} received a zero gradient"


def test_self_consistency_candidates_count():
    """generate_candidates produces n_candidates versions."""
    from fractus.reasoning.self_consistency import SelfConsistencyCheck
    scc = SelfConsistencyCheck(n_candidates=4, noise_scale=0.1)
    h = torch.randn(2, 5, 16)
    cands = scc.generate_candidates(h)
    assert len(cands) == 4
    for c in cands:
        assert c.shape == h.shape


def test_self_consistency_select_best():
    """select_best returns a valid idx and a score in [-1, 1]."""
    from fractus.reasoning.self_consistency import SelfConsistencyCheck
    scc = SelfConsistencyCheck(n_candidates=3, noise_scale=0.05)
    ref = torch.randn(2, 4, 16)
    cands = scc.generate_candidates(ref)
    idx, score = scc.select_best(cands, ref)
    assert 0 <= idx < 3
    assert -1.0 <= score <= 1.0
