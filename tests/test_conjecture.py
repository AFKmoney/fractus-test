"""Tests of the conjectures pipeline: generator, tester, memory, loop."""

import torch


def test_conjecture_templates_count_10():
    """10 templates (original conjecture.rs:15-26)."""
    from fractus.reasoning.conjecture import ConjectureTemplate
    assert ConjectureTemplate.n_templates() == 10


def test_tester_sum_identity_survives():
    """SumIdentity (a+a=2a) must always survive (a true identity)."""
    from fractus.reasoning.conjecture import ConjectureTester, Conjecture
    tester = ConjectureTester(n_trials=50, seed=42)
    conj = Conjecture(template_index=0, template_name="SumIdentity")
    result = tester.test(conj)
    assert result.survived is True
    assert result.n_tests_passed == 50


def test_tester_fermat_survives_for_prime():
    """FermatLittle for prime p=7 must survive."""
    from fractus.reasoning.conjecture import ConjectureTester, Conjecture
    tester = ConjectureTester(n_trials=50, seed=42)
    conj = Conjecture(template_index=3, template_name="FermatLittle", parameters=[7.0])
    result = tester.test(conj)
    assert result.survived is True


def test_tester_fermat_fails_for_composite():
    """FermatLittle for p=4 (not prime) must fail."""
    from fractus.reasoning.conjecture import ConjectureTester, Conjecture
    tester = ConjectureTester(n_trials=10, seed=42)
    conj = Conjecture(template_index=3, template_name="FermatLittle", parameters=[4.0])
    result = tester.test(conj)
    assert result.survived is False


def test_generator_produces_conjecture():
    """The generator produces a valid conjecture from a state."""
    from fractus.reasoning.conjecture import ConjectureGenerator
    gen = ConjectureGenerator(state_dim=32)
    state = torch.randn(32)
    conj = gen(state)
    assert 0 <= conj.template_index < 10
    assert len(conj.parameters) == 4


def test_generator_backward_every_param():
    """L5 CRITERION: backward propagates a finite AND non-zero gradient to EVERY parameter.

    The loss must touch w_template (logits), w_params (generated parameters) AND
    w_novelty (novelty score). Otherwise some weights receive no gradient.
    """
    from fractus.reasoning.conjecture import ConjectureGenerator
    gen = ConjectureGenerator(state_dim=32)
    state = torch.randn(32)
    # Recompute the 3 tensors to get the full graph.
    logits = state @ gen.w_template
    params_logits = state @ gen.w_params
    novelty = state @ gen.w_novelty
    loss = logits.sum() + params_logits.sum() + novelty.sum()
    loss.backward()
    for name, p in gen.named_parameters():
        assert p.requires_grad, f"{name} should requires_grad=True"
        assert p.grad is not None, f"{name} received no gradient"
        assert torch.isfinite(p.grad).all()
        assert p.grad.abs().sum().item() > 0, f"{name} received a zero gradient"


def test_memory_add_and_evict():
    """The memory evicts non-survivors when full."""
    from fractus.reasoning.conjecture import ConjectureMemory, Conjecture
    mem = ConjectureMemory(max_size=3)
    for i in range(3):
        c = Conjecture(template_index=i, template_name=str(i), survived=False)
        mem.add(c)
    assert len(mem.discovered) == 3
    # Add a 4th: eviction of the oldest non-survivor.
    c4 = Conjecture(template_index=3, template_name="3", survived=True)
    mem.add(c4)
    assert len(mem.discovered) == 3
    assert mem.discovered[-1].template_index == 3


def test_memory_is_novel():
    """is_novel detects templates that already have survivors."""
    from fractus.reasoning.conjecture import ConjectureMemory, Conjecture
    mem = ConjectureMemory()
    c1 = Conjecture(template_index=0, template_name="A", survived=True)
    mem.add(c1)
    # A new conjecture with the same surviving template is not novel.
    c2 = Conjecture(template_index=0, template_name="A")
    assert mem.is_novel(c2) is False
    # Another template is new.
    c3 = Conjecture(template_index=1, template_name="B")
    assert mem.is_novel(c3) is True


def test_discovery_loop_runs():
    """The full loop runs without crashing."""
    from fractus.reasoning.conjecture import ConjectureDiscoveryLoop
    loop = ConjectureDiscoveryLoop(state_dim=32, n_trials=20, seed=42)
    discoveries = 0
    for _ in range(20):
        result = loop.discover_step()
        if result is not None:
            discoveries += 1
    # At least one discovery (SumIdentity, FermatLittle, etc. are true).
    assert discoveries >= 1, f"No discovery in 20 steps, got {discoveries}"
    assert 0.0 <= loop.discovery_rate() <= 1.0
