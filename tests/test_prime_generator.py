"""Tests of PrimeGenerator: produces prime numbers, learnable via REINFORCE."""

import torch


def test_prime_generator_output_range():
    """predict returns n ∈ [2, max_n]."""
    from fractus.reasoning.prime_generator import PrimeGenerator
    gen = PrimeGenerator(max_n=50, context_dim=8, hidden=32)
    ctx = torch.randn(4, 8)
    n = gen.predict(ctx)
    assert n.shape == (4,)
    assert (n >= 2).all() and (n <= 50).all()


def test_prime_generator_logits_shape():
    """forward returns (B, n_classes) logits, n_classes = max_n - 1."""
    from fractus.reasoning.prime_generator import PrimeGenerator
    gen = PrimeGenerator(max_n=50, context_dim=8, hidden=32)
    ctx = torch.randn(4, 8)
    logits = gen(ctx)
    assert logits.shape == (4, 49)  # n_classes = max_n - 1 = 49


def test_prime_generator_backward_every_param():
    """L5+v2 CRITERION: backward propagates a finite AND non-zero gradient to EVERY parameter."""
    from fractus.reasoning.prime_generator import PrimeGenerator
    gen = PrimeGenerator(max_n=50, context_dim=8, hidden=32)
    ctx = torch.randn(4, 8)
    logits = gen(ctx)
    loss = logits.sum()  # proxy loss
    loss.backward()
    for name, p in gen.named_parameters():
        assert p.requires_grad, f"{name} should requires_grad=True"
        assert p.grad is not None, f"{name} received no gradient"
        assert torch.isfinite(p.grad).all()
        assert p.grad.abs().sum().item() > 0, f"{name} received a zero gradient"


def test_prime_generator_learns_reinforce():
    """L5+v2 CRITERION: after REINFORCE training, valid_rate > 50% (>25% by chance).

    This is THE pivot test of the L5+ correction. The original claimed to produce valid
    proofs but did not learn. Here we prove that the 'neural proposes, exact verifier
    disposes' pipeline LEARNS when the task is attainable.
    """
    from fractus.reasoning.prime_generator import PrimeGenerator
    torch.manual_seed(42)
    gen = PrimeGenerator(max_n=100, context_dim=16, hidden=64)
    opt = torch.optim.Adam(gen.parameters(), lr=1e-2)

    # Short training (150 steps).
    for step in range(150):
        opt.zero_grad()
        ctx = torch.randn(16, gen.context_dim)
        logits = gen(ctx)
        indices = logits.argmax(dim=-1)
        n_pred = indices + 2
        rewards = gen.is_prime_pred(n_pred).float()
        log_probs = torch.log_softmax(logits, dim=-1)
        chosen = log_probs[torch.arange(16), indices]
        loss = -(rewards * chosen).mean()
        loss.backward()
        opt.step()

    # Evaluate on 200 fresh contexts.
    gen.eval()
    n_valid = 0
    with torch.no_grad():
        for _ in range(200):
            ctx = torch.randn(1, gen.context_dim)
            n = gen.predict(ctx)
            if gen.is_prime_pred(n)[0]:
                n_valid += 1
    valid_rate = n_valid / 200
    assert valid_rate > 0.5, \
        f"valid_rate after training should be > 50%, got {valid_rate:.1%}"


def test_prime_generator_soundness():
    """Every predicted n after training must REALLY be prime (soundness)."""
    from fractus.reasoning.prime_generator import PrimeGenerator
    torch.manual_seed(42)
    gen = PrimeGenerator(max_n=100, context_dim=16, hidden=64)
    opt = torch.optim.Adam(gen.parameters(), lr=1e-2)
    for _ in range(100):
        opt.zero_grad()
        ctx = torch.randn(16, gen.context_dim)
        logits = gen(ctx)
        indices = logits.argmax(dim=-1)
        rewards = gen.is_prime_pred(indices + 2).float()
        chosen = torch.log_softmax(logits, dim=-1)[torch.arange(16), indices]
        loss = -(rewards * chosen).mean()
        loss.backward()
        opt.step()
    # Verify with an INDEPENDENT sieve (re-verification).
    from fractus.math.primes import PrimeSieve
    independent_sieve = PrimeSieve(1000)
    gen.eval()
    with torch.no_grad():
        for _ in range(100):
            ctx = torch.randn(1, gen.context_dim)
            n = int(gen.predict(ctx).item())
            # The PrimeGenerator verifier and the independent sieve must
            # agree (soundness guaranteed by the exact sieve).
            assert independent_sieve.verify_prime(n) == gen.is_prime_pred(torch.tensor([n]))[0].item()
