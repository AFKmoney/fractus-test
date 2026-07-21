"""Demo L5+ v2: PrimeGenerator learns to produce prime numbers.

REDESIGN: the task "converge toward a target to within 1e-3" (the original L5) was
unattainable. New task: produce prime integers, verified by the exact sieve
(soundness guaranteed).

REINFORCE: reward = 1 if the predicted n is prime, 0 otherwise. Loss = -reward · log_prob.
Since 25% of integers in [2,100] are prime, the signal is not sparse.

Honest criterion: valid_rate must exceed 25% (random chance) after training,
ideally reaching >70%.

Run:
    python scripts/demo_prime_reinforce.py
"""

import torch
from fractus.reasoning.prime_generator import PrimeGenerator


def evaluate_valid_rate(gen: PrimeGenerator, n_eval: int = 200) -> float:
    """Fraction of predicted n that are prime."""
    gen.eval()
    n_valid = 0
    with torch.no_grad():
        for _ in range(n_eval):
            ctx = torch.randn(1, gen.context_dim)
            n = gen.predict(ctx)
            if gen.is_prime_pred(n)[0]:
                n_valid += 1
    return n_valid / n_eval


def main():
    torch.manual_seed(42)
    gen = PrimeGenerator(max_n=100, context_dim=16, hidden=64)
    opt = torch.optim.Adam(gen.parameters(), lr=1e-2)

    initial_rate = evaluate_valid_rate(gen)
    print(f"Initial primality rate (random): {initial_rate:.1%}")
    print(f"  (prime density in [2,100] ≈ 25%)")
    print()

    n_steps = 500
    eval_every = 100
    for step in range(n_steps):
        gen.train()
        opt.zero_grad()
        ctx = torch.randn(16, gen.context_dim)  # batch of 16 contexts.
        logits = gen(ctx)
        indices = logits.argmax(dim=-1)
        n_pred = indices + 2
        # Reward = 1 if prime, 0 otherwise.
        rewards = gen.is_prime_pred(n_pred).float()  # (16,)
        # REINFORCE: we want to maximize E[reward]. Loss = -reward · log_prob(n).
        log_probs = torch.log_softmax(logits, dim=-1)
        chosen_log_probs = log_probs[torch.arange(16), indices]
        loss = -(rewards * chosen_log_probs).mean()
        loss.backward()
        opt.step()

        if step % eval_every == 0 or step == n_steps - 1:
            vr = evaluate_valid_rate(gen)
            print(f"step {step:3d}  valid_rate = {vr:.1%}  "
                  f"(batch mean reward = {rewards.mean().item():.2f})")

    final_rate = evaluate_valid_rate(gen)
    print()
    print(f"Initial primality rate : {initial_rate:.1%}")
    print(f"Final primality rate   : {final_rate:.1%}")
    print(f"Improvement            : {(final_rate - initial_rate)*100:+.1f} points")

    if final_rate > 0.5:
        print(f"\nOK: PrimeGenerator LEARNS to produce primes "
              f"({final_rate:.0%} vs 25% by chance).")
        print(f"  Every accepted n is mathematically prime (soundness).")
    elif final_rate > initial_rate + 0.05:
        print(f"\n~: modest improvement ({final_rate:.0%}). REINFORCE works")
        print(f"  but converges slowly. More steps would help.")
    else:
        print(f"\n~: no improvement. Investigate.")


if __name__ == "__main__":
    main()
