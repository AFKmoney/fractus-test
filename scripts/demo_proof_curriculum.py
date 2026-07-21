"""Demo L5+: ProofGenerator learns via curriculum + reward shaping + baseline.

L5 VERDICT CORRECTION: pure REINFORCE did not learn because the task was
too hard right from the start (targets ±5, error 1.7, reward crushed to 0).

This demo uses ProofTrainer with:
    - continuous reward shaping (-log(1+err), non-zero gradient everywhere)
    - baseline subtraction (reduces REINFORCE variance)
    - curriculum (targets ±0.1 → ±5 progressively)

Honest criterion: the median error at ±5 must drop by at least 30% after
training, and/or the validity rate must increase significantly.

Run:
    python scripts/demo_proof_curriculum.py
"""

import torch
from fractus.reasoning.proof import ProofGenerator, ProofVerifier
from fractus.reasoning.proof_trainer import ProofTrainer


def main():
    torch.manual_seed(42)
    generator = ProofGenerator(hidden_dim=32, max_steps=6)
    verify = ProofVerifier()
    trainer = ProofTrainer(generator, verify, lr=1e-2)

    metrics = trainer.train(verbose=True)

    print()
    print("=" * 60)
    print("VERDICT L5+ (curriculum + reward shaping + baseline)")
    print("=" * 60)
    print(f"Median error (±5): {metrics['initial_error']:.4f} -> {metrics['final_error']:.4f}")
    reduction = (1 - metrics['final_error'] / max(metrics['initial_error'], 1e-9)) * 100
    print(f"Reduction: {reduction:.1f}%")
    print(f"Validity rate (±5): {metrics['initial_valid_rate']:.1%} -> {metrics['final_valid_rate']:.1%}")
    print(f"Stages reached: {metrics['levels_reached']}/5")

    if reduction >= 30.0 or metrics['final_valid_rate'] > metrics['initial_valid_rate'] + 0.05:
        print("\nOK: the generator LEARNS (curriculum + reward shaping + baseline).")
        print("  Correction of the L5 verdict: pure REINFORCE was not the problem,")
        print("  it was the task being too hard without a curriculum.")
    else:
        print("\n~: modest improvement. Consider: a higher lr, more stages,")
        print("  or PPO to reduce variance further.")


if __name__ == "__main__":
    main()
