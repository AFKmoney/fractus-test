"""Demo L5: ProofGenerator learns via REINFORCE to converge toward a target.

"Neural proposes, exact verifier disposes" pipeline:
    1. ProofGenerator produces a proof for a random target.
    2. ProofVerifier says whether it is valid (soundness guaranteed).
    3. ProofReward computes the reward.
    4. The generator is updated via REINFORCE (policy gradient).

HONEST SCIENTIFIC POSITION (updated after diagnosis):
The validity threshold of the original (|conclusion - target| < 1e-3) is UNREACHABLE for
this architecture: a GRU with 6 steps of EMA 0.8/0.2 cannot converge to an arbitrary
target to within 1e-3. Median error without training ≈ 1.72 (over range [-5,5] of width 10).
The original never measured this (no autodiff).

We therefore measure an HONEST CONTINUOUS METRIC: the median error |conclusion -
target| over 200 targets, before/after training. Success criterion: the median
error must drop by at least 30% after 500 REINFORCE steps.

Additional honesty: the verifier imposes no logical structure.
A "valid proof" here = the GRU converged to the numeric target. Not a logical
derivation in the sense of inference rules. Faithful to the original (proof.rs:341-360).

Run:
    python scripts/demo_proof_reinforce.py
"""

import torch
from fractus.reasoning.proof import ProofGenerator, ProofVerifier, ProofReward


def evaluate_median_error(generator: ProofGenerator, n_eval: int = 200) -> float:
    """Median error |conclusion - target| over n_eval random targets."""
    errors = []
    with torch.no_grad():
        for _ in range(n_eval):
            target = float(torch.empty(1).uniform_(-5.0, 5.0).item())
            proof, _ = generator.generate(target)
            errors.append(abs(proof.conclusion - proof.target))
    errors.sort()
    return errors[len(errors) // 2]


def reinforce_update(
    generator: ProofGenerator,
    target: float,
    reward_fn: ProofReward,
    verify: ProofVerifier,
    optimizer: torch.optim.Optimizer,
) -> tuple[float, bool]:
    """One REINFORCE step. Returns (reward, is_valid)."""
    optimizer.zero_grad()
    proof, info = generator.generate(target)
    is_valid = verify.verify_proof(proof)
    reward = reward_fn.compute_reward(proof, is_valid)

    # REINFORCE: we want to maximize E[reward]. reward is a non-diff scalar.
    # We use the reward as a weight on the log-probs of the chosen rules.
    loss = torch.tensor(0.0, requires_grad=True)
    for logits, selected_idx in zip(info["logits_per_step"], info["selected_indices"]):
        log_probs = torch.log_softmax(logits, dim=-1)
        term = -reward * log_probs[selected_idx]
        loss = loss + term
    loss.backward()
    optimizer.step()
    return reward, is_valid


def main():
    torch.manual_seed(42)
    generator = ProofGenerator(hidden_dim=32, max_steps=6)
    verify = ProofVerifier()
    reward_fn = ProofReward()
    optimizer = torch.optim.Adam(generator.parameters(), lr=1e-2)

    n_steps = 500
    eval_every = 100

    initial_error = evaluate_median_error(generator)
    print(f"Initial median error: {initial_error:.4f}")
    print(f"  (target range [-5,5], width 10; median error = "
          f"{initial_error/10*100:.1f}% of the range)")
    print()

    for step in range(n_steps):
        target = float(torch.empty(1).uniform_(-5.0, 5.0).item())
        reward, _ = reinforce_update(generator, target, reward_fn, verify, optimizer)

        if step % eval_every == 0 or step == n_steps - 1:
            med_err = evaluate_median_error(generator)
            print(f"step {step:3d}  median error = {med_err:.4f}  "
                  f"(reward ~{reward:.3f})")

    final_error = evaluate_median_error(generator)
    print()
    print(f"Initial median error : {initial_error:.4f}")
    print(f"Final median error   : {final_error:.4f}")
    reduction = (1 - final_error / initial_error) * 100 if initial_error > 0 else 0
    print(f"Reduction            : {reduction:.1f}%")

    if reduction >= 30.0:
        print(f"\nOK: the generator learns to converge better "
              f"(median error divided by {initial_error/max(final_error,1e-9):.2f}).")
    else:
        print(f"\nHONEST VERDICT: pure REINFORCE does NOT learn this task.")
        print(f"  Diagnosed reason: the reward is nearly constant (~0.15-0.25).")
        print(f"  The 'correctness' component is ~0 (error >> 1e-3), and the")
        print(f"  'efficiency'/'diversity' components do not depend on convergence")
        print(f"  quality. REINFORCE with a constant reward = no signal.")
        print(f"  ")
        print(f"  This is NOT a code bug — it is a scientific diagnosis:")
        print(f"  the proof pipeline, even corrected (native autodiff), is not enough")
        print(f"  to learn this task. Future directions:")
        print(f"    - Reward shaping: continuous penalty on the error (not just binary)")
        print(f"    - PPO or A2C (lower variance than REINFORCE)")
        print(f"    - A more expressive architecture (wider GRU, more steps)")
        print(f"    - An easier task first (targets in [-1,1], threshold 0.1)")
        print(f"  ")
        print(f"  The rebuild has done its job: we now KNOW that this pipeline")
        print(f"  is not enough, whereas the original claimed success without measuring it.")


if __name__ == "__main__":
    main()
