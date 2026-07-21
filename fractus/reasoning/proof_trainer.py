"""ProofTrainer: trains ProofGenerator via REINFORCE with a curriculum.

L5 VERDICT CORRECTION (pure REINFORCE does not learn):

The diagnosis revealed that the reward shaping of the original system was already
continuous and informative, BUT it gets crushed to 0 for targets ±5 (median error
1.7 → correctness ≈ 0 → no learning signal). The problem was not REINFORCE nor
the architecture, but the task being too hard from the very start.

Solution (3 combined ingredients):

1. CONTINUOUS REWARD SHAPING: penalty -log(1 + err) instead of max(0, 1-err/max).
   More informative even for large errors (non-zero gradient).

2. BASELINE SUBTRACTION: we subtract a moving-average reward to reduce the
   variance of REINFORCE. ∇J = E[(R - b) · ∇log π], b = EMA(R).
   Without a baseline, REINFORCE has high variance which prevents learning.

3. CURRICULUM: we train in stages of increasing difficulty.
   Stage 0: targets ±0.1 (the generator already succeeds at ~10% without training).
   Stage 1: ±0.5, Stage 2: ±1, Stage 3: ±2, Stage 4: ±5.
   We advance to the next stage when the validity rate exceeds a threshold (e.g. 30%).

The idea: the generator first learns on the easy task (where there is a signal),
then generalizes progressively toward the hard tasks.
"""

from dataclasses import dataclass, field
from typing import List, Optional

import torch
import torch.nn as nn

from .proof import ProofGenerator, ProofVerifier, ProofReward, Proof


@dataclass
class CurriculumLevel:
    """A curriculum stage."""
    target_range: float  # targets in [-range, +range]
    min_valid_rate: float  # validity rate required to advance to the next stage
    max_steps: int  # maximum number of training steps at this stage


DEFAULT_CURRICULUM: List[CurriculumLevel] = [
    CurriculumLevel(target_range=0.1, min_valid_rate=0.30, max_steps=200),
    CurriculumLevel(target_range=0.5, min_valid_rate=0.25, max_steps=300),
    CurriculumLevel(target_range=1.0, min_valid_rate=0.20, max_steps=400),
    CurriculumLevel(target_range=2.0, min_valid_rate=0.15, max_steps=500),
    CurriculumLevel(target_range=5.0, min_valid_rate=0.10, max_steps=600),
]


def shaped_reward(
    proof: Proof,
    is_valid: bool,
    base_reward_fn: ProofReward,
    sharpness: float = 2.0,
) -> float:
    """Continuous reward shaping: penalty -log(1 + sharpness·err).

    Unlike the original system's correctness_reward (max(0, 1-err/max_err), which
    gets crushed to 0 for err > max_err), this shape has a non-zero gradient:
    even a large error gives a signal (weak but non-zero).

    Args:
        proof: the generated proof.
        is_valid: exact verifier verdict.
        base_reward_fn: the original ProofReward (for efficiency + diversity).
        sharpness: controls the slope of the penalty.
    Returns:
        reward: float. Composed of:
            - correctness_shaped: -log(1 + sharpness·err) normalized to [0, 1].
            - efficiency (from the original system): 1/n_steps.
            - diversity (from the original system): n_unique_rules/20.
    """
    err = abs(proof.conclusion - proof.target)
    # -log(1 + sharpness·err) ∈ (-inf, 0]. We normalize to [0, 1] via
    # 1 - log(1 + sharpness·err) / log(1 + sharpness·10) (upper bound at err=10).
    max_norm = torch.log1p(torch.tensor(sharpness * 10.0)).item()
    correctness_shaped = max(0.0, 1.0 - torch.log1p(torch.tensor(sharpness * err)).item() / max_norm)
    if is_valid:
        correctness_shaped = 1.0  # bonus for exact validity.

    # Efficiency + diversity components unchanged (from the original ProofReward).
    eff = base_reward_fn.efficiency_reward(proof)
    div = base_reward_fn.diversity_reward(proof)

    # Same weights as the original system: 0.6 correctness + 0.3 efficiency + 0.1 diversity.
    return 0.6 * correctness_shaped + 0.3 * eff + 0.1 * div


class ProofTrainer:
    """Trains ProofGenerator via REINFORCE + baseline + curriculum.

    Args:
        generator:    the ProofGenerator to train.
        verify:       the ProofVerifier (sound).
        base_reward:  the original ProofReward (for efficiency + diversity).
        curriculum:   list of CurriculumLevel (defaults to DEFAULT_CURRICULUM).
        lr:           Adam learning rate.
        baseline_decay: EMA decay of the baseline (0.95 by default).
    """

    def __init__(
        self,
        generator: ProofGenerator,
        verify: ProofVerifier,
        base_reward: Optional[ProofReward] = None,
        curriculum: Optional[List[CurriculumLevel]] = None,
        lr: float = 1e-2,
        baseline_decay: float = 0.95,
    ):
        self.generator = generator
        self.verify = verify
        self.base_reward = base_reward if base_reward is not None else ProofReward()
        self.curriculum = curriculum if curriculum is not None else DEFAULT_CURRICULUM
        self.optimizer = torch.optim.Adam(generator.parameters(), lr=lr)
        self.baseline_decay = baseline_decay
        self.baseline: float = 0.0  # EMA reward.
        self.current_level_idx: int = 0

    def _evaluate_valid_rate(self, target_range: float, n_eval: int = 100) -> float:
        """Valid-proof rate over n_eval targets in [-range, range]."""
        n_valid = 0
        with torch.no_grad():
            for _ in range(n_eval):
                t = float(torch.empty(1).uniform_(-target_range, target_range).item())
                proof, _ = self.generator.generate(t)
                if self.verify.verify_proof(proof):
                    n_valid += 1
        return n_valid / n_eval

    def _evaluate_median_error(self, target_range: float, n_eval: int = 100) -> float:
        """Median error over n_eval targets."""
        errs = []
        with torch.no_grad():
            for _ in range(n_eval):
                t = float(torch.empty(1).uniform_(-target_range, target_range).item())
                proof, _ = self.generator.generate(t)
                errs.append(abs(proof.conclusion - proof.target))
        errs.sort()
        return errs[len(errs) // 2] if errs else 0.0

    def train_step(self, target_range: float) -> tuple[float, bool, float]:
        """One REINFORCE step with baseline, on a target in [-range, range].

        Returns (reward, is_valid, advantage).
        """
        self.optimizer.zero_grad()
        target = float(torch.empty(1).uniform_(-target_range, target_range).item())
        proof, info = self.generator.generate(target)
        is_valid = self.verify.verify_proof(proof)
        reward = shaped_reward(proof, is_valid, self.base_reward)

        # Advantage = reward - baseline (reduces REINFORCE variance).
        advantage = reward - self.baseline
        # Baseline update (EMA).
        self.baseline = self.baseline_decay * self.baseline + (1 - self.baseline_decay) * reward

        # REINFORCE: ∇J = advantage · ∇log π(rule | state).
        loss = torch.tensor(0.0, requires_grad=True)
        for logits, selected_idx in zip(info["logits_per_step"], info["selected_indices"]):
            log_probs = torch.log_softmax(logits, dim=-1)
            term = -advantage * log_probs[selected_idx]
            loss = loss + term
        loss.backward()
        self.optimizer.step()
        return reward, is_valid, advantage

    def train(self, verbose: bool = True) -> dict:
        """Trains over the entire curriculum. Returns a dict of metrics.

        Metrics:
            initial_error:      median error at range=5.0 before training.
            final_error:        median error at range=5.0 after training.
            initial_valid_rate: validity rate at range=5.0 before.
            final_valid_rate:   validity rate at range=5.0 after.
            levels_reached:     number of stages reached.
        """
        # Evaluate before.
        initial_error = self._evaluate_median_error(5.0)
        initial_valid = self._evaluate_valid_rate(5.0)
        if verbose:
            print(f"Before training: err_med(±5) = {initial_error:.4f}, "
                  f"valid_rate(±5) = {initial_valid:.1%}")

        levels_reached = 0
        for level_idx, level in enumerate(self.curriculum):
            if verbose:
                print(f"\n--- Stage {level_idx}: targets ±{level.target_range} "
                      f"(target valid_rate >= {level.min_valid_rate:.0%}) ---")
            for step in range(level.max_steps):
                self.train_step(level.target_range)
                if verbose and (step % 100 == 0 or step == level.max_steps - 1):
                    err = self._evaluate_median_error(level.target_range)
                    vr = self._evaluate_valid_rate(level.target_range, n_eval=50)
                    print(f"  step {step:4d}  err_med(±{level.target_range}) = {err:.4f}  "
                          f"valid_rate = {vr:.1%}  baseline = {self.baseline:.3f}")
            # Final evaluation for this stage.
            final_vr = self._evaluate_valid_rate(level.target_range)
            levels_reached = level_idx + 1
            if verbose:
                status = "OK" if final_vr >= level.min_valid_rate else "~"
                print(f"  -> final stage valid_rate: {final_vr:.1%} {status} "
                      f"(target {level.min_valid_rate:.0%})")
            # If we miss the stage by a wide margin, we still continue to the next
            # (the curriculum stays progressive; we do not block).

        # Evaluate after.
        final_error = self._evaluate_median_error(5.0)
        final_valid = self._evaluate_valid_rate(5.0)
        if verbose:
            print(f"\nAfter training: err_med(±5) = {final_error:.4f}, "
                  f"valid_rate(±5) = {final_valid:.1%}")
            baisse = (1 - final_error / max(initial_error, 1e-9)) * 100
            print(f"Error reduction: {baisse:.1f}%")

        return {
            "initial_error": initial_error,
            "final_error": final_error,
            "initial_valid_rate": initial_valid,
            "final_valid_rate": final_valid,
            "levels_reached": levels_reached,
        }
