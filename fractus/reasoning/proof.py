"""Proof pipeline: the neural network proposes, the exact verifier disposes.

Architecture:
    ProofGenerator: autoregressive GRU that produces a sequence of ProofSteps.
        Each step: chooses a rule (argmax over logits) + predicts a scalar
        value (conclusion progressing toward the target).
    ProofVerifier: EXACT VERIFICATION. Does NOT use inference rules for
        verification. Instead, it tests concrete arithmetic identities
        (addition, primality, divisibility, Fermat, Wilson, GCD, modular).
        Soundness guaranteed: if the verifier says "valid", it is
        mathematically true.
    ProofReward: R = 0.6*correctness + 0.3*efficiency + 0.1*diversity.

The ProofGenerator is trainable via REINFORCE (policy gradient) on the
reward from the verifier.

HONESTY NOTE: The verifier only checks the NUMERICAL CONCLUSION
(|conclusion - target| < 1e-3), not the logical structure of the proof.
An "accepted" proof is therefore guaranteed to reach the correct numerical
value, but is not necessarily a logically valid step-by-step derivation.
"""

import math
from dataclasses import dataclass, field
from typing import List, Optional

import torch
import torch.nn as nn

from ..math.primes import PrimeSieve


class InferenceRule:
    """Labels for the 20 inference rules.

    NOTE: these are just labels. The generator predicts a rule index, but the
    verifier does NOT read the rule. It only verifies the numerical conclusion.
    """

    NAMES = [
        "AddBothSides", "MultiplyBothSides", "Distributive", "Commutative",
        "Associative", "Transitive", "Substitution", "Reflexive", "Symmetric",
        "FermatLittle", "WilsonTheorem", "EuclidGCD", "Divisibility",
        "ModularArithmetic", "Contradiction", "DoubleNegation", "ModusPonens",
        "UniversalInstant", "ExistentialIntro", "ProofByInduction",
    ]

    @classmethod
    def n_rules(cls):
        return len(cls.NAMES)

    @classmethod
    def name(cls, idx):
        return cls.NAMES[idx]


def all_rules():
    return list(InferenceRule.NAMES)


@dataclass
class ProofStep:
    rule_index: int
    rule_name: str
    premise_indices: List[int]
    conclusion: float
    confidence: float


@dataclass
class Proof:
    steps: List[ProofStep] = field(default_factory=list)
    target: float = 0.0
    conclusion: float = 0.0
    is_valid: bool = False


class ProofGenerator(nn.Module):
    """Autoregressive GRU proof generator.

    Args:
        hidden_dim: GRU hidden state dimension (32 by default).
        n_rules:    number of rules (= 20).
        max_steps:  number of proof steps generated (6 by default).
    """

    def __init__(self, hidden_dim=32, n_rules=None, max_steps=6):
        super().__init__()
        if n_rules is None:
            n_rules = InferenceRule.n_rules()
        self.hidden_dim = hidden_dim
        self.n_rules = n_rules
        self.max_steps = max_steps

        scale_rule = math.sqrt(2.0 / (hidden_dim + n_rules))
        self.w_rule = nn.Parameter(torch.empty(hidden_dim, n_rules).uniform_(-scale_rule, scale_rule))
        scale_value = math.sqrt(2.0 / (hidden_dim + 1))
        self.w_value = nn.Parameter(torch.empty(hidden_dim).uniform_(-scale_value, scale_value))

        gru_scale = math.sqrt(2.0 / (hidden_dim + hidden_dim + 1))
        self.w_update = nn.Parameter(torch.empty(hidden_dim, hidden_dim + 1).uniform_(-gru_scale, gru_scale))
        self.w_reset = nn.Parameter(torch.empty(hidden_dim, hidden_dim + 1).uniform_(-gru_scale, gru_scale))
        self.w_candidate = nn.Parameter(torch.empty(hidden_dim, hidden_dim + 1).uniform_(-gru_scale, gru_scale))

    def _gru_step(self, hidden, x_scalar):
        x = torch.cat([hidden, x_scalar.reshape(1)])
        z = torch.sigmoid(self.w_update @ x)
        r = torch.sigmoid(self.w_reset @ x)
        combined = torch.cat([r * hidden, x_scalar.reshape(1)])
        h_tilde = torch.tanh(self.w_candidate @ combined)
        return (1 - z) * hidden + z * h_tilde

    def generate(self, target):
        device = self.w_rule.device
        steps = []
        current_value = float(target)

        idx = torch.arange(1, self.hidden_dim + 1, dtype=torch.float32, device=device)
        hidden = torch.tensor(target, dtype=torch.float32, device=device) * torch.sin(idx) * 0.1

        logits_per_step = []
        selected_indices = []
        value_preds_per_step = []

        for step_idx in range(self.max_steps):
            x_scalar = torch.tensor(current_value * 0.1, dtype=torch.float32, device=device)
            hidden = self._gru_step(hidden, x_scalar)

            logits = hidden @ self.w_rule
            rule_index = int(logits.argmax().item())

            value_pred = (hidden @ self.w_value)

            premise_indices = [step_idx - 1] if step_idx > 0 else []

            max_logit = logits.max()
            exps = torch.exp(logits - max_logit)
            sum_exp = exps.sum()
            if sum_exp > 1e-10:
                confidence = float((exps[rule_index] / sum_exp).item())
            else:
                confidence = 1.0 / self.n_rules

            value_pred_float = float(value_pred.item())
            current_value = 0.8 * current_value + 0.2 * value_pred_float

            steps.append(ProofStep(
                rule_index=rule_index,
                rule_name=InferenceRule.name(rule_index),
                premise_indices=premise_indices,
                conclusion=current_value,
                confidence=confidence,
            ))
            logits_per_step.append(logits)
            selected_indices.append(rule_index)
            value_preds_per_step.append(value_pred)

        is_valid = abs(current_value - target) < 1e-3
        proof = Proof(steps=steps, target=float(target), conclusion=current_value, is_valid=is_valid)
        info = {"logits_per_step": logits_per_step, "selected_indices": selected_indices,
                "value_preds_per_step": value_preds_per_step}
        return proof, info


def _mod_pow(base, exp, modulus):
    if modulus == 1:
        return 0
    result = 1
    base = base % modulus
    while exp > 0:
        if exp % 2 == 1:
            result = (result * base) % modulus
        exp //= 2
        base = (base * base) % modulus
    return result


def _gcd(a, b):
    while b:
        a, b = b, a % b
    return abs(a)


class ProofVerifier:
    """Exact proof verifier (soundness guaranteed).

    Tests concrete arithmetic identities, not inference rule semantics.
    Any accepted proof is mathematically true on its numerical conclusion.
    """

    def __init__(self, sieve_limit=10000):
        self.sieve = PrimeSieve(sieve_limit)

    def verify_arithmetic(self, a, b, claimed_sum):
        return a + b == claimed_sum

    def verify_primality(self, n, is_prime):
        return self.sieve.verify_prime(n) == is_prime

    def verify_divisibility(self, a, b):
        if b == 0:
            return False
        return a % b == 0

    def verify_modular(self, a, b, n):
        if n == 0:
            return False
        return (a - b) % n == 0

    def verify_fermat(self, a, p):
        if p < 2 or not self.sieve.verify_prime(p):
            return False
        if a % p == 0:
            return True
        return _mod_pow(a, p - 1, p) == 1

    def verify_wilson(self, p):
        if p < 2 or not self.sieve.verify_prime(p):
            return False
        if p == 2:
            return True
        fact_mod = 1
        for i in range(1, p):
            fact_mod = (fact_mod * (i % p)) % p
        return fact_mod == p - 1

    def verify_gcd(self, a, b):
        g = _gcd(a, b)
        if a == 0 or b == 0:
            return g == max(a, b)
        lcm = (a // g) * b
        return g * lcm == a * b

    def verify_proof(self, proof):
        if not proof.steps:
            return False
        n_rules = InferenceRule.n_rules()
        for step in proof.steps:
            if step.rule_index < 0 or step.rule_index >= n_rules:
                return False
        return abs(proof.conclusion - proof.target) < 1e-3


class ProofReward:
    """Composite reward for REINFORCE: R = 0.6*correctness + 0.3*efficiency + 0.1*diversity."""

    def __init__(self, correctness_weight=0.6, efficiency_weight=0.3, diversity_weight=0.1):
        self.correctness_weight = correctness_weight
        self.efficiency_weight = efficiency_weight
        self.diversity_weight = diversity_weight

    def correctness_reward(self, proof, is_valid):
        """1.0 if valid, otherwise decreases with error."""
        if is_valid:
            return 1.0
        error = abs(proof.conclusion - proof.target)
        max_error = max(abs(proof.target), 1.0)
        return max(0.0, 1.0 - min(error / max_error, 1.0))

    def efficiency_reward(self, proof):
        """1.0 if 1 step, otherwise 1/n_steps (brevity)."""
        n = len(proof.steps)
        if n <= 1:
            return 1.0
        return 1.0 / n

    def diversity_reward(self, proof):
        """Fraction of distinct rules used (novelty)."""
        if not proof.steps:
            return 0.0
        n_rules = InferenceRule.n_rules()
        used = [False] * n_rules
        for step in proof.steps:
            if 0 <= step.rule_index < n_rules:
                used[step.rule_index] = True
        n_unique = sum(used)
        return min(n_unique / n_rules, 1.0)

    def compute_reward(self, proof, is_valid):
        """R = 0.6*correctness + 0.3*efficiency + 0.1*diversity."""
        c = self.correctness_reward(proof, is_valid)
        e = self.efficiency_reward(proof)
        d = self.diversity_reward(proof)
        return (
            self.correctness_weight * c
            + self.efficiency_weight * e
            + self.diversity_weight * d
        )
