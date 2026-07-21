"""Tests of the proofs pipeline: generator, exact verifier, reward."""

import torch
import pytest


# --- InferenceRule ---

def test_inference_rules_count_20():
    """There are exactly 20 rules (original proof.rs:14-36)."""
    from fractus.reasoning.proof import InferenceRule, all_rules
    assert InferenceRule.n_rules() == 20
    assert len(all_rules()) == 20


def test_inference_rule_names_consistent():
    """Index and name are consistent."""
    from fractus.reasoning.proof import InferenceRule
    assert InferenceRule.name(0) == "AddBothSides"
    assert InferenceRule.name(19) == "ProofByInduction"


# --- ProofGenerator ---

def test_generator_produces_proof():
    """generate(target) returns a proof with max_steps steps."""
    from fractus.reasoning.proof import ProofGenerator
    gen = ProofGenerator(hidden_dim=32, max_steps=6)
    proof, info = gen.generate(target=5.0)
    assert len(proof.steps) == 6
    assert proof.target == 5.0
    # Each step has a valid rule_index.
    for step in proof.steps:
        assert 0 <= step.rule_index < 20


def test_generator_backward_every_param():
    """L5 CRITERION: backward propagates a finite AND non-zero gradient to EVERY parameter.

    The loss must touch BOTH the rule logits (w_rule) AND the predicted values
    (w_value), otherwise w_value receives no gradient. We therefore build a loss
    that combines both: REINFORCE-like.
    """
    from fractus.reasoning.proof import ProofGenerator
    gen = ProofGenerator(hidden_dim=32, max_steps=6)
    proof, info = gen.generate(target=3.0)
    # Loss = sum of logits (touches w_rule) + sum of value_preds (touches w_value).
    # Both are tensors in the autodiff graph.
    loss = sum(logits.sum() for logits in info["logits_per_step"])
    loss = loss + sum(vp for vp in info["value_preds_per_step"])
    loss.backward()

    for name, p in gen.named_parameters():
        assert p.requires_grad, f"{name} should requires_grad=True"
        assert p.grad is not None, f"{name} received no gradient"
        assert torch.isfinite(p.grad).all(), f"{name} has a non-finite gradient"
        assert p.grad.abs().sum().item() > 0, f"{name} received a zero gradient"


# --- ProofVerifier ---

def test_verify_arithmetic_correct():
    from fractus.reasoning.proof import ProofVerifier
    v = ProofVerifier()
    assert v.verify_arithmetic(2, 3, 5) is True
    assert v.verify_arithmetic(2, 3, 6) is False


def test_verify_primality():
    from fractus.reasoning.proof import ProofVerifier
    v = ProofVerifier()
    assert v.verify_primality(7, True) is True
    assert v.verify_primality(8, True) is False
    assert v.verify_primality(8, False) is True


def test_verify_fermat():
    """Small Fermat: 2^(7-1) ≡ 1 mod 7."""
    from fractus.reasoning.proof import ProofVerifier
    v = ProofVerifier()
    assert v.verify_fermat(2, 7) is True
    assert v.verify_fermat(3, 7) is True


def test_verify_wilson():
    """Wilson: 6! ≡ 6 mod 7."""
    from fractus.reasoning.proof import ProofVerifier
    v = ProofVerifier()
    assert v.verify_wilson(7) is True
    assert v.verify_wilson(4) is False  # 4 is not prime


def test_verify_gcd():
    from fractus.reasoning.proof import ProofVerifier
    v = ProofVerifier()
    assert v.verify_gcd(12, 18) is True  # gcd=6, lcm=36, 6*36=216=12*18


def test_verify_rejects_invalid_proof():
    """A proof whose conclusion is far from the target must be rejected."""
    from fractus.reasoning.proof import ProofVerifier, Proof, ProofStep
    v = ProofVerifier()
    steps = [ProofStep(rule_index=0, rule_name="AddBothSides", premise_indices=[],
                       conclusion=99.0, confidence=0.5)]
    proof = Proof(steps=steps, target=5.0, conclusion=99.0, is_valid=False)
    assert v.verify_proof(proof) is False


def test_verify_accepts_valid_proof():
    """A proof whose conclusion reaches the target must be accepted.

    Note: the 1e-3 threshold is strict (<, not <=), so |conclusion - target| must
    be STRICTLY less than 0.001. We use 0.0005 to be safe.
    """
    from fractus.reasoning.proof import ProofVerifier, Proof, ProofStep
    v = ProofVerifier()
    steps = [ProofStep(rule_index=0, rule_name="AddBothSides", premise_indices=[],
                       conclusion=5.0005, confidence=0.9)]
    proof = Proof(steps=steps, target=5.0, conclusion=5.0005, is_valid=True)
    assert v.verify_proof(proof) is True


# --- ProofReward ---

def test_reward_valid_proof_high():
    """A valid proof gets a high reward."""
    from fractus.reasoning.proof import ProofReward, Proof, ProofStep
    r = ProofReward()
    steps = [ProofStep(rule_index=0, rule_name="AddBothSides", premise_indices=[],
                       conclusion=5.0, confidence=0.9)]
    proof = Proof(steps=steps, target=5.0, conclusion=5.0, is_valid=True)
    reward = r.compute_reward(proof, is_valid=True)
    assert reward > 0.8  # 0.6 + 0.3 + 0.1·(1/20)


def test_reward_diversity_increases_with_rules():
    """More distinct rules → higher diversity reward."""
    from fractus.reasoning.proof import ProofReward, Proof, ProofStep
    r = ProofReward()
    steps1 = [ProofStep(rule_index=0, rule_name="A", premise_indices=[], conclusion=5.0, confidence=1.0)]
    steps20 = [ProofStep(rule_index=i, rule_name=str(i), premise_indices=[], conclusion=5.0, confidence=1.0)
               for i in range(20)]
    proof1 = Proof(steps=steps1, target=5.0, conclusion=5.0, is_valid=True)
    proof20 = Proof(steps=steps20, target=5.0, conclusion=5.0, is_valid=True)
    assert r.diversity_reward(proof20) > r.diversity_reward(proof1)


def test_reward_weights_sum_to_one():
    """The default weights sum to 1.0 (0.6 + 0.3 + 0.1)."""
    from fractus.reasoning.proof import ProofReward
    r = ProofReward()
    total = r.correctness_weight + r.efficiency_weight + r.diversity_weight
    assert abs(total - 1.0) < 1e-6
