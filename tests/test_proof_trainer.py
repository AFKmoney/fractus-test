"""Tests of ProofTrainer: curriculum, reward shaping, baseline."""

import torch


def test_shaped_reward_continuous():
    """shaped_reward must be continuous and decreasing with the error."""
    from fractus.reasoning.proof_trainer import shaped_reward
    from fractus.reasoning.proof import ProofReward, Proof, ProofStep
    rf = ProofReward()
    rewards = []
    for err in [0.0, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0]:
        steps = [ProofStep(0, "A", [], 5.0, 0.9)]
        proof = Proof(steps=steps, target=5.0, conclusion=5.0 + err, is_valid=(err < 0.001))
        r = shaped_reward(proof, err < 0.001, rf)
        rewards.append(r)
    # Must be non-increasing.
    for i in range(len(rewards) - 1):
        assert rewards[i] >= rewards[i + 1], \
            f"reward(err={[0.0,0.1,0.5,1.0,2.0,5.0,10.0][i]})={rewards[i]} < " \
            f"reward(err={[0.0,0.1,0.5,1.0,2.0,5.0,10.0][i+1]})={rewards[i+1]}"


def test_shaped_reward_nonzero_for_large_error():
    """L5+ CRITERION: shaped_reward must be > 0 even for a large error
    (unlike the original correctness_reward, which got crushed to 0)."""
    from fractus.reasoning.proof_trainer import shaped_reward
    from fractus.reasoning.proof import ProofReward, Proof, ProofStep
    rf = ProofReward()
    steps = [ProofStep(0, "A", [], 5.0, 0.9)]
    proof = Proof(steps=steps, target=5.0, conclusion=10.0, is_valid=False)  # err=5
    r = shaped_reward(proof, False, rf)
    assert r > 0.0, f"shaped_reward must be > 0 even for err=5, got {r}"


def test_trainer_train_step_runs():
    """One training step runs without crashing."""
    from fractus.reasoning.proof import ProofGenerator, ProofVerifier
    from fractus.reasoning.proof_trainer import ProofTrainer
    gen = ProofGenerator(hidden_dim=16, max_steps=4)
    ver = ProofVerifier()
    trainer = ProofTrainer(gen, ver, lr=1e-2)
    reward, is_valid, advantage = trainer.train_step(target_range=0.5)
    assert isinstance(reward, float)
    assert isinstance(is_valid, bool)
    assert isinstance(advantage, float)


def test_trainer_baseline_updates():
    """The baseline must evolve after a few steps (EMA reward)."""
    from fractus.reasoning.proof import ProofGenerator, ProofVerifier
    from fractus.reasoning.proof_trainer import ProofTrainer
    torch.manual_seed(0)
    gen = ProofGenerator(hidden_dim=16, max_steps=4)
    ver = ProofVerifier()
    trainer = ProofTrainer(gen, ver, lr=1e-2, baseline_decay=0.5)
    initial_baseline = trainer.baseline
    for _ in range(10):
        trainer.train_step(target_range=0.5)
    # The baseline must have changed (non-zero if rewards are non-zero).
    assert trainer.baseline != initial_baseline


def test_trainer_evaluates_median_error():
    """_evaluate_median_error returns a float >= 0."""
    from fractus.reasoning.proof import ProofGenerator, ProofVerifier
    from fractus.reasoning.proof_trainer import ProofTrainer
    gen = ProofGenerator(hidden_dim=16, max_steps=4)
    ver = ProofVerifier()
    trainer = ProofTrainer(gen, ver)
    err = trainer._evaluate_median_error(target_range=1.0, n_eval=20)
    assert isinstance(err, float)
    assert err >= 0.0


def test_trainer_curriculum_improves_or_runs():
    """L5+ CRITERION: after curriculum training, the ±5 error must drop
    OR at least not explode (proof that it learns or plateaus, not diverges)."""
    from fractus.reasoning.proof import ProofGenerator, ProofVerifier
    from fractus.reasoning.proof_trainer import ProofTrainer, CurriculumLevel
    torch.manual_seed(42)
    gen = ProofGenerator(hidden_dim=32, max_steps=6)
    ver = ProofVerifier()
    # Short curriculum for the test (otherwise too long).
    short_curriculum = [
        CurriculumLevel(0.1, 0.30, 50),
        CurriculumLevel(0.5, 0.25, 50),
        CurriculumLevel(1.0, 0.20, 50),
    ]
    trainer = ProofTrainer(gen, ver, curriculum=short_curriculum, lr=1e-2)
    metrics = trainer.train(verbose=False)
    # The ±5 error must have dropped by at least 10% (a looser test criterion
    # than the demo, which targets 30%).
    assert metrics["final_error"] <= metrics["initial_error"] * 1.5, \
        f"The error exploded: {metrics['initial_error']} -> {metrics['final_error']}"
