"""Tests of SurpriseGatedTrainer: the energy-proportional trainer (L8)."""

import torch
import torch.nn as nn


class _ToyModel(nn.Module):
    """Tiny model: linear (B, L, in) → (B, L, V). Returns (logits, aux=0)."""
    def __init__(self, v=6, din=4):
        super().__init__()
        self.lin = nn.Linear(din, v)
    def forward(self, x):
        return self.lin(x.float()), torch.tensor(0.0)


def test_surprise_trainer_runs_and_selects():
    """The trainer runs and, after warmup, selects < 100% of tokens (gating active)."""
    from fractus.train import SurpriseGatedTrainer
    torch.manual_seed(0)
    model = _ToyModel()
    trainer = SurpriseGatedTrainer(model, lr=1e-2, percentile=70.0,
                                   warmup_full=3, ema_decay=0.5)
    info = trainer.info()
    assert info["method"] == "surprise-gated"

    inputs = torch.randint(0, 4, (8, 6, 4))   # (B=8, L=6, din=4)
    targets = torch.randint(0, 6, (8, 6))     # (B=8, L=6)
    # Warmup steps (full gradient).
    for _ in range(3):
        m = trainer.train_step(inputs, targets, vocab_size=6)
        assert m["selectivity"] == 1.0
    # Post-warmup: gating kicks in. Selectivity must be < 1 at least sometimes.
    selectivities = []
    for _ in range(10):
        m = trainer.train_step(inputs, targets, vocab_size=6)
        selectivities.append(m["selectivity"])
        assert torch.isfinite(torch.tensor(m["ce"]))
    # At least one step must have gated (selectivity < 1).
    assert min(selectivities) < 1.0, \
        f"gating never engaged; selectivities={selectivities}"
    assert 0.0 <= min(selectivities)


def test_surprise_trainer_converges():
    """L8 CRITERION: despite the biased gradient (dropping some tokens), the
    trainer must still CONVERGE on a learnable target (loss drops)."""
    from fractus.train import SurpriseGatedTrainer
    torch.manual_seed(42)
    model = _ToyModel(v=4, din=4)
    trainer = SurpriseGatedTrainer(model, lr=5e-2, percentile=60.0,
                                   warmup_full=2, ema_decay=0.7)

    # Fixed learnable target: a linear mapping.
    W_true = torch.randn(4, 4)
    inputs = torch.randint(0, 4, (16, 5, 4))
    with torch.no_grad():
        targets = (inputs.float() @ W_true).argmax(dim=-1)  # (16, 5) in [0,4)

    losses = []
    for _ in range(40):
        m = trainer.train_step(inputs, targets, vocab_size=4)
        losses.append(m["ce"])
    # Loss must drop substantially (target is learnable).
    assert losses[-1] < losses[0] * 0.6, \
        f"surprise-gated trainer did not converge: {losses[0]:.3f} -> {losses[-1]:.3f}"


def test_surprise_threshold_adapts():
    """The threshold EMA must evolve (not stay frozen)."""
    from fractus.train import SurpriseGatedTrainer
    torch.manual_seed(0)
    model = _ToyModel()
    trainer = SurpriseGatedTrainer(model, lr=1e-2, warmup_full=1, ema_decay=0.5)
    inputs = torch.randint(0, 4, (8, 6, 4))
    targets = torch.randint(0, 6, (8, 6))
    thresholds = []
    for _ in range(15):
        m = trainer.train_step(inputs, targets, vocab_size=6)
        thresholds.append(m["threshold"])
    # Post-warmup thresholds should be > 0 and changing.
    post = thresholds[2:]
    assert all(t >= 0 for t in post)
    assert len(set(round(t, 4) for t in post)) > 1, "threshold never adapted"
