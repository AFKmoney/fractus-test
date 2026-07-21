"""Tests of honest_perplexity: true perplexity, not a proxy."""

import math
import torch
import torch.nn as nn


class _DummyModel(nn.Module):
    """Trivial model: logits = linear embedding."""

    def __init__(self, vocab, d):
        super().__init__()
        self.emb = nn.Embedding(vocab, d)
        self.head = nn.Linear(d, vocab)

    def forward(self, ids):
        return self.head(self.emb(ids))


def test_perplexity_returns_float_ge_one():
    """ppl >= 1.0 (1.0 = perfect prediction)."""
    from fractus.metrics.perplexity import honest_perplexity
    model = _DummyModel(vocab=10, d=4)
    inp = torch.randint(0, 10, (2, 5))
    tgt = torch.randint(0, 10, (2, 5))
    ppl = honest_perplexity(model, inp, tgt)
    assert isinstance(ppl, float)
    assert ppl >= 1.0


def test_perplexity_uniform_init_close_to_vocab():
    """An untrained model (logits ~ uniform) → ppl ≈ vocab."""
    from fractus.metrics.perplexity import honest_perplexity
    # With a small random embedding init, logits ~ uniform → CE ≈ log(vocab).
    torch.manual_seed(0)
    model = _DummyModel(vocab=10, d=4)
    inp = torch.randint(0, 10, (4, 8))
    tgt = torch.randint(0, 10, (4, 8))
    ppl = honest_perplexity(model, inp, tgt)
    # ppl ≈ exp(log(10)) = 10. Wide tolerance because the init is not perfectly uniform.
    assert 3.0 < ppl < 30.0, f"expected ppl ~10, got {ppl}"


def test_perplexity_perfect_model_close_to_one():
    """A model that predicts exactly → ppl ≈ 1.0."""
    from fractus.metrics.perplexity import honest_perplexity
    model = _DummyModel(vocab=5, d=8)

    # Perfect overfit: force head so logits[arg] = large.
    inp = torch.tensor([[0, 1, 2]])
    tgt = torch.tensor([[0, 1, 2]])
    with torch.no_grad():
        # Bias so the correct token gets a huge logit.
        for i in range(3):
            model.head.weight[i, :] = 0
            model.head.weight[i, i] = 100.0
            model.head.bias[i] = 0
            model.emb.weight[i, i] = 1.0
    ppl = honest_perplexity(model, inp, tgt)
    assert ppl < 1.5, f"expected ppl ~1.0 for a perfect model, got {ppl}"


def test_perplexity_not_just_embedding_norm():
    """L6 CRITERION: honest_perplexity must do a REAL forward + cross_entropy,
    not a proxy based on the embedding norm (the original model.rs:537 falsehood)."""
    import inspect
    from fractus.metrics import perplexity as ppl_mod
    src = inspect.getsource(ppl_mod)
    assert "cross_entropy" in src, "Must use cross_entropy (not a proxy)"
    assert "model(input_ids)" in src or "model(" in src, "Must do a real forward"
