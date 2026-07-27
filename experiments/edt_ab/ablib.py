"""Helper library for the EDT AB-test.

Pure functions operating on ContinuousThoughtEngine + tensors.
No I/O, no side effects — every function is unit-testable.
"""

import torch

from fractus.continuous_engine import ContinuousThoughtEngine


def build_engine(seed: int = 42) -> ContinuousThoughtEngine:
    """Construct a fresh 13M ContinuousThoughtEngine with deterministic init."""
    torch.manual_seed(seed)
    return ContinuousThoughtEngine(
        vocab_size=50257, d_model=128, n_heads=2, d_head=64,
        n_levels=2, n_oscillators=8, coupling_rank=4,
        n_experts=4, top_k=2, expert_d_ff=128, siren_rank=32,
    )


def load_corpus(path: str, *, n_train: int = 400_000, n_holdout: int = 30_000,
                n_phase1: int = 60_000, seed: int = 42) -> dict:
    """Load + shuffle (fixed seed) + split the corpus.

    Returns dict with:
      train       : (n_train,) int64  — Phase-3 budget (also Arm A's whole budget)
      holdout     : (n_holdout,) int64 — never seen in training, identical across arms
      phase1      : (n_phase1,) int64 — Phase-1 budget for arms B and C
      domain_split: list of 4 int64 tensors, contiguous disjoint slices of phase1
    """
    tokens = torch.load(path, weights_only=False).to(torch.int64)
    g = torch.Generator().manual_seed(seed)
    perm = torch.randperm(tokens.numel(), generator=g)
    tokens = tokens[perm]

    # Layout: [phase1 | train | holdout]  (phase1 first so domain partition is clean)
    need = n_phase1 + n_train + n_holdout
    assert tokens.numel() >= need, f"corpus has {tokens.numel()} tokens, need {need}"
    phase1 = tokens[:n_phase1].clone()
    train = tokens[n_phase1:n_phase1 + n_train].clone()
    holdout = tokens[n_phase1 + n_train:n_phase1 + n_train + n_holdout].clone()

    assert n_phase1 % 4 == 0, "n_phase1 must be divisible by 4 (4 experts)"
    stride = n_phase1 // 4
    domain_split = [phase1[i * stride:(i + 1) * stride].clone() for i in range(4)]

    return {"train": train, "holdout": holdout,
            "phase1": phase1, "domain_split": domain_split}


def make_hidden_bank(engine: ContinuousThoughtEngine, tokens: torch.Tensor,
                     *, chunk_len: int = 32, n_chunks: int = 1000,
                     seed: int = 0) -> dict:
    """Build (h_t, h_{t+1}) pairs from engine.observe over consecutive positions.

    For each of n_chunks randomly placed chunks of length chunk_len, emit
    (chunk_len - 1) aligned pairs. Pairs never cross chunk boundaries.
    Returns {"h_in": (P, d_model), "h_target": (P, d_model)} where
    P = n_chunks * (chunk_len - 1).
    """
    engine.eval()
    g = torch.Generator().manual_seed(seed)
    n = tokens.numel()
    assert n > chunk_len, "need at least chunk_len tokens"
    starts = torch.randint(0, n - chunk_len + 1, (n_chunks,), generator=g)
    h_in_list, h_tgt_list = [], []
    with torch.no_grad():
        for s in starts.tolist():
            chunk = tokens[s:s + chunk_len]              # (chunk_len,) consecutive
            h = engine.observe(chunk)                     # (chunk_len, d_model)
            h_in_list.append(h[:-1].clone())              # (chunk_len-1, d_model)
            h_tgt_list.append(h[1:].clone())              # (chunk_len-1, d_model)
    return {"h_in": torch.cat(h_in_list, dim=0),
            "h_target": torch.cat(h_tgt_list, dim=0)}
