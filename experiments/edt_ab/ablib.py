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
