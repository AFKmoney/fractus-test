"""Helper library for the EDT 1B AB-test.

Pure functions operating on Fractus1B + tensors. Mirrors the 13M ablib.py
structure but adapts to the 1B's 16-block stack: per-layer Phase 1, Phase 2a
attention pre-training, tied embedding Phase 2b, PGSU+AMP Phase 3.

The full run requires GPU; tests run on CPU at reduced config.
"""

import torch

from fractus1B.model_1b import Fractus1B


def build_engine_1b(seed: int = 42, **config) -> Fractus1B:
    """Construct a deterministic Fractus1B.

    Default = full 1B config (~1.05B params, needs ~4GB RAM + GPU for training).
    Pass reduced kwargs (e.g. n_layers=2, n_experts=4) for CPU tests.
    """
    torch.manual_seed(seed)
    cfg = dict(vocab_size=50257, d_model=1280, n_layers=16, n_heads=20, d_head=64,
               n_levels=2, n_experts=128, top_k=2, expert_d_ff=2048,
               siren_rank=64, max_seq_len=512)
    cfg.update(config)
    return Fractus1B(**cfg)
