"""fractus1B — TRUE 1 billion parameter Fractus model.

This is the scaled-up version of Fractus: a full 1B-parameter model using
LazyStructuredSiren (rank 64) as native weight storage. Every parameter is
real, trainable, and counts toward the 1B total.

Config "K" (default): d_model=1280, n_layers=16, n_heads=20, d_head=64,
n_experts=128, top_k=2, expert_d_ff=2048, siren_rank=64.
Measured at 0.956B trainable parameters, ~4 GB RAM.
"""

__version__ = "1.0.0"
