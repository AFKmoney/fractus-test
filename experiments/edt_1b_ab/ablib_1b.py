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


def load_corpus_1b(path: str, *, n_train: int, n_holdout: int,
                   n_phase1: int, n_experts: int, seed: int = 42) -> dict:
    """Load + shuffle (fixed seed) + split. domain_split is n_experts contiguous slices.

    Returns: train, holdout, phase1, domain_split (list of n_experts tensors).
    """
    tokens = torch.load(path, weights_only=False).to(torch.int64)
    g = torch.Generator().manual_seed(seed)
    perm = torch.randperm(tokens.numel(), generator=g)
    tokens = tokens[perm]

    need = n_phase1 + n_train + n_holdout
    assert tokens.numel() >= need, f"corpus {tokens.numel()} < need {need}"
    phase1 = tokens[:n_phase1].clone()
    train = tokens[n_phase1:n_phase1 + n_train].clone()
    holdout = tokens[n_phase1 + n_train:n_phase1 + n_train + n_holdout].clone()

    assert n_phase1 % n_experts == 0, "n_phase1 must be divisible by n_experts"
    stride = n_phase1 // n_experts
    domain_split = [phase1[i * stride:(i + 1) * stride].clone() for i in range(n_experts)]
    return {"train": train, "holdout": holdout,
            "phase1": phase1, "domain_split": domain_split}


def _partial_forward_to_block_input(model: Fractus1B,
                                    chunk_ids: torch.Tensor,
                                    after_block: int) -> torch.Tensor:
    """Run embedding + blocks[0..after_block-1], return the INPUT to block `after_block`.

    after_block=0 -> just the embedding.
    after_block=l -> output of blocks[0..l-1].
    chunk_ids: (1, L) token ids. Returns (1, L, d_model). No grad.
    """
    with torch.no_grad():
        x = model.embed(chunk_ids)
        for l in range(after_block):
            x, _ = model.blocks[l](x)
    return x


def make_hidden_bank_1b(model: Fractus1B, tokens: torch.Tensor, *,
                        after_block: int, chunk_len: int = 32,
                        n_chunks: int = 1000, seed: int = 0) -> dict:
    """Build (h_t, h_{t+1}) aligned pairs from the input of block `after_block`.

    For each sampled chunk of chunk_len+1 tokens, embed+forward through blocks
    [0..after_block-1] (no_grad), then emit (chunk_len) aligned pairs.
    Returns {"h_in": (P, d_model), "h_target": (P, d_model)} where
    P = n_chunks * chunk_len.
    """
    model.eval()
    g = torch.Generator().manual_seed(seed)
    n = tokens.numel()
    assert n > chunk_len + 1
    starts = torch.randint(0, n - chunk_len - 1, (n_chunks,), generator=g)
    h_in_list, h_tgt_list = [], []
    for s in starts.tolist():
        chunk_ids = tokens[s:s + chunk_len + 1].unsqueeze(0)  # (1, chunk_len+1)
        h = _partial_forward_to_block_input(model, chunk_ids, after_block)  # (1, L, D)
        h = h.squeeze(0)  # (L, D)
        h_in_list.append(h[:-1].clone())
        h_tgt_list.append(h[1:].clone())
    return {"h_in": torch.cat(h_in_list, dim=0),
            "h_target": torch.cat(h_tgt_list, dim=0)}
