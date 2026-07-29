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


# ============================================================================
# Phase 1: per-expert pre-training on real hidden states (MSE on h_t -> h_{t+1}).
#
# At the 1B scale each expert is its OWN LazyStructuredSirenLinear module
# (moe.experts_w1[e], moe.experts_w2[e] live in an nn.ModuleList). Training
# expert i therefore physically cannot touch expert j — no gradient masking
# is needed (unlike the 13M redesign, where experts shared one (E,...) tensor).
# This natural isolation is pinned by test_phase1_experts_1b_natural_isolation.
# ============================================================================


def _expert_forward_1b(model: Fractus1B, block_idx: int, expert_idx: int,
                       h: torch.Tensor) -> torch.Tensor:
    """Output of one expert: gelu(w1(h)) -> w2. h: (..., d_model)."""
    moe = model.blocks[block_idx].moe
    w1 = moe.experts_w1[expert_idx]
    w2 = moe.experts_w2[expert_idx]
    h1 = w1(h)
    h1_act = torch.nn.functional.gelu(h1)
    return w2(h1_act)


def _eval_expert_mse_1b(model: Fractus1B, block_idx: int, expert_idx: int,
                        bank: dict) -> float:
    model.eval()
    with torch.no_grad():
        out = _expert_forward_1b(model, block_idx, expert_idx, bank["h_in"])
        return torch.nn.functional.mse_loss(out, bank["h_target"]).item()


def _train_one_expert_1b(model: Fractus1B, block_idx: int, expert_idx: int,
                         bank: dict, *, steps: int, lr: float,
                         batch_size: int = 64, seed: int = 0) -> list:
    """Train one expert (block_idx, expert_idx) on bank with MSE.

    Natural isolation: each expert is a separate LazyStructuredSirenLinear module,
    so the optimizer over that expert's params cannot touch other experts.
    No gradient masking needed (unlike the 13M redesign).
    """
    model.train()
    moe = model.blocks[block_idx].moe
    params = list(moe.experts_w1[expert_idx].parameters()) + \
             list(moe.experts_w2[expert_idx].parameters())
    opt = torch.optim.AdamW(params, lr=lr, weight_decay=0.0)
    g = torch.Generator().manual_seed(seed)
    n = bank["h_in"].size(0)
    losses = []
    for _ in range(steps):
        idx_b = torch.randint(0, n, (batch_size,), generator=g)
        h_in = bank["h_in"][idx_b]
        h_tgt = bank["h_target"][idx_b]
        opt.zero_grad()
        out = _expert_forward_1b(model, block_idx, expert_idx, h_in)
        loss = torch.nn.functional.mse_loss(out, h_tgt)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step()
        losses.append(loss.item())
    return losses


def phase1_experts_1b_shared(model: Fractus1B, bank_per_block: list, *,
                             steps: int = 2000, lr: float = 1e-3,
                             batch_size: int = 64, seed: int = 0,
                             log_every_block: bool = True) -> list:
    """Arm B Phase 1: all experts trained on the SAME per-layer bank.

    bank_per_block[l] is the hidden bank for the input of block l.
    Trains every (block, expert) pair. Returns a flat losses list.
    """
    losses = []
    for l in range(len(model.blocks)):
        bank = bank_per_block[l]
        for e in range(model.blocks[l].moe.n_experts):
            losses.extend(_train_one_expert_1b(model, l, e, bank, steps=steps,
                                               lr=lr, batch_size=batch_size,
                                               seed=seed + l * 1000 + e))
    return losses


def phase1_experts_1b_partitioned(model: Fractus1B,
                                  bank_per_block_per_expert: list, *,
                                  steps: int = 2000, lr: float = 1e-3,
                                  batch_size: int = 64, seed: int = 0) -> list:
    """Arm C Phase 1: expert (l,e) trained only on bank_per_block_per_expert[l][e].

    Disjoint per-expert data -> specialization.
    """
    losses = []
    for l in range(len(model.blocks)):
        for e in range(model.blocks[l].moe.n_experts):
            bank = bank_per_block_per_expert[l][e]
            losses.extend(_train_one_expert_1b(model, l, e, bank, steps=steps,
                                               lr=lr, batch_size=batch_size,
                                               seed=seed + l * 1000 + e))
    return losses


# ============================================================================
# Phase 2a: standalone attention pre-training (per layer).
#
# Each FractalBlockSparse has its own FractalLinearAttention + norm1. Phase 2a
# trains each layer's attention INDEPENDENTLY as a denoising autoencoder:
# input h (random hidden states), target h + 0.1·noise. The attention learns to
# preserve h's structure while suppressing the noise (self-supervised, as in the
# EDT doc). This mirrors FractalBlockSparse.forward: h_out = h + attn(norm1(h)).
# An optimizer scoped to attn.parameters() + norm1.parameters() means every other
# parameter in the model receives no gradient and stays bit-identical (pinned by
# test_phase2a_attention_1b_reduces_loss_and_isolates).
# ============================================================================


def phase2a_attention_1b(model: Fractus1B, *, n_steps_per_layer: int = 5000,
                         lr: float = 1e-3, seq_len: int = 16,
                         batch_size: int = 16, seed: int = 0) -> list:
    """Phase 2a: train each block's attention + norm1 standalone (denoising target).

    Target = h + 0.1·noise (self-supervised denoising, as in the EDT doc).
    Returns per-step losses of the FIRST block (representative; full loss list
    would be huge for 16 layers × 5000 steps).
    """
    model.train()
    g = torch.Generator().manual_seed(seed)
    d = model.d_model
    first_block_losses = []
    for l in range(len(model.blocks)):
        attn = model.blocks[l].attn
        norm = model.blocks[l].norm1
        params = list(attn.parameters()) + list(norm.parameters())
        opt = torch.optim.AdamW(params, lr=lr, weight_decay=0.0)
        for step in range(n_steps_per_layer):
            h = torch.randn(batch_size, seq_len, d, generator=g)
            target = h + 0.1 * torch.randn_like(h)
            opt.zero_grad()
            h_normed = norm(h)
            attn_out = attn(h_normed)
            h_out = h + attn_out
            loss = torch.nn.functional.mse_loss(h_out, target)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            opt.step()
            if l == 0:
                first_block_losses.append(loss.item())
    return first_block_losses


# ============================================================================
# Phase 2b: tied embedding + lm_head next-token training.
#
# At 1B the lm_head is TIED to the token embedding: lm_head.weight IS
# embed.tok_embed.weight (same tensor object — Fractus1B.__init__ line 260).
# So an optimizer over embed.tok_embed.parameters() trains BOTH the embedding
# AND the head simultaneously — there is only one parameter to optimize. This
# differs from the 13M CTE where the head was a separate untied Linear.
# Phase 2b trains ONLY tok_embed (and its tied head); pos_embed and embed.norm
# stay frozen. All params are restored to requires_grad=True at the end so
# Phase 3 can train everything.
# ============================================================================


def _eval_ce_1b(model: Fractus1B, tokens: torch.Tensor,
                seq_len: int = 16) -> float:
    """Mean per-token CE of embed -> lm_head (no blocks). lm_head is tied to tok_embed."""
    model.eval()
    total, n = 0.0, 0
    with torch.no_grad():
        for s in range(0, tokens.numel() - seq_len - 1, seq_len):
            chunk = tokens[s:s + seq_len].unsqueeze(0)
            tgt = tokens[s + 1:s + 1 + seq_len].reshape(-1)
            h = model.embed(chunk)
            logits = model.lm_head(h).reshape(-1, model.vocab_size)
            total += torch.nn.functional.cross_entropy(logits, tgt, reduction="sum").item()
            n += tgt.numel()
    return total / max(n, 1)


def phase2b_embedding_1b(model: Fractus1B, tokens: torch.Tensor, *,
                         steps: int = 2000, lr: float = 1e-3,
                         seq_len: int = 16, seed: int = 0) -> list:
    """Phase 2b: train embed.tok_embed on next-token CE (lm_head is tied -> trains too)."""
    model.train()
    for p in model.parameters():
        p.requires_grad_(False)
    for p in model.embed.tok_embed.parameters():
        p.requires_grad_(True)  # lm_head.weight IS this tensor (tied).

    opt = torch.optim.AdamW(model.embed.tok_embed.parameters(), lr=lr, weight_decay=0.0)
    g = torch.Generator().manual_seed(seed)
    n = tokens.numel()
    losses = []
    for _ in range(steps):
        s = torch.randint(0, n - seq_len - 1, (1,), generator=g).item()
        chunk = tokens[s:s + seq_len].unsqueeze(0)
        tgt = tokens[s + 1:s + 1 + seq_len].reshape(-1)
        opt.zero_grad()
        h = model.embed(chunk)
        logits = model.lm_head(h).reshape(-1, model.vocab_size)
        loss = torch.nn.functional.cross_entropy(logits, tgt)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.embed.tok_embed.parameters(), 1.0)
        opt.step()
        losses.append(loss.item())
    for p in model.parameters():
        p.requires_grad_(True)
    return losses
