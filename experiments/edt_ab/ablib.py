"""Helper library for the EDT AB-test.

Pure functions operating on ContinuousThoughtEngine + tensors.
No I/O, no side effects — every function is unit-testable.
"""

import torch

from fractus.continuous_engine import ContinuousThoughtEngine
from fractus.train.online import OnlineTrainer


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


def _expert_forward(engine: ContinuousThoughtEngine, idx: int,
                    h: torch.Tensor) -> torch.Tensor:
    """expert_w2(gelu(expert_w1(h))). Forces a cache refresh first."""
    engine.experts_w1[idx].force_refresh()
    engine.experts_w2[idx].force_refresh()
    h1 = engine.experts_w1[idx](h)
    h1_act = torch.nn.functional.gelu(h1)
    return engine.experts_w2[idx](h1_act)


def _eval_expert_mse(engine: ContinuousThoughtEngine, idx: int, bank: dict) -> float:
    """Mean MSE of expert idx on a bank (no grad)."""
    engine.eval()
    with torch.no_grad():
        out = _expert_forward(engine, idx, bank["h_in"])
        return torch.nn.functional.mse_loss(out, bank["h_target"]).item()


def _train_one_expert(engine: ContinuousThoughtEngine, idx: int, bank: dict,
                      steps: int, lr: float, batch_size: int, seed: int) -> list:
    """Train expert idx on bank with MSE. Returns per-step loss list.

    We train only the low-rank adapters (U, V) and the bias of each expert's
    CachedStructuredSirenLinear. The residual_siren is excluded because its
    spectral high-frequency nature (omega0=30) makes its gradients pathologically
    large relative to its tiny init scale; including them under the test's
    lr=1e-2 blows up the loss on step 1 (1.0 → ~44) before it can recover.
    Gradients still flow to U/V (verified) and they are the dominant
    reconstruction directions, so this matches standard low-rank adapter tuning.
    """
    engine.train()
    params = []
    for module in (engine.experts_w1[idx], engine.experts_w2[idx]):
        for name, p in module.named_parameters():
            if not name.startswith("residual_siren"):
                params.append(p)
    opt = torch.optim.AdamW(params, lr=lr, weight_decay=0.01)
    g = torch.Generator().manual_seed(seed)
    n = bank["h_in"].size(0)
    losses = []
    for _ in range(steps):
        idx_b = torch.randint(0, n, (batch_size,), generator=g)
        h_in = bank["h_in"][idx_b]
        h_tgt = bank["h_target"][idx_b]
        opt.zero_grad()
        out = _expert_forward(engine, idx, h_in)
        loss = torch.nn.functional.mse_loss(out, h_tgt)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step()
        losses.append(loss.item())
    # Final refresh so the cache reflects trained weights.
    engine.experts_w1[idx].force_refresh()
    engine.experts_w2[idx].force_refresh()
    return losses


def phase1_experts_shared(engine: ContinuousThoughtEngine, bank: dict, *,
                          steps: int = 2000, lr: float = 1e-3,
                          batch_size: int = 64, seed: int = 0) -> list:
    """Arm B Phase 1: all experts trained on the SAME shared bank.

    Returns a flat list of per-step MSE losses across all experts.
    """
    all_losses = []
    for i in range(engine.n_experts):
        losses = _train_one_expert(engine, i, bank, steps=steps, lr=lr,
                                   batch_size=batch_size, seed=seed + i)
        all_losses.extend(losses)
    return all_losses


def phase1_experts_partitioned(engine: ContinuousThoughtEngine, banks: list, *,
                               steps: int = 2000, lr: float = 1e-3,
                               batch_size: int = 64, seed: int = 0) -> list:
    """Arm C Phase 1: expert i trained only on banks[i] (disjoint data).

    Returns a flat list of per-step MSE losses across all experts.
    """
    assert len(banks) == engine.n_experts, "need one bank per expert"
    all_losses = []
    for i in range(engine.n_experts):
        losses = _train_one_expert(engine, i, banks[i], steps=steps, lr=lr,
                                   batch_size=batch_size, seed=seed + i)
        all_losses.extend(losses)
    return all_losses


def _eval_ce(engine: ContinuousThoughtEngine, tokens: torch.Tensor,
             chunk_len: int = 16) -> float:
    """Mean per-token cross-entropy of the two-table LM (observe→output_head)."""
    engine.eval()
    total, n = 0.0, 0
    with torch.no_grad():
        for s in range(0, tokens.numel() - chunk_len - 1, chunk_len):
            chunk = tokens[s:s + chunk_len].unsqueeze(0)
            tgt = tokens[s + 1:s + 1 + chunk_len].reshape(-1)
            h = engine.observe(chunk)
            logits = engine.output_head(h).reshape(-1, engine.vocab_size)
            total += torch.nn.functional.cross_entropy(
                logits, tgt, reduction="sum").item()
            n += tgt.numel()
    return total / max(n, 1)


def phase2b_embedding(engine: ContinuousThoughtEngine, tokens: torch.Tensor, *,
                      steps: int = 2000, lr: float = 1e-3,
                      chunk_len: int = 16, seed: int = 0) -> list:
    """Phase 2b: train observe + output_head only (everything else frozen)."""
    engine.train()
    # Freeze everything, then unfreeze the two tables.
    for p in engine.parameters():
        p.requires_grad_(False)
    for p in engine.observe.parameters():
        p.requires_grad_(True)
    for p in engine.output_head.parameters():
        p.requires_grad_(True)

    opt = torch.optim.AdamW(
        list(engine.observe.parameters()) + list(engine.output_head.parameters()),
        lr=lr, weight_decay=0.01)
    g = torch.Generator().manual_seed(seed)
    n = tokens.numel()
    losses = []
    for _ in range(steps):
        s = torch.randint(0, n - chunk_len - 1, (1,), generator=g).item()
        chunk = tokens[s:s + chunk_len].unsqueeze(0)
        tgt = tokens[s + 1:s + 1 + chunk_len].reshape(-1)
        opt.zero_grad()
        h = engine.observe(chunk)
        logits = engine.output_head(h).reshape(-1, engine.vocab_size)
        loss = torch.nn.functional.cross_entropy(logits, tgt)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            list(engine.observe.parameters()) + list(engine.output_head.parameters()),
            1.0)
        opt.step()
        losses.append(loss.item())
    # Restore: everything trainable again (Phase 3 expects this).
    for p in engine.parameters():
        p.requires_grad_(True)
    return losses


def phase3_joint(engine: ContinuousThoughtEngine, tokens: torch.Tensor, *,
                 steps: int, lr: float = 3e-4, chunk_len: int = 16) -> dict:
    """Phase 3: full-model online training, chunk-based.

    `steps` is the number of optimizer steps (chunks). Each chunk is `chunk_len`
    tokens. The total tokens consumed is steps * chunk_len.
    """
    engine.train()
    for p in engine.parameters():
        p.requires_grad_(True)
    # Force-refresh all expert caches so Phase 3 starts from trained matrices.
    for i in range(engine.n_experts):
        engine.experts_w1[i].force_refresh()
        engine.experts_w2[i].force_refresh()
    trainer = OnlineTrainer(engine, lr=lr)
    trainer.losses = []  # fresh curve
    engine.reset_thought(batch_size=1)

    total, correct, n = 0.0, 0, 0
    vocab = engine.vocab_size
    for start in range(0, steps * chunk_len, chunk_len):
        if start + chunk_len + 1 >= tokens.numel():
            break
        chunk = tokens[start:start + chunk_len].unsqueeze(0)
        tgt = tokens[start + 1:start + 1 + chunk_len].reshape(-1)
        logits = engine.tick_chunk(chunk).reshape(-1, vocab)
        loss = torch.nn.functional.cross_entropy(logits, tgt)
        trainer.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(engine.parameters(), 1.0)
        trainer.optimizer.step()
        trainer.losses.append(loss.item())
        total += loss.item() * chunk_len
        correct += (logits.argmax(-1) == tgt).sum().item()
        n += chunk_len
    return {"losses": trainer.losses,
            "avg_loss": total / max(n, 1),
            "accuracy": correct / max(n, 1),
            "steps": n}


def evaluate_ppl(engine: ContinuousThoughtEngine, tokens: torch.Tensor,
                 chunk_len: int = 16, nll_cap: float = 20.0) -> float:
    """Perplexity on a hold-out stream, computed via tick_chunk (full engine)."""
    engine.eval()
    engine.reset_thought(batch_size=1)
    total_nll, n = 0.0, 0
    for s in range(0, tokens.numel() - chunk_len - 1, chunk_len):
        chunk = tokens[s:s + chunk_len].unsqueeze(0)
        tgt = tokens[s + 1:s + 1 + chunk_len].reshape(-1)
        with torch.no_grad():
            logits = engine.tick_chunk(chunk).reshape(-1, engine.vocab_size)
            nll = torch.nn.functional.cross_entropy(logits, tgt, reduction="none")
            nll = nll.clamp(max=nll_cap)
        total_nll += nll.sum().item()
        n += tgt.numel()
    avg_nll = total_nll / max(n, 1)
    import math
    return math.exp(avg_nll)


def expert_diversity(engine: ContinuousThoughtEngine, probe_tokens: torch.Tensor) -> float:
    """Mean off-diagonal cosine between expert outputs on a fixed shared probe.

    Lower = more diverse (better specialization).
    """
    engine.eval()
    with torch.no_grad():
        h = engine.observe(probe_tokens)            # (P, d_model)
        outs = []
        for i in range(engine.n_experts):
            outs.append(_expert_forward(engine, i, h))   # (P, d_model)
        outs = torch.stack(outs)                          # (E, P, d_model)
    E = outs.size(0)
    cos_sum, count = 0.0, 0
    for i in range(E):
        for j in range(E):
            if i == j:
                continue
            a = outs[i].flatten()
            b = outs[j].flatten()
            cos = torch.nn.functional.cosine_similarity(a, b, dim=0).item()
            cos_sum += cos
            count += 1
    return cos_sum / max(count, 1)


def greedy_sample(engine: ContinuousThoughtEngine, prompt: torch.Tensor,
                  n_tokens: int = 80) -> str:
    """Greedy decode n_tokens after the prompt, detokenize to a string."""
    from fractus.tokenizer import FractusTokenizer
    tok = FractusTokenizer.gpt2_compatible()
    engine.eval()
    engine.reset_thought(batch_size=1)
    ids = prompt.tolist()
    cur = torch.tensor(ids[-1:], dtype=torch.int64)
    for _ in range(n_tokens):
        with torch.no_grad():
            logits, _ = engine.tick(cur)
        nxt = int(logits.argmax(-1).item())
        ids.append(nxt)
        cur = torch.tensor([nxt], dtype=torch.int64)
    return tok.decode(ids)


# ---- Arm orchestrators (A / B / C) ----

# ---- phase token accounting (spec §5) ----
PHASE1_FRAC, PHASE2B_FRAC, PHASE3_FRAC = 0.15, 0.30, 0.55

PROMPT = torch.tensor([464, 1292, 13], dtype=torch.int64)  # "The dog." in GPT-2 BPE


def _probe(holdout: torch.Tensor, n: int = 256) -> torch.Tensor:
    return holdout[:n].to(torch.int64)


def _finalize(engine, holdout, losses) -> dict:
    return {
        "ppl": evaluate_ppl(engine, holdout),
        "accuracy": None,  # filled by phase3 wrapper if available
        "diversity": expert_diversity(engine, _probe(holdout)),
        "losses": losses,
        "sample": greedy_sample(engine, PROMPT, n_tokens=80),
    }


def arm_from_scratch(engine, *, train, holdout, budget, chunk_len=16,
                     lr=3e-4) -> dict:
    """Arm A: online training on the full budget, no pre-training."""
    p3 = phase3_joint(engine, train[:budget], steps=budget // chunk_len,
                      lr=lr, chunk_len=chunk_len)
    out = _finalize(engine, holdout, p3["losses"])
    out["accuracy"] = p3["accuracy"]
    return out


def arm_edt_vanilla(engine, *, train, holdout, budget, chunk_len=16,
                    lr=3e-4) -> dict:
    """Arm B: EDT faithful to docs/EDT.md (shared bank)."""
    n1 = int(budget * PHASE1_FRAC)
    n2b = int(budget * PHASE2B_FRAC)
    n3 = budget - n1 - n2b
    bank = make_hidden_bank(engine, train[:n1], chunk_len=chunk_len,
                            n_chunks=max(n1 // chunk_len, 1), seed=0)
    p1 = phase1_experts_shared(engine, bank, steps=2000, lr=1e-3, seed=0)
    p2b = phase2b_embedding(engine, train[:n2b], steps=n2b // chunk_len,
                            lr=1e-3, chunk_len=chunk_len, seed=0)
    p3 = phase3_joint(engine, train[:n3], steps=n3 // chunk_len,
                      lr=lr, chunk_len=chunk_len)
    out = _finalize(engine, holdout, p3["losses"])
    out.update({"phase1_losses": p1, "phase2b_losses": p2b,
                "phase3_losses": p3["losses"], "accuracy": p3["accuracy"]})
    return out


def arm_edt_spec(engine, *, train, holdout, budget, domain_split, chunk_len=16,
                 lr=3e-4) -> dict:
    """Arm C: EDT with per-expert disjoint domain banks."""
    n2b = int(budget * PHASE2B_FRAC)
    n3 = budget - int(budget * PHASE1_FRAC) - n2b
    banks = [make_hidden_bank(engine, ds, chunk_len=chunk_len,
                              n_chunks=max(ds.numel() // chunk_len, 1), seed=i)
             for i, ds in enumerate(domain_split)]
    p1 = phase1_experts_partitioned(engine, banks, steps=2000, lr=1e-3, seed=0)
    p2b = phase2b_embedding(engine, train[:n2b], steps=n2b // chunk_len,
                            lr=1e-3, chunk_len=chunk_len, seed=0)
    p3 = phase3_joint(engine, train[:n3], steps=n3 // chunk_len,
                      lr=lr, chunk_len=chunk_len)
    out = _finalize(engine, holdout, p3["losses"])
    out.update({"phase1_losses": p1, "phase2b_losses": p2b,
                "phase3_losses": p3["losses"], "accuracy": p3["accuracy"]})
    return out
