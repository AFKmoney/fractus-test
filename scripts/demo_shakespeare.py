"""Scaling demo: fractal transformer on tinyshakespeare (short training).

Trains TinyFractalLM on a SUBSET of tinyshakespeare (200 batches), measures
the honest perplexity, and generates text.

HONESTY ABOUT THE CPU LIMIT: the full training (1 epoch = ~900 batches) takes
~10 min on the Ryzen 5 because of the Kuramoto RK4 (4 sub-steps) and the
triangular attention mask. This demo does a SHORT training (200 batches, ~3 min)
that proves the model LEARNS on real text, but without full convergence.
For full training: a GPU or deeper Kuramoto vectorization (future work).

Setup (CPU-only):
    vocab = 65 (tinyshakespeare)
    d_model = 48
    n_blocks = 2
    seq_len = 32
    ~40k params

Run:
    python scripts/demo_shakespeare.py
"""

import os
import sys
import time

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fractus.nn import FractalEmbedding, FractalBlockFull
from fractus.metrics.perplexity import honest_perplexity
from data.text.tinyshakespeare import TinyShakespeareDataset


class ShakespeareFractalLM(nn.Module):
    def __init__(self, vocab, d_model, n_blocks):
        super().__init__()
        self.embed = FractalEmbedding(vocab, d_model, n_frequencies=12)
        self.blocks = nn.ModuleList([
            FractalBlockFull(
                d_model=d_model, n_heads=4, d_head=d_model // 4, n_levels=2,
                n_oscillators=8, coupling_rank=4,
                n_experts=4, top_k=2, kappa=4.0,
            )
            for _ in range(n_blocks)
        ])
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab, bias=False)

    def forward(self, ids):
        x = self.embed(ids)
        aux = torch.tensor(0.0, device=x.device)
        for b in self.blocks:
            x, lb = b(x)
            aux = aux + lb
        x = self.norm(x)
        return self.head(x), aux


def main():
    torch.manual_seed(42)
    seq_len = 32
    dataset = TinyShakespeareDataset(seq_len=seq_len)
    print(f"Dataset: {len(dataset)} sequences of {seq_len} tokens, vocab={dataset.vocab_size}")

    # Train/val split (90/10).
    n_train = int(0.9 * len(dataset))
    n_val = len(dataset) - n_train
    train_ds, val_ds = random_split(
        dataset, [n_train, n_val], generator=torch.Generator().manual_seed(42),
    )
    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=32)

    model = ShakespeareFractalLM(vocab=dataset.vocab_size, d_model=48, n_blocks=2)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model: {n_params} parameters ({n_params/1000:.0f}k)")

    opt = torch.optim.Adam(model.parameters(), lr=3e-3)

    # Evaluate initial perplexity.
    model.eval()
    with torch.no_grad():
        inp, tgt = next(iter(val_loader))
        initial_ppl = honest_perplexity(model, inp, tgt)
    print(f"Initial perplexity: {initial_ppl:.2f}  (= vocab ≈ {dataset.vocab_size})")

    # SHORT training: 200 batches (subset, ~3 min on CPU).
    n_batches = 200
    print(f"\nShort training: {n_batches} batches (dataset subset)...")
    t0 = time.time()
    model.train()
    losses = []
    batch_iter = iter(train_loader)
    for batch_idx in range(n_batches):
        try:
            inp, tgt = next(batch_iter)
        except StopIteration:
            batch_iter = iter(train_loader)
            inp, tgt = next(batch_iter)
        opt.zero_grad()
        logits, aux = model(inp)
        ce = nn.functional.cross_entropy(
            logits.reshape(-1, dataset.vocab_size), tgt.reshape(-1)
        )
        loss = ce + 0.1 * aux
        loss.backward()
        opt.step()
        losses.append(ce.item())
        if batch_idx % 50 == 0 or batch_idx == n_batches - 1:
            avg_recent = sum(losses[-50:]) / max(len(losses[-50:]), 1)
            print(f"  batch {batch_idx:4d}/{n_batches}  ce={ce.item():.4f}  "
                  f"(recent avg={avg_recent:.4f}, ppl={torch.exp(torch.tensor(avg_recent)):.2f})")

    elapsed = time.time() - t0
    print(f"\nTime: {elapsed:.0f}s ({elapsed/60:.1f} min)")

    # Final perplexity on validation.
    model.eval()
    val_ces = []
    with torch.no_grad():
        for inp, tgt in val_loader:
            logits, _ = model(inp)
            ce = nn.functional.cross_entropy(
                logits.reshape(-1, dataset.vocab_size), tgt.reshape(-1)
            )
            val_ces.append(ce.item())
    final_ppl = torch.exp(torch.tensor(sum(val_ces) / len(val_ces))).item()
    print(f"Final perplexity (val): {final_ppl:.2f}")
    print(f"Reduction: {(1 - final_ppl/initial_ppl)*100:.1f}%")

    # Generation.
    print("\n=== Generation (greedy, 150 chars) ===")
    prompt = "ROMEO:\n"
    ctx = torch.tensor([[dataset.char_to_id[c] for c in prompt]])
    with torch.no_grad():
        for _ in range(150):
            if ctx.shape[1] > seq_len:
                ctx = ctx[:, -seq_len:]
            logits, _ = model(ctx)
            nxt = int(logits[0, -1].argmax().item())
            ctx = torch.cat([ctx, torch.tensor([[nxt]])], dim=1)
    generated = dataset.decode(ctx[0])
    print(generated)

    if final_ppl < initial_ppl * 0.7:
        print(f"\nOK: the fractal transformer learns on real tinyshakespeare "
              f"(ppl {initial_ppl:.1f} -> {final_ppl:.1f}, ÷{initial_ppl/final_ppl:.1f}).")
        print(f"  Note: short training (200 batches). Full convergence")
        print(f"  would require a GPU or Kuramoto vectorization (future work).")
    else:
        print(f"\n~: ppl drops little. More batches / a larger model would help.")


if __name__ == "__main__":
    main()
