"""Demo L2b: COMPLETE fractal transformer (with Kuramoto + MoE).

Assemble FractalEmbedding + N×FractalBlockFull + logit projection, and train
on a toy text sequence. FractalBlockFull integrates:
  - multi-level causal linear attention (L2a)
  - low-rank Kuramoto RK4 oscillators (L2b)
  - von Mises/Farey-routed mixture-of-experts (L2b)

Corrects the central error of the original system (training.rs:399 = noise): here Adam
receives real gradients across the ENTIRE pipeline (including Kuramoto U/Λ and expert
W1/W2) and the loss drops.

Small setup (CPU-only):
    vocab  = 128 (printable ASCII: ord(c)-32 ∈ [0,95] ⊂ [0,128))
    d_model = 32
    n_blocks = 2
    seq_len  = 16

Run:
    python scripts/demo_transformer.py
"""

import torch
import torch.nn as nn
from fractus.nn import FractalEmbedding, FractalBlockFull


class TinyFractalLM(nn.Module):
    """Embedding + FractalBlockFull blocks + logit projection."""

    def __init__(self, vocab, d_model, n_heads, d_head, n_levels, n_blocks,
                 n_oscillators, coupling_rank, n_experts, top_k):
        super().__init__()
        self.embed = FractalEmbedding(vocab, d_model, n_frequencies=8)
        self.blocks = nn.ModuleList([
            FractalBlockFull(
                d_model=d_model, n_heads=n_heads, d_head=d_head, n_levels=n_levels,
                n_oscillators=n_oscillators, coupling_rank=coupling_rank,
                n_experts=n_experts, top_k=top_k, kappa=4.0,
            )
            for _ in range(n_blocks)
        ])
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab, bias=False)

    def forward(self, ids):
        x = self.embed(ids)  # (B, L, d_model)
        aux_loss = torch.tensor(0.0)
        for block in self.blocks:
            x, lb = block(x)
            aux_loss = aux_loss + lb
        x = self.norm(x)
        return self.head(x), aux_loss  # (logits (B,L,vocab), aux_loss scalar)


def main():
    torch.manual_seed(42)

    text = "hello world " * 8
    vocab = 128
    ids = torch.tensor([ord(c) - 32 for c in text if 0 <= ord(c) - 32 < vocab])
    print(f"Sequence: {len(ids)} tokens, vocab={vocab}")
    print(f"Excerpt: {''.join(chr(int(i)+32) for i in ids[:24])!r}")

    seq_len = 16
    n_seqs = len(ids) // seq_len
    ids = ids[:n_seqs * seq_len].view(n_seqs, seq_len)
    print(f"Batches: {n_seqs} sequences of length {seq_len}")

    model = TinyFractalLM(
        vocab=vocab, d_model=32, n_heads=4, d_head=8, n_levels=2, n_blocks=2,
        n_oscillators=8, coupling_rank=4, n_experts=4, top_k=2,
    )
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {n_params}")

    opt = torch.optim.Adam(model.parameters(), lr=3e-3)

    initial_loss = None
    for epoch in range(40):
        opt.zero_grad()
        logits, aux_loss = model(ids)
        # Cross-entropy + 0.1 * auxiliary load-balance loss.
        ce_loss = nn.functional.cross_entropy(
            logits[:, :-1].reshape(-1, vocab),
            ids[:, 1:].reshape(-1),
        )
        loss = ce_loss + 0.1 * aux_loss
        if initial_loss is None:
            initial_loss = ce_loss.item()
        loss.backward()
        opt.step()
        if epoch % 8 == 0 or epoch == 39:
            print(f"epoch {epoch:2d}  ce_loss = {ce_loss.item():.4f}  aux = {aux_loss.item():.4f}")

    final_loss = ce_loss.item()
    print()
    print(f"Initial CE loss : {initial_loss:.4f}  (= log({vocab}) ≈ {torch.log(torch.tensor(float(vocab))).item():.3f})")
    print(f"Final CE loss   : {final_loss:.4f}")
    print(f"Reduction       : {(1 - final_loss / initial_loss) * 100:.1f}%")

    print()
    print("Generation (greedy):")
    model.eval()
    with torch.no_grad():
        context = torch.tensor([[ord(c) - 32 for c in "hello"]])
        for _ in range(20):
            logits, _ = model(context)
            next_id = logits[0, -1].argmax().item()
            next_id = max(0, min(vocab - 1, next_id))
            context = torch.cat([context, torch.tensor([[next_id]])], dim=1)
        generated = "".join(chr(int(i) + 32) for i in context[0].tolist())
    print(f"  '{generated}'")

    if final_loss < initial_loss * 0.5:
        print("\n✓ SUCCESS: the complete fractal transformer (Kuramoto+MoE) learns.")
    else:
        print("\n✗ FAILURE: the loss does not drop enough.")


if __name__ == "__main__":
    main()
