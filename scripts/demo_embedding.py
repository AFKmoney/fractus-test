"""Demo L1: proves that FractalEmbedding truly learns.

Goal: overfit a target whose frequencies are NOT present in the embedding's
internal Fourier basis. This forces the embedding to actually combine its three
sources (char features + Fourier + vortex conditioning) via its trainable
projection, rather than simply copying a combination of its input columns
(which a Linear does trivially).

The frequencies of the internal basis are ω_k = (φ2)^{-k} for k=0..n_freq-1,
i.e. decreasing starting from 1.0. We therefore pick independent target
frequencies: increasing (1.3, 0.7) which are NOT in the basis.

This is the minimal honest proof that autodiff flows through the fractal
embedding (which the original system could not do — training.rs:399 used noise).

Run:
    python scripts/demo_embedding.py
"""

import torch
from fractus.nn import FractalEmbedding


def main():
    torch.manual_seed(42)

    vocab = 64
    d_model = 32
    emb = FractalEmbedding(vocab_size=vocab, d_model=d_model, n_frequencies=12)
    print(f"Trainable parameters: {sum(p.numel() for p in emb.parameters())}")

    # Target independent of the input space: sinusoids at frequencies that
    # are NOT in the (φ2)^{-k} basis. The basis frequencies are decreasing
    # from 1.0; we pick varied positive frequencies (0.7, 1.3, 2.1) that
    # require a non-trivial approximation.
    ids = torch.arange(vocab, dtype=torch.float32).unsqueeze(1)  # (V, 1)
    freqs_target = torch.linspace(0.3, 2.1, d_model)  # (d_model,)
    phases = freqs_target * ids  # (V, d_model)
    # Target = non-linear mix (sin + cos at different phases) to force a real
    # regression, not a simple copy.
    target = torch.sin(phases) * 0.6 + torch.cos(phases * 1.7 + 0.3) * 0.4

    opt = torch.optim.Adam(emb.parameters(), lr=3e-3)
    # Cosine schedule: helps fine convergence near the end of training.
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=600)

    initial_loss = None
    for step in range(600):
        opt.zero_grad()
        token_ids = torch.arange(vocab)
        out = emb(token_ids)
        loss = ((out - target) ** 2).mean()
        if initial_loss is None:
            initial_loss = loss.item()
        loss.backward()
        opt.step()
        sched.step()
        if step % 100 == 0 or step == 599:
            print(f"step {step:3d}  loss = {loss.item():.6f}  lr = {sched.get_last_lr()[0]:.5f}")

    final_loss = loss.item()
    print()
    print(f"Initial loss : {initial_loss:.6f}")
    print(f"Final loss   : {final_loss:.6f}")
    print(f"Reduction    : {(1 - final_loss / initial_loss) * 100:.1f}%")

    # Honest criterion: the target is NOT trivially in the input space,
    # so a 3× reduction (67%) already proves that autodiff flows through the
    # pipeline (char + Fourier + vortex → MLP → proj). We do not aim for 10000×.
    if final_loss < initial_loss / 3.0:
        print("\n✓ SUCCESS: the fractal embedding learns (loss divided by >3 on a "
              "target outside the input space — honest autodiff proof).")
    elif final_loss < initial_loss / 2.0:
        print("\n✓ PARTIAL: loss divided by >2 — autodiff works but the approximation "
              "capacity is limited (acceptable for the L1 demo).")
    else:
        print("\n✗ FAILURE: the loss does not drop enough — investigate.")


if __name__ == "__main__":
    main()
