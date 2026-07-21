"""Demo L3: measure the TRUE SIREN compression on a model.

We build two variants of a mini-MLP:
    (A) 100% dense (nn.Linear)
    (B) Hidden layers in SirenLinear, last layer dense.

We measure:
    - The compression ratio (SIREN params vs dense equivalent).
    - The learning capacity: can we overfit a target with (B) as well
      as with (A)?

HONEST SCIENTIFIC POSITION:
We expect a MODEST ratio (~2× to 5×) and a loss in learning quality
(SIREN weights are smooth, which limits the capacity to express arbitrary
functions). This is the truth — to be compared with the prior false 20.4× claim.

Run:
    python scripts/demo_siren_compression.py
"""

import torch
import torch.nn as nn
from fractus.nn import SirenLinear
from fractus.metrics.compression import measure_compression_ratio


def make_dense_model(d_in, d_hidden, d_out):
    return nn.Sequential(
        nn.Linear(d_in, d_hidden), nn.ReLU(),
        nn.Linear(d_hidden, d_hidden), nn.ReLU(),
        nn.Linear(d_hidden, d_out),
    )


def make_siren_model(d_in, d_hidden, d_out, siren_hidden=16):
    return nn.Sequential(
        SirenLinear(d_in, d_hidden, hidden=siren_hidden), nn.ReLU(),
        SirenLinear(d_hidden, d_hidden, hidden=siren_hidden), nn.ReLU(),
        nn.Linear(d_hidden, d_out),  # last layer dense
    )


def train_and_eval(model, X, Y, n_steps=300, lr=1e-2):
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    initial = None
    for step in range(n_steps):
        opt.zero_grad()
        pred = model(X)
        loss = ((pred - Y) ** 2).mean()
        if initial is None:
            initial = loss.item()
        loss.backward()
        opt.step()
    final = loss.item()
    return initial, final


def main():
    torch.manual_seed(42)
    d_in, d_hidden, d_out = 16, 32, 8
    n_samples = 64

    # Target: a non-trivial function (sinusoid at a non-aligned frequency).
    X = torch.randn(n_samples, d_in)
    Y = torch.sin(X[:, :d_out] * 1.3) + 0.5 * torch.cos(X[:, :d_out] * 0.7)

    dense = make_dense_model(d_in, d_hidden, d_out)
    siren = make_siren_model(d_in, d_hidden, d_out, siren_hidden=16)

    n_dense = sum(p.numel() for p in dense.parameters())
    n_siren = sum(p.numel() for p in siren.parameters())
    ratio_dense = measure_compression_ratio(dense)
    ratio_siren = measure_compression_ratio(siren)

    print("=== Measured compression ===")
    print(f"Dense model : {n_dense} params, ratio = {ratio_dense:.2f}x")
    print(f"SIREN model : {n_siren} params, ratio = {ratio_siren:.2f}x")
    print(f"Savings     : {(1 - n_siren/n_dense)*100:.1f}% fewer params")
    print()

    print("=== Learning capacity (overfit sinusoidal target) ===")
    i_d, f_d = train_and_eval(dense, X, Y)
    i_s, f_s = train_and_eval(siren, X, Y)
    print(f"Dense : loss {i_d:.4f} -> {f_d:.4f}  (drop {(1-f_d/i_d)*100:.1f}%)")
    print(f"SIREN : loss {i_s:.4f} -> {f_s:.4f}  (drop {(1-f_s/i_s)*100:.1f}%)")
    print()

    print("=== Honest verdict ===")
    print(f"Real compression ratio: {ratio_siren:.2f}x")
    print(f"  (compare to the '20.4x' hardcoded in the original design, which was false)")
    print(f"Learning quality loss: {(f_s - f_d):.4f} (SIREN - Dense)")
    if ratio_siren > 1.5 and f_s < i_s * 0.5:
        print("\nOK: the SIREN compresses (>1.5x) AND learns — honest and useful.")
    elif ratio_siren > 1.5:
        print("\n~: the SIREN compresses but learns less well — a trade-off to document.")
    else:
        print("\n~: weak compression (<1.5x) — the SIREN is not suited to these weights.")
    print("\nConclusion: the '20.4x without loss' claim of the original is not reproduced.")
    print("The SIREN is useful for smooth functions, not for dense weights.")


if __name__ == "__main__":
    main()
