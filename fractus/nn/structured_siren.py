"""StructuredSirenLinear: low-rank + spectral-residual SIREN weight compression.

THE L9 INNOVATION. Instead of representing a weight matrix W as a single
SIREN smooth field (which compresses poorly on trained weights = noise),
we decompose:

    W ≈ U @ V^T + SIREN(R)

where:
    - U @ V^T (low-rank, rank r) captures the dominant singular directions.
      Storage: (out + in) * r instead of out * in.
    - SIREN(R) learns only the residual R = W - U @ V^T. The residual has
      exploitable spectral structure (dominant singular values already captured),
      so the SIREN compresses it far better than W directly.

Target: 10-30× compression over dense, on real trained weights.

The forward pass reconstructs W on-the-fly (in-graph, differentiable), then
does y = x @ W + b. Both (U, V) and the SIREN params receive gradients.
"""

import math
import torch
import torch.nn as nn


class _ResidualSiren(nn.Module):
    """A compact SIREN that learns a residual matrix (out, in).

    Takes a coordinate (u, v) in [0, 1) and outputs a scalar. Evaluated on
    an (out × in) grid to reconstruct the residual matrix. Uses sin(ω0·)
    nonlinearity (Sitzmann 2020), ω0=30.
    """

    def __init__(self, out_features: int, in_features: int, hidden: int = 64, omega0: float = 30.0):
        super().__init__()
        self.out_features = out_features
        self.in_features = in_features
        self.hidden = hidden
        self.omega0 = omega0

        self.fc1 = nn.Linear(2, hidden)
        self.fc2 = nn.Linear(hidden, hidden)
        self.fc3 = nn.Linear(hidden, 1)
        self._init_siren()

        # Precompute the coordinate grid (off-graph constant).
        grid = self._build_grid(out_features, in_features)
        self.register_buffer("grid", grid)

    def _init_siren(self):
        with torch.no_grad():
            nn.init.uniform_(self.fc1.weight, -1.0 / self.omega0, 1.0 / self.omega0)
            nn.init.zeros_(self.fc1.bias)
            for layer in [self.fc2, self.fc3]:
                fan_in = layer.weight.shape[1]
                bound = math.sqrt(6.0 / (self.omega0 ** 2 * fan_in))
                nn.init.uniform_(layer.weight, -bound, bound)
                nn.init.zeros_(layer.bias)

    @staticmethod
    def _build_grid(h, w):
        u = torch.arange(h, dtype=torch.float32) / h
        v = torch.arange(w, dtype=torch.float32) / w
        grid = torch.stack(torch.meshgrid(u, v, indexing="ij"), dim=-1)
        return grid.reshape(-1, 2)

    def forward(self) -> torch.Tensor:
        """Returns the residual matrix (out, in)."""
        x = self.grid
        x = torch.sin(self.omega0 * self.fc1(x))
        x = torch.sin(self.omega0 * self.fc2(x))
        x = self.fc3(x)
        return x.squeeze(-1).reshape(self.out_features, self.in_features)


class StructuredSirenLinear(nn.Module):
    """Linear layer whose weight W ≈ U @ V^T + SIREN(residual).

    Args:
        in_features:  input dimension.
        out_features: output dimension.
        rank:         low-rank dimension r (captures dominant directions).
        siren_hidden: width of the residual SIREN.
        bias:         whether to add a trainable bias.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        rank: int = 128,
        siren_hidden: int = 64,
        bias: bool = True,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.rank = rank

        # Low-rank core: U (out, r), V (in, r). W_lr = U @ V^T.
        scale_u = math.sqrt(2.0 / (out_features + rank))
        scale_v = math.sqrt(2.0 / (in_features + rank))
        self.U = nn.Parameter(torch.empty(out_features, rank).uniform_(-scale_u, scale_u))
        self.V = nn.Parameter(torch.empty(in_features, rank).uniform_(-scale_v, scale_v))

        # Spectral residual SIREN.
        self.residual_siren = _ResidualSiren(out_features, in_features, hidden=siren_hidden)

        # Bias.
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features))
        else:
            self.register_parameter("bias", None)

    def reconstruct_weight(self) -> torch.Tensor:
        """Reconstruct the full weight matrix W = U @ V^T + SIREN(R)."""
        W_lr = self.U @ self.V.T  # (out, in)
        W_res = self.residual_siren()  # (out, in)
        return W_lr + W_res

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (..., in_features) → (..., out_features)."""
        W = self.reconstruct_weight()  # (out, in), differentiable
        y = x @ W.T  # (..., out)
        if self.bias is not None:
            y = y + self.bias
        return y

    @property
    def n_dense_equivalent(self) -> int:
        """Size the weight matrix would be if stored dense."""
        return self.in_features * self.out_features

    @property
    def n_actual_params(self) -> int:
        """Actual trainable params in this layer."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    @property
    def compression_ratio(self) -> float:
        """Dense-equivalent / actual-params. > 1 means compression."""
        return self.n_dense_equivalent / max(self.n_actual_params, 1)
