"""TorusSirenWeight: a true SIREN sin(omega0*x) to represent a weight matrix.

Uses torch.sin(omega0*(Wx+b)) as nonlinearity (Sitzmann et al. 2020), omega0=30.
Not SiLU, not ReLU. Init follows Sitzmann section 3.2.
Compression ratio is MEASURED, never hardcoded.
"""

import math
import torch
import torch.nn as nn


class TorusSirenWeight(nn.Module):
    """SIREN that represents a weight matrix W[out_h, out_w] as a scalar field
    over the torus T2 = [0,1)2.

    Args:
        out_h, out_w : dimensions of the matrix to regenerate.
        hidden       : width of the SIREN hidden layers.
        omega0       : fundamental frequency (30.0 by default, Sitzmann 2020).
    """

    def __init__(
        self,
        out_h: int,
        out_w: int,
        hidden: int = 32,
        omega0: float = 30.0,
    ):
        super().__init__()
        if out_h < 1 or out_w < 1 or hidden < 1:
            raise ValueError("out_h, out_w, hidden must be >= 1")
        self.out_h = out_h
        self.out_w = out_w
        self.hidden = hidden
        self.omega0 = omega0

        # Three layers (as in the SIREN paper for scalar fields).
        self.fc1 = nn.Linear(2, hidden)
        self.fc2 = nn.Linear(hidden, hidden)
        self.fc3 = nn.Linear(hidden, 1)
        self._init_siren_weights()

        # Precomputed grid (off-graph because it is constant).
        grid = self._build_grid(out_h, out_w)
        self.register_buffer("grid", grid)

    def _init_siren_weights(self):
        """SIREN-specific init (Sitzmann 2020, section 3.2)."""
        with torch.no_grad():
            # First layer: U(-1/ω0, 1/ω0).
            nn.init.uniform_(self.fc1.weight, -1.0 / self.omega0, 1.0 / self.omega0)
            nn.init.zeros_(self.fc1.bias)
            # Subsequent layers: U(-√(6/(ω02·fan_in)), √(6/(ω02·fan_in))).
            for layer in [self.fc2, self.fc3]:
                fan_in = layer.weight.shape[1]
                bound = math.sqrt(6.0 / (self.omega0 ** 2 * fan_in))
                nn.init.uniform_(layer.weight, -bound, bound)
                nn.init.zeros_(layer.bias)

    @staticmethod
    def _build_grid(h: int, w: int) -> torch.Tensor:
        """Grid of coords (u,v) ∈ [0,1)2 on the torus, shape (h·w, 2).

        On the torus T2=[0,1)2, the point 1 ≡ 0 (identification). We therefore exclude
        the endpoint 1 to avoid edge duplication (i/h for i=0..h-1).
        """
        u = torch.arange(h, dtype=torch.float32) / h
        v = torch.arange(w, dtype=torch.float32) / w
        grid = torch.stack(torch.meshgrid(u, v, indexing="ij"), dim=-1)  # (h, w, 2)
        return grid.reshape(-1, 2)  # (h·w, 2)

    def forward(self) -> torch.Tensor:
        """Evaluates the SIREN on the grid → matrix W[out_h, out_w].

        This is the 'decompression': we regenerate W from the SIREN params.
        """
        x = self.grid  # (h·w, 2)
        # Layer 1 + sin(ω0·).
        x = torch.sin(self.omega0 * self.fc1(x))
        # Layer 2 + sin(ω0·).
        x = torch.sin(self.omega0 * self.fc2(x))
        # Output layer: linear (no sin).
        x = self.fc3(x)  # (h·w, 1)
        return x.squeeze(-1).reshape(self.out_h, self.out_w)
