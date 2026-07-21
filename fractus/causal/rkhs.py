"""RKHSCausalOperator: causal operator in an RKHS via Random Fourier Features.

A true RKHS with a Gaussian kernel approximated by RFF (Rahimi-Recht 2007).
Not a bare low-rank projection. W_rff are frozen (Rahimi-Recht method).
Only U, V, and decode are trained.
"""

import torch
import torch.nn as nn


class RKHSCausalOperator(nn.Module):
    """Causal operator in an RKHS approximated by Random Fourier Features.

    Args:
        dim:   input/output dimension (original space).
        rank:  rank of the low-rank decomposition A = U @ V^T in the RKHS.
        n_rff: number of random features K (more = better approximation).
        sigma: bandwidth of the Gaussian kernel (1.0 by default).
    """

    def __init__(self, dim, rank=16, n_rff=64, sigma=1.0):
        super().__init__()
        self.dim = dim
        self.rank = rank
        self.n_rff = n_rff
        self.sigma = sigma
        self.feature_dim = 2 * n_rff

        # Random RFF features: w_k ~ N(0, 1/sigma^2). FROZEN.
        W_rff = torch.randn(dim, n_rff) / sigma
        self.register_buffer("W_rff", W_rff)
        b_rff = torch.rand(n_rff) * 2 * 3.141592653589793
        self.register_buffer("b_rff", b_rff)

        # Low-rank operator A = U @ V^T in the RKHS. TRAINABLE.
        scale = 0.02
        self.U = nn.Parameter(torch.randn(self.feature_dim, rank) * scale)
        self.V = nn.Parameter(torch.randn(self.feature_dim, rank) * scale)

        # Decoder: maps from feature space back to dim. TRAINABLE.
        self.decode = nn.Linear(self.feature_dim, dim, bias=False)

    def features(self, x):
        """phi(x) = [cos(w.x + b), sin(w.x + b)] / sqrt(K). Shape (N, 2K)."""
        proj = x @ self.W_rff + self.b_rff
        sqrt_K = self.n_rff ** 0.5
        cos_part = torch.cos(proj) / sqrt_K
        sin_part = torch.sin(proj) / sqrt_K
        return torch.cat([cos_part, sin_part], dim=-1)

    def kernel(self, x, y):
        """Approximated Gaussian kernel: k(x, y) ~= phi(x) . phi(y). Shape (N_x, N_y)."""
        return self.features(x) @ self.features(y).T

    def forward(self, x):
        """x: (N, dim) -> y: (N, dim)."""
        phi = self.features(x)
        low_rank = phi @ self.U
        transformed = low_rank @ self.V.T
        y = self.decode(transformed)
        return y
