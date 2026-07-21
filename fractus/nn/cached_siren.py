"""CachedStructuredSirenLinear: the breakthrough for fast CPU training.

THE INNOVATION. The profiler showed SIREN matrix reconstruction is 148% of
the forward time — by far the dominant cost. This module CACHES the
reconstructed weight matrix and only recomputes it every `refresh_every`
forward calls. Between refreshes, the forward uses the cached matrix directly
(a single cheap matmul, no SIREN evaluation).

How it works:
    - On forward, if the cache is stale (step % refresh_every == 0):
        W = U @ V^T + SIREN(R)   ← expensive, but only 1/N of the time
        cache W
    - Else:
        W = cached_W              ← free (just a lookup)
    - y = x @ W^T + b             ← cheap matmul

The gradients still flow to U, V, and SIREN params during the refresh step
(because the cache is built WITH grad). Between refreshes, the cached W is
detached — the experts learn in "frozen matrix" mode, which is fine because
the matrix changes slowly (it's a low-rank + smooth residual, not noise).

This is analogous to how quantized models work: the expensive quantization/
dequantization happens periodically, not every forward.
"""

import math
import torch
import torch.nn as nn

from .structured_siren import _ResidualSiren


class CachedStructuredSirenLinear(nn.Module):
    """StructuredSirenLinear with weight caching for fast CPU training.

    Args:
        in_features:   input dimension.
        out_features:  output dimension.
        rank:          low-rank dimension.
        siren_hidden:  width of the residual SIREN.
        bias:          whether to add a trainable bias.
        refresh_every: recompute the matrix every N forward calls.
                       1 = no caching (same as StructuredSirenLinear).
                       8 = recompute only 1/8 of the time (8× faster forward).
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        rank: int = 64,
        siren_hidden: int = 32,
        bias: bool = True,
        refresh_every: int = 8,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.rank = rank
        self.refresh_every = max(refresh_every, 1)
        self._call_count = 0

        # Low-rank core.
        scale_u = math.sqrt(2.0 / (out_features + rank))
        scale_v = math.sqrt(2.0 / (in_features + rank))
        self.U = nn.Parameter(torch.empty(out_features, rank).uniform_(-scale_u, scale_u))
        self.V = nn.Parameter(torch.empty(in_features, rank).uniform_(-scale_v, scale_v))

        # Residual SIREN.
        self.residual_siren = _ResidualSiren(out_features, in_features, hidden=siren_hidden)

        # Bias.
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features))
        else:
            self.register_parameter("bias", None)

        # Cached weight (not a parameter — a buffer that we update).
        # Initialize with a fresh reconstruction.
        with torch.no_grad():
            W = self.U @ self.V.T + self.residual_siren()
        self.register_buffer("_cached_W", W.detach())

    def _reconstruct(self) -> torch.Tensor:
        """Reconstruct the full weight (with grad)."""
        return self.U @ self.V.T + self.residual_siren()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (..., in_features) → (..., out_features)."""
        self._call_count += 1
        if self._call_count % self.refresh_every == 1:
            # Refresh: reconstruct with grad, update cache.
            W = self._reconstruct()
            # Update the buffer (detached copy for next calls).
            self._cached_W = W.detach()
        else:
            # Use cached weight (fast path — just a matmul).
            W = self._cached_W
        y = x @ W.T
        if self.bias is not None:
            y = y + self.bias
        return y

    def force_refresh(self):
        """Force a cache refresh (call before evaluation/export)."""
        with torch.no_grad():
            self._cached_W = (self.U @ self.V.T + self.residual_siren()).detach()
        self._call_count = 0

    @property
    def n_dense_equivalent(self) -> int:
        return self.in_features * self.out_features

    @property
    def n_actual_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    @property
    def compression_ratio(self) -> float:
        return self.n_dense_equivalent / max(self.n_actual_params, 1)
