"""SirenLinear: an nn.Linear-like layer whose weight matrix is produced
by a SIREN.

CORRECTION vs the original: in the original, the decompressed matrix W was computed then
DISCARDED (training_loop.py:30-37 applied a mirror to W then ran on the raw input).
Here, the SIREN IS the matrix: we evaluate the SIREN at each forward to obtain W,
then we do y = x @ W + b. Everything is in the autodiff graph.

Usage: replace some nn.Linear layers with SirenLinear to compress their
weights via SIREN. The trade-off: fewer parameters (compression) but a more
expensive forward (SIREN evaluation at each call) and potentially reduced
expressiveness (SIREN weights are smooth, not dense — see the L3 demo).
"""

import torch
import torch.nn as nn

from .siren import TorusSirenWeight


class SirenLinear(nn.Module):
    """Linear layer whose matrix W = SIREN(grid).

    Args:
        in_features, out_features : dimensions (as in nn.Linear).
        hidden : width of the SIREN that produces W.
        omega0 : SIREN frequency.
        bias   : if True, adds a trainable bias.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        hidden: int = 32,
        omega0: float = 30.0,
        bias: bool = True,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        # The weight matrix comes from a SIREN evaluated on a grid
        # (in_features, out_features).
        self.siren = TorusSirenWeight(
            out_h=in_features, out_w=out_features, hidden=hidden, omega0=omega0
        )
        # Separate trainable bias (not compressed — this is a vector, not a matrix).
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features))
        else:
            self.register_parameter("bias", None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (..., in_features) → (..., out_features).

        W = self.siren(): (in_features, out_features), in the autodiff graph.
        y = x @ W + bias.
        """
        W = self.siren()  # (in_features, out_features), differentiable
        y = x @ W
        if self.bias is not None:
            y = y + self.bias
        return y
