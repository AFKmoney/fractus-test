"""FractalBlock: fractal transformer block (minimal L2a + full L2b).

L2a: x -> LayerNorm -> FractalLinearAttention -> Dropout -> + x (residual)
L2b: x -> LN -> attn -> +x -> LN -> Kuramoto -> phases -> LN -> MoE -> +x
"""

import torch
import torch.nn as nn

from .attention import FractalLinearAttention
from .phase_ode import KuramotoLayer
from .moe import PhaseRoutedMoE


class FractalBlock(nn.Module):
    """Minimal fractal transformer block (L2a).

    Args:
        d_model  : model dimension.
        n_heads  : number of attention heads.
        d_head   : dimension per head (n_heads·d_head == d_model required).
        n_levels : fractal levels of the attention.
        dropout  : dropout rate (0 by default in L2a, will be added in L7).
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        d_head: int,
        n_levels: int = 3,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.attn = FractalLinearAttention(d_model, n_heads, d_head, n_levels)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, L, d_model) → (B, L, d_model).

        Residual connection: out = x + dropout(attn(norm(x))).
        """
        return x + self.dropout(self.attn(self.norm(x)))


class FractalBlockFull(nn.Module):
    """Full fractal transformer block (L2b): integrates Kuramoto + MoE.

    Architecture:
        x → LN → FractalLinearAttention → + x (residual 1)
              → LN → KuramotoLayer → phases
              → LN → PhaseRoutedMoE(hidden, phases) → + x (residual 2)

    Returns (output, loss_aux) where loss_aux is the MoE load_balance_loss
    (to be added to the main loss by the caller).
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        d_head: int,
        n_levels: int,
        n_oscillators: int,
        coupling_rank: int,
        n_experts: int,
        top_k: int,
        kappa: float = 4.0,
        kuramoto_steps: int = 4,
        kuramoto_dt: float = 0.1,
        dropout: float = 0.0,
    ):
        super().__init__()
        # Attention sub-block.
        self.norm1 = nn.LayerNorm(d_model)
        self.attn = FractalLinearAttention(d_model, n_heads, d_head, n_levels)
        # Kuramoto + MoE.
        self.norm_kur = nn.LayerNorm(d_model)
        self.kuramoto = KuramotoLayer(d_model, n_oscillators, coupling_rank,
                                      n_steps=kuramoto_steps, dt=kuramoto_dt)
        self.norm_moe = nn.LayerNorm(d_model)
        self.moe = PhaseRoutedMoE(d_model, n_experts, top_k, kappa=kappa)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x: torch.Tensor):
        """x: (B, L, d_model) → (output (B, L, d_model), loss_aux scalar)."""
        # Residual 1: attention.
        x = x + self.dropout(self.attn(self.norm1(x)))
        # Kuramoto: phases from the normalized hidden state.
        phases = self.kuramoto(self.norm_kur(x))  # (B, L, N)
        # MoE: routing by phases.
        moe_out, lb_loss = self.moe(self.norm_moe(x), phases)
        x = x + self.dropout(moe_out)
        return x, lb_loss
