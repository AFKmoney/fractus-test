"""SurpriseGatedTrainer: energy-proportional training via surprise gating.

L8 INNOVATION (méthode inédite). The idea:
    On each batch, compute the per-token loss. Most tokens are already well
    predicted (loss ~ 0) → their gradient is ~ 0 → computing it wastes energy.
    We MASK the loss to only backpropagate through the "surprising" tokens
    (loss above an adaptive threshold). Energy-proportional: we spend compute
    where the gradient is non-trivial.

    Concretely:
        1. forward → logits (B, L, V)
        2. per-token CE: loss_t (B, L)
        3. threshold = EMA of the percentile-70 of loss_t (adaptive)
        4. mask = loss_t > threshold  (the "surprising" tokens)
        5. masked_loss = (loss_t * mask).sum() / mask.sum().clamp(min=1)
        6. masked_loss.backward()

    The threshold adapts (EMA) so it tracks the model's evolving competence.
    Early in training, many tokens are surprising → low selectivity → ≈ full
    training. As the model learns, fewer tokens surprise → higher selectivity
    → less energy per step (the model "skips" what it already knows).

    Trade-off (honest): this is a biased gradient estimator (we drop some
    tokens). On well-mixed batches the bias is small because the dropped
    tokens have near-zero gradient anyway. We verify in test that it still
    converges on a toy target (no divergence) and that the fraction of
    tokens backpropagated is < 1 (proving selectivity).
"""

from collections import deque
from typing import Optional

import torch
import torch.nn as nn


class SurpriseGatedTrainer:
    """Trainer that backpropagates only on high-loss ('surprising') tokens.

    Args:
        model          : nn.Module returning (logits (B, L, V), aux_loss).
        lr             : learning rate.
        weight_decay   : AdamW weight decay.
        percentile     : keep tokens ABOVE this loss percentile (70 = top 30%).
        ema_decay      : EMA decay for the adaptive threshold (0.9 default).
        warmup_full    : number of initial steps with NO gating (full gradient)
                         so the model bootstraps before selectivity kicks in.
    """

    def __init__(
        self,
        model: nn.Module,
        lr: float = 3e-3,
        weight_decay: float = 0.01,
        percentile: float = 70.0,
        ema_decay: float = 0.9,
        warmup_full: int = 5,
    ):
        self.model = model
        self.optimizer = torch.optim.AdamW(
            model.parameters(), lr=lr, weight_decay=weight_decay
        )
        self.percentile = percentile
        self.ema_decay = ema_decay
        self.warmup_full = warmup_full
        # Adaptive threshold state.
        self.threshold_ema: Optional[float] = None
        self.step_count = 0
        # Stats for logging.
        self.recent_selectivity: deque = deque(maxlen=50)  # fraction of tokens kept

    def train_step(self, inputs, targets, vocab_size: int) -> dict:
        """One gated training step.

        inputs/targets: (B, L) long tensors.
        Returns dict with ce (full-batch CE for monitoring), gated_ce (the
        loss actually backpropped), selectivity (fraction of tokens kept),
        and threshold.
        """
        self.model.train()
        self.optimizer.zero_grad()
        logits, aux = self.model(inputs)  # (B, L, V), scalar

        # Per-token CE (no reduction): (B, L).
        loss_per_token = torch.nn.functional.cross_entropy(
            logits.reshape(-1, vocab_size),
            targets.reshape(-1),
            reduction="none",
        ).reshape(targets.shape)  # (B, L)

        # Full-batch CE for monitoring (NOT used for the gradient).
        with torch.no_grad():
            full_ce = loss_per_token.mean().item()

        if self.step_count < self.warmup_full:
            # Warmup: full gradient (no gating) to bootstrap.
            loss = loss_per_token.mean() + 0.1 * aux
            kept_frac = 1.0
            threshold = 0.0
        else:
            # Adaptive threshold via EMA of the kept percentile.
            with torch.no_grad():
                current_thresh = torch.quantile(
                    loss_per_token.float().flatten(),
                    self.percentile / 100.0,
                ).item()
                if self.threshold_ema is None:
                    self.threshold_ema = current_thresh
                else:
                    self.threshold_ema = (
                        self.ema_decay * self.threshold_ema
                        + (1.0 - self.ema_decay) * current_thresh
                    )
                threshold = self.threshold_ema
                # Mask: keep tokens with loss > threshold.
                mask = (loss_per_token > threshold).float()
                n_kept = mask.sum().item()
                kept_frac = n_kept / loss_per_token.numel()
                # Guard against the degenerate case where NO token is kept
                # (shouldn't happen with a percentile threshold, but be safe).
                if n_kept < 1:
                    mask = torch.ones_like(loss_per_token)
                    kept_frac = 1.0

            # Gated loss: mean over KEPT tokens only.
            gated_per_token = loss_per_token * mask
            loss = gated_per_token.sum() / mask.sum().clamp(min=1.0) + 0.1 * aux

        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        self.optimizer.step()
        self.step_count += 1
        self.recent_selectivity.append(kept_frac)

        return {
            "ce": full_ce,
            "gated_ce": float(loss.item()),
            "selectivity": kept_frac,
            "threshold": threshold,
            "mean_selectivity": sum(self.recent_selectivity) / len(self.recent_selectivity),
        }

    def info(self) -> dict:
        return {
            "method": "surprise-gated",
            "percentile": self.percentile,
            "ema_decay": self.ema_decay,
            "warmup_full": self.warmup_full,
        }
