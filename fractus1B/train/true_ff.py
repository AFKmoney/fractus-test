"""TrueLayerWiseFF: the REAL Forward-Forward algorithm for the 1B model.

THE BREAKTHROUGH. My previous FF attempt kept the global backward — useless.
This is the REAL Hinton 2022 algorithm:

    Each layer is trained INDEPENDENTLY and SEQUENTIALLY:
    1. Layer N receives the (detached) output of layer N-1.
    2. Layer N computes a local "goodness" = mean(activations²).
    3. On POSITIVE data (real): maximize goodness.
    4. On NEGATIVE data (corrupted): minimize goodness.
    5. Layer N does its OWN backward + optimizer step.
    6. The output is DETACHED before passing to layer N+1.

    → NO global backward. NO gradient checkpointing. NO OOM.
    → Each backward touches ONE layer (~11M params) not the whole model (89M).
    → Cost = 2× forward (positive + negative) + N small backwards (one per layer).

For a language model, the "goodness" is the layer's ability to distinguish
real token sequences from corrupted ones. Each layer learns to be "active"
on real data and "quiet" on noise.
"""

import math
import random
import torch
import torch.nn as nn
import torch.nn.functional as F


def corrupt_tokens(input_ids: torch.Tensor, vocab_size: int, rate: float = 0.4) -> torch.Tensor:
    """Corrupt a fraction of tokens to create negative data."""
    corrupted = input_ids.clone()
    B, L = corrupted.shape
    n = int(L * rate)
    for b in range(B):
        pos = random.sample(range(L), min(n, L))
        corrupted[b, pos] = torch.randint(0, vocab_size, (len(pos),))
    return corrupted


def layer_goodness(h: torch.Tensor) -> torch.Tensor:
    """Goodness = mean of squared activations (per sample)."""
    return (h ** 2).mean(dim=-1)


def ff_loss(pos_good: torch.Tensor, neg_good: torch.Tensor, threshold: float = 1.0) -> torch.Tensor:
    """Forward-Forward loss: push positive > threshold, negative < threshold."""
    pos = F.softplus(-(pos_good - threshold)).mean()
    neg = F.softplus(neg_good - threshold).mean()
    return pos + neg


class TrueLayerWiseFF:
    """Layer-wise Forward-Forward trainer for Fractus-1B.

    Trains each block independently. No global backward. No OOM.

    Args:
        model: a Fractus1B model.
        lr: learning rate (applied per-layer).
        threshold: goodness threshold.
        corrupt_rate: fraction of tokens to corrupt for negatives.
    """

    def __init__(self, model, lr=1e-4, threshold=1.0, corrupt_rate=0.4):
        self.model = model
        self.threshold = threshold
        self.corrupt_rate = corrupt_rate
        self.vocab_size = model.vocab_size

        # One optimizer PER BLOCK — each layer is independent.
        self.block_optimizers = []
        for block in model.blocks:
            opt = torch.optim.AdamW(block.parameters(), lr=lr, weight_decay=0.01)
            self.block_optimizers.append(opt)

        # Separate optimizer for embedding + head.
        self.aux_optimizer = torch.optim.AdamW(
            list(model.embed.parameters()) + list(model.norm.parameters()) +
            list(model.lm_head.parameters()), lr=lr, weight_decay=0.01
        )

        self.step_count = 0

    def train_step(self, input_ids: torch.Tensor) -> dict:
        """One Forward-Forward training step.

        Returns dict with layer losses and total time.
        """
        self.model.train()
        vocab = self.vocab_size

        # Create positive and negative batches.
        pos_ids = input_ids
        neg_ids = corrupt_tokens(input_ids, vocab, self.corrupt_rate)

        layer_losses = []

        # === Phase 1: Forward through embedding (shared for pos and neg) ===
        h_pos = self.model.embed(pos_ids)
        h_neg = self.model.embed(neg_ids)

        # === Phase 2: Train each block independently ===
        for i, (block, opt) in enumerate(zip(self.model.blocks, self.block_optimizers)):
            # Forward pos through this block (gradients ON).
            x_pos, lb_pos = block(h_pos)
            # Forward neg through this block (gradients ON).
            x_neg, lb_neg = block(h_neg)

            # Compute goodness on the block's output.
            # Use the residual output (x) normalized by the block's norm.
            pos_good = layer_goodness(x_pos.flatten(1))
            neg_good = layer_goodness(x_neg.flatten(1))

            loss = ff_loss(pos_good, neg_good, self.threshold)

            # Local backward + step — ONLY this block's params.
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(block.parameters(), 1.0)
            opt.step()

            layer_losses.append(loss.item())

            # DETACH before passing to next layer (the key trick).
            # Recompute the forward with the UPDATED weights (no grad needed).
            with torch.no_grad():
                h_pos = block(h_pos)[0].detach()
                h_neg = block(h_neg)[0].detach()

        # === Phase 3: Train the LM head with standard CE on positive data ===
        # The head learns to predict next tokens from the final representation.
        h_final = h_pos  # detached, from the last block
        h_final.requires_grad_(True)
        x = self.model.norm(h_final)
        logits = self.model.lm_head(x)

        target = torch.cat([input_ids[:, 1:],
                           torch.zeros(input_ids.shape[0], 1, dtype=torch.long)], dim=1)
        ce_loss = F.cross_entropy(logits.reshape(-1, vocab), target.reshape(-1))

        self.aux_optimizer.zero_grad()
        ce_loss.backward()
        self.aux_optimizer.step()

        self.step_count += 1

        return {
            "layer_losses": layer_losses,
            "avg_layer_loss": sum(layer_losses) / max(len(layer_losses), 1),
            "ce_loss": ce_loss.item(),
        }

    def forward_eval(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Standard forward for evaluation (no FF, just inference)."""
        self.model.eval()
        with torch.no_grad():
            x = self.model.embed(input_ids)
            for block in self.model.blocks:
                x, _ = block(x)
            x = self.model.norm(x)
            return self.model.lm_head(x)

    def info(self) -> dict:
        return {
            "method": "true-layer-wise forward-forward (Hinton 2022)",
            "threshold": self.threshold,
            "corrupt_rate": self.corrupt_rate,
            "n_blocks": len(self.block_optimizers),
            "key_property": "detach between layers → no global backward → no OOM",
        }
