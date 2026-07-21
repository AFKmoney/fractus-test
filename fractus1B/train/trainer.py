"""Lightweight trainers for fractus (L8).

StateCarryTrainer:
    Demonstrates chunk-wise training carrying the linear-attention state (S,z)
    across chunk boundaries — O(chunk_len) memory instead of O(seq_len).
    This is the Mamba/RWKV trick, legitimate for linear attention (impossible
    for softmax). Provided at the single-attention level: the test
    test_state_carry.py proves that processing a sequence as 2 chunks (carrying
    state) gives the same output as processing it whole.

    NOTE: carrying state through the FULL model (embedding + N blocks + head)
    requires the block stack to expose the attention state API, which is future
    work. Here we prove the principle and provide the chunked-attention helper.

LightweightTrainer:
    A standard batched trainer that adds the cheap, non-invasive CPU wins:
        - torch.autocast('cpu', dtype=bfloat16) when supported (≈2× faster
          matmuls on Zen-class CPUs, energy halved)
        - fused AdamW (fused C++ kernel, less Python overhead per step)
        - cosine-annealing LR with warm restarts (faster convergence → fewer
          steps → less energy)
        - explicit thread pinning to all cores
    Works with the existing FractalBlockFull model unchanged.
"""

import math
import os
from typing import Optional

import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# StateCarryTrainer: chunk-wise attention with state carrying
# ---------------------------------------------------------------------------

class StateCarryTrainer:
    """Carry the linear-attention (S, z) state across chunks.

    This is a helper that splits a long sequence into chunks and runs the
    FractalLinearAttention on each chunk, passing the (S, z) state between
    chunks (detached, so no BPTT across the whole sequence). Memory becomes
    O(chunk_len) instead of O(seq_len).

    Args:
        attention: a FractalLinearAttention module.
        chunk_len: number of tokens per chunk.
    """

    def __init__(self, attention: nn.Module, chunk_len: int = 16):
        self.attention = attention
        self.chunk_len = chunk_len

    @torch.no_grad()
    def chunked_forward(self, q_all, k_all, v_all, B, L, H, D, nlev):
        """Run the attention's internal vectorized path over chunks, carrying
        the (S, z) state. Returns the full (B, L, d_model) output.

        This is the proof-of-concept path: it demonstrates state-carry at the
        attention level. For the full model, the block stack would need to
        expose this API (future work).
        """
        from fractus.nn.stats import elu_plus_one
        attn = self.attention
        offsets = attn.level_offsets
        outputs = []
        # Per (batch, level, head) carried state.
        # S: (B*nlev*H, D, D), z: (B*nlev*H, D). Init to zeros.
        S0 = None
        z0 = None
        for start in range(0, L, self.chunk_len):
            end = min(start + self.chunk_len, L)
            qc = q_all[:, start:end]
            kc = k_all[:, start:end]
            vc = v_all[:, start:end]
            qc_l = qc.unsqueeze(1) + offsets.view(nlev, 1, 1, 1)
            kc_l = kc.unsqueeze(1) + offsets.view(nlev, 1, 1, 1)
            qf = elu_plus_one(qc_l, alpha=1.0)
            kf = elu_plus_one(kc_l, alpha=1.0)
            vf = vc.unsqueeze(1).expand(B, nlev, end - start, H, D)
            qf = qf.permute(0, 1, 3, 2, 4).reshape(B * nlev * H, end - start, D)
            kf = kf.permute(0, 1, 3, 2, 4).reshape(B * nlev * H, end - start, D)
            vf = vf.permute(0, 1, 3, 2, 4).reshape(B * nlev * H, end - start, D)
            carry = (S0, z0) if S0 is not None else None
            y, (S0, z0) = attn._linear_attention_causal_vectorized(qf, kf, vf, carry=carry)
            S0 = S0.detach()
            z0 = z0.detach()
            y = y.reshape(B, nlev, H, end - start, D).permute(0, 1, 3, 2, 4) \
                .reshape(B, nlev, end - start, H * D)
            outputs.append(y)
        return torch.cat(outputs, dim=2)  # (B, nlev, L, H*D)


# ---------------------------------------------------------------------------
# LightweightTrainer: AMP + fused AdamW + cosine scheduler
# ---------------------------------------------------------------------------

def _bf16_supported() -> bool:
    """Detect whether CPU bfloat16 autocast works on this machine."""
    try:
        with torch.autocast("cpu", dtype=torch.bfloat16):
            _ = (torch.randn(2, 2) @ torch.randn(2, 2)).sum()
        return True
    except Exception:
        return False


class LightweightTrainer:
    """Batched trainer with the cheap non-invasive CPU wins.

    Adds (vs plain Adam + fp32):
        - bf16 autocast (when supported): ~2× faster matmuls, half the energy.
        - fused AdamW: C++ fused kernel, less per-step overhead.
        - cosine-annealing-with-warm-restarts LR: faster convergence.
        - explicit thread pinning.

    Args:
        model           : the nn.Module to train (must return (logits, aux_loss)).
        lr              : peak learning rate.
        weight_decay    : AdamW weight decay.
        warmup_steps    : linear LR warmup (stabilizes early training).
        t_max           : cosine period (in steps) for the first restart.
        use_amp         : force-enable/disable bf16 autocast (default: autodetect).
        use_fused       : use fused AdamW if available.
    """

    def __init__(
        self,
        model: nn.Module,
        lr: float = 3e-3,
        weight_decay: float = 0.01,
        warmup_steps: int = 20,
        t_max: int = 200,
        use_amp: Optional[bool] = None,
        use_fused: bool = True,
    ):
        self.model = model
        # Pin threads to all available cores (explicit, reproducible).
        n_threads = os.cpu_count() or 1
        torch.set_num_threads(n_threads)
        self.n_threads = n_threads

        # AMP detection.
        self.use_amp = _bf16_supported() if use_amp is None else use_amp

        # Fused AdamW (falls back gracefully).
        try:
            self.optimizer = torch.optim.AdamW(
                model.parameters(), lr=lr, weight_decay=weight_decay,
                fused=use_fused,
            )
            self.fused = use_fused
        except (TypeError, RuntimeError):
            self.optimizer = torch.optim.AdamW(
                model.parameters(), lr=lr, weight_decay=weight_decay,
            )
            self.fused = False

        self.warmup_steps = max(warmup_steps, 1)
        self.base_lr = lr
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            self.optimizer, T_0=t_max, T_mult=2,
        )
        self.step_count = 0

    def _warmup_lr(self):
        """Linear warmup for the first warmup_steps, then cosine takes over."""
        if self.step_count < self.warmup_steps:
            frac = (self.step_count + 1) / self.warmup_steps
            for pg in self.optimizer.param_groups:
                pg["lr"] = self.base_lr * frac

    def train_step(self, inputs, targets, vocab_size: int) -> dict:
        """One training step. inputs/targets: (B, L) long tensors.

        Returns a dict with the CE loss, aux loss, and effective LR.
        """
        self.model.train()
        self.optimizer.zero_grad()
        if self.use_amp:
            with torch.autocast("cpu", dtype=torch.bfloat16):
                logits, aux = self.model(inputs)
                ce = nn.functional.cross_entropy(
                    logits.reshape(-1, vocab_size), targets.reshape(-1)
                )
                loss = ce + 0.1 * aux.float()
        else:
            logits, aux = self.model(inputs)
            ce = nn.functional.cross_entropy(
                logits.reshape(-1, vocab_size), targets.reshape(-1)
            )
            loss = ce + 0.1 * aux
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        self._warmup_lr()
        self.optimizer.step()
        self.scheduler.step()
        self.step_count += 1
        cur_lr = self.optimizer.param_groups[0]["lr"]
        return {"ce": float(ce.item()), "aux": float(aux.item()), "lr": cur_lr}

    def info(self) -> dict:
        """Return the active optimization flags (for logging / benchmarking)."""
        return {
            "threads": self.n_threads,
            "amp_bf16": self.use_amp,
            "fused_adamw": self.fused,
            "scheduler": "CosineAnnealingWarmRestarts",
            "warmup_steps": self.warmup_steps,
        }
