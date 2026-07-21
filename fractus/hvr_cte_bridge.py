"""HVR-Enhanced CTE: the Continuous Thought Engine with a holographic
knowledge layer that it queries during generation.

ARCHITECTURE:
  ┌──────────────────────────────────────────────────┐
  │                HVR Memory (10k dim)               │
  │  Encodes the entire corpus in one pass.           │
  │  Stores token transitions via circular conv.      │
  │  Queryable: "what follows token X?" → top-k        │
  └────────────────────┬─────────────────────────────┘
                       │ query (during generation)
  ┌────────────────────▼─────────────────────────────┐
  │              CTE (88M or expanded)                │
  │  Generates next-token logits from attention +     │
  │  MoE. The HVR provides ADDITIONAL context:        │
  │  - Retrieved transitions are added to logits      │
  │  - Or fed as a context vector to the attention    │
  └──────────────────────────────────────────────────┘

WHY THIS REDUCES TRAINING:
  Without HVR: the 1B must memorize ALL knowledge in its weights → 90 days.
  With HVR: the 1B only learns to GENERATE + INTERROGATE → much less data
  needed. The knowledge lives in the HVR layer (one-pass encoded), not in
  the attention weights.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import List, Optional

from fractus.hvr import HolographicMemory


class HVRContextProvider:
    """Wraps an HVR memory and provides context vectors to the CTE.

    During generation, for each token, the provider:
    1. Queries the HVR memory: "what tokens tend to follow the current token?"
    2. Returns the top-k recalled token IDs + their similarity scores
    3. The CTE can use this to bias its logits (knowledge retrieval)
    """

    def __init__(self, hvr: HolographicMemory, top_k: int = 10,
                 vocab_size: int = 50257, bias_strength: float = 2.0):
        self.hvr = hvr
        self.top_k = top_k
        self.vocab_size = vocab_size
        self.bias_strength = bias_strength

    def get_logit_bias(self, token_id: int) -> torch.Tensor:
        """Get a logit bias vector for the current token.

        Queries the HVR: "what follows this token?" and creates a bias
        vector that boosts the logits of likely next tokens.

        Returns: (vocab_size,) tensor of biases to ADD to the CTE's logits.
        """
        recalled = self.hvr.recall_next(token_id, top_k=self.top_k)
        bias = torch.zeros(self.vocab_size, dtype=torch.float32)
        for tid, sim in recalled:
            # Scale similarity to a logit-space bias.
            bias[tid] += self.bias_strength * sim
        return bias

    def get_context_embedding(self, token_ids: List[int]) -> torch.Tensor:
        """Get a context vector for a sequence of tokens.

        Binds and superposes the HVR vectors for the token sequence,
        producing a single 10k-dim context vector that summarizes the
        sequence's "holographic fingerprint".

        Returns: (dim,) tensor.
        """
        if not token_ids:
            return torch.zeros(self.hvr.dim, dtype=torch.float32)
        ctx = np.zeros(self.hvr.dim, dtype=np.float32)
        for i in range(len(token_ids) - 1):
            tok = self.hvr.encode_token(token_ids[i])
            nxt = self.hvr.encode_token(token_ids[i + 1])
            ctx += self.hvr.bind(tok, nxt)
        return torch.from_numpy(ctx)


class HVREnhancedCTE(nn.Module):
    """A CTE wrapper that uses HVR as a knowledge layer.

    During generation:
    1. The CTE produces logits (from attention + MoE)
    2. The HVR provides a logit bias (from holographic memory)
    3. The final logits = CTE_logits + hvr_bias

    This means the CTE doesn't need to memorize the corpus — the HVR
    provides the factual knowledge, and the CTE provides the reasoning.
    """

    def __init__(self, cte, hvr_provider: HVRContextProvider,
                 use_bias: bool = True, use_context: bool = False):
        super().__init__()
        self.cte = cte
        self.hvr_provider = hvr_provider
        self.use_bias = use_bias
        self.use_context = use_context

    def tick_chunk(self, observations: torch.Tensor) -> torch.Tensor:
        """Generate with HVR knowledge bias.

        1. Run CTE tick_chunk → base logits (B, C, vocab)
        2. For each position, query HVR for the previous token
        3. Add the HVR bias to the logits
        """
        # 1. CTE base logits.
        base_logits = self.cte.tick_chunk(observations)  # (B, C, vocab)

        if self.use_bias:
            B, C, V = base_logits.shape
            # 2. For each position, get the HVR bias based on the input token.
            for b in range(B):
                for c in range(C):
                    token_id = observations[b, c].item()
                    bias = self.hvr_provider.get_logit_bias(token_id).to(base_logits.device)
                    base_logits[b, c] += bias

        return base_logits

    def forward(self, ids: torch.Tensor) -> tuple:
        """Forward that returns (enhanced_logits, aux_loss)."""
        # Get the hidden state from the CTE (skip lm_head).
        self.cte._return_hidden = True
        hidden, aux = self.cte(ids) if hasattr(self.cte, '__call__') else (None, None)
        self.cte._return_hidden = False

        if hidden is None:
            # Fallback: use tick_chunk
            logits = self.tick_chunk(ids)
            return logits, torch.tensor(0.0)

        # Compute logits with HVR bias.
        logits = self.cte.lm_head(hidden)  # (B, L, vocab)

        if self.use_bias:
            B, L, V = logits.shape
            for b in range(B):
                for l in range(L):
                    token_id = ids[b, l].item() if l < ids.shape[1] else 0
                    bias = self.hvr_provider.get_logit_bias(token_id).to(logits.device)
                    logits[b, l] += bias

        return logits, aux if aux is not None else torch.tensor(0.0)
