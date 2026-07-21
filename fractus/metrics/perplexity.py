"""honest_perplexity: true perplexity = exp(validation loss).

True perplexity = exp(average cross-entropy on a validation dataset).
Not a proxy based on embedding norms.
"""

import math
import torch
import torch.nn as nn


def honest_perplexity(
    model: nn.Module,
    input_ids: torch.Tensor,
    target_ids: torch.Tensor,
) -> float:
    """Computes the true perplexity = exp(average CE loss).

    Args:
        model      : a model that takes input_ids and returns logits.
        input_ids  : (B, L) tensor of input ids.
        target_ids : (B, L) tensor of target ids (typically input shifted by 1).
    Returns:
        ppl : float >= 1.0. ppl = 1.0 = perfect prediction.
    """
    model.eval()
    with torch.no_grad():
        logits = model(input_ids)  # (B, L, vocab) or other depending on the model
        if isinstance(logits, tuple):
            logits = logits[0]  # some models return (logits, aux_loss)
        B, L, V = logits.shape
        ce = nn.functional.cross_entropy(
            logits.reshape(-1, V),
            target_ids.reshape(-1),
        )
    return math.exp(float(ce.item()))
