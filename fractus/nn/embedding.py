"""FractalEmbedding: a trainable fractal codepoint embedding.

Combines three feature sources for each token id t:

    (A) 16 deterministic morphological features (CharClassFeatures)
    (B) Mandelbrot-decayed Fourier basis (MandelbrotFourierBasis)
    (C) Vortex conditioning: a 2-adic hash (Collatz, computed in Rust,
        outside the autodiff graph) is projected into phases via a trainable
        MLP (PyTorch, in the graph). The 2-adic vortex influences learning
        without pretending to be differentiable.

The final projection to d_model is a trainable nn.Linear. The entire forward
pass is differentiable end-to-end. The deterministic parts (A, B, and the
hash of C) are precomputed as buffers outside the graph; only the MLP of C
and the final projection carry trainable parameters.
"""

import torch
import torch.nn as nn

from .char_features import CharClassFeatures
from .fourier import MandelbrotFourierBasis


class FractalEmbedding(nn.Module):
    """Trainable fractal embedding.

    Args:
        vocab_size:    number of token ids covered.
        d_model:       output dimension.
        n_frequencies: number of frequencies for the Fourier basis.
        vortex_hidden: width of the MLP that projects the Collatz hash into phases.
        collatz_steps: number of Collatz iterations for the hash (deterministic).
    """

    def __init__(
        self,
        vocab_size: int,
        d_model: int,
        n_frequencies: int = 16,
        vortex_hidden: int = 32,
        collatz_steps: int = 7,
    ):
        super().__init__()
        if vocab_size <= 0 or d_model <= 0:
            raise ValueError("vocab_size and d_model must be > 0")

        self.vocab_size = vocab_size
        self.d_model = d_model
        self.collatz_steps = collatz_steps

        # (A) Morphological features: deterministic precomputation, outside graph.
        char_matrix = torch.stack(
            [CharClassFeatures.extract(t) for t in range(vocab_size)], dim=0
        )
        self.register_buffer("char_features", char_matrix)

        # (B) Mandelbrot-decayed Fourier basis: deterministic precomputation.
        self.fourier = MandelbrotFourierBasis(vocab_size, n_frequencies)
        fourier_matrix = self.fourier.matrix()
        self.register_buffer("fourier_features", fourier_matrix)

        # (C) Vortex conditioning: Collatz hash precomputed (outside graph),
        # then projected by a trainable MLP (in the graph).
        try:
            from fractus import _core
        except ImportError as e:
            raise ImportError(
                "fractus._core not found. Run `maturin develop`."
            ) from e
        hashes = torch.tensor(
            [_core.collatz_hash(t, collatz_steps) for t in range(vocab_size)],
            dtype=torch.float32,
        )
        max_h = hashes.max().item() + 1.0
        hashes_norm = hashes / max_h
        self.register_buffer("vortex_hashes", hashes_norm)

        self.vortex_phase_dim = vortex_hidden
        self.vortex_mlp = nn.Sequential(
            nn.Linear(1, vortex_hidden),
            nn.Tanh(),
            nn.Linear(vortex_hidden, vortex_hidden),
        )

        in_dim = 16 + fourier_matrix.shape[1] + vortex_hidden
        self.proj = nn.Linear(in_dim, d_model)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        """token_ids: (N,) or (N, L) of integers in [0, vocab_size).

        Returns (N, d_model) or (N, L, d_model).
        """
        if token_ids.max() >= self.vocab_size or token_ids.min() < 0:
            raise IndexError(
                f"token_id outside [0, {self.vocab_size}): "
                f"min={int(token_ids.min())}, max={int(token_ids.max())}"
            )

        original_shape = token_ids.shape
        flat = token_ids.reshape(-1)

        char = self.char_features[flat]
        fourier = self.fourier_features[flat]

        h = self.vortex_hashes[flat].unsqueeze(1)
        vortex_phases = self.vortex_mlp(h)

        x = torch.cat([char, fourier, vortex_phases], dim=1)
        out = self.proj(x)

        return out.reshape(*original_shape, self.d_model)
