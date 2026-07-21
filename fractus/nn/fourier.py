"""Fourier basis with Mandelbrot decay for the fractal embedding.

Inspired by the original system (src/math/mandelbrot.rs + src/embedding.rs) but renamed
honestly: the original called these "Mandelbrot frequencies" in reference to the Mandelbrot
set, but it is really just a geometric decay of base φ2 (the square of the golden ratio).
We therefore call it the "Mandelbrot-decayed Fourier basis" — the decay is real and
justified (multi-level scale separation), but the link to the Mandelbrot set is nil.

Mathematics:
    φ = (1 + √5) / 2  ≈ 1.618
    φ2 ≈ 2.618
    ω_k = (φ2)^{-k}    for k = 0, 1, ..., n_freq-1

The Fourier basis associates with each token id t and each frequency k the pair
(sin, cos) of ω_k · t :
    M[t, 2k]   = sin(ω_k · t)
    M[t, 2k+1] = cos(ω_k · t)

We store n_freq frequencies; the produced matrix has 2·n_freq columns (sin+cos per
frequency). The caller (FractalEmbedding) handles the final projection.

NO trainable parameters here: everything is deterministic, computed once.
"""

import math
import torch


class MandelbrotFourierBasis:
    """Deterministic Fourier basis with (φ2)^{-k} decay.

    Attributes:
        vocab_size   : number of covered token ids (0 .. vocab_size-1)
        n_frequencies : number of frequencies ω_k
        frequencies  : tensor (n_frequencies,) of ω_k, in float32
    """

    def __init__(self, vocab_size: int, n_frequencies: int):
        if vocab_size <= 0 or n_frequencies <= 0:
            raise ValueError("vocab_size and n_frequencies must be > 0")
        self.vocab_size = vocab_size
        self.n_frequencies = n_frequencies

        phi = (1.0 + math.sqrt(5.0)) / 2.0
        phi_sq = phi * phi  # ≈ 2.618
        ks = torch.arange(n_frequencies, dtype=torch.float32)
        # ω_k = (φ2)^{-k}
        self.frequencies = phi_sq ** (-ks)

        # Precompute the matrix (vocab_size, 2·n_frequencies).
        self._matrix = self._build_matrix()

    def _build_matrix(self) -> torch.Tensor:
        """Builds the matrix M[t, :] = [sin(ω_k·t), cos(ω_k·t)] for all k."""
        t = torch.arange(self.vocab_size, dtype=torch.float32).unsqueeze(1)  # (V, 1)
        omega = self.frequencies.unsqueeze(0)  # (1, K)
        phases = omega * t  # (V, K) broadcast
        sin_part = torch.sin(phases)  # (V, K)
        cos_part = torch.cos(phases)  # (V, K)
        # Interleave sin/cos: columns 0,2,4,... = sin; 1,3,5,... = cos
        M = torch.empty(self.vocab_size, 2 * self.n_frequencies, dtype=torch.float32)
        M[:, 0::2] = sin_part
        M[:, 1::2] = cos_part
        return M

    def matrix(self) -> torch.Tensor:
        """Returns the precomputed matrix (vocab_size, 2·n_frequencies)."""
        return self._matrix
