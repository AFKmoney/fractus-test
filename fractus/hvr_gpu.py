"""HVR-GPU: Holographic Vector Learning optimized for CUDA.

GPU speedup comes from:
  - torch.fft on GPU (cuFFT) instead of numpy FFT
  - Batched similarity: one big matmul (vocab × dim) instead of loop
  - Batched learning: process all token pairs at once with vectorized FFT
  - CUDA tensors (no CPU↔GPU transfer)

Expected: 30-100× faster than CPU numpy version.
"""
import torch
import numpy as np
from typing import List, Tuple


class HolographicMemoryGPU:
    """Holographic Reduced Representation memory — GPU-optimized.

    Same math as the CPU version, but using torch tensors on CUDA.
    The bottleneck operations (FFT, matmul) are 30-100× faster on GPU.
    """

    def __init__(self, dim: int = 10_000, vocab_size: int = 50257,
                 seed: int = 42, device: str = "cuda"):
        self.dim = dim
        self.vocab_size = vocab_size
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        if not torch.cuda.is_available() and device == "cuda":
            print(f"[HVR-GPU] CUDA not available, falling back to CPU")
            self.device = torch.device("cpu")

        torch.manual_seed(seed)

        # Item Memory: random bipolar vectors for each token (int8 = compact).
        # (vocab_size, dim) — stored on device.
        self.item_memory = (
            torch.randint(0, 2, (vocab_size, dim), device=self.device, dtype=torch.int8) * 2 - 1
        )  # int8: ±1

        # The global memory vector (float32 — accumulates sums).
        self.memory = torch.zeros(dim, device=self.device, dtype=torch.float32)

        # Pre-compute float32 version of item memory for similarity (lazily).
        self._item_memory_f32 = None

    @property
    def item_memory_f32(self):
        """Lazy float32 version of item memory for similarity computation."""
        if self._item_memory_f32 is None:
            self._item_memory_f32 = self.item_memory.float()
            # Pre-normalize rows for cosine similarity.
            norms = self._item_memory_f32.norm(dim=1, keepdim=True)  # (vocab, 1)
            self._item_memory_normalized = self._item_memory_f32 / (norms + 1e-10)
        return self._item_memory_normalized

    def encode_token(self, token_id: int) -> torch.Tensor:
        """Get the bipolar vector for a token id (float32, on device)."""
        return self.item_memory[token_id].float()

    def encode_tokens_batch(self, token_ids: torch.Tensor) -> torch.Tensor:
        """Get bipolar vectors for a batch of token ids.
        token_ids: (N,) long tensor
        Returns: (N, dim) float32 tensor
        """
        return self.item_memory[token_ids].float()  # (N, dim)

    # ===================================================================
    # Core HRR operations — GPU vectorized
    # ===================================================================

    def bind(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        """Bind via circular convolution using FFT (GPU).

        A ⊛ B = IFFT(FFT(A) * FFT(B))

        Works on single vectors (dim,) or batches (N, dim).
        """
        # rfft: real input → complex output of size dim//2+1.
        fa = torch.fft.rfft(a, dim=-1)
        fb = torch.fft.rfft(b, dim=-1)
        return torch.fft.irfft(fa * fb, n=self.dim, dim=-1)

    def bind_batch(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        """Batched binding: a and b are (N, dim) → result (N, dim)."""
        fa = torch.fft.rfft(a, dim=-1)  # (N, dim//2+1) complex
        fb = torch.fft.rfft(b, dim=-1)
        return torch.fft.irfft(fa * fb, n=self.dim, dim=-1)  # (N, dim)

    def unbind(self, bound: torch.Tensor, key: torch.Tensor) -> torch.Tensor:
        """Unbind: circular correlation.

        bound ⊛⁻¹ key = IFFT(conj(FFT(key)) * FFT(bound))
        """
        fb = torch.fft.rfft(bound, dim=-1)
        fk = torch.fft.rfft(key, dim=-1)
        return torch.fft.irfft(fb * torch.conj(fk), n=self.dim, dim=-1)

    # ===================================================================
    # High-level learning — VECTORIZED for GPU
    # ===================================================================

    def learn_sequence(self, token_ids: List[int] or torch.Tensor, weight: float = 1.0):
        """Encode a token sequence in ONE VECTORIZED pass.

        All adjacent pairs are bound simultaneously via batched FFT,
        then summed into memory. No Python loop over pairs.
        """
        if isinstance(token_ids, list):
            token_ids = torch.tensor(token_ids, dtype=torch.long, device=self.device)

        if len(token_ids) < 2:
            return

        # Create current/next pairs: (N-1, dim) each.
        current_vecs = self.encode_tokens_batch(token_ids[:-1])  # (N-1, dim)
        next_vecs = self.encode_tokens_batch(token_ids[1:])      # (N-1, dim)

        # Batched bind: all pairs at once.
        bound = self.bind_batch(current_vecs, next_vecs)  # (N-1, dim)

        # Superpose: sum all bound pairs into memory.
        self.memory += weight * bound.sum(dim=0)  # (dim,)

    def learn_corpus_chunked(self, token_ids: torch.Tensor, chunk_size: int = 10000,
                              weight: float = 1.0):
        """Encode a large corpus in chunks (avoids OOM on GPU).

        token_ids: (N,) long tensor of token ids.
        chunk_size: number of tokens per chunk (controls VRAM usage).
        """
        n = len(token_ids)
        for start in range(0, n - 1, chunk_size):
            end = min(start + chunk_size + 1, n)  # +1 for the overlap pair
            chunk = token_ids[start:end]
            self.learn_sequence(chunk, weight=weight)

    # ===================================================================
    # Recall — VECTORIZED for GPU
    # ===================================================================

    def recall_next(self, token_id: int, top_k: int = 5) -> List[Tuple[int, float]]:
        """Recall: what tokens follow this one?

        Uses one matmul (vocab × dim) to compute all similarities at once.
        """
        tok_vec = self.encode_token(token_id)  # (dim,)
        recalled = self.unbind(self.memory, tok_vec)  # (dim,)

        # Normalize recalled vector.
        recalled_norm = recalled / (recalled.norm() + 1e-10)  # (dim,)

        # Cosine similarity: one matmul (vocab, dim) @ (dim,) = (vocab,).
        sims = self.item_memory_normalized @ recalled_norm  # (vocab,)

        # Top-k.
        top_vals, top_idx = torch.topk(sims, top_k)
        return [(int(idx), float(val)) for idx, val in zip(top_idx.tolist(), top_vals.tolist())]

    def recall_next_batch(self, token_ids: torch.Tensor, top_k: int = 5) -> torch.Tensor:
        """Batched recall: for each token, get top-k next token IDs.

        token_ids: (N,) long tensor
        Returns: (N, top_k) long tensor of token IDs.
        """
        tok_vecs = self.encode_tokens_batch(token_ids)  # (N, dim)
        # Unbind each: memory ⊛⁻¹ tok_vec_i
        # Batch: conj(FFT(tok_vecs)) * FFT(memory)
        fm = torch.fft.rfft(self.memory, dim=-1)  # (dim//2+1,)
        ft = torch.fft.rfft(tok_vecs, dim=-1)     # (N, dim//2+1)
        recalled = torch.fft.irfft(ft * torch.conj(fm).unsqueeze(0), n=self.dim, dim=-1)  # (N, dim)

        # Normalize.
        norms = recalled.norm(dim=1, keepdim=True)  # (N, 1)
        recalled_norm = recalled / (norms + 1e-10)  # (N, dim)

        # Similarity: (N, dim) @ (dim, vocab) = (N, vocab).
        sims = recalled_norm @ self.item_memory_normalized.T  # (N, vocab)

        # Top-k per row.
        top_vals, top_idx = torch.topk(sims, top_k, dim=1)  # (N, top_k)
        return top_idx  # (N, top_k)

    # ===================================================================
    # Persistence
    # ===================================================================

    def save(self, path: str):
        """Save memory + item memory to disk (moves to CPU first)."""
        torch.save({
            "memory": self.memory.cpu(),
            "item_memory": self.item_memory.cpu(),
            "dim": self.dim,
            "vocab_size": self.vocab_size,
        }, path)

    def load(self, path: str):
        """Load from disk."""
        data = torch.load(path, weights_only=False)
        self.memory = data["memory"].to(self.device)
        self.item_memory = data["item_memory"].to(self.device)
        self._item_memory_f32 = None  # force recompute
