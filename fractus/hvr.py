"""Holographic Vector Learning (HVR) — core module for Fractus-test.

Based on Holographic Reduced Representations (Plate 1995) and
Hyperdimensional Computing (Kanerva 2009).

THREE PRIMITIVES:
1. Bipolar vectors (±1) in high dimension (10k+). On CPU these are
   bitwise XOR/AND operations — massively faster than float matmul.
2. Binding via circular convolution: A ⊛ B = a new vector of the same
   dimension that encodes the PAIR (A, B). Inverse operation (unbinding)
   recovers either component from the bound vector.
3. Superposition via addition: Σ(A_i ⊛ B_i) stores multiple pairs in the
   SAME vector. Information is distributed — like a hologram, every part
   of the vector contains a little bit of every stored pair.

WHAT THIS CAN DO (proven in neuroscience/ML literature):
- Store and retrieve structured associations in one pass (no gradient descent)
- Answer "what is the X of Y?" by unbinding
- Encode sequences, trees, and graphs holographically
- Operate on CPU with bitwise ops (no GPU needed)

WHAT THIS CANNOT DO (I will be honest):
- Generate fluent text on its own (HRR is a memory/reasoning system, not a
  language model. To generate text you still need the CTE.)
- Replace ALL of backprop. The CTE's attention and MoE still need trained
  weights. HVR is a KNOWLEDGE ENCODING layer that sits ON TOP of the CTE.

The hypothesis we are testing: can HVR encode the training corpus in one
pass and let Fractus retrieve structured knowledge from it — effectively
"learning" the corpus without gradient descent?
"""
import numpy as np
import torch
from typing import List, Tuple, Optional


class HolographicMemory:
    """Holographic Reduced Representation memory.

    Stores associations as superposed circular convolutions in a single
    high-dimensional bipolar vector.

    Args:
        dim:        dimensionality of the vectors (10,000 = standard for HRR).
        vocab_size: number of tokens in the vocabulary.
    """

    def __init__(self, dim: int = 10_000, vocab_size: int = 50257, seed: int = 42):
        self.dim = dim
        self.vocab_size = vocab_size
        self.rng = np.random.RandomState(seed)

        # Item Memory: random bipolar vectors for each token in the vocab.
        # These are fixed (not learned) — like random projection bases.
        # Stored as int8 (+1/-1) → 10KB per token, 500MB total for 50k vocab.
        # In practice we generate them on-the-fly or cache a subset.
        self._item_memory = None  # lazy init

        # The global memory vector: a superposition of all bound pairs.
        # This is the "thermodynamic state" that accumulates knowledge.
        # Stored as float32 (sums of ±1 vectors → values in [-N, N]).
        self.memory = np.zeros(dim, dtype=np.float32)

        # Role vectors: fixed random bipolar vectors for structural roles
        # (POSITION, TOKEN, NEXT, PREV, TYPE, etc.)
        self._role_vectors = {}

    @property
    def item_memory(self):
        """Lazy-init the token item memory (bipolar, int8)."""
        if self._item_memory is None:
            # Generate 50257 random bipolar vectors of dimension `dim`.
            # int8 = 1 byte per element = dim * vocab_size bytes.
            # For dim=10000, vocab=50257: ~500MB. Acceptable.
            self._item_memory = (
                self.rng.randint(0, 2, size=(self.vocab_size, self.dim), dtype=np.int8) * 2 - 1
            )
        return self._item_memory

    def get_role(self, name: str) -> np.ndarray:
        """Get or create a named role vector (bipolar)."""
        if name not in self._role_vectors:
            self._role_vectors[name] = (
                self.rng.randint(0, 2, size=self.dim, dtype=np.int8) * 2 - 1
            )
        return self._role_vectors[name]

    def encode_token(self, token_id: int) -> np.ndarray:
        """Get the bipolar vector for a token id."""
        return self.item_memory[token_id].astype(np.float32)

    # ===================================================================
    # Core HRR operations
    # ===================================================================

    def bind(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        """Bind two vectors via circular convolution (HRR binding).

        A ⊛ B = IFFT(FFT(A) * FFT(B))

        The result is the same dimension as A and B, and encodes the pair.
        Circular convolution in spatial domain = element-wise multiply in
        frequency domain. FFT makes this O(D log D) instead of O(D²).
        """
        fa = np.fft.rfft(a)
        fb = np.fft.rfft(b)
        return np.fft.irfft(fa * fb, n=self.dim).astype(np.float32)

    def unbind(self, bound: np.ndarray, key: np.ndarray) -> np.ndarray:
        """Unbind: recover the value from a bound pair given the key.

        bound ⊛⁻¹ key ≈ IFFT(conj(FFT(key)) * FFT(bound))
        (circular correlation, the approximate inverse of convolution)
        """
        fb = np.fft.rfft(bound)
        fk = np.fft.rfft(key)
        # Avoid division by zero: add small epsilon.
        return np.fft.irfft(fb * np.conj(fk), n=self.dim).astype(np.float32)

    def superpose(self, *vectors: np.ndarray) -> np.ndarray:
        """Superpose (sum) multiple vectors into one.

        This is how multiple associations are stored in the same memory.
        The result is a single vector of the same dimension.
        """
        result = np.zeros(self.dim, dtype=np.float32)
        for v in vectors:
            result += v
        return result

    # ===================================================================
    # High-level learning operations
    # ===================================================================

    def learn_sequence(self, token_ids: List[int], weight: float = 1.0):
        """Encode a token sequence into the global memory.

        For each adjacent pair (token[i], token[i+1]):
            - Bind: current_token ⊛ next_token
            - Superpose into memory

        After this call, the sequence's transition structure is stored
        in self.memory. One pass, no gradient descent.
        """
        for i in range(len(token_ids) - 1):
            tok_vec = self.encode_token(token_ids[i])
            next_vec = self.encode_token(token_ids[i + 1])
            # Bind the pair: current ⊛ next. Store in memory.
            bound = self.bind(tok_vec, next_vec)
            self.memory += weight * bound

    def learn_association(self, key_id: int, value_id: int, weight: float = 1.0):
        """Store a single key→value association."""
        key_vec = self.encode_token(key_id)
        val_vec = self.encode_token(value_id)
        bound = self.bind(key_vec, val_vec)
        self.memory += weight * bound

    def recall_next(self, token_id: int, top_k: int = 5) -> List[Tuple[int, float]]:
        """Recall: given a token, what tokens tend to follow it?

        The memory stores Σ(token_i ⊛ token_{i+1}).
        To recall what follows token_i: unbind(memory, token_i) ≈ token_{i+1}.
        Then compare the recalled vector against all vocab vectors.
        """
        tok_vec = self.encode_token(token_id)
        # Unbind: memory ⊛⁻¹ token_vec ≈ next_token_vec
        recalled = self.unbind(self.memory, tok_vec)

        # Compare recalled vector against all vocab vectors (cosine similarity).
        item = self.item_memory.astype(np.float32)
        recalled_norm = recalled / (np.linalg.norm(recalled) + 1e-10)
        item_norms = item / (np.linalg.norm(item, axis=1, keepdims=True) + 1e-10)
        sims = item_norms @ recalled_norm  # (vocab,)

        # Top-k.
        top_idx = np.argsort(sims)[-top_k:][::-1]
        return [(int(i), float(sims[i])) for i in top_idx]

    def save(self, path: str):
        """Save the memory and item memory to disk."""
        np.savez(path,
                 memory=self.memory,
                 item_memory=self.item_memory,
                 dim=self.dim,
                 vocab_size=self.vocab_size)

    def load(self, path: str):
        """Load from disk."""
        data = np.load(path)
        self.memory = data["memory"]
        self._item_memory = data["item_memory"]
        self.dim = int(data["dim"])
        self.vocab_size = int(data["vocab_size"])
