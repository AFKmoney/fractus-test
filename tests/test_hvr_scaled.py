"""Fix HVR for 50k BPE vocab — the real engineering solution.

Problems identified:
  1. dim=10000 too small for 50257 vocab → cross-talk kills recall
  2. Random item memory → tokens have no semantic relationship
  3. Test corpus too small (448 tokens) → signal drowned in noise

Solutions:
  1. dim=50000 (HDC standard for large vocab)
  2. Binarize the model's trained embeddings (semantic item memory)
  3. Test with 10k+ tokens
"""
import sys, os, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
from fractus.hvr import HolographicMemory
from fractus.tokenizer import FractusTokenizer


def build_semantic_item_memory(model_ckpt_path, vocab_size=50257, dim=50000):
    """Build item memory from the trained model's token embeddings.

    The trained 88M has an embedding layer (50257, 768) that encodes
    semantic relationships between tokens. We binarize (+1/-1) these
    embeddings and pad/truncate to the HVR dimension.

    Advantage: tokens that are semantically similar (e.g., "def" and "function")
    will have similar HVR vectors, making recall more robust.
    """
    print(f"  Building semantic item memory from model embeddings...", flush=True)
    ck = torch.load(model_ckpt_path, weights_only=False, map_location="cpu")
    model_sd = ck["model_state"]

    # Find the embedding weight.
    embed_key = None
    for k in model_sd:
        if "tok_embed" in k or "embed.weight" in k:
            embed_key = k
            break

    if embed_key is None:
        print(f"  No embedding found, falling back to random")
        rng = np.random.RandomState(42)
        return (rng.randint(0, 2, (vocab_size, dim), dtype=np.int8) * 2 - 1)

    embed = model_sd[embed_key]  # (vocab, 768)
    print(f"  Source embedding: {embed.shape}")

    # Project to dim dimensions using random projection (Johnson-Lindenstrauss).
    rng = np.random.RandomState(42)
    proj = rng.randn(768, dim).astype(np.float32) / np.sqrt(768)
    projected = embed.numpy() @ proj  # (vocab, dim)

    # Binarize: sign function → +1 or -1.
    item_memory = np.sign(projected).astype(np.int8)
    item_memory[item_memory == 0] = 1  # replace zeros with +1

    print(f"  Semantic item memory: {item_memory.shape}, "
          f"semantic similarity preserved")
    return item_memory


class ScaledHVR:
    """HVR with semantic item memory + higher dimensionality."""

    def __init__(self, dim=50000, vocab_size=50257, item_memory=None, seed=42):
        self.dim = dim
        self.vocab_size = vocab_size

        if item_memory is not None:
            self.item_memory = item_memory
        else:
            rng = np.random.RandomState(seed)
            self.item_memory = (rng.randint(0, 2, (vocab_size, dim), dtype=np.int8) * 2 - 1)

        self.memory = np.zeros(dim, dtype=np.float32)
        # Don't pre-compute float32 — too much RAM at 50k dim × 50k vocab.
        # Compute similarity on-the-fly using int16 dot product.

    def encode_token(self, token_id):
        return self.item_memory[token_id].astype(np.float32)

    def bind(self, a, b):
        fa = np.fft.rfft(a)
        fb = np.fft.rfft(b)
        return np.fft.irfft(fa * fb, n=self.dim).astype(np.float32)

    def unbind(self, bound, key):
        fb = np.fft.rfft(bound)
        fk = np.fft.rfft(key)
        return np.fft.irfft(fb * np.conj(fk), n=self.dim).astype(np.float32)

    def learn_sequence(self, token_ids, weight=1.0):
        for i in range(len(token_ids) - 1):
            tok_vec = self.encode_token(token_ids[i])
            next_vec = self.encode_token(token_ids[i + 1])
            bound = self.bind(tok_vec, next_vec)
            self.memory += weight * bound

    def recall_next(self, token_id, top_k=10):
        tok_vec = self.encode_token(token_id)
        recalled = self.unbind(self.memory, tok_vec)
        recalled_norm = recalled / (np.linalg.norm(recalled) + 1e-10)
        # Similarity via int16 dot product (chunked to avoid OOM).
        sims = np.zeros(self.vocab_size, dtype=np.float32)
        chunk = 5000  # process 5000 vocab items at a time
        for start in range(0, self.vocab_size, chunk):
            end = min(start + chunk, self.vocab_size)
            block = self.item_memory[start:end].astype(np.float32)
            sims[start:end] = block @ recalled_norm
        top_idx = np.argsort(sims)[-top_k:][::-1]
        return [(int(i), float(sims[i])) for i in top_idx]


def test_scaled_hvr():
    """Test HVR with dim=50000 + semantic item memory + large corpus."""
    print("="*70)
    print("FIXED TEST: HVR dim=50000 + semantic embeddings + large corpus", flush=True)
    print("="*70)

    tok = FractusTokenizer.gpt2_compatible()

    # Build a MUCH larger corpus (repeat texts to simulate more data).
    base_texts = [
        "def fibonacci(n):\n    if n <= 1:\n        return n\n    return fibonacci(n-1) + fibonacci(n-2)\n",
        "def factorial(n):\n    if n <= 1:\n        return 1\n    return n * factorial(n-1)\n",
        "def quicksort(arr):\n    if len(arr) <= 1:\n        return arr\n    pivot = arr[0]\n    return quicksort([x for x in arr[1:] if x < pivot])\n",
        "def binary_search(arr, target):\n    low, high = 0, len(arr) - 1\n    while low <= high:\n        mid = (low + high) // 2\n        return mid\n    return -1\n",
        "Python is a programming language created by Guido van Rossum.\n",
        "Python is widely used for web development and data science.\n",
        "Fractus is a decentralized AI that runs on your machine.\n",
        "Fractus has continuous thought and persistent memory.\n",
    ]

    # Tokenize + repeat to build a larger corpus (5000+ tokens).
    all_tokens = []
    for text in base_texts:
        ids = tok.encode(text)
        all_tokens.extend(ids)

    # Repeat 20× to get ~5000 tokens.
    corpus = all_tokens * 20
    print(f"  Corpus: {len(corpus)} BPE tokens (base {len(all_tokens)} × 20)")
    print(f"  Unique tokens in corpus: {len(set(corpus))}")

    # Try multiple configurations.
    configs = [
        ("random dim=50000", 50000, None),
        ("random dim=100000", 100000, None),
    ]

    # Try semantic item memory if checkpoint is available.
    ckpt_path = "checkpoints/fractus_1b_latest.pt"
    if os.path.exists(ckpt_path):
        configs.append(("semantic dim=50000", 50000, "semantic"))
    else:
        print(f"  No model checkpoint at {ckpt_path}, skipping semantic test")

    for name, dim, item_type in configs:
        print(f"\n  --- Config: {name} ---")

        if item_type == "semantic":
            item_mem = build_semantic_item_memory(ckpt_path, vocab_size=50257, dim=dim)
            hvr = ScaledHVR(dim=dim, vocab_size=50257, item_memory=item_mem)
        else:
            hvr = ScaledHVR(dim=dim, vocab_size=50257)

        t0 = time.time()
        # 10 passes.
        for _ in range(10):
            hvr.learn_sequence(corpus)
        elapsed = time.time() - t0
        print(f"  Encoded in {elapsed:.1f}s ({len(corpus)*10/elapsed:.0f} tok/s)")

        # Test recall.
        test_words = ["def", " return", "Python", "Fractus"]
        correct = 0
        total = 0

        for word in test_words:
            token_ids = tok.encode(word)
            if not token_ids:
                continue
            token_id = token_ids[0]

            # Find actual next tokens.
            actual_nexts = set()
            for i in range(len(corpus) - 1):
                if corpus[i] == token_id:
                    actual_nexts.add(corpus[i + 1])

            if not actual_nexts:
                continue

            recalled = hvr.recall_next(token_id, top_k=10)
            recalled_ids = [t for t, _ in recalled]
            hits = actual_nexts.intersection(recalled_ids)

            actual_decoded = [tok.decode([t])[:10] for t in list(actual_nexts)[:3]]
            recalled_decoded = [tok.decode([t])[:10] for t in recalled_ids[:5]]

            print(f"    '{word}': actual={actual_decoded} recalled={recalled_decoded} hits={len(hits)}/{len(actual_nexts)}")

            if hits:
                correct += 1
            total += 1

        acc = correct / max(total, 1) * 100
        print(f"  Result: {correct}/{total} ({acc:.0f}%) — {'PASS ✓' if acc >= 75 else 'MARGINAL' if acc >= 50 else 'FAIL ✗'}")


if __name__ == "__main__":
    print("\n" + "#"*70)
    print("# HVR FIX: higher dim + semantic memory + larger corpus")
    print("#"*70 + "\n")
    test_scaled_hvr()
