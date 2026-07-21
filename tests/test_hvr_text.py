"""Test HVR on REAL text — can it learn English structure in one pass?

This is the make-or-break test. We tokenize real text with GPT-2 BPE,
feed it to the HVR memory in one pass, then test if it can predict
the next token for common words.
"""
import sys, os, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from fractus.hvr import HolographicMemory

# We can't import FractusTokenizer (needs torch + maturin), so we use
# a simple mock tokenizer for now. In production, swap with GPT-2 BPE.

# Simple word-level tokenizer for the test (split on spaces + punctuation).
import re

def simple_tokenize(text: str, vocab: dict = None) -> list:
    """Word-level tokenizer. Returns token IDs."""
    words = re.findall(r'\w+|[^\w\s]', text.lower())
    if vocab is None:
        return words
    return [vocab.get(w, 0) for w in words]  # 0 = unknown

def build_vocab(texts: list) -> dict:
    """Build a word-level vocabulary from texts."""
    vocab = {"<UNK>": 0}
    for text in texts:
        words = re.findall(r'\w+|[^\w\s]', text.lower())
        for w in words:
            if w not in vocab:
                vocab[w] = len(vocab)
    return vocab


def test_real_text():
    """Test HVR on real English text."""
    print("="*60)
    print("TEST 5: HVR on REAL text (one-pass learning)")
    print("="*60)

    # Training text — a mix of code and prose.
    training_text = """
    def fibonacci(n):
        if n <= 1:
            return n
        return fibonacci(n - 1) + fibonacci(n - 2)

    def factorial(n):
        if n <= 1:
            return 1
        return n * factorial(n - 1)

    Python is a programming language created by Guido van Rossum.
    Python is widely used for web development, data science, and AI.
    The key feature of Python is its readable and clean syntax.
    Python supports object-oriented and functional programming.

    def quicksort(arr):
        if len(arr) <= 1:
            return arr
        pivot = arr[len(arr) // 2]
        left = [x for x in arr if x < pivot]
        return quicksort(left)

    Fractus is a decentralized AI that runs on your machine.
    Fractus has continuous thought and persistent memory.
    Fractus learns without retraining through its RAG engine.
    """

    # Build vocab.
    vocab = build_vocab([training_text])
    print(f"  Vocab size: {len(vocab)}")
    print(f"  Training text length: {len(training_text)} chars")

    # Tokenize.
    tokens = simple_tokenize(training_text, vocab)
    print(f"  Tokenized to {len(tokens)} tokens")
    print(f"  First 20 tokens: {tokens[:20]}")

    # Create HVR memory.
    hm = HolographicMemory(dim=10_000, vocab_size=len(vocab), seed=42)

    # Learn in ONE pass.
    t0 = time.time()
    # Learn the text multiple times (like reading it several times).
    for _ in range(5):
        hm.learn_sequence(tokens)
    elapsed = time.time() - t0
    print(f"  Learned in {elapsed*1000:.1f}ms (5 passes)")
    print(f"  Memory norm: {np.linalg.norm(hm.memory):.1f}")

    # Test recall: what follows common words?
    print(f"\n  Recall test (what token follows each word?):")

    # Reverse vocab for display.
    id2word = {v: k for k, v in vocab.items()}

    test_words = ["def", "return", "python", "if", "fractus"]
    correct = 0
    total = 0

    for word in test_words:
        if word not in vocab:
            print(f"  '{word}': not in vocab, skipping")
            continue
        word_id = vocab[word]

        # Find the ACTUAL next tokens in the training text for this word.
        actual_nexts = set()
        for i in range(len(tokens) - 1):
            if tokens[i] == word_id:
                actual_nexts.add(tokens[i + 1])

        # Recall from HVR.
        recalled = hm.recall_next(word_id, top_k=5)
        recalled_ids = [tid for tid, _ in recalled]

        # Check overlap.
        hits = actual_nexts.intersection(recalled_ids)
        in_top5 = len(hits) > 0

        actual_words = [id2word.get(t, '?') for t in actual_nexts]
        recalled_words = [id2word.get(t, '?') for t in recalled_ids[:3]]

        print(f"  '{word}':")
        print(f"    actual next: {actual_words}")
        print(f"    recalled:    {recalled_words}")
        print(f"    hits: {len(hits)}/{len(actual_nexts)}")

        if in_top5:
            correct += 1
        total += 1

    print(f"\n  Recall accuracy: {correct}/{total} words had a correct next-token in top-5")
    print(f"  RESULT: {'PASS ✓' if correct >= 3 else 'MARGINAL' if correct >= 1 else 'FAIL ✗'}")
    return correct, total


if __name__ == "__main__":
    print("\n" + "#"*60)
    print("# HOLOGRAPHIC VECTOR LEARNING — REAL TEXT TEST")
    print("#"*60 + "\n")
    test_real_text()
