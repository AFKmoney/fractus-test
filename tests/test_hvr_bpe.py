"""CRITICAL TEST: HVR with real GPT-2 BPE tokenizer (50257 vocab).

This is the test that validates or kills the HVR approach.
With 50257 tokens in bipolar vectors of 10000 dim, the cross-talk
(noise from superposition) could destroy recall accuracy.

If this test passes, HVR is viable for Fractus 1B training.
If it fails, we need higher dim (50k+) or a different encoding.
"""
import sys, os, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from fractus.hvr import HolographicMemory
from fractus.tokenizer import FractusTokenizer


def test_hvr_bpe_recall():
    """Test HVR recall with real GPT-2 BPE tokens."""
    print("="*70)
    print("CRITICAL TEST: HVR with GPT-2 BPE (50257 vocab)")
    print("="*70)

    tok = FractusTokenizer.gpt2_compatible()
    print(f"  Tokenizer: GPT-2 BPE, vocab={tok.vocab_size}")

    # Build a real corpus — code + prose.
    corpus_texts = [
        # Python code
        "def fibonacci(n):\n    if n <= 1:\n        return n\n    return fibonacci(n-1) + fibonacci(n-2)\n",
        "def factorial(n):\n    if n <= 1:\n        return 1\n    return n * factorial(n-1)\n",
        "def quicksort(arr):\n    if len(arr) <= 1:\n        return arr\n    pivot = arr[0]\n    return quicksort([x for x in arr[1:] if x < pivot])\n",
        "def binary_search(arr, target):\n    low, high = 0, len(arr) - 1\n    while low <= high:\n        mid = (low + high) // 2\n        if arr[mid] == target:\n            return mid\n        elif arr[mid] < target:\n            low = mid + 1\n        else:\n            high = mid - 1\n    return -1\n",
        "def merge_sort(arr):\n    if len(arr) <= 1:\n        return arr\n    mid = len(arr) // 2\n    left = merge_sort(arr[:mid])\n    right = merge_sort(arr[mid:])\n    return merge(left, right)\n",
        # Prose
        "Python is a programming language created by Guido van Rossum.\n",
        "Python is widely used for web development and data science.\n",
        "Fractus is a decentralized AI that runs on your machine.\n",
        "Fractus has continuous thought and persistent memory.\n",
        "The key feature of Python is its readable and clean syntax.\n",
    ]

    # Tokenize all texts with GPT-2 BPE.
    all_tokens = []
    text_boundaries = []
    for text in corpus_texts:
        ids = tok.encode(text)
        text_boundaries.append((len(all_tokens), len(all_tokens) + len(ids)))
        all_tokens.extend(ids)

    print(f"  Corpus: {len(corpus_texts)} texts, {len(all_tokens)} BPE tokens")
    print(f"  Vocab IDs range: {min(all_tokens)}-{max(all_tokens)}")

    # Build HVR with FULL GPT-2 vocab (50257).
    print(f"  Building HVR (dim=10000, vocab=50257)...", flush=True)
    t0 = time.time()
    hm = HolographicMemory(dim=10_000, vocab_size=50257, seed=42)
    print(f"  Item memory built in {time.time()-t0:.1f}s ({hm.item_memory.nbytes/1e6:.0f}MB)")

    # Learn the corpus (5 passes to strengthen).
    t0 = time.time()
    for _ in range(5):
        hm.learn_sequence(all_tokens)
    elapsed = time.time() - t0
    print(f"  Encoded in {elapsed*1000:.0f}ms ({len(all_tokens)*5/elapsed:.0f} tok/s)")

    # Test recall on key transitions.
    print(f"\n  Recall test (BPE token transitions):")

    # Key token IDs from GPT-2 BPE.
    def_id = tok.encode("def")[0]
    return_id = tok.encode(" return")[0] if tok.encode(" return") else tok.encode("return")[0]
    python_id = tok.encode("Python")[0]
    fractus_id = tok.encode("Fractus")[0]
    if_id = tok.encode(" if")[0] if tok.encode(" if") else tok.encode("if")[0]

    test_tokens = [
        ("def", def_id),
        (" return", return_id),
        ("Python", python_id),
        ("Fractus", fractus_id),
    ]

    correct = 0
    total = 0

    for name, token_id in test_tokens:
        # Find what actually follows this token in the corpus.
        actual_nexts = set()
        for i in range(len(all_tokens) - 1):
            if all_tokens[i] == token_id:
                actual_nexts.add(all_tokens[i + 1])

        if not actual_nexts:
            print(f"  '{name}' (id={token_id}): not found in corpus")
            continue

        # Recall from HVR.
        recalled = hm.recall_next(token_id, top_k=10)
        recalled_ids = [tid for tid, _ in recalled]

        # Check overlap.
        hits = actual_nexts.intersection(recalled_ids)

        # Decode for display.
        actual_words = [tok.decode([t])[:15] for t in list(actual_nexts)[:3]]
        recalled_words = [tok.decode([t])[:15] for t in recalled_ids[:5]]

        print(f"  '{name}' (id={token_id}):")
        print(f"    actual next (first 3): {actual_words}")
        print(f"    recalled top-5: {recalled_words}")
        print(f"    hits: {len(hits)}/{len(actual_nexts)}")

        if hits:
            correct += 1
        total += 1

    print(f"\n  Recall accuracy: {correct}/{total} words had at least 1 correct hit in top-10")

    # Also test with higher dim to see if cross-talk is the issue.
    print(f"\n  Testing with dim=20000 (double) to check cross-talk...")
    hm2 = HolographicMemory(dim=20_000, vocab_size=50257, seed=42)
    for _ in range(5):
        hm2.learn_sequence(all_tokens)

    correct2 = 0
    for name, token_id in test_tokens:
        actual_nexts = set()
        for i in range(len(all_tokens) - 1):
            if all_tokens[i] == token_id:
                actual_nexts.add(all_tokens[i + 1])
        if not actual_nexts:
            continue
        recalled = hm2.recall_next(token_id, top_k=10)
        hits = actual_nexts.intersection([t for t, _ in recalled])
        if hits:
            correct2 += 1

    print(f"  Recall with dim=20000: {correct2}/{total}")

    # Verdict.
    acc = correct / max(total, 1) * 100
    print(f"\n  VERDICT: {'PASS ✓ — HVR works with BPE at 50k vocab' if acc >= 75 else 'MARGINAL — may need higher dim' if acc >= 50 else 'FAIL ✗ — too much cross-talk at 50k vocab'}")
    print(f"  dim=10000: {correct}/{total} | dim=20000: {correct2}/{total}")
    return acc


if __name__ == "__main__":
    print("\n" + "#"*70)
    print("# CRITICAL VALIDATION: HVR × GPT-2 BPE (50257 vocab)")
    print("#"*70 + "\n")
    test_hvr_bpe_recall()
