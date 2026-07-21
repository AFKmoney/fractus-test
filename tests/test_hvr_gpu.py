"""Benchmark: HVR-GPU vs HVR-CPU speed.

Tests:
1. Equivalence: GPU and CPU produce the same recall results
2. Speed: encoding throughput (tok/s) on GPU vs CPU
3. Scaling: 1k, 10k, 100k, 1M tokens
4. Extrapolation: how long for 21B tokens on GPU
"""
import sys, os, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch

from fractus.hvr import HolographicMemory       # CPU version
from fractus.hvr_gpu import HolographicMemoryGPU  # GPU version


def test_equivalence():
    """Verify GPU and CPU give the same recall results."""
    print("="*60)
    print("TEST: GPU vs CPU equivalence")
    print("="*60)

    if not torch.cuda.is_available():
        print("  SKIP — no CUDA available")
        return False

    # CPU version.
    hm_cpu = HolographicMemory(dim=10_000, vocab_size=100, seed=42)
    seq = [1, 2, 3, 4, 5]
    for _ in range(10):
        hm_cpu.learn_sequence(seq)

    # GPU version — same seed.
    hm_gpu = HolographicMemoryGPU(dim=10_000, vocab_size=100, seed=42)
    for _ in range(10):
        hm_gpu.learn_sequence(seq)

    # Compare recall for token 1.
    cpu_recall = hm_cpu.recall_next(1, top_k=5)
    gpu_recall = hm_gpu.recall_next(1, top_k=5)

    cpu_ids = [t for t, _ in cpu_recall]
    gpu_ids = [t for t, _ in gpu_recall]

    print(f"  CPU recall for token 1: {cpu_ids}")
    print(f"  GPU recall for token 1: {gpu_ids}")
    match = cpu_ids == gpu_ids
    print(f"  Match: {'YES ✓' if match else 'NO ✗'}")
    return match


def benchmark_speed():
    """Benchmark encoding throughput on GPU vs CPU."""
    print("\n" + "="*60)
    print("BENCHMARK: HVR Encoding Speed")
    print("="*60)

    has_gpu = torch.cuda.is_available()
    print(f"  GPU available: {has_gpu}")
    if has_gpu:
        print(f"  GPU: {torch.cuda.get_device_name(0)}")

    sizes = [1000, 10000, 100000, 1000000]
    vocab = 50257

    for n_tokens in sizes:
        rng = np.random.RandomState(0)
        tokens = rng.randint(0, min(vocab, 5000), size=n_tokens).tolist()

        # CPU.
        hm_cpu = HolographicMemory(dim=10_000, vocab_size=min(vocab, 5000), seed=42)
        t0 = time.time()
        hm_cpu.learn_sequence(tokens)
        cpu_time = time.time() - t0
        cpu_rate = n_tokens / cpu_time if cpu_time > 0 else float('inf')

        line = f"  {n_tokens:>8} tokens — CPU: {cpu_time*1000:>8.1f}ms ({cpu_rate:>8.0f} tok/s)"

        # GPU.
        if has_gpu:
            torch.cuda.empty_cache()
            hm_gpu = HolographicMemoryGPU(dim=10_000, vocab_size=min(vocab, 5000), seed=42)
            tok_tensor = torch.tensor(tokens, dtype=torch.long, device=hm_gpu.device)

            # Warmup.
            hm_gpu.learn_sequence(tok_tensor[:100])
            hm_gpu.memory.zero_()

            torch.cuda.synchronize() if has_gpu else None
            t0 = time.time()
            hm_gpu.learn_corpus_chunked(tok_tensor, chunk_size=50000)
            if has_gpu:
                torch.cuda.synchronize()
            gpu_time = time.time() - t0
            gpu_rate = n_tokens / gpu_time if gpu_time > 0 else float('inf')

            line += f" | GPU: {gpu_time*1000:>8.1f}ms ({gpu_rate:>8.0f} tok/s) | speedup: {gpu_rate/cpu_rate:.1f}×"

        print(line)

    # Extrapolation.
    print("\n  Extrapolation to 21B tokens:")
    if has_gpu:
        # Use the 1M token rate as reference (most stable).
        hm_gpu2 = HolographicMemoryGPU(dim=10_000, vocab_size=5000, seed=42)
        tokens_1m = torch.randint(0, 5000, (1_000_000,), device=hm_gpu2.device)
        hm_gpu2.learn_sequence(tokens_1m[:100])  # warmup
        hm_gpu2.memory.zero_()
        if has_gpu: torch.cuda.synchronize()
        t0 = time.time()
        hm_gpu2.learn_corpus_chunked(tokens_1m, chunk_size=50000)
        if has_gpu: torch.cuda.synchronize()
        gpu_rate = 1_000_000 / (time.time() - t0)
        time_21b_gpu = 21e9 / gpu_rate / 3600 / 24
        cost_gpu = time_21b_gpu * 24 * 0.50  # ~$0.50/hr GPU
        print(f"    GPU ({gpu_rate:.0f} tok/s): {time_21b_gpu:.1f} days, ~${cost_gpu:.0f}")

    # CPU reference.
    hm_cpu2 = HolographicMemory(dim=10_000, vocab_size=5000, seed=42)
    tokens_100k = np.random.RandomState(0).randint(0, 5000, 100_000).tolist()
    t0 = time.time()
    hm_cpu2.learn_sequence(tokens_100k)
    cpu_rate = 100_000 / (time.time() - t0)
    time_21b_cpu = 21e9 / cpu_rate / 3600 / 24
    print(f"    CPU 1-thread ({cpu_rate:.0f} tok/s): {time_21b_cpu:.0f} days")
    print(f"    CPU 176 threads (est.): {time_21b_cpu/176*5:.1f} days")


def test_recall_accuracy_scale():
    """Test recall accuracy at scale (100k tokens)."""
    print("\n" + "="*60)
    print("TEST: Recall accuracy at scale (100k tokens)")
    print("="*60)

    # Generate a pseudo-language with strong transition patterns.
    # Pattern: [A, B, C, D, E] repeated with variations.
    rng = np.random.RandomState(42)
    vocab = 501
    tokens = []
    patterns = [
        [1, 2, 3, 4, 5],
        [10, 20, 30, 40, 50],
        [100, 200, 300, 400, 500],
        [1, 11, 21, 31, 41],
    ]
    for _ in range(20000):  # 100k tokens total
        p = rng.choice(len(patterns))
        tokens.extend(patterns[p])

    print(f"  Corpus: {len(tokens)} tokens, {vocab} vocab")
    print(f"  Patterns: 4 repeating sequences")

    has_gpu = torch.cuda.is_available()

    if has_gpu:
        hm = HolographicMemoryGPU(dim=10_000, vocab_size=vocab, seed=42)
        tok_tensor = torch.tensor(tokens, dtype=torch.long, device=hm.device)
        t0 = time.time()
        hm.learn_corpus_chunked(tok_tensor, chunk_size=50000)
        torch.cuda.synchronize()
        elapsed = time.time() - t0
        print(f"  Encoded in {elapsed*1000:.1f}ms ({len(tokens)/elapsed:.0f} tok/s)")
    else:
        hm = HolographicMemory(dim=10_000, vocab_size=vocab, seed=42)
        t0 = time.time()
        hm.learn_sequence(tokens)
        elapsed = time.time() - t0
        print(f"  Encoded in {elapsed*1000:.1f}ms ({len(tokens)/elapsed:.0f} tok/s)")

    # Test recall on all pattern transitions.
    correct = 0
    total = 0
    test_cases = [(1,2), (2,3), (3,4), (4,5),
                  (10,20), (20,30), (30,40), (40,50),
                  (100,200), (200,300), (300,400), (400,500),
                  (1,11), (11,21), (21,31), (31,41)]

    for token, expected in test_cases:
        recalled = hm.recall_next(token, top_k=5)
        in_top5 = expected in [t for t, _ in recalled]
        if in_top5:
            correct += 1
        total += 1

    acc = correct / total * 100
    print(f"  Recall accuracy: {correct}/{total} ({acc:.0f}%)")
    print(f"  RESULT: {'PASS ✓' if acc >= 75 else 'MARGINAL' if acc >= 50 else 'FAIL ✗'}")
    return acc


if __name__ == "__main__":
    print("\n" + "#"*60)
    print("# HVR-GPU BENCHMARK")
    print("#"*60 + "\n")

    test_equivalence()
    benchmark_speed()
    test_recall_accuracy_scale()
