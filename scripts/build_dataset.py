#!/usr/bin/env python
"""Build a large multi-domain dataset for Fractus-1B training.

Streams and filters datasets from HuggingFace, tokenizes with BPE,
and writes to a local shard cache for fast training loops.

Domains:
    - Code (Python, JS, C++, Rust, Go) — 40%
    - Universal knowledge (Wikipedia, science, history) — 30%
    - Math (problems + solutions) — 15%
    - Books / literature — 15%

Usage:
    python scripts/build_dataset.py --max-tokens 500000
    python scripts/build_dataset.py --max-tokens 2000000
"""

import argparse
import os
import sys
import json
import random

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from fractus.tokenizer import FractusTokenizer


# Datasets by domain. (name, config, field, domain, weight)
DATASET_CONFIG = [
    # === CODE (40%) ===
    ("codeparrot/github-code", None, "code", "code", 0.12),
    ("code_search_net", "python", "func_code_string", "code", 0.06),
    ("code_search_net", "javascript", "func_code_string", "code", 0.04),
    ("code_search_net", "go", "func_code_string", "code", 0.04),
    ("code_search_net", "rust", "func_code_string", "code", 0.04),
    ("codeparrot/github-code", None, "code", "code", 0.10),

    # === UNIVERSAL KNOWLEDGE (30%) ===
    ("wikitext", "wikitext-103-raw-v1", "text", "knowledge", 0.15),
    ("scientific_papers", "arxiv", "abstract", "science", 0.08),
    ("scientific_papers", "pubmed", "abstract", "science", 0.07),

    # === MATH (15%) ===
    ("meta-math/MetaMathQA", None, "query", "math", 0.04),
    ("meta-math/MetaMathQA", None, "response", "math", 0.06),
    ("hendrycks/competition_math", None, "problem", "math", 0.025),
    ("hendrycks/competition_math", None, "solution", "math", 0.025),

    # === BOOKS / LITERATURE (15%) ===
    ("tiny_shakespeare", None, "text", "literature", 0.05),
    ("wikitext", "wikitext-2-raw-v1", "text", "literature", 0.05),
    ("bookcorpus", None, "text", "literature", 0.05),
]


# Inline high-quality code samples (always available as fallback).
INLINE_CODE = [
    "def fibonacci(n):\n    if n <= 1: return n\n    a, b = 0, 1\n    for _ in range(n-1):\n        a, b = b, a + b\n    return b\n",
    "def quicksort(arr):\n    if len(arr) <= 1: return arr\n    pivot = arr[len(arr)//2]\n    left = [x for x in arr if x < pivot]\n    middle = [x for x in arr if x == pivot]\n    right = [x for x in arr if x > pivot]\n    return quicksort(left) + middle + quicksort(right)\n",
    "class Node:\n    def __init__(self, value):\n        self.value = value\n        self.left = None\n        self.right = None\n\nclass BST:\n    def __init__(self):\n        self.root = None\n    def insert(self, value):\n        if not self.root:\n            self.root = Node(value)\n        else:\n            self._insert(self.root, value)\n    def _insert(self, node, value):\n        if value < node.value:\n            if node.left: self._insert(node.left, value)\n            else: node.left = Node(value)\n        else:\n            if node.right: self._insert(node.right, value)\n            else: node.right = Node(value)\n    def search(self, value):\n        return self._search(self.root, value)\n    def _search(self, node, value):\n        if not node: return False\n        if node.value == value: return True\n        if value < node.value: return self._search(node.left, value)\n        return self._search(node.right, value)\n",
    "def binary_search(arr, target):\n    low, high = 0, len(arr) - 1\n    while low <= high:\n        mid = (low + high) // 2\n        if arr[mid] == target: return mid\n        elif arr[mid] < target: low = mid + 1\n        else: high = mid - 1\n    return -1\n",
    "import math\ndef is_prime(n):\n    if n < 2: return False\n    for i in range(2, int(math.sqrt(n)) + 1):\n        if n % i == 0: return False\n    return True\n\ndef prime_sieve(limit):\n    sieve = [True] * (limit + 1)\n    sieve[0] = sieve[1] = False\n    for i in range(2, int(math.sqrt(limit)) + 1):\n        if sieve[i]:\n            for j in range(i*i, limit+1, i):\n                sieve[j] = False\n    return [i for i in range(limit+1) if sieve[i]]\n",
    "def merge_sort(arr):\n    if len(arr) <= 1: return arr\n    mid = len(arr) // 2\n    left = merge_sort(arr[:mid])\n    right = merge_sort(arr[mid:])\n    return merge(left, right)\n\ndef merge(left, right):\n    result, i, j = [], 0, 0\n    while i < len(left) and j < len(right):\n        if left[i] <= right[j]:\n            result.append(left[i]); i += 1\n        else:\n            result.append(right[j]); j += 1\n    result.extend(left[i:])\n    result.extend(right[j:])\n    return result\n",
    "def dijkstra(graph, start):\n    distances = {node: float('inf') for node in graph}\n    distances[start] = 0\n    visited = set()\n    while len(visited) < len(graph):\n        current = min((n for n in graph if n not in visited),\n                      key=lambda n: distances[n])\n        visited.add(current)\n        for neighbor, weight in graph[current]:\n            distance = distances[current] + weight\n            if distance < distances[neighbor]:\n                distances[neighbor] = distance\n    return distances\n",
    "class NeuralNetwork:\n    def __init__(self, input_size, hidden_size, output_size):\n        self.W1 = [[random.random() * 0.1 for _ in range(hidden_size)]\n                   for _ in range(input_size)]\n        self.W2 = [[random.random() * 0.1 for _ in range(output_size)]\n                   for _ in range(hidden_size)]\n    def forward(self, X):\n        hidden = self._relu([sum(X[j] * self.W1[j][k] for j in range(len(X)))\n                             for k in range(len(self.W1[0]))])\n        output = [sum(hidden[j] * self.W2[j][k] for j in range(len(hidden)))\n                  for k in range(len(self.W2[0]))]\n        return output\n    def _relu(self, x):\n        return [max(0, v) for v in x]\n",
    "import socket\n\ndef create_server(host='localhost', port=8080):\n    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)\n    server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)\n    server.bind((host, port))\n    server.listen(5)\n    print(f'Server listening on {host}:{port}')\n    while True:\n        client, addr = server.accept()\n        data = client.recv(1024)\n        client.sendall(b'HTTP/1.1 200 OK\\r\\n\\r\\nHello World')\n        client.close()\n",
    "def AES_encrypt(plaintext, key):\n    # Simplified AES round structure (educational)\n    state = [list(plaintext[i:i+4]) for i in range(0, 16, 4)]\n    round_keys = key_expansion(key)\n    add_round_key(state, round_keys[0])\n    for i in range(1, 10):\n        sub_bytes(state)\n        shift_rows(state)\n        mix_columns(state)\n        add_round_key(state, round_keys[i])\n    sub_bytes(state)\n    shift_rows(state)\n    add_round_key(state, round_keys[10])\n    return flatten(state)\n",
]

# Universal knowledge text (always available as fallback).
UNIVERSAL_KNOWLEDGE = """
The speed of light in vacuum is approximately 299,792,458 meters per second.
This is a fundamental constant of nature, denoted by the letter c.

The theory of relativity, proposed by Albert Einstein in 1905, states that
the laws of physics are the same for all non-accelerating observers, and
that the speed of light in a vacuum is independent of the motion of all observers.

Quantum mechanics describes the behavior of matter at the atomic and subatomic level.
The Heisenberg uncertainty principle states that we cannot simultaneously know
both the exact position and momentum of a particle.

DNA (deoxyribonucleic acid) is the molecule that carries genetic information
in all living organisms. It consists of two strands wound into a double helix,
connected by base pairs: adenine with thymine, and guanine with cytosine.

The French Revolution began in 1789 and fundamentally changed the political
landscape of Europe. It established the principles of liberty, equality,
and fraternity, which remain core values of modern democratic societies.

The Internet was developed from ARPANET, a project of the US Department of
Defense in the late 1960s. The World Wide Web was invented by Tim Berners-Lee
at CERN in 1989, making the Internet accessible to the general public.

Artificial intelligence is the field of computer science focused on creating
systems that can perform tasks requiring human intelligence. Machine learning,
a subset of AI, enables systems to learn from data without explicit programming.

The Pythagorean theorem states that in a right triangle, the square of the
hypotenuse equals the sum of the squares of the other two sides: a squared
plus b squared equals c squared.

Euler's identity connects five fundamental mathematical constants:
e to the power of i times pi plus one equals zero.

The golden ratio, approximately 1.618, appears throughout nature and art.
It is defined as phi equals one plus the square root of five, divided by two.

A prime number is a natural number greater than one that has no positive
divisors other than one and itself. The distribution of primes is described
by the prime number theorem.

The Fibonacci sequence is defined by F of n equals F of n minus one plus
F of n minus two, with F of zero equals zero and F of one equals one.
The ratio of consecutive Fibonacci numbers converges to the golden ratio.

In cybersecurity, encryption transforms data into an unreadable format
to protect it from unauthorized access. Symmetric encryption uses the same
key for encryption and decryption, while asymmetric encryption uses a
public-private key pair.
"""


def build_dataset(tokenizer, max_tokens=500000, samples_per_dataset=200):
    """Build a multi-domain tokenized dataset."""
    all_tokens = []
    domain_counts = {}

    random.seed(42)

    print("Building multi-domain dataset...", flush=True)

    # 1. Inline data (always available).
    print("  Inline code...", flush=True)
    for code in INLINE_CODE:
        ids = tokenizer.encode(code * 3)  # repeat for more coverage
        all_tokens.extend(ids)
        domain_counts["code"] = domain_counts.get("code", 0) + len(ids)

    print("  Inline knowledge...", flush=True)
    ids = tokenizer.encode(UNIVERSAL_KNOWLEDGE * 20)
    all_tokens.extend(ids)
    domain_counts["knowledge"] = domain_counts.get("knowledge", 0) + len(ids)

    # 2. tinyshakespeare (local).
    ts_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "data", "text", "tinyshakespeare.txt",
    )
    if os.path.exists(ts_path):
        with open(ts_path, "r", encoding="utf-8") as f:
            text = f.read()
        ids = tokenizer.encode(text)
        all_tokens.extend(ids)
        domain_counts["literature"] = domain_counts.get("literature", 0) + len(ids)
        print(f"  tinyshakespeare: {len(ids)} tokens", flush=True)

    # 3. Stream HF datasets.
    for name, config, field, domain, weight in DATASET_CONFIG:
        target = int(samples_per_dataset * weight * 10)
        try:
            from datasets import load_dataset
            if config:
                ds = load_dataset(name, config, split="train", streaming=True,
                                  trust_remote_code=True)
            else:
                ds = load_dataset(name, split="train", streaming=True,
                                  trust_remote_code=True)
            count = 0
            for example in ds:
                if count >= target or len(all_tokens) >= max_tokens:
                    break
                text = example.get(field, "")
                if not text or len(str(text)) < 20:
                    continue
                ids = tokenizer.encode(str(text))
                all_tokens.extend(ids)
                domain_counts[domain] = domain_counts.get(domain, 0) + len(ids)
                count += 1
            print(f"  {name} ({domain}): {count} samples", flush=True)
        except Exception as e:
            print(f"  SKIP {name}: {str(e)[:80]}", flush=True)
            continue

        if len(all_tokens) >= max_tokens:
            break

    # Truncate to max_tokens.
    all_tokens = all_tokens[:max_tokens]

    print(f"\nTotal: {len(all_tokens):,} tokens", flush=True)
    print("Domain breakdown:", flush=True)
    for domain, count in sorted(domain_counts.items(), key=lambda x: -x[1]):
        print(f"  {domain:15s}: {count:>8,} tokens ({count/max(len(all_tokens),1)*100:.1f}%)", flush=True)

    # Shuffle (but keep domain balance roughly).
    random.shuffle(all_tokens)

    return all_tokens, domain_counts


def save_dataset(tokens, path):
    """Save tokenized dataset as a binary file for fast loading."""
    tensor = torch.tensor(tokens, dtype=torch.int32)
    torch.save(tensor, path)
    size_mb = os.path.getsize(path) / 1e6
    print(f"Saved to {path} ({size_mb:.1f} MB)", flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-tokens", type=int, default=500000)
    parser.add_argument("--samples", type=int, default=200)
    parser.add_argument("--output", type=str,
                        default=os.path.join(
                            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                            "data", "fractus_corpus.pt"
                        ))
    args = parser.parse_args()

    tok = FractusTokenizer.gpt2_compatible()
    print(f"Tokenizer: vocab={tok.vocab_size}", flush=True)

    tokens, domains = build_dataset(tok, max_tokens=args.max_tokens,
                                     samples_per_dataset=args.samples)

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    save_dataset(tokens, args.output)

    print(f"\nDone. {len(tokens):,} tokens ready for training.", flush=True)
    print(f"Load with: torch.load('{args.output}')", flush=True)


if __name__ == "__main__":
    main()
