#!/usr/bin/env python
"""Train the Continuous Thought Engine on real data.

Streams tinyshakespeare (text) + code samples, trains the engine continuously,
checkpoints to HuggingFace Hub, benchmarks perplexity, and exports ONNX.

This is the real training run for Fractus-1B (Continuous Thought Engine edition).

Usage:
    python scripts/train_continuous.py
    python scripts/train_continuous.py --epochs 50 --d-model 128
"""

import argparse
import math
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from fractus.continuous_engine import ContinuousThoughtEngine
from fractus.train.online import OnlineTrainer
from fractus.tokenizer import FractusTokenizer


HF_TOKEN = os.environ.get("HF_TOKEN", "")
HF_REPO_ID = "AFKmoney/Fractus-1B"


def load_training_data(tokenizer, max_tokens=50000):
    """Load and tokenize training data from multiple sources."""
    all_tokens = []

    # 1. tinyshakespeare (local, always available).
    ts_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "data", "text", "tinyshakespeare.txt",
    )
    if os.path.exists(ts_path):
        with open(ts_path, "r", encoding="utf-8") as f:
            text = f.read()
        ids = tokenizer.encode(text)
        all_tokens.extend(ids)
        print(f"  tinyshakespeare: {len(ids)} tokens", flush=True)

    # 2. Python code samples (inline for guaranteed availability).
    code_samples = [
        "def fibonacci(n):\n    if n <= 1:\n        return n\n    return fibonacci(n-1) + fibonacci(n-2)\n",
        "def quicksort(arr):\n    if len(arr) <= 1:\n        return arr\n    pivot = arr[len(arr)//2]\n    left = [x for x in arr if x < pivot]\n    middle = [x for x in arr if x == pivot]\n    right = [x for x in arr if x > pivot]\n    return quicksort(left) + middle + quicksort(right)\n",
        "class Node:\n    def __init__(self, value):\n        self.value = value\n        self.next = None\n\nclass LinkedList:\n    def __init__(self):\n        self.head = None\n    def append(self, value):\n        if not self.head:\n            self.head = Node(value)\n        else:\n            current = self.head\n            while current.next:\n                current = current.next\n            current.next = Node(value)\n",
        "def binary_search(arr, target):\n    low, high = 0, len(arr) - 1\n    while low <= high:\n        mid = (low + high) // 2\n        if arr[mid] == target:\n            return mid\n        elif arr[mid] < target:\n            low = mid + 1\n        else:\n            high = mid - 1\n    return -1\n",
        "import math\ndef is_prime(n):\n    if n < 2:\n        return False\n    for i in range(2, int(math.sqrt(n)) + 1):\n        if n % i == 0:\n            return False\n    return True\n",
        "def merge_sort(arr):\n    if len(arr) <= 1:\n        return arr\n    mid = len(arr) // 2\n    left = merge_sort(arr[:mid])\n    right = merge_sort(arr[mid:])\n    return merge(left, right)\n\ndef merge(left, right):\n    result = []\n    i = j = 0\n    while i < len(left) and j < len(right):\n        if left[i] <= right[j]:\n            result.append(left[i])\n            i += 1\n        else:\n            result.append(right[j])\n            j += 1\n    result.extend(left[i:])\n    result.extend(right[j:])\n    return result\n",
    ]
    for code in code_samples:
        ids = tokenizer.encode(code)
        all_tokens.extend(ids)
    print(f"  code samples: {sum(len(tokenizer.encode(c)) for c in code_samples)} tokens", flush=True)

    # 3. Math/science text.
    math_text = (
        "The Pythagorean theorem states that in a right triangle, "
        "the square of the hypotenuse equals the sum of the squares of the other two sides. "
        "a squared plus b squared equals c squared. "
        "The golden ratio phi equals one plus square root of five divided by two. "
        "Euler's identity connects five fundamental constants: e to the i pi plus one equals zero. "
        "The derivative of x squared is two x. The integral of x squared is x cubed over three. "
        "A prime number is a natural number greater than one that has no positive divisors other than one and itself. "
        "The Fibonacci sequence is defined by F(n) = F(n-1) + F(n-2) with F(0)=0 and F(1)=1."
    )
    ids = tokenizer.encode(math_text * 5)  # repeat for more data.
    all_tokens.extend(ids)
    print(f"  math/science: {len(ids)} tokens", flush=True)

    # Truncate.
    all_tokens = all_tokens[:max_tokens]
    print(f"  Total: {len(all_tokens)} tokens", flush=True)
    return all_tokens


def export_onnx(engine, tokenizer, path="fractus_continuous.onnx"):
    """Export the engine to ONNX."""
    print(f"\nExporting ONNX to {path}...", flush=True)
    engine.eval()
    engine.reset_thought(1)

    # Create a wrapper that does one tick for ONNX export.
    class TickWrapper(torch.nn.Module):
        def __init__(self, engine):
            super().__init__()
            self.engine = engine
        def forward(self, obs):
            logits, conf = self.engine.tick(obs)
            return logits, conf

    wrapper = TickWrapper(engine)
    dummy = torch.tensor([100], dtype=torch.long)
    try:
        torch.onnx.export(
            wrapper, dummy, path,
            input_names=["observation"],
            output_names=["logits", "confidence"],
            opset_version=17,
        )
        print(f"ONNX exported: {os.path.getsize(path) / 1e6:.1f} MB", flush=True)
        return True
    except Exception as e:
        print(f"ONNX export failed: {e}", flush=True)
        return False


def upload_to_hf(engine, tokenizer, perplexity, step):
    """Upload checkpoint to HuggingFace Hub. NEVER crashes the training."""
    # ALWAYS save locally first (so we never lose progress).
    local_dir = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "checkpoints",
    )
    os.makedirs(local_dir, exist_ok=True)
    local_path = os.path.join(local_dir, f"checkpoint_{step}.pt")
    torch.save({
        "model_state": engine.state_dict(),
        "config": {"d_model": engine.d_model, "vocab_size": engine.vocab_size},
        "step": step,
        "perplexity": perplexity,
    }, local_path)
    print(f"  Local checkpoint saved: {local_path} ({os.path.getsize(local_path)/1e6:.1f} MB)", flush=True)

    # Try HF upload (non-fatal if it fails).
    if not HF_TOKEN:
        return
    try:
        from huggingface_hub import HfApi
        api = HfApi(token=HF_TOKEN)
        # Check if we can access the repo (create if needed).
        try:
            api.repo_info(repo_id=HF_REPO_ID, repo_type="model")
        except Exception:
            try:
                api.create_repo(repo_id=HF_REPO_ID, repo_type="model", exist_ok=True)
            except Exception as ce:
                print(f"  HF: cannot create repo ({ce}), keeping local only.", flush=True)
                return

        api.upload_file(
            path_or_fileobj=local_path,
            path_in_repo=f"checkpoints/checkpoint_{step}.pt",
            repo_id=HF_REPO_ID,
            repo_type="model",
        )
        print(f"  HF uploaded checkpoint_{step}", flush=True)
    except Exception as e:
        print(f"  HF upload skipped ({type(e).__name__}), local checkpoint is safe.", flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--d-model", type=int, default=128)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--accum-steps", type=int, default=16)
    parser.add_argument("--max-tokens", type=int, default=50000)
    parser.add_argument("--checkpoint-every", type=int, default=5000)
    args = parser.parse_args()

    torch.set_num_threads(os.cpu_count() or 6)
    torch.manual_seed(42)

    # Tokenizer.
    tok = FractusTokenizer.gpt2_compatible()
    print(f"Tokenizer: vocab={tok.vocab_size}", flush=True)

    # Data.
    print("Loading data...", flush=True)
    tokens_list = load_training_data(tok, max_tokens=args.max_tokens)
    tokens = torch.tensor(tokens_list, dtype=torch.long)

    # Engine.
    d_model = args.d_model
    d_head = d_model // 4 if d_model >= 64 else 32
    n_heads = max(2, d_model // d_head)
    engine = ContinuousThoughtEngine(
        vocab_size=tok.vocab_size, d_model=d_model, n_heads=n_heads, d_head=d_head,
        n_levels=2, n_oscillators=8, coupling_rank=4,
        n_experts=4, top_k=2, expert_d_ff=d_model, siren_rank=min(32, d_model//4),
    )
    n_params = sum(p.numel() for p in engine.parameters())
    print(f"Engine params: {n_params:,} ({n_params/1e6:.1f}M)", flush=True)

    # Trainer with cosine LR.
    trainer = OnlineTrainer(engine, lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        trainer.optimizer, T_max=args.epochs, eta_min=1e-5,
    )

    # Training loop.
    print(f"\nTraining {args.epochs} epochs on {len(tokens)} tokens...", flush=True)
    print("=" * 70, flush=True)
    initial_loss = None
    global_step = 0

    for epoch in range(args.epochs):
        t0 = time.perf_counter()
        m = trainer.train_on_stream_chunked(tokens, chunk_len=args.accum_steps)
        elapsed = time.perf_counter() - t0
        tps = len(tokens) / elapsed

        if initial_loss is None:
            initial_loss = m["avg_loss"]
        scheduler.step()
        global_step += m["optimizer_steps"]

        # Perplexity.
        ppl = math.exp(min(m["avg_loss"], 20))
        lr = trainer.optimizer.param_groups[0]["lr"]
        print(f"Epoch {epoch+1:3d}/{args.epochs}: loss={m['avg_loss']:.2f}  "
              f"ppl={ppl:.1f}  acc={m['accuracy']:.2%}  "
              f"{tps:.0f} tok/s  lr={lr:.5f}  {elapsed:.1f}s", flush=True)

        # Checkpoint.
        if (epoch + 1) % 10 == 0 or epoch == args.epochs - 1:
            print(f"  [checkpoint] Uploading step {global_step}...", flush=True)
            upload_to_hf(engine, tok, ppl, global_step)

    # Final generation.
    print("\n=== Generation ===", flush=True)
    engine.reset_thought(1)
    seed = tok.encode("The ")
    for tid in seed:
        engine.tick(torch.tensor([tid]))
    generated = list(seed)
    for _ in range(80):
        logits, _ = engine.tick()
        generated.append(logits.argmax(dim=-1).item())
    print(tok.decode(generated), flush=True)

    # Export ONNX.
    onnx_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "fractus_continuous.onnx",
    )
    export_onnx(engine, tok, onnx_path)
    upload_to_hf(engine, tok, ppl, global_step)
    print("\nDone.", flush=True)


if __name__ == "__main__":
    main()
