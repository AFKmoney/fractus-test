#!/usr/bin/env python
"""Training script for fractus on HuggingFace datasets.

Supports any HF text dataset (tinyshakespeare, wikitext, OpenWebText, FineWeb, ...)
with presets adapted to different hardware.

⚠️ HARDWARE HONESTY:
    - cpu-tiny / cpu-small: work on a laptop/desktop CPU.
    - gpu-*: require a CUDA GPU (or MPS on Apple Silicon).
    - gpu-1b: requires an A100 80GB or H100. IMPOSSIBLE on a consumer CPU or GPU.
      The main bottleneck is the Kuramoto RK4 (not vectorized) + SIREN.

Usage:
    # Small model on tinyshakespeare (CPU, ~2 min)
    python scripts/train_hf.py --preset cpu-small --dataset tinyshakespeare

    # Medium on wikitext-2 (GPU)
    python scripts/train_hf.py --preset gpu-medium --dataset wikitext-2

    # Arbitrary HF dataset
    python scripts/train_hf.py --dataset HuggingFaceFW/fineweb \
        --text-field text --preset gpu-small --max-samples 100000

    # Fully custom configuration
    python scripts/train_hf.py --dataset tinyshakespeare \
        --d-model 256 --n-blocks 6 --seq-len 128 --batch-size 16 --epochs 3
"""

import argparse
import math
import os
import sys
import time
from dataclasses import dataclass

import torch
import torch.nn as nn

# Allow importing from the project root.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fractus.nn import FractalEmbedding, FractalBlockFull
from fractus.metrics.perplexity import honest_perplexity


# ---------------------------------------------------------------------------
# Presets (model sizes + hyperparameters per hardware)
# ---------------------------------------------------------------------------

@dataclass
class Preset:
    name: str
    d_model: int
    n_blocks: int
    n_heads: int
    n_levels: int
    n_frequencies: int
    n_oscillators: int
    coupling_rank: int
    n_experts: int
    top_k: int
    seq_len: int
    batch_size: int
    lr: float
    epochs: int
    description: str

    def n_params_approx(self, vocab: int) -> int:
        """Rough parameter-count estimate."""
        # Embedding: 16 char + 2*n_freq Fourier + vortex_mlp ~ (16 + 2*n_freq + 32) * d_model
        emb = (16 + 2 * self.n_frequencies + 32) * self.d_model + 32 * (32 + self.n_frequencies)
        # Per block: attention (3 * d_model^2 + d_model^2) + Kuramoto + MoE
        attn = 3 * self.d_model * self.d_model + self.d_model * self.d_model + 4 * self.d_model
        kur = self.n_oscillators + self.n_oscillators * self.coupling_rank + self.coupling_rank
        moe = self.n_experts * (self.d_model * 64 + 64 * self.d_model + 64 + self.d_model)
        block = attn + kur + moe + 2 * self.d_model  # + 2 LayerNorms
        # Head.
        head = self.d_model * vocab
        return int(emb + self.n_blocks * block + head + self.d_model)


PRESETS = {
    "cpu-tiny": Preset(
        name="cpu-tiny", d_model=48, n_blocks=2, n_heads=4, n_levels=2,
        n_frequencies=12, n_oscillators=8, coupling_rank=4, n_experts=4, top_k=2,
        seq_len=32, batch_size=32, lr=3e-3, epochs=1,
        description="CPU laptop, ~80k params, ~2 min/epoch",
    ),
    "cpu-small": Preset(
        name="cpu-small", d_model=96, n_blocks=3, n_heads=4, n_levels=2,
        n_frequencies=16, n_oscillators=8, coupling_rank=4, n_experts=4, top_k=2,
        seq_len=48, batch_size=16, lr=3e-3, epochs=2,
        description="CPU desktop, ~500k params, ~30 min/epoch",
    ),
    "gpu-small": Preset(
        name="gpu-small", d_model=256, n_blocks=4, n_heads=8, n_levels=3,
        n_frequencies=32, n_oscillators=16, coupling_rank=8, n_experts=8, top_k=2,
        seq_len=128, batch_size=32, lr=3e-4, epochs=3,
        description="Entry GPU (RTX 3060), ~5M params, ~5 min/epoch",
    ),
    "gpu-medium": Preset(
        name="gpu-medium", d_model=512, n_blocks=6, n_heads=8, n_levels=3,
        n_frequencies=64, n_oscillators=32, coupling_rank=16, n_experts=8, top_k=4,
        seq_len=256, batch_size=16, lr=3e-4, epochs=3,
        description="Mid GPU (RTX 4090), ~50M params, ~30 min/epoch",
    ),
    "gpu-large": Preset(
        name="gpu-large", d_model=1024, n_blocks=12, n_heads=16, n_levels=4,
        n_frequencies=128, n_oscillators=64, coupling_rank=32, n_experts=16, top_k=4,
        seq_len=512, batch_size=8, lr=3e-4, epochs=3,
        description="Datacenter GPU (A100 40GB), ~300M params, ~2h/epoch",
    ),
    "gpu-1b": Preset(
        name="gpu-1b", d_model=2048, n_blocks=24, n_heads=16, n_levels=4,
        n_frequencies=256, n_oscillators=128, coupling_rank=64, n_experts=32, top_k=8,
        seq_len=1024, batch_size=4, lr=3e-4, epochs=3,
        description=("Datacenter GPU (A100 80GB / H100), ~1B params, ~8h/epoch. "
                     "IMPOSSIBLE on a consumer CPU or GPU — OOM guaranteed."),
    ),
}


# ---------------------------------------------------------------------------
# HuggingFace dataset
# ---------------------------------------------------------------------------

# Aliases for local vs HF datasets.
LOCAL_DATASETS = {
    "tinyshakespeare": None,  # special: local file.
}

HF_DATASET_DEFAULTS = {
    "wikitext-2": ("wikitext", "wikitext-2-raw-v1", "text"),
    "wikitext-103": ("wikitext", "wikitext-103-raw-v1", "text"),
    "tinyshakespeare": ("tiny_shakespeare", None, "text"),
}


def load_text_dataset(
    dataset_name: str,
    text_field: str,
    max_samples: int,
) -> tuple[str, dict]:
    """Loads a dataset and returns (full_text, vocab_char_to_id).

    Character-level (like tinyshakespeare).
    """
    # tinyshakespeare local (already downloaded).
    if dataset_name == "tinyshakespeare":
        local_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "data", "text", "tinyshakespeare.txt",
        )
        if os.path.exists(local_path):
            with open(local_path, "r", encoding="utf-8") as f:
                text = f.read()
        else:
            # HF fallback.
            try:
                from datasets import load_dataset
                ds = load_dataset("tiny_shakespeare", split="train")
                text = "\n".join(ds[text_field])
            except Exception as e:
                raise RuntimeError(
                    f"Could not load tinyshakespeare (neither local nor HF): {e}"
                ) from e
    else:
        # Arbitrary HF dataset.
        try:
            from datasets import load_dataset
        except ImportError as e:
            raise RuntimeError(
                "The `datasets` library is not installed. Run: pip install datasets"
            ) from e

        # Parse "name" or "name:config".
        if ":" in dataset_name:
            name, config = dataset_name.split(":", 1)
        elif dataset_name in HF_DATASET_DEFAULTS:
            name, config, default_field = HF_DATASET_DEFAULTS[dataset_name]
            if not text_field:
                text_field = default_field
        else:
            name, config = dataset_name, None

        print(f"Loading HF dataset: {name} (config={config})...")
        ds = load_dataset(name, config, split="train")
        texts = []
        for i, ex in enumerate(ds):
            if max_samples and i >= max_samples:
                break
            texts.append(ex[text_field])
        text = "\n".join(texts)

    if max_samples and len(text) > max_samples * 1000:  # heuristic: ~1000 chars/sample.
        text = text[:max_samples * 1000]

    # Character-level vocabulary.
    chars = sorted(set(text))
    char_to_id = {c: i for i, c in enumerate(chars)}
    print(f"Text: {len(text):,} characters, vocabulary: {len(chars)} characters")
    return text, char_to_id


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

class FractalLM(nn.Module):
    """Fractal language model: Embedding + N×FractalBlockFull + head."""

    def __init__(self, vocab, preset: Preset):
        super().__init__()
        self.embed = FractalEmbedding(
            vocab, preset.d_model, n_frequencies=preset.n_frequencies,
        )
        self.blocks = nn.ModuleList([
            FractalBlockFull(
                d_model=preset.d_model,
                n_heads=preset.n_heads,
                d_head=preset.d_model // preset.n_heads,
                n_levels=preset.n_levels,
                n_oscillators=preset.n_oscillators,
                coupling_rank=preset.coupling_rank,
                n_experts=preset.n_experts,
                top_k=preset.top_k,
                kappa=4.0,
            )
            for _ in range(preset.n_blocks)
        ])
        self.norm = nn.LayerNorm(preset.d_model)
        self.head = nn.Linear(preset.d_model, vocab, bias=False)

    def forward(self, ids):
        x = self.embed(ids)
        aux = torch.tensor(0.0, device=x.device)
        for b in self.blocks:
            x, lb = b(x)
            aux = aux + lb
        x = self.norm(x)
        return self.head(x), aux


# ---------------------------------------------------------------------------
# Character-level tokenizer
# ---------------------------------------------------------------------------

class CharTokenizer:
    def __init__(self, char_to_id: dict):
        self.char_to_id = char_to_id
        self.id_to_char = {i: c for c, i in char_to_id.items()}
        self.vocab_size = len(char_to_id)

    def encode(self, text: str) -> torch.Tensor:
        return torch.tensor(
            [self.char_to_id[c] for c in text if c in self.char_to_id],
            dtype=torch.long,
        )

    def decode(self, ids: torch.Tensor) -> str:
        return "".join(self.id_to_char.get(int(i), "?") for i in ids)


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def train(args):
    # Device.
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"Device: {device}")

    # Preset.
    if args.preset:
        preset = PRESETS[args.preset]
        # Override with explicit args.
        if args.d_model: preset.d_model = args.d_model
        if args.n_blocks: preset.n_blocks = args.n_blocks
        if args.seq_len: preset.seq_len = args.seq_len
        if args.batch_size: preset.batch_size = args.batch_size
        if args.epochs: preset.epochs = args.epochs
        if args.lr: preset.lr = args.lr
    else:
        # Build a preset from the args.
        preset = Preset(
            name="custom",
            d_model=args.d_model or 64,
            n_blocks=args.n_blocks or 2,
            n_heads=args.n_heads or 4,
            n_levels=args.n_levels or 2,
            n_frequencies=args.n_frequencies or 16,
            n_oscillators=args.n_oscillators or 8,
            coupling_rank=args.coupling_rank or 4,
            n_experts=args.n_experts or 4,
            top_k=args.top_k or 2,
            seq_len=args.seq_len or 64,
            batch_size=args.batch_size or 16,
            lr=args.lr or 3e-3,
            epochs=args.epochs or 1,
            description="custom",
        )

    print(f"\nPreset: {preset.name} — {preset.description}")
    print(f"  d_model={preset.d_model}, n_blocks={preset.n_blocks}, "
          f"seq_len={preset.seq_len}, batch_size={preset.batch_size}")

    # ⚠️ 1B disclaimer.
    if preset.name == "gpu-1b" and device.type == "cpu":
        print("\n" + "=" * 70)
        print("⚠️  CRITICAL WARNING")
        print("=" * 70)
        print("You are attempting to train a ~1B-parameter model on CPU.")
        print("This is IMPOSSIBLE in practice:")
        print("  - Memory: ~16 GB minimum required, you likely have < 16 GB RAM.")
        print("  - Time: weeks to months per epoch.")
        print("The gpu-1b preset requires an A100 80GB or H100.")
        print("Aborting. Use --preset cpu-small or --preset gpu-small.")
        sys.exit(1)

    # Dataset.
    print(f"\nDataset: {args.dataset}")
    text, char_to_id = load_text_dataset(
        args.dataset, args.text_field, args.max_samples,
    )
    tokenizer = CharTokenizer(char_to_id)
    vocab = tokenizer.vocab_size

    # Parameter estimate.
    n_params_est = preset.n_params_approx(vocab)
    print(f"Estimated parameters: {n_params_est:,} ({n_params_est/1e6:.1f}M)")
    if n_params_est > 100_000_000 and device.type == "cpu":
        print(f"\n⚠️  {n_params_est/1e6:.0f}M params on CPU will be VERY slow.")
        print("Consider --preset cpu-small or a GPU.")

    # Encode.
    all_ids = tokenizer.encode(text)
    seq_len = preset.seq_len
    n_seqs = (len(all_ids) - 1) // seq_len
    all_ids = all_ids[:n_seqs * seq_len + 1]
    print(f"Sequences: {n_seqs:,} of length {seq_len}")

    # Train/val split (95/5).
    n_train = int(0.95 * n_seqs)
    train_ids = all_ids[:n_train * seq_len + 1]
    val_ids = all_ids[n_train * seq_len:]

    def make_batches(ids_tensor, batch_size, shuffle):
        n = (len(ids_tensor) - 1) // seq_len
        ids_tensor = ids_tensor[:n * seq_len + 1]
        seqs = ids_tensor[:n * seq_len].view(n, seq_len)
        targets = ids_tensor[1:n * seq_len + 1].view(n, seq_len)
        if shuffle:
            perm = torch.randperm(n)
            seqs, targets = seqs[perm], targets[perm]
        # Batch.
        for i in range(0, n - batch_size + 1, batch_size):
            yield seqs[i:i+batch_size].to(device), targets[i:i+batch_size].to(device)

    # Model.
    model = FractalLM(vocab, preset).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Real parameters: {n_params:,} ({n_params/1e6:.2f}M)")

    opt = torch.optim.AdamW(model.parameters(), lr=preset.lr, weight_decay=0.01)

    # Checkpoint dir.
    ckpt_dir = args.checkpoint_dir or os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "checkpoints",
    )
    os.makedirs(ckpt_dir, exist_ok=True)

    # Loop.
    print(f"\nTraining: {preset.epochs} epochs, lr={preset.lr}")
    best_val_ppl = float("inf")
    for epoch in range(preset.epochs):
        model.train()
        t0 = time.time()
        epoch_loss = 0.0
        n_batches = 0
        for inp, tgt in make_batches(train_ids, preset.batch_size, shuffle=True):
            opt.zero_grad()
            logits, aux = model(inp)
            ce = nn.functional.cross_entropy(
                logits.reshape(-1, vocab), tgt.reshape(-1)
            )
            loss = ce + 0.1 * aux
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            epoch_loss += ce.item()
            n_batches += 1
            if n_batches % 50 == 0:
                print(f"  epoch {epoch+1} batch {n_batches}  ce={ce.item():.4f}")
        train_ce = epoch_loss / max(n_batches, 1)
        elapsed = time.time() - t0

        # Validation.
        model.eval()
        val_ces = []
        with torch.no_grad():
            for inp, tgt in make_batches(val_ids, preset.batch_size, shuffle=False):
                logits, _ = model(inp)
                ce = nn.functional.cross_entropy(
                    logits.reshape(-1, vocab), tgt.reshape(-1)
                )
                val_ces.append(ce.item())
        val_ce = sum(val_ces) / max(len(val_ces), 1)
        val_ppl = math.exp(val_ce) if val_ce < 20 else float("inf")

        print(f"\nEpoch {epoch+1}/{preset.epochs}: "
              f"train_ce={train_ce:.4f} (ppl={math.exp(min(train_ce,20)):.2f})  "
              f"val_ce={val_ce:.4f} (ppl={val_ppl:.2f})  "
              f"time={elapsed:.0f}s")

        # Checkpoint if better.
        if val_ppl < best_val_ppl:
            best_val_ppl = val_ppl
            ckpt_path = os.path.join(ckpt_dir, f"fractal_{preset.name}_best.pt")
            torch.save({
                "model_state": model.state_dict(),
                "preset": preset.__dict__,
                "vocab": char_to_id,
                "epoch": epoch + 1,
                "val_ppl": val_ppl,
            }, ckpt_path)
            print(f"  → Checkpoint saved: {ckpt_path}")

    # Final generation.
    print("\n=== Generation (greedy) ===")
    model.eval()
    prompt = args.prompt or "The "
    if all(c in char_to_id for c in prompt[:3]):
        ctx = torch.tensor([[char_to_id[c] for c in prompt]], device=device)
    else:
        ctx = torch.tensor([[0]], device=device)
    with torch.no_grad():
        for _ in range(200):
            if ctx.shape[1] > seq_len:
                ctx = ctx[:, -seq_len:]
            logits, _ = model(ctx)
            nxt = int(logits[0, -1].argmax().item())
            ctx = torch.cat([ctx, torch.tensor([[nxt]], device=device)], dim=1)
    print(tokenizer.decode(ctx[0]))

    print(f"\nDone. Best validation perplexity: {best_val_ppl:.2f}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser(
        description="Train fractus on HuggingFace datasets.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Available presets:
  cpu-tiny    ~80k params, CPU laptop, ~2 min/epoch
  cpu-small   ~500k params, CPU desktop, ~30 min/epoch
  gpu-small   ~5M params, entry GPU (RTX 3060)
  gpu-medium  ~50M params, mid GPU (RTX 4090)
  gpu-large   ~300M params, A100 40GB
  gpu-1b      ~1B params, A100 80GB/H100 (IMPOSSIBLE on CPU)

Predefined datasets:
  tinyshakespeare  (local, 1.1 MB)
  wikitext-2       (HF, ~5 MB)
  wikitext-103     (HF, ~500 MB)
  Or any HF dataset: --dataset name --text-field field

Examples:
  python scripts/train_hf.py --preset cpu-small --dataset tinyshakespeare
  python scripts/train_hf.py --preset gpu-medium --dataset wikitext-2
  python scripts/train_hf.py --dataset HuggingFaceFW/fineweb --text-field text --max-samples 10000 --preset gpu-small
""",
    )
    p.add_argument("--preset", choices=list(PRESETS.keys()), default=None,
                   help="Size/hardware preset (default: cpu-tiny).")
    p.add_argument("--dataset", default="tinyshakespeare",
                   help="Dataset: tinyshakespeare, wikitext-2, or an HF name.")
    p.add_argument("--text-field", default=None,
                   help="Text field of the HF dataset (default: depends on the dataset).")
    p.add_argument("--max-samples", type=int, default=0,
                   help="Max samples (0 = all).")
    p.add_argument("--prompt", default=None,
                   help="Prompt for the final generation.")
    p.add_argument("--checkpoint-dir", default=None,
                   help="Checkpoint folder (default: ./checkpoints).")
    # Custom overrides.
    p.add_argument("--d-model", type=int, default=None)
    p.add_argument("--n-blocks", type=int, default=None)
    p.add_argument("--n-heads", type=int, default=None)
    p.add_argument("--n-levels", type=int, default=None)
    p.add_argument("--n-frequencies", type=int, default=None)
    p.add_argument("--n-oscillators", type=int, default=None)
    p.add_argument("--coupling-rank", type=int, default=None)
    p.add_argument("--n-experts", type=int, default=None)
    p.add_argument("--top-k", type=int, default=None)
    p.add_argument("--seq-len", type=int, default=None)
    p.add_argument("--batch-size", type=int, default=None)
    p.add_argument("--lr", type=float, default=None)
    p.add_argument("--epochs", type=int, default=None)

    args = p.parse_args()
    if args.preset is None:
        args.preset = "cpu-tiny"

    train(args)


if __name__ == "__main__":
    main()
