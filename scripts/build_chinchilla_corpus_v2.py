#!/usr/bin/env python
"""Build a Chinchilla-optimal corpus for Fractus-1B (88M params). v2 — crash-safe.

Target: 1.76B tokens (20x params per Chinchilla scaling law).
Uses FineWeb (web text) + existing communication/ultimate corpora.

v2 fixes:
  - int32 dtype (int16 overflowed at token id 32768; GPT-2 vocab is 50257)
  - Sharded incremental saving every 100M tokens (crash = max 100M lost, not all)
  - Resume from existing shards (re-run continues where it stopped)
  - Memory-bounded: flushes shard buffers to disk instead of holding 1.76B in RAM
"""
import os, sys, time, glob, json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
import numpy as np
from datasets import load_dataset
from fractus.tokenizer import FractusTokenizer

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
OUTPUT = os.path.join(DATA_DIR, "chinchilla_corpus.pt")
SHARD_DIR = os.path.join(DATA_DIR, "chinchilla_shards")
META = os.path.join(SHARD_DIR, "progress.json")

EOS = 50256
TARGET_TOKENS = 1_760_000_000  # Chinchilla optimal for 88M params
SHARD_SIZE = 100_000_000        # save every 100M tokens (crash safety)


def load_progress():
    """Return (existing_tokens, fineweb_examples_consumed)."""
    if os.path.exists(META):
        with open(META) as f:
            d = json.load(f)
        return d.get("total_tokens", 0), d.get("fw_examples", 0)
    return 0, 0


def save_progress(total_tokens, fw_examples):
    os.makedirs(SHARD_DIR, exist_ok=True)
    with open(META, "w") as f:
        json.dump({"total_tokens": total_tokens, "fw_examples": fw_examples}, f)


def existing_shard_tokens():
    """Count tokens already saved across all shards."""
    total = 0
    for p in sorted(glob.glob(os.path.join(SHARD_DIR, "shard_*.npy"))):
        arr = np.load(p, mmap_mode="r")
        total += len(arr)
    return total


def main():
    tok = (FractusTokenizer.gpt2Compatible()
           if hasattr(FractusTokenizer, 'gpt2Compatible')
           else FractusTokenizer.gpt2_compatible())
    os.makedirs(SHARD_DIR, exist_ok=True)

    print(f"Tokenizer: vocab={tok.vocab_size}", flush=True)
    print(f"Target: {TARGET_TOKENS/1e9:.2f}B tokens (Chinchilla optimal for 88M params)", flush=True)
    print(f"Output format: int32 (GPT-2 max id 50256 needs > int16)", flush=True)
    print(f"Shard size: {SHARD_SIZE/1e6:.0f}M tokens (crash-safe)", flush=True)
    print(flush=True)

    # ---- Phase 1: load existing quality corpora into shards (only once) ----
    quality_shard = os.path.join(SHARD_DIR, "shard_000_quality.npy")
    progress = load_progress()
    shard_disk_tokens = existing_shard_tokens()

    if not os.path.exists(quality_shard):
        buf = []
        for name in ["communication_corpus.pt", "ultimate_corpus.pt"]:
            path = os.path.join(DATA_DIR, name)
            if os.path.exists(path):
                t = torch.load(path, weights_only=False).long()
                buf.extend(t.tolist())
                print(f"  Loaded {name}: {len(t):,} tokens", flush=True)
        if buf:
            arr = np.array(buf, dtype=np.int32)
            np.save(quality_shard, arr)
            print(f"  Saved quality shard: {len(arr):,} tokens", flush=True)
            shard_disk_tokens = existing_shard_tokens()
            save_progress(shard_disk_tokens, 0)
    else:
        print(f"  Quality shard exists, skipping reload", flush=True)

    base_tokens = existing_shard_tokens()
    print(f"\nBase tokens on disk: {base_tokens/1e6:.1f}M", flush=True)
    remaining = TARGET_TOKENS - base_tokens
    print(f"Need from FineWeb: {remaining/1e6:.1f}M more tokens", flush=True)

    if remaining <= 0:
        print("Target already reached, skipping FineWeb.", flush=True)
    else:
        # ---- Phase 2: stream FineWeb, save shards incrementally ----
        print(f"\nStreaming FineWeb (sample-10BT)...", flush=True)
        ds = load_dataset("HuggingFaceFW/fineweb", "sample-10BT",
                          split="train", streaming=True)

        # Resume: skip examples already consumed.
        _, fw_skip = load_progress()
        if fw_skip > 0:
            print(f"  Resuming: skipping first {fw_skip:,} FineWeb examples", flush=True)
            ds = ds.skip(fw_skip)

        buf = []
        buf_tokens = 0
        count = 0
        skipped = 0
        shard_idx = len(glob.glob(os.path.join(SHARD_DIR, "shard_*.npy")))
        t0 = time.perf_counter()

        for ex in ds:
            total_so_far = existing_shard_tokens() + buf_tokens
            if total_so_far >= TARGET_TOKENS:
                break

            text = ex.get("text", "")
            if not isinstance(text, str) or len(text) < 100:
                skipped += 1
                continue
            if len(text) > 5000:
                text = text[:5000]

            ids = tok.encode(text)
            ids.append(EOS)
            buf.extend(ids)
            buf_tokens += len(ids)
            count += 1

            # Flush shard to disk when buffer hits SHARD_SIZE.
            if buf_tokens >= SHARD_SIZE:
                spath = os.path.join(SHARD_DIR, f"shard_{shard_idx:03d}.npy")
                arr = np.array(buf, dtype=np.int32)
                np.save(spath, arr)
                shard_idx += 1
                disk_now = existing_shard_tokens()
                save_progress(disk_now, fw_skip + count)
                buf = []
                buf_tokens = 0
                elapsed = time.perf_counter() - t0
                pct = disk_now / TARGET_TOKENS * 100
                eta = (TARGET_TOKENS - disk_now) / (disk_now / max(elapsed, 1)) / 60
                print(f"  [shard {shard_idx}] {disk_now/1e6:.0f}M tokens "
                      f"({pct:.1f}%), {count:,} ex, ETA {eta:.0f} min", flush=True)

        # Flush remainder.
        if buf_tokens > 0:
            spath = os.path.join(SHARD_DIR, f"shard_{shard_idx:03d}.npy")
            arr = np.array(buf[:max(0, remaining)], dtype=np.int32)
            if len(arr) > 0:
                np.save(spath, arr)
                save_progress(existing_shard_tokens() + len(arr), fw_skip + count)
                print(f"  [final shard] flushed {len(arr):,} tokens", flush=True)

    # ---- Phase 3: concatenate shards into single corpus, truncate to target ----
    print(f"\n{'='*60}", flush=True)
    shards = sorted(glob.glob(os.path.join(SHARD_DIR, "shard_*.npy")))
    print(f"Concatenating {len(shards)} shards...", flush=True)
    arrays = [np.load(s) for s in shards]
    full = np.concatenate(arrays)[:TARGET_TOKENS]
    print(f"FINAL CORPUS: {len(full):,} tokens ({len(full)/1e9:.2f}B)", flush=True)

    unique = len(set(full[:1_000_000].tolist()))
    print(f"  Vocab coverage (sampled 1M): {unique}/{tok.vocab_size} "
          f"({unique/tok.vocab_size*100:.1f}%)", flush=True)

    tensor = torch.from_numpy(full.astype(np.int32))
    torch.save(tensor, OUTPUT)
    size_gb = os.path.getsize(OUTPUT) / 1e9
    print(f"\nSaved: {OUTPUT} ({size_gb:.1f} GB)", flush=True)
    print(f"Load with: torch.load('{OUTPUT}').long()", flush=True)
    print(f"\nAt 5000 tok/s (GPU): {len(full)/5000/3600:.0f}h per epoch", flush=True)
    print(f"At 10000 tok/s (GPU): {len(full)/10000/3600:.0f}h per epoch", flush=True)
    print("\nDONE.", flush=True)


if __name__ == "__main__":
    main()
