#!/usr/bin/env python
"""Build the Fractus1B Chinchilla corpus — 21B tokens for 1B params.

Target: ~21B tokens (20× params per Chinchilla scaling law for 1.05B params).
All datasets verified NON-GATED with HF Pro token on 2026-07-20.

Mix for a true 1B Fractus — coder that reasons, talks, knows science:
  WEB EDU      ~8B   38%   FineWeb-Edu (quality-filtered web)
  CODE         ~4B   19%   codeparrot-clean + CodeFeedback
  INSTRUCT     ~3B   14%   OpenOrca, Tulu-3 SFT, FLAN-v2
  WEB          ~2B   10%   FineWeb (general web)
  MATH         ~1.5B  7%   Open-Web-Math + Cosmopedia math
  SCIENCE      ~1B    5%   SciQ + Cosmopedia textbooks
  WIKI         ~1B    5%   Wikipedia 2023
  CREATIVE     ~0.5B  2%   TinyStories + Cosmopedia stories

Engineering:
  - int32 dtype (GPT-2 vocab 50257)
  - Sharded incremental save every 200M tokens (crash-safe, less overhead at scale)
  - Resume support via progress.json per source
  - Memory-bounded: flushes to disk
  - EOS (50256) between documents
"""
import os, sys, time, glob, json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
import numpy as np
from datasets import load_dataset
from fractus.tokenizer import FractusTokenizer

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
OUTPUT = os.path.join(DATA_DIR, "fractus_1b_corpus.pt")
SHARD_DIR = os.path.join(DATA_DIR, "fractus_1b_shards")
META = os.path.join(SHARD_DIR, "progress.json")

EOS = 50256
TARGET_TOKENS = 21_000_000_000  # 21B — Chinchilla optimal for 1.05B params
SHARD_SIZE = 200_000_000         # flush every 200M tokens (less I/O at scale)

# Per-source token budgets (sum to ~21B).
SOURCES = [
    # ---- WEB EDU 38% (~8B) — quality-filtered educational web ----
    ("fineweb_edu",   "HuggingFaceFW/fineweb-edu", "sample-10BT", "train", "text", 6000, 8_000_000_000),

    # ---- CODE 19% (~4B) ----
    ("code_python",   "codeparrot/codeparrot-clean", None, "train", "content", 8000, 2_500_000_000),
    ("code_qa",       "m-a-p/CodeFeedback-Filtered-Instruction", None, "train", "query", 5000, 1_000_000_000),
    ("code_py_doc",   "google/code_x_glue_ct_code_to_text", "python", "train", "code", 5000,  250_000_000),
    ("code_go_doc",   "google/code_x_glue_ct_code_to_text", "go",     "train", "code", 5000,   80_000_000),
    ("code_java_doc", "google/code_x_glue_ct_code_to_text", "java",   "train", "code", 5000,   80_000_000),
    ("code_js_doc",   "google/code_x_glue_ct_code_to_text", "javascript", "train", "code", 5000, 80_000_000),

    # ---- INSTRUCT 14% (~3B) — high-quality SFT data ----
    ("openorca",      "Open-Orca/OpenOrca", None, "train", "question", 5000, 1_500_000_000),
    ("tulu3_sft",     "allenai/tulu-3-sft-mixture", None, "train", None, 5000, 800_000_000),
    ("flan_v2",       "SirNeural/flan_v2", None, "train", "inputs", 5000, 700_000_000),

    # ---- WEB 10% (~2B) — general web ----
    ("fineweb",       "HuggingFaceFW/fineweb", "sample-100BT", "train", "text", 6000, 2_000_000_000),

    # ---- MATH 7% (~1.5B) ----
    ("openweb_math",  "open-web-math/open-web-math", None, "train", "text", 6000, 1_000_000_000),
    ("cosmo_math",    "HuggingFaceTB/cosmopedia", "auto_math_text", "train", "text", 5000, 500_000_000),

    # ---- SCIENCE 5% (~1B) ----
    ("sciq",          "allenai/sciq", None, "train", "support", 4000, 100_000_000),
    ("helpsteer",     "nvidia/HelpSteer", None, "train", "response", 4000, 200_000_000),
    ("cosmo_stories", "HuggingFaceTB/cosmopedia", "stories", "train", "text", 4000, 700_000_000),

    # ---- WIKI 5% (~1B) ----
    ("wiki",          "wikimedia/wikipedia", "20231101.en", "train", "text", 6000, 1_000_000_000),

    # ---- CREATIVE 2% (~0.5B) ----
    ("tiny_stories",  "roneneldan/TinyStories", None, "train", "text", 3000, 300_000_000),
    ("openbookqa",    "allenai/openbookqa", "main", "train", "question_stem", 2000, 50_000_000),
    ("dolly",         "databricks/databricks-dolly-15k", None, "train", "response", 4000, 100_000_000),
]


def tok():
    return (FractusTokenizer.gpt2Compatible()
            if hasattr(FractusTokenizer, 'gpt2Compatible')
            else FractusTokenizer.gpt2_compatible())


def load_progress():
    if os.path.exists(META):
        with open(META) as f:
            return json.load(f)
    return {}


def save_progress(progress):
    os.makedirs(SHARD_DIR, exist_ok=True)
    with open(META, "w") as f:
        json.dump(progress, f, indent=2)


def shard_tokens_for(source_name):
    total = 0
    for p in glob.glob(os.path.join(SHARD_DIR, f"{source_name}_*.npy")):
        arr = np.load(p, mmap_mode="r")
        total += len(arr)
    return total


def total_shard_tokens():
    total = 0
    for p in glob.glob(os.path.join(SHARD_DIR, "*_*.npy")):
        arr = np.load(p, mmap_mode="r")
        total += len(arr)
    return total


def extract_text(name, ex, field):
    """Pull the best text out of an example, with per-source formatting."""
    def g(k):
        v = ex.get(k, "") if isinstance(ex, dict) else ""
        if isinstance(v, list):
            # tulu-3 / chat-style: join message contents
            parts = []
            for m in v:
                if isinstance(m, dict) and "content" in m:
                    parts.append(str(m.get("content", "")))
            return "\n".join(parts)
        return v if isinstance(v, str) else ""

    if name == "tulu3_sft":
        return g("messages")
    if name in ("openorca", "flan_v2"):
        instr = g("question") or g("inputs") or g("prompt")
        out = g("response") or g("targets") or g("answer")
        return f"{instr}\n\n{out}" if instr and out else (out or instr)
    if name == "helpsteer":
        return g("response") or g("prompt")
    if name in ("sciq", "openbookqa"):
        q = g("question") or g("question_stem") or g("support")
        a = g("correct_answer") or g("answer") or g("support")
        return f"{q}\n\n{a}" if q and a else (a or q)
    if name.startswith("cosmo"):
        return g("text")
    # default
    return g(field)


def stream_source(name, dataset, config, split, field, max_chars, target,
                  tokenizer, progress):
    already = shard_tokens_for(name)
    if already >= target:
        print(f"  [{name}] already {already/1e6:.0f}M >= target {target/1e6:.0f}M, skip", flush=True)
        return already

    consumed = progress.get(name + "_examples", 0)
    if consumed:
        print(f"  [{name}] resuming, skip first {consumed:,} examples", flush=True)

    try:
        if config:
            ds = load_dataset(dataset, config, split=split, streaming=True)
        else:
            ds = load_dataset(dataset, split=split, streaming=True)
    except Exception as e:
        print(f"  [{name}] FAILED to load dataset: {e}", flush=True)
        return already

    if consumed:
        try:
            ds = ds.skip(consumed)
        except Exception:
            pass

    buf = []
    buf_tokens = 0
    shard_idx = len(glob.glob(os.path.join(SHARD_DIR, f"{name}_*.npy")))
    count = 0
    t0 = time.perf_counter()

    for ex in ds:
        have = shard_tokens_for(name) + buf_tokens
        if have >= target:
            break

        text = extract_text(name, ex, field)
        if not isinstance(text, str) or len(text) < 20:
            continue
        if len(text) > max_chars:
            text = text[:max_chars]

        ids = tokenizer.encode(text)
        ids.append(EOS)
        buf.extend(ids)
        buf_tokens += len(ids)
        count += 1

        if buf_tokens >= SHARD_SIZE:
            spath = os.path.join(SHARD_DIR, f"{name}_{shard_idx:03d}.npy")
            np.save(spath, np.array(buf, dtype=np.int32))
            shard_idx += 1
            disk_now = shard_tokens_for(name)
            progress[name + "_examples"] = consumed + count
            save_progress(progress)
            buf = []
            buf_tokens = 0
            elapsed = time.perf_counter() - t0
            pct = disk_now / target * 100
            eta = (target - disk_now) / (disk_now / max(elapsed, 1)) / 60
            print(f"  [{name}] {disk_now/1e6:.0f}M/{target/1e6:.0f}M tokens "
                  f"({pct:.1f}%), {count:,} ex, ETA {eta:.0f}min", flush=True)

    if buf_tokens > 0:
        spath = os.path.join(SHARD_DIR, f"{name}_{shard_idx:03d}.npy")
        remaining = target - shard_tokens_for(name)
        if remaining > 0:
            np.save(spath, np.array(buf[:remaining], dtype=np.int32))
        progress[name + "_examples"] = consumed + count
        save_progress(progress)

    have = shard_tokens_for(name)
    print(f"  [{name}] done: {have/1e6:.0f}M tokens ({count:,} examples)", flush=True)
    return have


def main():
    os.makedirs(SHARD_DIR, exist_ok=True)
    tokenizer = tok()
    progress = load_progress()

    print("="*64, flush=True)
    print("FRACTUS1B CHINCHILLA CORPUS — 21B tokens for true 1B params", flush=True)
    print(f"Target: {TARGET_TOKENS/1e9:.1f}B tokens | int32 | shards every {SHARD_SIZE/1e6:.0f}M", flush=True)
    print(f"Tokenizer vocab: {tokenizer.vocab_size}", flush=True)
    print("="*64, flush=True)

    grand_total_budget = sum(s[6] for s in SOURCES)
    print(f"\nSource budgets sum: {grand_total_budget/1e9:.1f}B tokens\n", flush=True)

    for src in SOURCES:
        name, dataset, config, split, field, max_chars, target = src
        print(f"[Source] {name}: target {target/1e6:.0f}M tokens", flush=True)
        try:
            stream_source(name, dataset, config, split, field, max_chars, target,
                          tokenizer, progress)
        except Exception as e:
            print(f"  [{name}] ERROR: {e} — continuing to next source", flush=True)
        progress[name + "_done"] = True
        save_progress(progress)
        print(f"  Running total on disk: {total_shard_tokens()/1e9:.2f}B tokens\n", flush=True)

    # Concatenate all shards into the final corpus.
    print("="*64, flush=True)
    shards = sorted(glob.glob(os.path.join(SHARD_DIR, "*_*.npy")))
    print(f"Concatenating {len(shards)} shards...", flush=True)
    arrays = [np.load(s) for s in shards]
    full = np.concatenate(arrays)[:TARGET_TOKENS]
    print(f"FINAL CORPUS: {len(full):,} tokens ({len(full)/1e9:.2f}B)", flush=True)

    sample = full[:2_000_000].tolist()
    unique = len(set(sample))
    print(f"  Vocab coverage (2M sample): {unique}/{tokenizer.vocab_size} "
          f"({unique/tokenizer.vocab_size*100:.1f}%)", flush=True)

    tensor = torch.from_numpy(full.astype(np.int32))
    torch.save(tensor, OUTPUT)
    size_gb = os.path.getsize(OUTPUT) / 1e9
    print(f"\nSaved: {OUTPUT} ({size_gb:.1f} GB)", flush=True)
    print(f"\nAt 10000 tok/s (GPU): {len(full)/10000/3600/24:.1f} days per epoch", flush=True)
    print(f"At 50000 tok/s (multi-GPU): {len(full)/50000/3600/24:.1f} days per epoch", flush=True)
    print("\nDONE.", flush=True)


if __name__ == "__main__":
    main()
