#!/usr/bin/env python
"""Build the Fractus Chinchilla corpus — diversified for a coder that talks.

Target: 1.76B tokens (20x params per Chinchilla scaling law for 88M params).

Mix (so Fractus is a real coder that talks and reasons, not a web text mimic):
  CODE         ~700M  40%   the-stack (Python/JS/Rust/Go/C++)   -> real code
  INSTRUCT     ~440M  25%   OpenOrca, OASST1, Dolly              -> talk/answer
  WEB          ~350M  20%   FineWeb sample                       -> natural text
  WIKI         ~100M   6%   wikipedia                            -> knowledge
  CREATIVE     ~130M  7.5%  TinyStories + narratives             -> narrative
  EXISTING      ~40M  2.5%  communication_corpus + ultimate      -> seeded quality

Engineering:
  - int32 dtype (GPT-2 vocab 50257 > int16 max 32767 — v1 crashed here)
  - Sharded incremental save every 50M tokens (crash = max 50M lost)
  - Resume support via progress.json per source
  - Memory-bounded: flushes to disk instead of holding 1.76B tokens in RAM
  - EOS (50256) between documents
"""
import os, sys, time, glob, json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
import numpy as np
from datasets import load_dataset
from fractus.tokenizer import FractusTokenizer

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
OUTPUT = os.path.join(DATA_DIR, "fractus_corpus.pt")          # final unified corpus
SHARD_DIR = os.path.join(DATA_DIR, "fractus_shards")           # per-source shards
META = os.path.join(SHARD_DIR, "progress.json")

EOS = 50256
TARGET_TOKENS = 1_760_000_000
SHARD_SIZE = 50_000_000  # 50M tokens per shard flush

# Per-source token budgets (sum to ~1.76B).
# All datasets verified NON-GATED and accessible with HF_TOKEN on 2026-07-17.
SOURCES = [
    # name, dataset, config, split, text_field, max_chars, target_tokens
    # ---- CODE 40% (~700M) ----
    ("code_python",   "codeparrot/codeparrot-clean", None, "train", "content", 6000, 350_000_000),
    ("code_qa",       "m-a-p/CodeFeedback-Filtered-Instruction", None, "train", "query", 4000, 150_000_000),
    ("code_py_doc",   "google/code_x_glue_ct_code_to_text", "python", "train", "code",     5000,  40_000_000),
    ("code_go_doc",   "google/code_x_glue_ct_code_to_text", "go",     "train", "code",     5000,  35_000_000),
    ("code_java_doc", "google/code_x_glue_ct_code_to_text", "java",   "train", "code",     5000,  35_000_000),
    ("code_js_doc",   "google/code_x_glue_ct_code_to_text", "javascript","train","code",  5000,  35_000_000),
    ("code_ruby_doc", "google/code_x_glue_ct_code_to_text", "ruby",   "train", "code",     5000,  30_000_000),
    ("code_php_doc",  "google/code_x_glue_ct_code_to_text", "php",    "train", "code",     5000,  25_000_000),
    # ---- INSTRUCT/CHAT 25% (~440M) ----
    ("instruct_orca",  "Open-Orca/OpenOrca",       None,      "train", "question", 4000, 160_000_000),
    ("instruct_oasst", "OpenAssistant/oasst1",     None,      "train", "text",     4000, 100_000_000),
    ("instruct_dolly", "databricks/databricks-dolly-15k", None, "train", "response", 4000, 60_000_000),
    ("instruct_alpaca","yahma/alpaca-cleaned",     None,      "train", "output",   4000, 120_000_000),
    # ---- WEB 20% (~350M) ----
    ("web_fineweb",   "HuggingFaceFW/fineweb",     "sample-10BT", "train", "text", 5000, 350_000_000),
    # ---- KNOWLEDGE 6% (~100M) ----
    ("wiki",          "wikimedia/wikipedia",       "20231101.en", "train", "text", 5000, 100_000_000),
    # ---- CREATIVE 7.5% (~130M) ----
    ("creative_stories","roneneldan/TinyStories",  None,      "train", "text",   3000, 100_000_000),
    ("creative_cosmo","HuggingFaceTB/cosmopedia-100k", None,  "train", "text",   5000,  30_000_000),
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
    """Pull the best text out of an example, with per-source formatting.

    For Q&A / instruct datasets we combine the prompt + answer so the model
    learns the conversation pattern, not just the answer in isolation.
    """
    def g(k):
        v = ex.get(k, "") if isinstance(ex, dict) else ""
        return v if isinstance(v, str) else ""

    if name == "code_qa":                       # query + answer
        q, a = g("query"), g("answer")
        return f"Question: {q}\n\nAnswer: {a}" if q or a else ""

    if name.startswith("instruct_"):            # instruction + output
        instr = g("instruction") or g("question") or g("prompt") or g("text")
        out = g("output") or g("response") or g("answer")
        inp = g("input")
        if instr and out:
            return (f"{instr}\n{inp}\n\n{out}" if inp else f"{instr}\n\n{out}")
        return out or instr

    if name.startswith("code_") and "docstring" in ex:
        # code_x_glue: pair code with its docstring so model learns the mapping.
        code = g("code") or g("original_string")
        doc = g("docstring")
        return f"{doc}\n\n{code}" if doc else code

    if name == "creative_cosmo":
        return g("text")

    # Default: just the named field.
    return g(field)


def stream_source(name, dataset, config, split, field, max_chars, target,
                  tokenizer, progress, quiet=False):
    """Stream one source until we have `target` tokens saved for it."""
    already = shard_tokens_for(name)
    if already >= target:
        if not quiet:
            print(f"  [{name}] already {already/1e6:.0f}M >= target {target/1e6:.0f}M, skip", flush=True)
        return already

    # Resume: skip examples already consumed.
    consumed = progress.get(name + "_examples", 0)
    if not quiet:
        print(f"  [{name}] loading dataset (config={config}, split={split})...", flush=True)
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
    skipped = 0
    t0 = time.perf_counter()

    for ex in ds:
        have = shard_tokens_for(name) + buf_tokens
        if have >= target:
            break

        text = extract_text(name, ex, field)

        if not isinstance(text, str) or len(text) < 20:
            skipped += 1
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
            buf = []
            buf_tokens = 0
            progress[name + "_examples"] = consumed + count
            save_progress(progress)
            have = shard_tokens_for(name)
            elapsed = time.perf_counter() - t0
            eta = (target - have) / (have / max(elapsed, 1)) / 60
            print(f"  [{name}] {have/1e6:.0f}M/{target/1e6:.0f}M tokens "
                  f"({have/target*100:.1f}%), {count:,} ex, ETA {eta:.0f}min", flush=True)

    # Flush remainder.
    if buf_tokens > 0:
        spath = os.path.join(SHARD_DIR, f"{name}_{shard_idx:03d}.npy")
        np.save(spath, np.array(buf, dtype=np.int32))
        progress[name + "_examples"] = consumed + count
        save_progress(progress)

    have = shard_tokens_for(name)
    print(f"  [{name}] done: {have/1e6:.0f}M tokens ({count:,} examples)", flush=True)
    return have


def load_existing_local(progress):
    """Phase 0: pull in communication + ultimate corpora as shards."""
    name = "existing"
    if shard_tokens_for(name) > 0:
        print(f"  [existing] already loaded, skip", flush=True)
        return
    buf = []
    for fname in ["communication_corpus.pt", "ultimate_corpus.pt"]:
        path = os.path.join(DATA_DIR, fname)
        if os.path.exists(path):
            t = torch.load(path, weights_only=False).long()
            # Clip to int32 range defensively.
            t = t.clamp(0, EOS)
            buf.extend(t.tolist())
            print(f"  [existing] loaded {fname}: {len(t):,} tokens", flush=True)
    if buf:
        np.save(os.path.join(SHARD_DIR, f"{name}_000.npy"),
                np.array(buf, dtype=np.int32))
        print(f"  [existing] saved: {len(buf)/1e6:.1f}M tokens", flush=True)
    progress["existing_done"] = True
    save_progress(progress)


def main():
    os.makedirs(SHARD_DIR, exist_ok=True)
    tokenizer = tok()
    progress = load_progress()

    print("="*64, flush=True)
    print("FRACTUS CHINCHILLA CORPUS — diversified for a coder that talks", flush=True)
    print(f"Target: {TARGET_TOKENS/1e9:.2f}B tokens | int32 | shards every {SHARD_SIZE/1e6:.0f}M", flush=True)
    print(f"Tokenizer vocab: {tokenizer.vocab_size}", flush=True)
    print("="*64, flush=True)

    # Phase 0: existing quality corpora.
    print("\n[Phase 0] Loading existing local corpora...", flush=True)
    load_existing_local(progress)

    # Phases 1..N: each diversified source.
    grand_total_budget = sum(s[6] for s in SOURCES)
    print(f"\nSource budgets sum: {grand_total_budget/1e6:.0f}M tokens", flush=True)
    print(f"Plus existing ~40M => ~{(grand_total_budget+40e6)/1e9:.2f}B total\n", flush=True)

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
        print(f"  Running total on disk: {total_shard_tokens()/1e6:.0f}M tokens\n", flush=True)

    # Final phase: concatenate all shards, truncate to exact Chinchilla target.
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
    print(f"\nSaved: {OUTPUT} ({size_gb:.2f} GB)", flush=True)
    print(f"Load with: torch.load('{OUTPUT}').long()", flush=True)
    print(f"\nAt  5000 tok/s (GPU): {len(full)/5000/3600:.1f}h per epoch", flush=True)
    print(f"At 10000 tok/s (GPU): {len(full)/10000/3600:.1f}h per epoch", flush=True)
    print(f"At 20000 tok/s (4090): {len(full)/20000/3600:.1f}h per epoch", flush=True)
    print("\nDONE.", flush=True)


if __name__ == "__main__":
    main()
