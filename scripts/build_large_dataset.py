#!/usr/bin/env python
"""Build a LARGE quality corpus for Fractus-1B (target: ~45M tokens, ~180MB).

Sources (all tested, all work without trust_remote_code):
  - FineWeb sample (web text, diverse knowledge): ~20M tokens
  - Alpaca (instruction QA): ~6M tokens
  - OpenAssistant (human chat): ~10M tokens
  - TinyStories (creative writing): ~8M tokens
  - Dolly (instruction tuning): ~1.5M tokens
"""

import os, sys, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
from datasets import load_dataset
from fractus.tokenizer import FractusTokenizer

OUTPUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                      "data", "quality_corpus_large.pt")

# (name, config, split, field, max_tokens, description)
SOURCES = [
    ("HuggingFaceFW/fineweb", "sample-10BT", "train", "text", 20_000_000, "Web text (FineWeb)"),
    ("tatsu-lab/alpaca", None, "train", "text", 6_000_000, "Instruction QA (Alpaca)"),
    ("OpenAssistant/oasst1", None, "train", "text", 10_000_000, "Human chat (OASST)"),
    ("roneneldan/TinyStories", None, "train", "text", 8_000_000, "Creative writing (TinyStories)"),
    ("databricks/databricks-dolly-15k", None, "train", "response", 1_500_000, "Instruction (Dolly)"),
]

EOS = 50256  # GPT-2 <|endoftext|>


def main():
    tok = FractusTokenizer.gpt2_compatible()
    print(f"Tokenizer: vocab={tok.vocab_size}", flush=True)

    all_tokens = []
    total_collected = 0
    grand_total = sum(s[4] for s in SOURCES)

    for name, config, split, field, max_tokens, desc in SOURCES:
        print(f"\n{'='*60}", flush=True)
        print(f"Loading {desc} ({name})...", flush=True)
        print(f"Target: {max_tokens:,} tokens", flush=True)

        try:
            if config:
                ds = load_dataset(name, config, split=split, streaming=True)
            else:
                ds = load_dataset(name, split=split, streaming=True)

            collected = 0
            examples = 0
            t0 = time.perf_counter()

            for example in ds:
                if collected >= max_tokens:
                    break

                text = example.get(field, "")
                if not isinstance(text, str) or len(text) < 20:
                    continue

                # Tokenize.
                ids = tok.encode(text)
                ids.append(EOS)
                all_tokens.extend(ids)
                collected += len(ids)
                examples += 1

                if examples % 10000 == 0:
                    elapsed = time.perf_counter() - t0
                    rate = collected / max(elapsed, 1)
                    print(f"  {examples:,} examples, {collected:,} tokens "
                          f"({rate:.0f} tok/s)", flush=True)

            total_collected += collected
            elapsed = time.perf_counter() - t0
            print(f"  Done: {examples:,} examples, {collected:,} tokens in {elapsed:.0f}s", flush=True)

        except Exception as e:
            print(f"  FAILED: {e}", flush=True)

        print(f"  Running total: {total_collected:,} tokens", flush=True)

    # Final stats.
    print(f"\n{'='*60}", flush=True)
    print(f"FINAL CORPUS: {len(all_tokens):,} tokens", flush=True)
    print(f"Target was: {grand_total:,} tokens", flush=True)

    # Save.
    os.makedirs(os.path.dirname(OUTPUT), exist_ok=True)
    tensor = torch.tensor(all_tokens, dtype=torch.int32)
    torch.save(tensor, OUTPUT)
    size_mb = os.path.getsize(OUTPUT) / 1e6
    print(f"Saved: {OUTPUT} ({size_mb:.1f} MB)", flush=True)

    unique = len(set(all_tokens))
    print(f"Unique tokens: {unique:,} / {tok.vocab_size} ({unique/tok.vocab_size*100:.1f}% coverage)", flush=True)
    print(f"\nReady: torch.load('{OUTPUT}').long()", flush=True)
    print(f"At 5 tok/s on 1B: {len(all_tokens)/5/3600:.0f} hours per epoch", flush=True)


if __name__ == "__main__":
    main()
