#!/usr/bin/env python
"""Download quality datasets, tokenize, and cache for Fractus-1B training.

Sources:
  - Alpaca (52k QA instruction-response pairs)
  - OpenAssistant (82k human chat messages)  
  - Databricks Dolly (15k instruction tuning)

Target: 500k tokens of high-quality instruction/conversational data.
"""

import os, sys, time, json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from datasets import load_dataset
from fractus.tokenizer import FractusTokenizer

OUTPUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                      "data", "quality_corpus.pt")

SOURCES = [
    # (name, split, field, max_examples, format)
    ("tatsu-lab/alpaca", "train", "text", 52000, "direct"),
    ("OpenAssistant/oasst1", "train", "text", 80000, "direct"),
    ("databricks/databricks-dolly-15k", "train", "response", 15000, "direct"),
]


def extract_text(example, field, fmt):
    """Extract text from a dataset example."""
    text = example.get(field, "")
    if isinstance(text, str):
        return text
    if isinstance(text, list):
        # Chat format: list of message dicts.
        return " ".join(m.get("content", "") for m in text if isinstance(m, dict))
    return str(text)


def main():
    tok = FractusTokenizer.gpt2_compatible()
    print(f"Tokenizer: vocab={tok.vocab_size}", flush=True)

    all_tokens = []
    
    for name, split, field, max_ex, fmt in SOURCES:
        print(f"\nLoading {name}...", flush=True)
        try:
            ds = load_dataset(name, split=split, streaming=True)
            count = 0
            t0 = time.perf_counter()
            
            for example in ds:
                if count >= max_ex:
                    break
                text = extract_text(example, field, fmt)
                if len(text) < 20:
                    continue
                # Tokenize.
                ids = tok.encode(text)
                # Add a separator token (EOS-like).
                ids.append(50256)  # GPT-2 <|endoftext|>
                all_tokens.extend(ids)
                count += 1
                
                if count % 10000 == 0:
                    elapsed = time.perf_counter() - t0
                    print(f"  {count:,} examples, {len(all_tokens):,} tokens, "
                          f"{elapsed:.0f}s", flush=True)
            
            print(f"  Done: {count:,} examples, {len(all_tokens):,} total tokens", flush=True)
        except Exception as e:
            print(f"  FAILED: {e}", flush=True)

    # Truncate to 500k tokens (keep the first 500k for consistency).
    TARGET = 500000
    if len(all_tokens) > TARGET:
        all_tokens = all_tokens[:TARGET]
    
    print(f"\nFinal corpus: {len(all_tokens):,} tokens", flush=True)
    
    # Save.
    os.makedirs(os.path.dirname(OUTPUT), exist_ok=True)
    tensor = torch.tensor(all_tokens, dtype=torch.int32)
    torch.save(tensor, OUTPUT)
    size_mb = os.path.getsize(OUTPUT) / 1e6
    print(f"Saved: {OUTPUT} ({size_mb:.1f} MB)", flush=True)
    
    # Stats.
    unique = len(set(all_tokens))
    print(f"Unique tokens: {unique:,} / {tok.vocab_size}", flush=True)
    print(f"Vocab coverage: {unique/tok.vocab_size*100:.1f}%", flush=True)
    print(f"\nReady for training: torch.load('{OUTPUT}').long()", flush=True)


if __name__ == "__main__":
    main()
