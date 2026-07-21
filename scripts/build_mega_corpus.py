#!/usr/bin/env python
"""Build the BIGGEST quality corpus for Fractus-1B.

Combines:
  - FineWeb (web text / universal knowledge)
  - Alpaca (instruction QA)
  - OpenAssistant (human chat)
  - TinyStories (creative writing)
  - Python code instructions (18k code examples)
  - CodeAlpaca (prompt + completion format)
  - Dolly (instruction tuning)

Target: 2M+ tokens with heavy code/knowledge focus.
"""
import os, sys, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
from datasets import load_dataset
from fractus.tokenizer import FractusTokenizer

OUTPUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                      "data", "mega_corpus.pt")
EOS = 50256

# (name, config, field, max_tokens, desc, is_code)
SOURCES = [
    # CODE (40%)
    ("iamtarun/python_code_instructions_18k_alpaca", None, "output", 1_000_000, "Python code", True),
    ("iamtarun/python_code_instructions_18k_alpaca", None, "instruction", 500_000, "Code prompts", True),
    ("HuggingFaceH4/CodeAlpaca_20K", None, "completion", 1_500_000, "Code completions", True),
    ("HuggingFaceH4/CodeAlpaca_20K", None, "prompt", 500_000, "Code prompts", True),

    # KNOWLEDGE (30%)
    ("HuggingFaceFW/fineweb", "sample-10BT", "text", 3_000_000, "Web knowledge", False),

    # INSTRUCTIONS (15%)
    ("tatsu-lab/alpaca", None, "text", 2_000_000, "Instruction QA", False),
    ("databricks/databricks-dolly-15k", None, "response", 1_000_000, "Dolly", False),

    # CHAT + CREATIVE (15%)
    ("OpenAssistant/oasst1", None, "text", 2_000_000, "Chat", False),
    ("roneneldan/TinyStories", None, "text", 1_500_000, "Stories", False),
]


def main():
    tok = FractusTokenizer.gpt2_compatible()
    print(f"Tokenizer: vocab={tok.vocab_size}", flush=True)

    all_tokens = []
    total = 0
    code_tokens = 0
    knowledge_tokens = 0

    for name, config, field, max_tok, desc, is_code in SOURCES:
        print(f"\n{'='*60}", flush=True)
        print(f"[{'CODE' if is_code else 'TEXT'}] {desc} ({name})", flush=True)
        print(f"Target: {max_tok:,} tokens", flush=True)

        try:
            if config:
                ds = load_dataset(name, config, split="train", streaming=True)
            else:
                ds = load_dataset(name, split="train", streaming=True)

            collected = 0
            examples = 0
            t0 = time.perf_counter()

            for example in ds:
                if collected >= max_tok:
                    break
                text = example.get(field, "")
                if not isinstance(text, str) or len(text) < 15:
                    continue

                ids = tok.encode(text)
                ids.append(EOS)
                all_tokens.extend(ids)
                collected += len(ids)
                examples += 1

                if examples % 10000 == 0:
                    elapsed = time.perf_counter() - t0
                    rate = collected / max(elapsed, 1)
                    print(f"  {examples:,} ex, {collected:,} tok ({rate:.0f} tok/s)", flush=True)

            total += collected
            if is_code:
                code_tokens += collected
            else:
                knowledge_tokens += collected

            elapsed = time.perf_counter() - t0
            print(f"  Done: {examples:,} ex, {collected:,} tok in {elapsed:.0f}s", flush=True)

        except Exception as e:
            print(f"  FAILED: {str(e)[:100]}", flush=True)

        print(f"  Running total: {total:,} tokens ({code_tokens:,} code, {knowledge_tokens:,} text)", flush=True)

    print(f"\n{'='*60}", flush=True)
    print(f"FINAL: {len(all_tokens):,} tokens", flush=True)
    print(f"  Code: {code_tokens:,} ({code_tokens/total*100:.0f}%)", flush=True)
    print(f"  Text: {knowledge_tokens:,} ({knowledge_tokens/total*100:.0f}%)", flush=True)

    os.makedirs(os.path.dirname(OUTPUT), exist_ok=True)
    torch.save(torch.tensor(all_tokens, dtype=torch.int32), OUTPUT)
    size_mb = os.path.getsize(OUTPUT) / 1e6
    print(f"Saved: {OUTPUT} ({size_mb:.1f} MB)", flush=True)

    unique = len(set(all_tokens))
    print(f"Vocab coverage: {unique:,}/{tok.vocab_size} ({unique/tok.vocab_size*100:.1f}%)", flush=True)
    print(f"At 19 tok/s on 1B: {len(all_tokens)/19/3600:.0f}h per epoch", flush=True)


if __name__ == "__main__":
    main()
