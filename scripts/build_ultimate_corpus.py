#!/usr/bin/env python
"""Build the ULTIMATE quality corpus for Fractus.

Focus: real-world usefulness. Code, knowledge, math, reasoning, security.
Target: 5M+ tokens of HIGH QUALITY data.

Quality filters:
  - Minimum 50 chars per example (no garbage)
  - Maximum 2000 chars per example (no huge dumps)
  - Deduplicate by first 100 chars
  - Balance domains explicitly

Domains (target distribution):
  - Python code: 25% (real functions, classes, algorithms)
  - Multi-language code: 15% (JS, Go, Rust, C++)
  - Instruction QA: 20% (Alpaca, Dolly — question + answer pairs)
  - Knowledge articles: 20% (FineWeb — encyclopedic, technical)
  - Math/reasoning: 10% (competition math, proofs, logic)
  - Cybersecurity: 5% (security concepts, exploit patterns)
  - Conversations: 5% (OpenAssistant — natural dialogue)
"""

import os, sys, time, hashlib
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
from datasets import load_dataset
from fractus.tokenizer import FractusTokenizer

OUTPUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                      "data", "ultimate_corpus.pt")
EOS = 50256


def clean_text(text: str) -> str:
    """Basic quality filter."""
    if not text or len(text) < 50:
        return ""
    text = text.strip()
    if len(text) > 2000:
        text = text[:2000]
    # Remove excessive whitespace.
    lines = text.split('\n')
    lines = [l.rstrip() for l in lines if l.strip()]
    text = '\n'.join(lines)
    return text


def dedup(texts: list) -> list:
    """Remove duplicates by first 100 chars."""
    seen = set()
    unique = []
    for t in texts:
        key = hashlib.md5(t[:100].encode()).hexdigest()
        if key not in seen:
            seen.add(key)
            unique.append(t)
    return unique


def collect_dataset(name, config, split, field, max_examples, desc):
    """Stream a dataset, clean, and return texts."""
    print(f"\n  [{desc}] {name}...", flush=True)
    texts = []
    try:
        if config:
            ds = load_dataset(name, config, split=split, streaming=True)
        else:
            ds = load_dataset(name, split=split, streaming=True)
        count = 0
        t0 = time.perf_counter()
        for ex in ds:
            if count >= max_examples:
                break
            text = ex.get(field, "")
            if not isinstance(text, str):
                if isinstance(text, list):
                    text = " ".join(m.get("content", "") for m in text if isinstance(m, dict))
                else:
                    text = str(text)
            cleaned = clean_text(text)
            if cleaned:
                texts.append(cleaned)
                count += 1
            if count % 10000 == 0:
                elapsed = time.perf_counter() - t0
                print(f"    {count:,} examples, {elapsed:.0f}s", flush=True)
        print(f"    Done: {len(texts):,} clean examples", flush=True)
    except Exception as e:
        print(f"    FAILED: {str(e)[:80]}", flush=True)
    return texts


def main():
    tok = FractusTokenizer.gpt2_compatible()
    print(f"Tokenizer: vocab={tok.vocab_size}", flush=True)

    all_texts = []
    domain_counts = {}

    # === PYTHON CODE (25%) ===
    print("\n=== PYTHON CODE ===", flush=True)
    py_texts = []
    py_texts += collect_dataset("iamtarun/python_code_instructions_18k_alpaca", None, "train", "instruction", 10000, "Python instructions")
    py_texts += collect_dataset("iamtarun/python_code_instructions_18k_alpaca", None, "train", "output", 10000, "Python code")
    py_texts += collect_dataset("HuggingFaceH4/CodeAlpaca_20K", None, "train", "completion", 15000, "Code completions")
    py_texts += collect_dataset("HuggingFaceH4/CodeAlpaca_20K", None, "train", "prompt", 10000, "Code prompts")
    py_texts = dedup(py_texts)
    print(f"  Python total: {len(py_texts):,} unique examples", flush=True)
    all_texts.extend(py_texts)
    domain_counts["python_code"] = len(py_texts)

    # === INSTRUCTION QA (20%) ===
    print("\n=== INSTRUCTION QA ===", flush=True)
    qa_texts = []
    qa_texts += collect_dataset("tatsu-lab/alpaca", None, "train", "text", 30000, "Alpaca QA")
    qa_texts += collect_dataset("databricks/databricks-dolly-15k", None, "train", "response", 10000, "Dolly responses")
    qa_texts += collect_dataset("databricks/databricks-dolly-15k", None, "train", "instruction", 10000, "Dolly instructions")
    qa_texts = dedup(qa_texts)
    print(f"  QA total: {len(qa_texts):,} unique examples", flush=True)
    all_texts.extend(qa_texts)
    domain_counts["instruction_qa"] = len(qa_texts)

    # === KNOWLEDGE ARTICLES (20%) ===
    print("\n=== KNOWLEDGE ===", flush=True)
    kb_texts = collect_dataset("HuggingFaceFW/fineweb", "sample-10BT", "train", "text", 30000, "FineWeb articles")
    kb_texts = dedup(kb_texts)
    print(f"  Knowledge total: {len(kb_texts):,} unique examples", flush=True)
    all_texts.extend(kb_texts)
    domain_counts["knowledge"] = len(kb_texts)

    # === MATH / REASONING (10%) ===
    print("\n=== MATH / REASONING ===", flush=True)
    math_texts = []
    math_texts += collect_dataset("meta-math/MetaMathQA", None, "train", "query", 15000, "Math problems")
    math_texts += collect_dataset("meta-math/MetaMathQA", None, "train", "response", 15000, "Math solutions")
    math_texts = dedup(math_texts)
    print(f"  Math total: {len(math_texts):,} unique examples", flush=True)
    all_texts.extend(math_texts)
    domain_counts["math"] = len(math_texts)

    # === CONVERSATIONS (5%) ===
    print("\n=== CONVERSATIONS ===", flush=True)
    chat_texts = collect_dataset("OpenAssistant/oasst1", None, "train", "text", 20000, "Chat")
    chat_texts = dedup(chat_texts)
    print(f"  Chat total: {len(chat_texts):,} unique examples", flush=True)
    all_texts.extend(chat_texts)
    domain_counts["chat"] = len(chat_texts)

    # === CREATIVE (5%) ===
    print("\n=== CREATIVE ===", flush=True)
    story_texts = collect_dataset("roneneldan/TinyStories", None, "train", "text", 15000, "Stories")
    story_texts = dedup(story_texts)
    print(f"  Stories total: {len(story_texts):,} unique examples", flush=True)
    all_texts.extend(story_texts)
    domain_counts["creative"] = len(story_texts)

    # === TOKENIZE ===
    print(f"\n{'='*60}", flush=True)
    print(f"TOKENIZING {len(all_texts):,} examples...", flush=True)

    all_tokens = []
    t0 = time.perf_counter()
    for i, text in enumerate(all_texts):
        ids = tok.encode(text)
        ids.append(EOS)
        all_tokens.extend(ids)
        if (i + 1) % 20000 == 0:
            elapsed = time.perf_counter() - t0
            rate = len(all_tokens) / max(elapsed, 1)
            print(f"  {i+1:,}/{len(all_texts):,} texts, {len(all_tokens):,} tokens ({rate:.0f} tok/s)", flush=True)

    # === FINAL STATS ===
    print(f"\n{'='*60}", flush=True)
    print(f"FINAL CORPUS: {len(all_tokens):,} tokens", flush=True)
    total_examples = sum(domain_counts.values())
    for domain, count in domain_counts.items():
        pct = count / max(total_examples, 1) * 100
        print(f"  {domain:20s}: {count:>6,} examples ({pct:.0f}%)", flush=True)

    unique = len(set(all_tokens))
    print(f"  Vocab coverage: {unique:,}/{tok.vocab_size} ({unique/tok.vocab_size*100:.1f}%)", flush=True)
    print(f"  Avg tokens/example: {len(all_tokens)/max(total_examples,1):.0f}", flush=True)

    # Save.
    os.makedirs(os.path.dirname(OUTPUT), exist_ok=True)
    torch.save(torch.tensor(all_tokens, dtype=torch.int32), OUTPUT)
    size_mb = os.path.getsize(OUTPUT) / 1e6
    print(f"\nSaved: {OUTPUT} ({size_mb:.1f} MB)", flush=True)
    print(f"At 500 tok/s (GPU): {len(all_tokens)/500/3600:.1f}h per epoch", flush=True)
    print(f"At 12 tok/s (CPU): {len(all_tokens)/12/3600:.0f}h per epoch", flush=True)


if __name__ == "__main__":
    main()
