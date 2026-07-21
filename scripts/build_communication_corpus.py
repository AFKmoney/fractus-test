#!/usr/bin/env python
"""Build the COMMUNICATION corpus for Fractus.

Focus: code communication (explanations, docs, reviews, dev conversations)
+ creative communication (rap lyrics, storytelling).

Datasets that work:
  - Suzhen/CodeChat: dev-LLM conversations (multi-turn, code context)
  - kaanrkaraman/code2doc: function + documentation pairs
  - codeparrot/github-jupyter-code-to-text: notebooks with explanations
  - Cropinky/rap_lyrics_english: rap lyrics for flow/style
  - tatsu-lab/alpaca: instruction QA
  - iamtarun/python_code_instructions_18k_alpaca: Python code + instructions
  - OpenAssistant/oasst1: human chat
  - roneneldan/TinyStories: creative writing

Mix: 40% code communication, 30% code, 15% instruction, 10% creative, 5% chat
"""

import os, sys, time, hashlib, json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
from datasets import load_dataset
from fractus.tokenizer import FractusTokenizer

OUTPUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                      "data", "communication_corpus.pt")
EOS = 50256


def clean(text):
    if not text or not isinstance(text, str) or len(text) < 40:
        return ""
    text = text.strip()[:3000]
    lines = [l.rstrip() for l in text.split('\n') if l.strip()]
    return '\n'.join(lines)


def dedup(texts):
    seen = set()
    unique = []
    for t in texts:
        key = hashlib.md5(t[:80].encode()).hexdigest()
        if key not in seen:
            seen.add(key)
            unique.append(t)
    return unique


def collect(name, config, field, max_ex, desc, formatter=None):
    """Stream dataset, extract + clean text."""
    print(f"\n  [{desc}] {name}...", flush=True)
    texts = []
    try:
        ds = load_dataset(name, config, split="train", streaming=True) if config \
            else load_dataset(name, split="train", streaming=True)
        count = 0
        for ex in ds:
            if count >= max_ex:
                break
            if formatter:
                text = formatter(ex)
            else:
                text = ex.get(field, "")
                if isinstance(text, list):
                    text = " ".join(m.get("content", "") for m in text if isinstance(m, dict))
            cleaned = clean(text)
            if cleaned:
                texts.append(cleaned)
                count += 1
            if count % 10000 == 0:
                print(f"    {count:,} examples", flush=True)
        print(f"    Done: {len(texts):,}", flush=True)
    except Exception as e:
        print(f"    FAILED: {str(e)[:80]}", flush=True)
    return texts


def codechat_formatter(ex):
    """Extract conversation turns from CodeChat list format."""
    conv = ex.get('conversation', [])
    if not isinstance(conv, list):
        return ""
    turns = []
    for msg in conv:
        if isinstance(msg, dict):
            role = msg.get('role', 'user')
            content = msg.get('content', '')
            if content and len(content) > 10:
                turns.append(f"{role}: {content}")
    return '\n'.join(turns[:6])  # limit to 6 turns


def main():
    tok = FractusTokenizer.gpt2_compatible()
    print(f"Tokenizer: vocab={tok.vocab_size}", flush=True)
    all_texts = []
    domains = {}

    # === CODE COMMUNICATION (40%) ===
    print("\n=== CODE COMMUNICATION ===", flush=True)
    comm = []

    # Dev conversations (multi-turn with code context).
    comm += collect("Suzhen/CodeChat", None, "conversation", 20000,
                     "Dev-LLM conversations", formatter=codechat_formatter)

    # Code + documentation pairs (function → explanation).
    docs = collect("kaanrkaraman/code2doc", None, "documentation", 10000, "Code docs")
    code_for_docs = collect("kaanrkaraman/code2doc", None, "function_code", 10000, "Code for docs")
    # Combine: "def foo():\n  ...\n\n# Documentation\n explanation"
    for i in range(min(len(docs), len(code_for_docs))):
        combined = f"{code_for_docs[i]}\n\n# Explanation\n{docs[i]}"
        comm.append(clean(combined))

    # Jupyter notebooks (code + markdown narrative).
    comm += collect("codeparrot/github-jupyter-code-to-text", None, "content", 5000,
                     "Jupyter notebooks (code+narrative)")

    comm = dedup(comm)
    print(f"  Communication total: {len(comm):,}", flush=True)
    all_texts.extend(comm)
    domains["code_communication"] = len(comm)

    # === RAW CODE (30%) ===
    print("\n=== RAW CODE ===", flush=True)
    code = []
    code += collect("iamtarun/python_code_instructions_18k_alpaca", None, "output", 15000, "Python code")
    code += collect("iamtarun/python_code_instructions_18k_alpaca", None, "instruction", 10000, "Python instructions")
    code += collect("HuggingFaceH4/CodeAlpaca_20K", None, "completion", 15000, "Multi-lang code")
    code += collect("HuggingFaceH4/CodeAlpaca_20K", None, "prompt", 10000, "Code prompts")
    code = dedup(code)
    print(f"  Code total: {len(code):,}", flush=True)
    all_texts.extend(code)
    domains["raw_code"] = len(code)

    # === INSTRUCTION QA (15%) ===
    print("\n=== INSTRUCTION QA ===", flush=True)
    qa = []
    qa += collect("tatsu-lab/alpaca", None, "text", 20000, "Alpaca")
    qa += collect("databricks/databricks-dolly-15k", None, "response", 10000, "Dolly")
    qa += collect("databricks/databricks-dolly-15k", None, "instruction", 10000, "Dolly instructions")
    qa = dedup(qa)
    print(f"  QA total: {len(qa):,}", flush=True)
    all_texts.extend(qa)
    domains["instruction_qa"] = len(qa)

    # === CREATIVE (10%) ===
    print("\n=== CREATIVE ===", flush=True)
    creative = []
    # Rap lyrics for flow/style.
    creative += collect("Cropinky/rap_lyrics_english", None, "text", 10000, "Rap lyrics")
    # Stories for narrative.
    creative += collect("roneneldan/TinyStories", None, "text", 10000, "Stories")
    creative = dedup(creative)
    print(f"  Creative total: {len(creative):,}", flush=True)
    all_texts.extend(creative)
    domains["creative"] = len(creative)

    # === CHAT (5%) ===
    print("\n=== CHAT ===", flush=True)
    chat = collect("OpenAssistant/oasst1", None, "text", 15000, "Human chat")
    chat = dedup(chat)
    print(f"  Chat total: {len(chat):,}", flush=True)
    all_texts.extend(chat)
    domains["chat"] = len(chat)

    # === TOKENIZE ===
    print(f"\n{'='*60}", flush=True)
    total_ex = len(all_texts)
    print(f"TOKENIZING {total_ex:,} examples...", flush=True)

    all_tokens = []
    t0 = time.perf_counter()
    for i, text in enumerate(all_texts):
        ids = tok.encode(text)
        ids.append(EOS)
        all_tokens.extend(ids)
        if (i + 1) % 20000 == 0:
            elapsed = time.perf_counter() - t0
            print(f"  {i+1:,}/{total_ex:,} ({len(all_tokens):,} tokens, {elapsed:.0f}s)", flush=True)

    print(f"\n{'='*60}", flush=True)
    print(f"FINAL: {len(all_tokens):,} tokens", flush=True)
    for d, c in domains.items():
        print(f"  {d:25s}: {c:>6,} examples ({c/total_ex*100:.0f}%)", flush=True)
    unique = len(set(all_tokens))
    print(f"  Vocab coverage: {unique:,}/{tok.vocab_size} ({unique/tok.vocab_size*100:.1f}%)", flush=True)
    print(f"  Avg tokens/example: {len(all_tokens)/max(total_ex,1):.0f}", flush=True)

    os.makedirs(os.path.dirname(OUTPUT), exist_ok=True)
    torch.save(torch.tensor(all_tokens, dtype=torch.int32), OUTPUT)
    size_mb = os.path.getsize(OUTPUT) / 1e6
    print(f"\nSaved: {OUTPUT} ({size_mb:.1f} MB)", flush=True)
    print(f"At 500 tok/s (GPU): {len(all_tokens)/500/3600:.1f}h/epoch", flush=True)


if __name__ == "__main__":
    main()
