#!/usr/bin/env python
"""Build a Chinchilla-optimal corpus for Fractus-1B (88M params).

Target: 1.76B tokens (20x params per Chinchilla scaling law).
Uses FineWeb (web text) + existing communication/ultimate corpora.
Saves as int16 to save disk (~3.5 GB instead of 7 GB).
"""
import os, sys, time, hashlib
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
from datasets import load_dataset
from fractus.tokenizer import FractusTokenizer

OUTPUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                      "data", "chinchilla_corpus.pt")
EOS = 50256
TARGET_TOKENS = 1_760_000_000  # Chinchilla optimal for 88M params


def main():
    tok = FractusTokenizer.gpt2Compatible() if hasattr(FractusTokenizer, 'gpt2Compatible') else FractusTokenizer.gpt2_compatible()
    print(f"Tokenizer: vocab={tok.vocab_size}", flush=True)
    print(f"Target: {TARGET_TOKENS/1e9:.2f}B tokens (Chinchilla optimal for 88M params)", flush=True)
    print(f"Output format: int16 (saves 50% disk)", flush=True)
    print(flush=True)

    # Start with existing high-quality corpora.
    all_tokens = []
    
    for name in ["communication_corpus.pt", "ultimate_corpus.pt"]:
        path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", name)
        if os.path.exists(path):
            t = torch.load(path, weights_only=False).long()
            all_tokens.extend(t.tolist())
            print(f"  Loaded {name}: {len(t):,} tokens (total: {len(all_tokens):,})", flush=True)
    
    current = len(all_tokens)
    remaining = TARGET_TOKENS - current
    print(f"\nExisting: {current/1e6:.1f}M tokens", flush=True)
    print(f"Need from FineWeb: {remaining/1e6:.1f}M more tokens", flush=True)
    
    # Stream FineWeb until we hit the target.
    print(f"\nStreaming FineWeb...", flush=True)
    ds = load_dataset("HuggingFaceFW/fineweb", "sample-10BT", split="train", streaming=True)
    
    count = 0
    t0 = time.perf_counter()
    skipped = 0
    
    for ex in ds:
        if len(all_tokens) >= TARGET_TOKENS:
            break
        
        text = ex.get("text", "")
        if not isinstance(text, str) or len(text) < 100:
            skipped += 1
            continue
        
        # Truncate very long texts (keep diversity).
        if len(text) > 5000:
            text = text[:5000]
        
        ids = tok.encode(text)
        ids.append(EOS)
        all_tokens.extend(ids)
        count += 1
        
        if count % 50000 == 0:
            elapsed = time.perf_counter() - t0
            rate = count / max(elapsed, 1)
            total_tokens = len(all_tokens)
            pct = total_tokens / TARGET_TOKENS * 100
            eta = (TARGET_TOKENS - total_tokens) / (total_tokens / max(elapsed, 1)) / 60
            print(f"  {count:,} examples, {total_tokens/1e6:.0f}M tokens "
                  f"({pct:.1f}%), {rate:.0f} ex/s, ETA {eta:.0f} min", flush=True)
    
    # Truncate to exact target.
    all_tokens = all_tokens[:TARGET_TOKENS]
    elapsed = time.perf_counter() - t0
    
    print(f"\n{'='*60}", flush=True)
    print(f"FINAL CORPUS: {len(all_tokens):,} tokens ({len(all_tokens)/1e9:.2f}B)", flush=True)
    print(f"  Existing quality data: {current/1e6:.1f}M ({current/len(all_tokens)*100:.1f}%)", flush=True)
    print(f"  FineWeb: {(len(all_tokens)-current)/1e6:.1f}M ({(len(all_tokens)-current)/len(all_tokens)*100:.1f}%)", flush=True)
    
    unique = len(set(all_tokens[:500000]))  # sample for speed
    print(f"  Vocab coverage (sampled): {unique}/{tok.vocab_size} ({unique/tok.vocab_size*100:.1f}%)", flush=True)
    print(f"  Download time: {elapsed/60:.0f} min", flush=True)
    
    # Save as int16 (max token id 50256 fits in int16).
    tensor = torch.tensor(all_tokens, dtype=torch.int16)
    torch.save(tensor, OUTPUT)
    size_gb = os.path.getsize(OUTPUT) / 1e9
    print(f"\nSaved: {OUTPUT} ({size_gb:.1f} GB)", flush=True)
    print(f"Load with: torch.load('{OUTPUT}').long()", flush=True)
    print(f"\nAt 5000 tok/s (GPU): {len(all_tokens)/5000/3600:.0f}h per epoch", flush=True)
    print(f"At 10000 tok/s (GPU): {len(all_tokens)/10000/3600:.0f}h per epoch", flush=True)


if __name__ == "__main__":
    main()
