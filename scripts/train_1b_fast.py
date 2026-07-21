#!/usr/bin/env python
"""Train Fractus-1B on quality data — NO gradient checkpointing (5x faster).

24.8 tok/s. 200k tokens × 5 epochs = ~11h on CPU.
"""
import argparse, gc, math, os, sys, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch, torch.nn as nn, torch.nn.functional as F
from fractus.model_1b import Fractus1B
from fractus.tokenizer import FractusTokenizer

HF_TOKEN = os.environ.get("HF_TOKEN", "")
HF_REPO = "thefinalboss/Fractus"

def save_ckpt(model, epoch, loss, acc, d):
    p = os.path.join(d, f"fractus_1b_epoch{epoch}.pt")
    torch.save({"model_state": model.state_dict(),
                "config": model.config, "epoch": epoch,
                "loss": loss, "accuracy": acc}, p)
    sz = os.path.getsize(p)/1e6
    print(f"  [ckpt] {p} ({sz:.0f}MB)", flush=True)
    if HF_TOKEN:
        try:
            from huggingface_hub import HfApi
            api = HfApi(token=HF_TOKEN)
            api.upload_file(path_or_fileobj=p, path_in_repo=f"checkpoints/fractus_1b_epoch{epoch}.pt",
                          repo_id=HF_REPO, repo_type="model")
            print(f"  [HF] Uploaded", flush=True)
        except: pass

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--epochs", type=int, default=15)
    p.add_argument("--seq-len", type=int, default=32)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--max-tokens", type=int, default=200000)
    args = p.parse_args()
    torch.set_num_threads(os.cpu_count() or 6)
    torch.manual_seed(42)

    corpus = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "mega_500k.pt")
    tokens = torch.load(corpus, weights_only=False).long()[:args.max_tokens]
    print(f"Corpus: {len(tokens):,} tokens", flush=True)

    print("Building Fractus-1B...", flush=True)
    model = Fractus1B(
        vocab_size=50257, d_model=768, n_layers=8, n_heads=12, d_head=64,
        n_levels=2, n_experts=64, top_k=2, expert_d_ff=1024, siren_rank=16,
        max_seq_len=args.seq_len,
    )
    n = model.n_params()
    cap = model.n_effective_capacity()
    print(f"  {n/1e6:.0f}M params, {cap/1e9:.2f}B capacity, {n*4/1e9:.1f}GB RAM", flush=True)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs, eta_min=1e-5)
    tok = FractusTokenizer.gpt2_compatible()
    seq = args.seq_len
    ckpt_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)

    n_steps = len(tokens) // seq
    print(f"\nTraining {args.epochs} epochs, {n_steps:,} steps/epoch", flush=True)
    print("=" * 70, flush=True)

    initial_loss = None
    for epoch in range(args.epochs):
        model.train()
        t0 = time.perf_counter()
        ep_loss = 0.0
        ep_n = 0

        for i in range(0, len(tokens) - seq - 1, seq):
            inp = tokens[i:i+seq].unsqueeze(0)
            tgt = tokens[i+1:i+seq+1].unsqueeze(0)
            opt.zero_grad()
            logits, aux = model(inp)
            ce = nn.functional.cross_entropy(logits.reshape(-1, 50257), tgt.reshape(-1))
            loss = ce + 0.001 * aux
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            ep_loss += ce.item()
            ep_n += 1
            if initial_loss is None:
                initial_loss = ce.item()
            if ep_n % 1000 == 0:
                el = time.perf_counter() - t0
                tps = (i + seq) / el
                print(f"  e{epoch+1} step {ep_n:5d}/{n_steps} ce={ce.item():.3f} "
                      f"{tps:.0f} tok/s {el/60:.0f}min", flush=True)

        el = time.perf_counter() - t0
        avg = ep_loss / max(ep_n, 1)
        ppl = math.exp(min(avg, 20))
        print(f"\nEpoch {epoch+1}/{args.epochs}: loss={avg:.3f} ppl={ppl:.1f} "
              f"{len(tokens)/el:.0f} tok/s {el/60:.0f}min", flush=True)
        save_ckpt(model, epoch+1, avg, 0, ckpt_dir)

        # Sample.
        model.eval()
        with torch.no_grad():
            p_ids = tok.encode("def fibonacci")[:seq]
            inp = torch.tensor([p_ids], dtype=torch.long)
            logits, _ = model(inp)
            gen = p_ids + logits[0].argmax(dim=-1).tolist()[:20]
        print(f"  Sample: {tok.decode(gen)[:120]}", flush=True)
        gc.collect()

    # Coherence test.
    print("\n=== COHERENCE TEST ===", flush=True)
    model.eval()
    for prompt in ["What is Python?", "def sort", "The sun is", "Explain AI"]:
        with torch.no_grad():
            p_ids = tok.encode(prompt)[:seq]
            inp = torch.tensor([p_ids], dtype=torch.long)
            logits, _ = model(inp)
            gen = p_ids + logits[0].argmax(dim=-1).tolist()[:30]
        print(f"  [{prompt}] -> {tok.decode(gen)[:120]}", flush=True)

    print(f"\nDone. Loss: {initial_loss:.3f} -> {avg:.3f}", flush=True)

if __name__ == "__main__":
    main()
