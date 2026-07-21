#!/usr/bin/env python
"""Train Fractus 13M engine — fixed version that resumes from checkpoint.

Fixes the stalling issue:
  1. GC between epochs (clear accumulated graph state)
  2. Progress logging every 5000 chunks (so we know it's alive)
  3. Resumes from existing checkpoint if available
  4. Saves checkpoint after every epoch (not just at the end)
"""
import argparse, gc, math, os, sys, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch, torch.nn as nn, torch.nn.functional as F
from fractus.continuous_engine import ContinuousThoughtEngine
from fractus.tokenizer import FractusTokenizer

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--chunk-len", type=int, default=16)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--resume", type=str, default=None)
    args = p.parse_args()

    torch.set_num_threads(os.cpu_count() or 6)
    torch.manual_seed(42)

    # Load corpus.
    corpus = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                          "data", "quality_500k.pt")
    tokens = torch.load(corpus, weights_only=False).long()
    print(f"Corpus: {len(tokens):,} tokens", flush=True)

    # Build engine.
    engine = ContinuousThoughtEngine(
        vocab_size=50257, d_model=128, n_heads=2, d_head=64,
        n_levels=2, n_oscillators=8, coupling_rank=4,
        n_experts=4, top_k=2, expert_d_ff=128, siren_rank=32,
    )
    n_params = sum(p.numel() for p in engine.parameters())
    print(f"Engine: {n_params/1e6:.1f}M params", flush=True)

    # Resume from checkpoint if available.
    ckpt_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)
    start_epoch = 0

    resume_path = args.resume
    if resume_path is None:
        # Auto-find latest checkpoint.
        ckpts = [f for f in os.listdir(ckpt_dir) if f.startswith("fractus_13m_") and f.endswith(".pt")]
        if ckpts:
            ckpts.sort()
            resume_path = os.path.join(ckpt_dir, ckpts[-1])

    if resume_path and os.path.exists(resume_path):
        print(f"Resuming from: {resume_path}", flush=True)
        ckpt = torch.load(resume_path, weights_only=False)
        # Fix buffer shape mismatches.
        sd = ckpt['model_state']
        sd['kuramoto_phases'] = engine.kuramoto_phases
        sd['attn_S'] = engine.attn_S
        sd['attn_z'] = engine.attn_z
        sd['thought_state'] = engine.thought_state
        engine.load_state_dict(sd, strict=False)
        start_epoch = ckpt.get('epoch', 0)
        print(f"  Resumed from epoch {start_epoch}, loss={ckpt.get('loss','?')}", flush=True)

    # Optimizer.
    opt = torch.optim.AdamW(engine.parameters(), lr=args.lr, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs, eta_min=1e-5)

    # Skip scheduler to the right position.
    for _ in range(start_epoch):
        sched.step()

    seq = args.chunk_len
    n_chunks = len(tokens) // seq
    tok = FractusTokenizer.gpt2_compatible()
    vocab = 50257

    print(f"\nTraining epochs {start_epoch+1}-{args.epochs}, {n_chunks:,} chunks/epoch, seq={seq}", flush=True)
    print("=" * 70, flush=True)

    for epoch in range(start_epoch, args.epochs):
        engine.train()
        engine.reset_thought(batch_size=1)

        t0 = time.perf_counter()
        ep_loss = 0.0
        ep_correct = 0
        ep_total = 0

        for i in range(0, len(tokens) - seq - 1, seq):
            chunk = tokens[i:i+seq].unsqueeze(0)
            target = tokens[i+1:i+seq+1].unsqueeze(0)

            opt.zero_grad()
            logits = engine.tick_chunk(chunk)
            loss = F.cross_entropy(logits.reshape(-1, vocab), target.reshape(-1))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(engine.parameters(), 1.0)
            opt.step()

            ep_loss += loss.item() * seq
            ep_correct += (logits.argmax(dim=-1) == target).sum().item()
            ep_total += seq

            # Progress every 5000 chunks.
            if (i // seq) % 5000 == 0 and i > 0:
                elapsed = time.perf_counter() - t0
                tps = i / elapsed
                print(f"  epoch {epoch+1} chunk {i//seq}/{n_chunks} "
                      f"loss={ep_loss/ep_total:.3f} acc={ep_correct/ep_total:.1%} "
                      f"{tps:.0f} tok/s {elapsed/60:.0f}min", flush=True)

        elapsed = time.perf_counter() - t0
        avg_loss = ep_loss / max(ep_total, 1)
        acc = ep_correct / max(ep_total, 1)
        ppl = math.exp(min(avg_loss, 20))
        sched.step()

        print(f"\nEpoch {epoch+1}/{args.epochs}: loss={avg_loss:.3f} ppl={ppl:.1f} "
              f"acc={acc:.1%} {len(tokens)/elapsed:.0f} tok/s {elapsed/60:.1f}min", flush=True)

        # Save checkpoint every epoch.
        ckpt_path = os.path.join(ckpt_dir, f"fractus_13m_epoch{epoch+1}.pt")
        torch.save({
            "model_state": engine.state_dict(),
            "config": {"d_model": 128, "vocab_size": vocab},
            "epoch": epoch + 1,
            "loss": avg_loss,
            "accuracy": acc,
        }, ckpt_path)
        print(f"  [ckpt] {ckpt_path} ({os.path.getsize(ckpt_path)/1e6:.0f}MB)", flush=True)

        # Sample generation.
        engine.eval()
        engine.reset_thought(1)
        prompt = "def fibonacci"
        pids = tok.encode(prompt)[:16]
        for tid in pids:
            engine.tick(torch.tensor([tid]))
        gen = list(pids)
        for _ in range(30):
            logits, _ = engine.tick()
            l = logits[0] / 0.8
            tv, ti = l.topk(40)
            gen.append(ti[torch.multinomial(F.softmax(tv, dim=-1), 1).item()].item())
        print(f"  Sample: {tok.decode(gen)[:100]}", flush=True)

        # GC between epochs (prevent memory accumulation).
        gc.collect()
        print(flush=True)

    print("Training complete.", flush=True)

    # Final coherence test.
    print("\n=== COHERENCE TEST ===", flush=True)
    engine.eval()
    for prompt in ["What is Python?", "def sort", "The sun is", "Explain AI"]:
        engine.reset_thought(1)
        ids = tok.encode(prompt)[:16]
        for tid in ids:
            engine.tick(torch.tensor([tid]))
        gen = list(ids)
        for _ in range(40):
            logits, _ = engine.tick()
            l = logits[0] / 0.8
            tv, ti = l.topk(40)
            gen.append(ti[torch.multinomial(F.softmax(tv, dim=-1), 1).item()].item())
        print(f"  [{prompt}] -> {tok.decode(gen)[:120]}", flush=True)

    print("\nDone.", flush=True)

if __name__ == "__main__":
    main()
