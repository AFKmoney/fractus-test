#!/usr/bin/env python
"""Train the 13M ContinuousThoughtEngine on quality data.

117 tok/s on CPU. 500k quality tokens × 3 epochs = ~3.5 hours.
After training, the engine produces coherent text + accurate embeddings
for the RAG system (retrieval + online learning).
"""
import argparse, math, os, sys, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch, torch.nn as nn, torch.nn.functional as F
from fractus.continuous_engine import ContinuousThoughtEngine
from fractus.tokenizer import FractusTokenizer

HF_TOKEN = os.environ.get("HF_TOKEN", "")
HF_REPO = "thefinalboss/Fractus-1B"

def save_ckpt(engine, step, loss, acc, d):
    os.makedirs(d, exist_ok=True)
    p = os.path.join(d, f"fractus_13m_step_{step}.pt")
    torch.save({"model_state": engine.state_dict(),
                "config": {"d_model": engine.d_model, "vocab_size": engine.vocab_size},
                "step": step, "loss": loss, "accuracy": acc}, p)
    print(f"  [ckpt] {p} ({os.path.getsize(p)/1e6:.0f}MB)", flush=True)
    if HF_TOKEN:
        try:
            from huggingface_hub import HfApi
            api = HfApi(token=HF_TOKEN)
            api.upload_file(path_or_fileobj=p,
                          path_in_repo=f"checkpoints/fractus_13m_step_{step}.pt",
                          repo_id=HF_REPO, repo_type="model")
            print(f"  [HF] Uploaded", flush=True)
        except Exception as e:
            print(f"  [HF] Skipped: {type(e).__name__}", flush=True)

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--chunk-len", type=int, default=16)
    p.add_argument("--lr", type=float, default=3e-4)
    args = p.parse_args()
    torch.set_num_threads(os.cpu_count() or 6)
    torch.manual_seed(42)

    # Load quality data.
    corpus = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                          "data", "quality_500k.pt")
    tokens = torch.load(corpus, weights_only=False).long()
    print(f"Corpus: {len(tokens):,} quality tokens", flush=True)

    # Build engine.
    engine = ContinuousThoughtEngine(
        vocab_size=50257, d_model=128, n_heads=2, d_head=64,
        n_levels=2, n_oscillators=8, coupling_rank=4,
        n_experts=4, top_k=2, expert_d_ff=128, siren_rank=32,
    )
    print(f"Engine: {sum(p.numel() for p in engine.parameters())/1e6:.1f}M params", flush=True)

    # Trainer.
    from fractus.train.online import OnlineTrainer
    trainer = OnlineTrainer(engine, lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        trainer.optimizer, T_max=args.epochs, eta_min=1e-5)

    ckpt_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                            "checkpoints")
    print(f"\nTraining {args.epochs} epochs, chunk={args.chunk_len}", flush=True)
    print("=" * 70, flush=True)

    initial_loss = None
    for epoch in range(args.epochs):
        t0 = time.perf_counter()
        m = trainer.train_on_stream_chunked(tokens, chunk_len=args.chunk_len)
        elapsed = time.perf_counter() - t0
        tps = len(tokens) / elapsed

        if initial_loss is None:
            initial_loss = m["avg_loss"]
        scheduler.step()

        ppl = math.exp(min(m["avg_loss"], 20))
        print(f"Epoch {epoch+1}/{args.epochs}: loss={m['avg_loss']:.3f} ppl={ppl:.1f} "
              f"acc={m['accuracy']:.1%} {tps:.0f} tok/s {elapsed/60:.1f}min", flush=True)

        # Checkpoint + sample.
        save_ckpt(engine, (epoch+1) * (len(tokens) // args.chunk_len),
                  m["avg_loss"], m["accuracy"], ckpt_dir)

        # Generate sample every epoch.
        engine.eval()
        engine.reset_thought(1)
        tok = FractusTokenizer.gpt2_compatible()
        prompt = "def fibonacci"
        for tid in tok.encode(prompt)[:16]:
            engine.tick(torch.tensor([tid]))
        generated = list(tok.encode(prompt)[:16])
        for _ in range(30):
            logits, _ = engine.tick()
            generated.append(logits.argmax(dim=-1).item())
        print(f"  Sample: {tok.decode(generated)[:100]}", flush=True)

    # Final coherence test.
    print("\n=== COHERENCE TEST ===", flush=True)
    tok = FractusTokenizer.gpt2_compatible()
    engine.eval()
    for prompt in ["What is Python?", "Explain machine learning", "def sort", "The sun is"]:
        engine.reset_thought(1)
        ids = tok.encode(prompt)[:16]
        for tid in ids:
            engine.tick(torch.tensor([tid]))
        gen = list(ids)
        for _ in range(40):
            logits, _ = engine.tick()
            l = logits[0] / 0.8
            tv, ti = l.topk(40)
            probs = F.softmax(tv, dim=-1)
            gen.append(ti[torch.multinomial(probs, 1).item()].item())
        print(f"  [{prompt}] -> {tok.decode(gen)[:120]}", flush=True)

    print(f"\nDone. Loss: {initial_loss:.3f} -> {m['avg_loss']:.3f}", flush=True)

if __name__ == "__main__":
    main()
