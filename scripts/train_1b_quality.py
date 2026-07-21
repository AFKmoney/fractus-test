#!/usr/bin/env python
"""Train Fractus-1B on the quality corpus (Alpaca + OpenAssistant + Dolly).

500k tokens of real instruction/conversational data.
5 epochs at ~5 tok/s = ~6 days. Checkpoints every 250 steps.
"""
import argparse, math, os, sys, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch, torch.nn as nn
from fractus.model_1b import Fractus1B

HF_TOKEN = os.environ.get("HF_TOKEN", "")
HF_REPO = "thefinalboss/Fractus-1B"

def save_ckpt(model, step, loss, config, cap_b, train_m):
    d = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "checkpoints")
    os.makedirs(d, exist_ok=True)
    p = os.path.join(d, f"fractus_1b_quality_step_{step}.pt")
    torch.save({"model_state": model.state_dict(), "config": config,
                "step": step, "loss": loss, "capacity_b": cap_b,
                "trainable_m": train_m, "dataset": "quality_corpus"},
               p)
    print(f"  [ckpt] {p} ({os.path.getsize(p)/1e6:.0f}MB)", flush=True)
    if HF_TOKEN:
        try:
            from huggingface_hub import HfApi
            api = HfApi(token=HF_TOKEN)
            api.upload_file(path_or_fileobj=p, path_in_repo=f"checkpoints/fractus_1b_quality_step_{step}.pt",
                          repo_id=HF_REPO, repo_type="model")
            print(f"  [HF] Uploaded step {step}", flush=True)
        except Exception as e:
            print(f"  [HF] Skipped: {type(e).__name__}", flush=True)

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--seq-len", type=int, default=32)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--corpus", type=str, default=None)
    args = p.parse_args()
    torch.set_num_threads(os.cpu_count() or 6)
    torch.manual_seed(42)

    # Load quality corpus.
    cpath = args.corpus or os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                        "data", "quality_500k.pt")
    print(f"Loading quality corpus: {cpath}", flush=True)
    tokens = torch.load(cpath, weights_only=False).long()
    print(f"Corpus: {len(tokens):,} tokens (Alpaca + OpenAssistant + Dolly)", flush=True)

    # Build 1B model.
    print("Building Fractus-1B...", flush=True)
    model = Fractus1B(
        vocab_size=50257, d_model=768, n_layers=8, n_heads=12, d_head=64,
        n_levels=2, n_experts=64, top_k=2, expert_d_ff=1024, siren_rank=16,
        max_seq_len=args.seq_len,
    )
    n = model.n_params()
    cap = model.n_effective_capacity()
    print(f"  Trainable: {n:,} ({n/1e6:.0f}M)  Capacity: {cap/1e9:.2f}B  RAM: {n*4/1e9:.1f}GB", flush=True)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs, eta_min=1e-6)

    seq = args.seq_len
    n_chunks = len(tokens) // seq
    print(f"\nTraining {args.epochs} epochs, {n_chunks} chunks/epoch, seq={seq}", flush=True)
    print("=" * 70, flush=True)

    step = 0
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
            with torch.amp.autocast('cpu', dtype=torch.bfloat16):
                logits, aux = model(inp)
                ce = nn.functional.cross_entropy(logits.reshape(-1, 50257), tgt.reshape(-1))
                loss = ce + 0.001 * aux.float()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            step += 1
            ep_loss += ce.item()
            ep_n += 1
            if initial_loss is None:
                initial_loss = ce.item()
            if step % 25 == 0:
                el = time.perf_counter() - t0
                tps = (i + seq) / el
                print(f"  step {step:5d}  ce={ce.item():.3f}  "
                      f"{tps:.0f} tok/s  lr={opt.param_groups[0]['lr']:.6f}  "
                      f"({el/60:.1f}min)  epoch {epoch+1}/{args.epochs}", flush=True)
            if step % 250 == 0:
                save_ckpt(model, step, ep_loss/max(ep_n,1), model.config, cap/1e9, n/1e6)

        el = time.perf_counter() - t0
        avg = ep_loss / max(ep_n, 1)
        ppl = math.exp(min(avg, 20))
        print(f"\nEpoch {epoch+1}/{args.epochs}: loss={avg:.3f} ppl={ppl:.1f} "
              f"{len(tokens)/el:.0f} tok/s {el/60:.1f}min\n", flush=True)
        save_ckpt(model, step, avg, model.config, cap/1e9, n/1e6)

    # Coherence test.
    print("\n=== Coherence Test ===", flush=True)
    from fractus.tokenizer import FractusTokenizer
    import torch.nn.functional as F
    tok = FractusTokenizer.gpt2_compatible()
    model.eval()
    prompts = ["What is Python?", "Explain machine learning", "def fibonacci", "How to secure a website"]
    for prompt in prompts:
        ids = tok.encode(prompt)[:seq]
        if len(ids) < 2:
            continue
        inp = torch.tensor([ids], dtype=torch.long)
        with torch.no_grad():
            logits, _ = model(inp)
        pred = logits[0].argmax(dim=-1).tolist()
        text = tok.decode(ids + pred[:20])
        ce = F.cross_entropy(logits.reshape(-1, 50257),
                             torch.cat([inp[:,1:], torch.zeros(1,1,dtype=torch.long)],dim=1).reshape(-1))
        print(f"  [{prompt}] ppl={math.exp(min(ce.item(),20)):.1f}", flush=True)
        print(f"    {text[:120]}", flush=True)

    print(f"\nFractus-1B training complete. {cap/1e9:.2f}B capacity, {n/1e6:.0f}M params.", flush=True)

if __name__ == "__main__":
    main()
