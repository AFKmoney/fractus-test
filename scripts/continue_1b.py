#!/usr/bin/env python
"""Continue 1B training from epoch 9 checkpoint. Targets epoch 15."""
import argparse, gc, math, os, sys, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch, torch.nn as nn, torch.nn.functional as F
from fractus.model_1b import Fractus1B
from fractus.tokenizer import FractusTokenizer

def main():
    torch.set_num_threads(os.cpu_count() or 6)
    torch.manual_seed(42)

    corpus = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "mega_500k.pt")
    tokens = torch.load(corpus, weights_only=False).long()
    print(f"Corpus: {len(tokens):,} tokens", flush=True)

    # Resume from epoch 9.
    ckpt_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "checkpoints", "fractus_1b_epoch9.pt")
    print(f"Loading checkpoint: {ckpt_path}", flush=True)
    ckpt = torch.load(ckpt_path, weights_only=False)
    print(f"  Epoch: {ckpt['epoch']}, Loss: {ckpt['loss']:.3f}", flush=True)

    model = Fractus1B(
        vocab_size=ckpt['config']['vocab_size'], d_model=ckpt['config']['d_model'],
        n_layers=ckpt['config'].get('n_layers', 8), n_heads=ckpt['config'].get('n_heads', 12),
        d_head=ckpt['config'].get('d_head', 64), n_levels=ckpt['config'].get('n_levels', 2),
        n_experts=ckpt['config'].get('n_experts', 64), top_k=ckpt['config'].get('top_k', 2),
        expert_d_ff=ckpt['config'].get('expert_d_ff', 1024), siren_rank=ckpt['config'].get('siren_rank', 16),
        max_seq_len=ckpt['config'].get('max_seq_len', 32),
    )
    model.load_state_dict(ckpt['model_state'])
    n = model.n_params()
    print(f"  Loaded: {n/1e6:.0f}M params", flush=True)

    opt = torch.optim.AdamW(model.parameters(), lr=5e-5, weight_decay=0.01)  # lower LR for fine-tuning
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=6, eta_min=1e-6)  # 6 more epochs (10→15)
    tok = FractusTokenizer.gpt2_compatible()
    seq = 32
    ckpt_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "checkpoints")

    n_chunks = len(tokens) // seq
    print(f"\nContinuing training: epochs 10-15, {n_chunks:,} chunks/epoch, lr=5e-5", flush=True)
    print("=" * 70, flush=True)

    for epoch in range(10, 16):
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

            if ep_n % 2000 == 0:
                el = time.perf_counter() - t0
                tps = (i + seq) / el
                print(f"  e{epoch} step {ep_n:5d}/{n_chunks} ce={ce.item():.3f} "
                      f"{tps:.0f} tok/s {el/60:.0f}min", flush=True)

        el = time.perf_counter() - t0
        avg = ep_loss / max(ep_n, 1)
        ppl = math.exp(min(avg, 20))
        print(f"\nEpoch {epoch}/15: loss={avg:.3f} ppl={ppl:.1f} "
              f"{len(tokens)/el:.0f} tok/s {el/60:.0f}min", flush=True)

        # Save checkpoint.
        ckpt_path = os.path.join(ckpt_dir, f"fractus_1b_epoch{epoch}.pt")
        torch.save({"model_state": model.state_dict(), "config": model.config,
                    "epoch": epoch, "loss": avg}, ckpt_path)
        print(f"  [ckpt] {ckpt_path} ({os.path.getsize(ckpt_path)/1e6:.0f}MB)", flush=True)

        # Upload to HF.
        HF_TOKEN = os.environ.get("HF_TOKEN", "")
        if HF_TOKEN:
            try:
                from huggingface_hub import HfApi
                api = HfApi(token=HF_TOKEN)
                api.upload_file(path_or_fileobj=ckpt_path,
                              path_in_repo=f"checkpoints/fractus_1b_epoch{epoch}.pt",
                              repo_id="thefinalboss/Fractus", repo_type="model")
                print(f"  [HF] Uploaded", flush=True)
            except: pass

        # Sample.
        model.eval()
        with torch.no_grad():
            p_ids = tok.encode("def fibonacci")[:seq]
            inp = torch.tensor([p_ids], dtype=torch.long)
            logits, _ = model(inp)
            gen = p_ids + logits[0].argmax(dim=-1).tolist()[:20]
        print(f"  Sample: {tok.decode(gen)[:120]}", flush=True)
        gc.collect()

    print("\nTraining complete. Epochs 10-15 done.", flush=True)

if __name__ == "__main__":
    main()
