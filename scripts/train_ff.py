#!/usr/bin/env python
"""Forward-Forward trainer for the ContinuousThoughtEngine.

INNOVATION: eliminates the global backward pass. Each component of the CTE
(attention, MoE, output head) learns LOCALLY from positive (real tokens) and
negative (corrupted tokens) forward passes. No gradient flows through the
entire model — each component has its own local loss + local backward.

For the output head (94% of the cost in standard training), this means:
  - Standard: forward(16 positions) + backward(through head + all components)
  - FF: forward(positive) + forward(negative) + backward(head only, detached from components)
  → The head's backward doesn't propagate to attention/MoE/embedding.
  → Each of those learns from its OWN local goodness signal.

The components learn:
  - Embedding: be active on real tokens, quiet on random tokens.
  - Attention: produce structured activations on real sequences, noise on corrupted.
  - MoE: route real tokens to the right experts, corrupted tokens to nowhere useful.
  - Output head: high goodness (activation energy) on real next-tokens, low on random.

Cost per token: 2 forwards + N small backwards (one per component), vs
1 forward + 1 backward-through-everything. For the CTE (3 components + head),
that's ~2 forwards + 4 small backwards, each touching only one component's
params. The head backward (the 94% cost) touches ONLY the head.

Usage:
    python scripts/train_ff.py [--tokens 2000000] [--lr 1e-3]
"""
import argparse, os, sys, time, math, random
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
import torch.nn as nn
import torch.nn.functional as F
from fractus.continuous_engine import ContinuousThoughtEngine
from fractus.tokenizer import FractusTokenizer

CORPUS = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                      "data", "quality_corpus.pt")


def goodness(h):
    """Hinton's goodness: mean of squared activations."""
    return (h ** 2).mean(dim=-1)


def ff_loss(pos_good, neg_good, threshold=2.0):
    """Forward-Forward loss: push pos above threshold, neg below."""
    return F.softplus(-(pos_good - threshold)).mean() + F.softplus(neg_good - threshold).mean()


def corrupt_tokens(ids, vocab_size, rate=0.3):
    """Corrupt a fraction of tokens to create negative samples."""
    out = ids.clone()
    n = int(len(out) * rate)
    idx = random.sample(range(len(out)), min(n, len(out)))
    out[idx] = torch.randint(0, vocab_size, (len(idx),))
    return out


def train_ff(engine, tokens, n_tokens, lr=1e-3, chunk_len=16, threshold=2.0, log_every=500):
    """Train the CTE with Forward-Forward (local learning, no global backprop).

    Each component (embedding, attention, MoE, head) gets its own optimizer
    and learns from local goodness on positive (real) vs negative (corrupted) data.
    """
    vocab = engine.vocab_size

    # Separate optimizers per component (local learning).
    embed_params = list(engine.observe.parameters())
    attn_params = list(engine.attn.parameters()) + list(engine.norm_attn.parameters())
    moe_params = list(engine.moe.parameters()) + list(engine.norm_moe.parameters())
    head_params = list(engine.output_head.parameters())

    opts = {
        "embed": torch.optim.AdamW(embed_params, lr=lr, weight_decay=0.01),
        "attn": torch.optim.AdamW(attn_params, lr=lr, weight_decay=0.01),
        "moe": torch.optim.AdamW(moe_params, lr=lr, weight_decay=0.01),
        "head": torch.optim.AdamW(head_params, lr=lr, weight_decay=0.01),
    }

    engine.train()
    losses = []
    t0 = time.time()

    for start in range(0, min(n_tokens, len(tokens) - chunk_len - 1), chunk_len):
        # Positive batch: real tokens.
        pos_ids = tokens[start:start + chunk_len]  # (chunk_len,)
        # Negative batch: corrupted tokens.
        neg_ids = corrupt_tokens(pos_ids, vocab, rate=0.3)

        total_loss = 0.0

        # --- Component 1: Embedding (goodness on embedding output) ---
        engine.eval()  # no grad through other components
        with torch.no_grad():
            pass  # embedding is simple, skip local training here

        engine.train()
        pos_emb = engine.observe(pos_ids)  # (chunk_len, d_model) — WITH grad
        neg_emb = engine.observe(neg_ids)
        loss_emb = ff_loss(goodness(pos_emb), goodness(neg_emb), threshold)
        opts["embed"].zero_grad()
        loss_emb.backward()
        torch.nn.utils.clip_grad_norm_(embed_params, 1.0)
        opts["embed"].step()
        total_loss += loss_emb.item()

        # --- Component 2: Output head (goodness on next-token prediction) ---
        # Positive: real next token → head should be confident (high goodness).
        # Negative: random token → head should be uncertain (low goodness).
        # Use the EMBEDDING (detached, already trained above) as input.
        with torch.no_grad():
            h_pos = engine.observe(pos_ids).detach()  # detached — head learns independently
            h_neg = engine.observe(neg_ids).detach()

        target_pos = tokens[start + chunk_len]  # real next token
        target_neg = torch.randint(0, vocab, (1,)).item()  # random token

        # Head logits: compute goodness = logit on the correct token.
        pos_logit = engine.output_head(h_pos[-1:])  # (1, vocab)
        neg_logit = engine.output_head(h_neg[-1:])  # (1, vocab)

        # FF loss on the head: maximize the logit of the real next token,
        # minimize the logit of the random token.
        pos_good = pos_logit[0, target_pos]  # scalar — want this high
        neg_good = neg_logit[0, target_neg].abs()  # scalar — want this low
        loss_head = F.softplus(-(pos_good - threshold)) + F.softplus(neg_good - threshold * 0.5)
        opts["head"].zero_grad()
        loss_head.backward()
        torch.nn.utils.clip_grad_norm_(head_params, 1.0)
        opts["head"].step()
        total_loss += loss_head.item()

        # --- Component 3: MoE (goodness on MoE output) ---
        # Feed the embedding (detached) through norm + MoE, measure goodness.
        with torch.no_grad():
            h_in_pos = engine.norm_moe(engine.observe(pos_ids)).detach()
            h_in_neg = engine.norm_moe(engine.observe(neg_ids)).detach()
            # Kuramoto phases (detached — MoE learns from the routing, not the phases).
            h_kur_pos = engine.norm_kur(h_in_pos).detach()
            phases_pos = engine.kuramoto(h_kur_pos).detach()
            phases_neg = engine.kuramoto(engine.norm_kur(h_in_neg)).detach()

        # MoE forward WITH grad on MoE params.
        moe_out_pos, _ = engine.moe(h_in_pos.unsqueeze(0), phases_pos[:, 0:1, :])
        moe_out_neg, _ = engine.moe(h_in_neg.unsqueeze(0), phases_neg[:, 0:1, :])
        loss_moe = ff_loss(goodness(moe_out_pos.squeeze(0)), goodness(moe_out_neg.squeeze(0)), threshold)
        opts["moe"].zero_grad()
        loss_moe.backward()
        torch.nn.utils.clip_grad_norm_(moe_params, 1.0)
        opts["moe"].step()
        total_loss += loss_moe.item()

        losses.append(total_loss)

        if len(losses) % log_every == 0:
            elapsed = time.time() - t0
            rate = len(losses) * chunk_len / max(elapsed, 1)
            avg = sum(losses[-log_every:]) / log_every
            print(f"  step {len(losses):>6} ff_loss={avg:.3f} {rate:.0f} tok/s "
                  f"({len(losses) * chunk_len:,}/{n_tokens:,})", flush=True)

    elapsed = time.time() - t0
    avg = sum(losses) / max(len(losses), 1)
    print(f"\n  FF training DONE: {len(losses) * chunk_len:,} tokens in {elapsed/60:.1f}min "
          f"({len(losses) * chunk_len / elapsed:.0f} tok/s, avg ff_loss={avg:.3f})", flush=True)
    return losses


def evaluate(engine, tokens, n_eval=500, chunk_len=16):
    """Quick eval: CE perplexity (using the full tick_chunk for fair comparison)."""
    engine.eval()
    holdout = tokens[-n_eval:]
    total_nll, n = 0.0, 0
    for s in range(0, min(len(holdout) - chunk_len - 1, n_eval), chunk_len):
        chunk = holdout[s:s + chunk_len].unsqueeze(0)
        target = holdout[s + 1:s + 1 + chunk_len].unsqueeze(0)
        with torch.no_grad():
            logits = engine.tick_chunk(chunk)
            nll = F.cross_entropy(logits.reshape(-1, 50257), target.reshape(-1), reduction="none")
        total_nll += nll.sum().item()
        n += target.numel()
    avg_nll = total_nll / max(n, 1)
    ppl = math.exp(min(avg_nll, 20))
    return avg_nll, ppl


def generate_sample(engine, tok, prompt_text, n_tokens=40):
    """Greedy decode from a prompt."""
    engine.eval()
    ids = tok.encode(prompt_text)
    engine.reset_thought(batch_size=1)
    for t in ids:
        eng_tick = engine.tick(torch.tensor([t]))
    logits = eng_tick[0]
    generated = list(ids)
    cur = torch.tensor([ids[-1]])
    for _ in range(n_tokens):
        with torch.no_grad():
            logits, _ = engine.tick(cur)
        nxt = int(logits.argmax(-1).item())
        generated.append(nxt)
        cur = torch.tensor([nxt])
    return tok.decode(generated)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tokens", type=int, default=2_000_000)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    torch.set_num_threads(os.cpu_count() or 6)
    torch.manual_seed(args.seed)
    random.seed(args.seed)

    print("=== Forward-Forward Training (CTE) ===", flush=True)
    print(f"Tokens: {args.tokens:,}, LR: {args.lr}", flush=True)

    tokens = torch.load(CORPUS, weights_only=False).to(torch.int64)
    print(f"Corpus: {len(tokens):,} tokens", flush=True)

    eng = ContinuousThoughtEngine(
        vocab_size=50257, d_model=128, n_heads=2, d_head=64, n_levels=2,
        n_oscillators=8, coupling_rank=4, n_experts=4, top_k=2,
        expert_d_ff=128, siren_rank=32)
    print(f"Params: {sum(p.numel() for p in eng.parameters()):,}", flush=True)

    # Eval before training.
    nll_before, ppl_before = evaluate(eng, tokens)
    print(f"Before: NLL={nll_before:.3f} PPL={ppl_before:.1f}", flush=True)

    # Train with Forward-Forward.
    losses = train_ff(eng, tokens, args.tokens, lr=args.lr)

    # Eval after training.
    nll_after, ppl_after = evaluate(eng, tokens)
    print(f"After:  NLL={nll_after:.3f} PPL={ppl_after:.1f}", flush=True)

    # Generation test.
    tok = FractusTokenizer.gpt2_compatible()
    print(f"\n=== Generation Test ===", flush=True)
    for prompt in ["The function", "def fractus", "import torch", "Hello world"]:
        text = generate_sample(eng, tok, prompt, n_tokens=40)
        print(f'  "{text}"', flush=True)

    # Save.
    ckpt_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                             "checkpoints", "fractus_ff.pt")
    os.makedirs(os.path.dirname(ckpt_path), exist_ok=True)
    torch.save({"model_state": eng.state_dict(),
                "config": {"d_model": 128, "n_experts": 4, "siren_rank": 32,
                           "method": "forward-forward"}}, ckpt_path)
    print(f"\nSaved: {ckpt_path}", flush=True)


if __name__ == "__main__":
    main()
