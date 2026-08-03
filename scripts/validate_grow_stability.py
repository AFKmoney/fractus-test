#!/usr/bin/env python
"""Validate stability after runtime expert growth.

Measures, for 500 ticks before and 1000 ticks after a forced add_expert():
  - loss (CE) trajectory
  - routing distribution (which experts get traffic)
  - gradient norm
  - does the new expert receive routing hits?

If the loss spikes and never recovers, or the new expert gets 0 traffic,
the growth is unstable and needs fixing.
"""
import os, sys, math
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
import torch.nn.functional as F
import experiments.edt_ab.ablib as ablib

CORPUS = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                      "data", "communication_corpus.pt")


def measure_routing(engine, phases_in):
    """Return per-expert hit counts for one tick."""
    with torch.no_grad():
        gates = engine.moe._compute_gates(phases_in)
        topk_idx = gates.topk(engine.moe.top_k, dim=-1).indices
        hits = torch.zeros(engine.moe.n_experts)
        for e in range(engine.moe.n_experts):
            hits[e] = (topk_idx == e).sum().item()
    return hits


def main():
    torch.set_num_threads(8)
    torch.manual_seed(42)
    print("=== Grow Stability Validation ===\n", flush=True)

    tokens = torch.load(CORPUS, weights_only=False).to(torch.int64)[:5000]
    eng = ablib.build_engine(seed=42)
    opt = torch.optim.AdamW(eng.parameters(), lr=3e-4, weight_decay=0.01)
    eng.train(); eng.reset_thought(batch_size=1)

    PRE_GROW = 500
    POST_GROW = 1000
    GROW_AT = PRE_GROW

    losses = []
    grad_norms = []
    routing = []  # list of (tick, hits_tensor)

    for t in range(PRE_GROW + POST_GROW):
        obs = tokens[t % len(tokens):t % len(tokens) + 1]
        if obs.numel() == 0:
            obs = tokens[:1]
        target = tokens[(t % len(tokens)) + 1:(t % len(tokens)) + 2]
        if target.numel() == 0:
            target = tokens[1:2]

        # Forward + loss.
        logits, _ = eng.tick(obs)
        ce = F.cross_entropy(logits, target)
        opt.zero_grad()
        ce.backward()

        # Gradient norm (before clip, for monitoring).
        grad_norm = torch.cat([p.grad.flatten() for p in eng.parameters()
                               if p.grad is not None]).norm().item()

        torch.nn.utils.clip_grad_norm_(eng.parameters(), 1.0)
        opt.step()

        losses.append(ce.item())
        grad_norms.append(grad_norm)

        # Track routing (cheap proxy: recompute gates on current phases).
        if hasattr(eng, 'kuramoto_phases'):
            phases_in = eng.kuramoto_phases[:, 0:1, :]  # (1, 1, n_osc)
            if phases_in.shape[-1] == eng.kuramoto_phases.shape[-1]:
                hits = measure_routing(eng, phases_in)
                routing.append((t, hits.tolist()))

        # Force growth at GROW_AT.
        if t == GROW_AT:
            print(f"\n--- GROWING at tick {t} (was {eng.moe.n_experts} experts) ---", flush=True)
            dominant_idx = eng._expert_hits.argmax().item() if hasattr(eng, '_expert_hits') else None
            new_idx = eng.moe.add_expert(dominant_idx=dominant_idx)
            print(f"    New expert {new_idx} added near dominant {dominant_idx}. "
                  f"Now {eng.moe.n_experts} experts. Zero-init.", flush=True)
            # Rebuild optimizer but copy AdamW state for existing params (preserve momentum).
            old_state = opt.state
            opt = torch.optim.AdamW(eng.parameters(), lr=3e-4, weight_decay=0.01)
            # Restore AdamW state for params that existed before (matched by shape + position).
            for old_p, old_s in old_state.items():
                for new_p in opt.param_groups[0]['params']:
                    if new_p.shape == old_p.shape and new_p.data_ptr() != old_p.data_ptr():
                        # Same shape — likely the same param (data was reallocated by cat).
                        # We can't perfectly match, but AdamW state starts empty for new params
                        # which is fine — the new expert is zero-init and warms up from scratch.
                        pass
            print(f"    Optimizer rebuilt (new expert starts fresh, old params lose AdamW momentum).", flush=True)

        if (t + 1) % 250 == 0:
            phase = "PRE" if t < GROW_AT else "POST"
            window = losses[-250:]
            print(f"  t={t+1:>5} [{phase}] loss={sum(window)/len(window):.3f} "
                  f"grad={sum(grad_norms[-250:])/250:.2f} "
                  f"E={eng.moe.n_experts}", flush=True)

    # Analysis.
    print(f"\n{'='*60}", flush=True)
    pre_losses = losses[:GROW_AT]
    post_losses = losses[GROW_AT:]

    # Loss around the grow point (±50 ticks).
    around = losses[max(0, GROW_AT-50):GROW_AT+50]
    pre_mean = sum(around[:50]) / 50
    post_50 = sum(around[50:100]) / 50
    post_500 = sum(post_losses[450:550]) / min(100, len(post_losses) - 450)

    print(f"Loss (mean):", flush=True)
    print(f"  50 ticks before grow:  {pre_mean:.3f}", flush=True)
    print(f"  50 ticks after grow:   {post_50:.3f}  (spike: {(post_50/pre_mean - 1)*100:+.1f}%)", flush=True)
    print(f"  500 ticks after grow:  {post_500:.3f}  (recovery: {(post_500/pre_mean - 1)*100:+.1f}%)", flush=True)

    # Did the new expert get traffic?
    post_routing = [r for t, r in routing if t >= GROW_AT and len(r) > 4]
    if post_routing:
        new_expert_traffic = [r[-1] for r in post_routing]  # last expert = new one
        total_traffic = [sum(r) for r in post_routing]
        pct = [ne / max(t, 1) * 100 for ne, t in zip(new_expert_traffic, total_traffic)]
        print(f"\nNew expert traffic:", flush=True)
        print(f"  Hits over {len(post_routing)} post-grow ticks: {sum(new_expert_traffic):.0f}", flush=True)
        print(f"  Avg % of routing: {sum(pct)/len(pct):.1f}%", flush=True)
        if sum(new_expert_traffic) > 0:
            print(f"  VERDICT: new expert IS being routed to.", flush=True)
        else:
            print(f"  VERDICT: new expert gets ZERO traffic — routing didn't adapt.", flush=True)

    # Gradient stability.
    post_grads = grad_norms[GROW_AT:]
    max_grad = max(post_grads)
    print(f"\nGradient stability:", flush=True)
    print(f"  Max grad norm post-grow: {max_grad:.2f}", flush=True)
    print(f"  Mean grad norm post-grow: {sum(post_grads)/len(post_grads):.2f}", flush=True)
    if max_grad > 100:
        print(f"  WARNING: gradient spike detected — growth may destabilize training.", flush=True)
    else:
        print(f"  Gradients stable.", flush=True)

    print(f"\n{'='*60}", flush=True)
    if post_500 <= pre_mean * 1.1:
        print(f"OVERALL: STABLE — loss recovered within 10% of pre-grow level.", flush=True)
    else:
        print(f"OVERALL: UNSTABLE — loss did not recover within 10%.", flush=True)


if __name__ == "__main__":
    main()
