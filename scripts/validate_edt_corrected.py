#!/usr/bin/env python
"""Validate corrected EDT: routing filter + objective search at 13M.

Runs 5 variants at equal budget:
  A from-scratch (no Phase 1)
  B-nh      EDT vanilla (next_hidden objective, all experts) — reference
  B-denoise denoise objective, routed experts only
  B-identity identity objective, routed experts only
  B-residual residual objective, routed experts only

Verdict: does any corrected variant beat from-scratch by >=5% ppl?
"""
import os, sys, time, json, math
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
import torch.nn.functional as F
import experiments.edt_ab.ablib as ablib

BUDGET = 200_000      # tokens/variant (~25 min each on CPU)
N_HOLDOUT = 20_000
CHUNK_LEN = 16
SEED = 42
PHASE1_STEPS = 2000
CORPUS = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                      "data", "communication_corpus.pt")
OUTDIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                      "experiments", "edt_ab")

# Phase fractions (same as ablib arms).
P1, P2B, P3 = 0.15, 0.30, 0.55


def detect_routed_experts(engine, probe_tokens, seq_len=16, n_chunks=50):
    """Probe the model: which experts does the Kuramoto router actually select?"""
    engine.eval()
    engine.reset_thought(batch_size=1)
    moe = engine.moe
    hits = torch.zeros(moe.n_experts, dtype=torch.long)
    # Monkeypatch _compute_gates to capture topk_idx.
    orig = moe._compute_gates
    captured = []
    def spy_gates(phases):
        gates = orig(phases)
        topk_idx = gates.topk(moe.top_k, dim=-1).indices
        captured.append(topk_idx)
        return gates
    moe._compute_gates = spy_gates
    with torch.no_grad():
        for s in range(0, min(n_chunks * seq_len, probe_tokens.numel() - seq_len - 1), seq_len):
            chunk = probe_tokens[s:s+seq_len].unsqueeze(0)
            engine.tick_chunk(chunk)
    moe._compute_gates = orig
    for tk in captured:
        for e in range(moe.n_experts):
            hits[e] += (tk == e).sum().item()
    routed = [e for e in range(moe.n_experts) if hits[e] > 0]
    return routed, hits


def make_bank_for_objective(bank, objective, seed=0):
    """Derive (h_in, h_target) from the base (h_t, h_{t+1}) bank per objective."""
    h_t = bank["h_in"]
    h_next = bank["h_target"]
    g = torch.Generator().manual_seed(seed)
    if objective == "next_hidden":
        return {"h_in": h_t, "h_target": h_next}
    elif objective == "denoise":
        noise = 0.1 * torch.randn(*h_t.shape, generator=g)
        return {"h_in": h_t + noise, "h_target": h_t}
    elif objective == "identity":
        return {"h_in": h_t, "h_target": h_t}
    elif objective == "residual":
        return {"h_in": h_t, "h_target": h_next - h_t}
    raise ValueError(f"unknown objective {objective}")


def run_variant(name, split, objective=None, routed_only=False):
    """Run one variant. Returns dict with ppl, diversity, phase info."""
    eng = ablib.build_engine(seed=SEED)
    t0 = time.time()

    n1 = int(BUDGET * P1)
    n2b = int(BUDGET * P2B)
    n3 = BUDGET - n1 - n2b  # for EDT arms; Arm A uses full BUDGET

    if name == "A_from_scratch":
        # Full budget in Phase 3.
        p3 = ablib.phase3_joint(eng, split["train"][:BUDGET],
                                steps=BUDGET // CHUNK_LEN, lr=3e-4, chunk_len=CHUNK_LEN)
        losses = p3["losses"]
    else:
        # Phase 1: build bank, optionally filter to routed experts, train with objective.
        bank = ablib.make_hidden_bank(eng, split["train"][:n1],
                                      chunk_len=CHUNK_LEN, n_chunks=max(n1 // CHUNK_LEN, 1), seed=0)
        obj_bank = make_bank_for_objective(bank, objective) if objective else bank

        experts_to_train = list(range(eng.moe.n_experts))
        if routed_only:
            routed, hits = detect_routed_experts(eng, split["train"][:1000])
            experts_to_train = routed
            print(f"    routed experts: {routed} (hits={hits.tolist()})", flush=True)

        for e in experts_to_train:
            ablib._train_one_expert(eng, e, obj_bank, steps=PHASE1_STEPS, lr=1e-3,
                                    batch_size=64, seed=0)

        # Phase 2b: embedding + output_head.
        ablib.phase2b_embedding(eng, split["train"][:n2b],
                                steps=n2b // CHUNK_LEN, lr=1e-3, chunk_len=CHUNK_LEN, seed=0)

        # Phase 3: joint.
        p3 = ablib.phase3_joint(eng, split["train"][:n3],
                                steps=n3 // CHUNK_LEN, lr=3e-4, chunk_len=CHUNK_LEN)
        losses = p3["losses"]

    ppl = ablib.evaluate_ppl(eng, split["holdout"])
    div = ablib.expert_diversity(eng, split["holdout"][:256])
    elapsed = time.time() - t0
    print(f"  {name}: ppl={ppl:.2f}  div={div:.3f}  time={elapsed:.0f}s", flush=True)
    return {"ppl": ppl, "diversity": div, "time_s": elapsed,
            "phase3_final_loss": losses[-1] if losses else None}


def main():
    torch.set_num_threads(os.cpu_count() or 6)
    os.makedirs(OUTDIR, exist_ok=True)
    print(f"=== EDT Corrected Validation (budget={BUDGET}/variant) ===\n", flush=True)

    split = ablib.load_corpus(CORPUS, n_train=BUDGET, n_holdout=N_HOLDOUT,
                              n_phase1=int(BUDGET * P1))
    print(f"train={split['train'].numel():,} holdout={split['holdout'].numel():,}\n", flush=True)

    results = {}
    # Arm A: from-scratch.
    print("-- A_from_scratch --", flush=True)
    results["A_from_scratch"] = run_variant("A_from_scratch", split)

    # B-nh: reference (old objective, all experts).
    print("\n-- B_next_hidden (reference) --", flush=True)
    results["B_next_hidden"] = run_variant("B_nh", split, objective="next_hidden", routed_only=False)

    # Corrected variants: new objectives + routing filter.
    for obj in ["denoise", "identity", "residual"]:
        print(f"\n-- B_{obj} (corrected) --", flush=True)
        results[f"B_{obj}"] = run_variant(f"B_{obj}", split, objective=obj, routed_only=True)

    # Verdict.
    ppl_A = results["A_from_scratch"]["ppl"]
    print(f"\n{'='*60}", flush=True)
    print(f"{'variant':<20} {'ppl':>10} {'vs_A':>8} {'beats_5%?':>10}", flush=True)
    print(f"{'-'*60}", flush=True)
    for name, r in results.items():
        delta = (r["ppl"] - ppl_A) / ppl_A * 100
        beats = "YES" if r["ppl"] <= 0.95 * ppl_A else "no"
        print(f"{name:<20} {r['ppl']:>10.1f} {delta:>+7.1f}% {beats:>10}", flush=True)
    threshold = 0.95 * ppl_A
    print(f"\nFrom-scratch ppl={ppl_A:.1f}  5% threshold={threshold:.1f}", flush=True)

    winners = [n for n, r in results.items() if r["ppl"] <= threshold and n != "A_from_scratch"]
    if winners:
        best = min(winners, key=lambda n: results[n]["ppl"])
        print(f"\nVERDICT: corrected EDT WORKS — {best} beats from-scratch by >=5%.", flush=True)
        print(f"  Winning objective: {best.replace('B_', '')}", flush=True)
        print(f"  -> Port to 1B with this objective + routing filter.", flush=True)
    else:
        print(f"\nVERDICT: no corrected variant beats from-scratch by 5%.", flush=True)
        print(f"  EDT is fundamentally incompatible with Fractus at this scale.", flush=True)

    with open(os.path.join(OUTDIR, "corrected_results.json"), "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nWrote {os.path.join(OUTDIR, 'corrected_results.json')}", flush=True)


if __name__ == "__main__":
    main()
