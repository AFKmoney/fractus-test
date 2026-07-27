#!/usr/bin/env python
"""Run the EDT AB-test on the 13M model.

Three arms (from-scratch / EDT vanilla / EDT+specialization), equal token
budget, hold-out perplexity comparison. Writes results.json + plots to
experiments/edt_ab/.

Usage:
    python scripts/ab_edt_13m.py [--budget 400000] [--chunk-len 16]
"""
import argparse, json, os, sys, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import experiments.edt_ab.ablib as ablib

CORPUS = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                      "data", "communication_corpus.pt")
OUTDIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                      "experiments", "edt_ab")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--budget", type=int, default=400_000)
    ap.add_argument("--n-holdout", type=int, default=30_000)
    ap.add_argument("--chunk-len", type=int, default=16)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--corpus", type=str, default=CORPUS)
    args = ap.parse_args()

    torch.set_num_threads(os.cpu_count() or 6)
    os.makedirs(OUTDIR, exist_ok=True)
    os.makedirs(os.path.join(OUTDIR, "samples"), exist_ok=True)

    print(f"Loading corpus {args.corpus} ...", flush=True)
    split = ablib.load_corpus(args.corpus, n_train=args.budget,
                              n_holdout=args.n_holdout,
                              n_phase1=int(args.budget * 0.15))
    print(f"  train={split['train'].numel():,} "
          f"holdout={split['holdout'].numel():,} "
          f"phase1={split['phase1'].numel():,}", flush=True)

    results = {}
    for name, fn in [("A_from_scratch", None),
                     ("B_edt_vanilla", None),
                     ("C_edt_spec", None)]:
        print(f"\n=== Arm {name} ===", flush=True)
        eng = ablib.build_engine(seed=42)
        t0 = time.time()
        if name == "A_from_scratch":
            out = ablib.arm_from_scratch(
                eng, train=split["train"], holdout=split["holdout"],
                budget=args.budget, chunk_len=args.chunk_len, lr=args.lr)
        elif name == "B_edt_vanilla":
            out = ablib.arm_edt_vanilla(
                eng, train=split["train"], holdout=split["holdout"],
                budget=args.budget, chunk_len=args.chunk_len, lr=args.lr)
        else:
            out = ablib.arm_edt_spec(
                eng, train=split["train"], holdout=split["holdout"],
                budget=args.budget, domain_split=split["domain_split"],
                chunk_len=args.chunk_len, lr=args.lr)
        out["wall_clock_s"] = time.time() - t0
        results[name] = out
        with open(os.path.join(OUTDIR, "samples", f"{name}.txt"), "w") as f:
            f.write(out.get("sample", ""))
        print(f"  ppl={out['ppl']:.2f}  div={out['diversity']:.3f}  "
              f"time={out['wall_clock_s']:.0f}s", flush=True)

    # results.json — strip the long loss lists into summary stats.
    summary = {}
    for name, out in results.items():
        summary[name] = {k: v for k, v in out.items()
                         if k not in ("losses", "phase1_losses",
                                       "phase2b_losses", "phase3_losses")}
        summary[name]["phase3_final_loss"] = (out.get("phase3_losses", out.get("losses", [0])) or [0])[-1]
    with open(os.path.join(OUTDIR, "results.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nWrote {os.path.join(OUTDIR, 'results.json')}", flush=True)

    # Plots.
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(8, 5))
        for name, out in results.items():
            curve = out.get("phase3_losses", out.get("losses", []))
            if curve:
                ax.plot(curve, label=name, alpha=0.8)
        ax.set_xlabel("Phase-3 optimizer step")
        ax.set_ylabel("loss")
        ax.set_title("EDT AB-test — Phase-3 loss curves")
        ax.legend()
        fig.tight_layout()
        fig.savefig(os.path.join(OUTDIR, "loss_curves.png"), dpi=110)
        print(f"Wrote {os.path.join(OUTDIR, 'loss_curves.png')}", flush=True)
    except Exception as e:
        print(f"  [plot] skipped: {e}", flush=True)


if __name__ == "__main__":
    main()
