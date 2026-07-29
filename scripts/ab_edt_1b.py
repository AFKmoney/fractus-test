#!/usr/bin/env python
"""Run the EDT AB-test on Fractus1B.

Three arms (from-scratch / EDT vanilla / EDT+specialization), equal token
budget, hold-out perplexity comparison. Writes results.json + plots to
experiments/edt_1b_ab/.

WARNING: the full 1B config needs a GPU (1.6 tok/s on CPU → weeks). Pass
--force-cpu to override (e.g. for smoke tests with --budget < 5000).

Usage:
    python scripts/ab_edt_1b.py --budget 2000000000 --corpus data/fractus_1b_corpus.pt
    python scripts/ab_edt_1b.py --budget 1000 --force-cpu      # smoke test
"""
import argparse, json, os, sys, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import experiments.edt_1b_ab.ablib_1b as ablib_1b

CORPUS = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                      "data", "communication_corpus.pt")
OUTDIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                      "experiments", "edt_1b_ab")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--budget", type=int, default=2_000_000_000)
    ap.add_argument("--n-holdout", type=int, default=100_000)
    ap.add_argument("--seq-len", type=int, default=64)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--corpus", type=str, default=CORPUS)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--force-cpu", action="store_true",
                    help="run on CPU even for large budgets (smoke tests)")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cpu" and args.budget >= 5000 and not args.force_cpu:
        print(f"ERROR: budget {args.budget} on CPU would take "
              f"~{args.budget / 1.6 / 3600:.0f} hours. Use a GPU, or pass "
              f"--force-cpu for a small smoke run.", flush=True)
        sys.exit(1)
    print(f"Device: {device}", flush=True)

    torch.set_num_threads(os.cpu_count() or 6)
    torch.manual_seed(args.seed)
    os.makedirs(OUTDIR, exist_ok=True)
    os.makedirs(os.path.join(OUTDIR, "samples"), exist_ok=True)

    print(f"Loading corpus {args.corpus} ...", flush=True)
    # Full 1B config for the real run; pass --config-reduced for CPU smoke.
    n_experts = 128
    split = ablib_1b.load_corpus_1b(args.corpus, n_train=args.budget,
                                    n_holdout=args.n_holdout,
                                    n_phase1=int(args.budget * ablib_1b.PHASE1_FRAC),
                                    n_experts=n_experts, seed=args.seed)
    print(f"  train={split['train'].numel():,} holdout={split['holdout'].numel():,} "
          f"phase1={split['phase1'].numel():,}", flush=True)

    results = {}
    for name in ["A_from_scratch", "B_edt_vanilla", "C_edt_spec"]:
        print(f"\n=== Arm {name} ===", flush=True)
        eng = ablib_1b.build_engine_1b(seed=args.seed).to(device)
        t0 = time.time()
        if name == "A_from_scratch":
            out = ablib_1b.arm_from_scratch_1b(
                eng, train=split["train"], holdout=split["holdout"],
                budget=args.budget, seq_len=args.seq_len, lr=args.lr)
        elif name == "B_edt_vanilla":
            out = ablib_1b.arm_edt_vanilla_1b(
                eng, train=split["train"], holdout=split["holdout"],
                budget=args.budget, seq_len=args.seq_len, lr=args.lr)
        else:
            out = ablib_1b.arm_edt_spec_1b(
                eng, train=split["train"], holdout=split["holdout"],
                budget=args.budget, domain_split=split["domain_split"],
                seq_len=args.seq_len, lr=args.lr)
        out["wall_clock_s"] = time.time() - t0
        results[name] = out
        with open(os.path.join(OUTDIR, "samples", f"{name}.txt"), "w") as f:
            f.write(out.get("sample", ""))
        print(f"  ppl={out['ppl']:.2f}  div={out['diversity']:.3f}  "
              f"time={out['wall_clock_s']:.0f}s", flush=True)

    summary = {}
    for name, out in results.items():
        summary[name] = {k: v for k, v in out.items()
                         if k not in ("losses", "phase1_losses", "phase2a_losses",
                                       "phase2b_losses", "phase3_losses")}
        summary[name]["phase3_final_loss"] = (out.get("phase3_losses", out.get("losses", [0])) or [0])[-1]
    with open(os.path.join(OUTDIR, "results.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nWrote {os.path.join(OUTDIR, 'results.json')}", flush=True)

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
        ax.set_title("EDT 1B AB-test — Phase-3 loss curves")
        ax.legend()
        fig.tight_layout()
        fig.savefig(os.path.join(OUTDIR, "loss_curves.png"), dpi=110)
    except Exception as e:
        print(f"  [plot] skipped: {e}", flush=True)


if __name__ == "__main__":
    main()
