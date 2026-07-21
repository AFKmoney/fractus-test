"""Benchmark: measure training throughput and energy on CPU.

This is the L8 baseline + verification tool. It runs a fixed, reproducible
training workload on the cpu-tiny preset and reports:

    - tokens_per_second   (training throughput)
    - ms_per_batch        (latency)
    - cpu_seconds         (user+sys CPU time = energy-normalized work)
    - cpu_seconds_per_token  (energy-efficiency metric; lower = better)

Run:
    python scripts/bench_train.py                   # baseline before L8 opts
    python scripts/bench_train.py --tag optimized   # after L8 opts
    python scripts/bench_train.py --compare          # diff baseline vs optimized

Output: writes/reads bench_<tag>.json next to this script.
"""

import argparse
import json
import os
import sys
import time

# Allow importing from the project root.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn

from fractus.nn import FractalEmbedding, FractalBlockFull


# ---------------------------------------------------------------------------
# Model (mirror of TinyFractalLM / FractalLM at the cpu-tiny size)
# ---------------------------------------------------------------------------

class BenchLM(nn.Module):
    def __init__(self, vocab=65, d_model=48, n_blocks=2):
        super().__init__()
        self.embed = FractalEmbedding(vocab, d_model, n_frequencies=12)
        self.blocks = nn.ModuleList([
            FractalBlockFull(
                d_model=d_model, n_heads=4, d_head=12, n_levels=2,
                n_oscillators=8, coupling_rank=4,
                n_experts=4, top_k=2, kappa=4.0,
            )
            for _ in range(n_blocks)
        ])
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab, bias=False)

    def forward(self, ids):
        x = self.embed(ids)
        aux = torch.tensor(0.0, device=x.device)
        for b in self.blocks:
            x, lb = b(x)
            aux = aux + lb
        x = self.norm(x)
        return self.head(x), aux


def run_bench(n_batches=80, batch_size=16, seq_len=32, vocab=65, seed=0):
    """Run a fixed training workload and return measured metrics.

    Returns a dict with the metrics. Deterministic (fixed seed, fixed data).
    """
    torch.manual_seed(seed)
    # Pin threads for reproducibility of CPU-time measurement.
    torch.set_num_threads(torch.get_num_threads())

    model = BenchLM(vocab=vocab, d_model=48, n_blocks=2)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-3, weight_decay=0.01)

    # Fixed synthetic data (deterministic).
    gen = torch.Generator().manual_seed(seed)
    inp = torch.randint(0, vocab, (batch_size, seq_len), generator=gen)
    tgt = torch.randint(0, vocab, (batch_size, seq_len), generator=gen)

    # Warmup (3 batches, untimed — let PyTorch lazily initialize / JIT kernels).
    for _ in range(3):
        opt.zero_grad()
        logits, aux = model(inp)
        ce = nn.functional.cross_entropy(
            logits.reshape(-1, vocab), tgt.reshape(-1)
        )
        loss = ce + 0.1 * aux
        loss.backward()
        opt.step()

    # Timed run.
    model.train()
    cpu0 = time.perf_counter()
    children0 = os.times()  # user, system, children_user, children_system, elapsed
    t0 = time.perf_counter()
    batch_times = []
    for i in range(n_batches):
        bt0 = time.perf_counter()
        opt.zero_grad()
        logits, aux = model(inp)
        ce = nn.functional.cross_entropy(
            logits.reshape(-1, vocab), tgt.reshape(-1)
        )
        loss = ce + 0.1 * aux
        loss.backward()
        opt.step()
        batch_times.append(time.perf_counter() - bt0)
    wall = time.perf_counter() - t0
    children1 = os.times()

    cpu_user = children1[0] - children0[0]   # user CPU seconds (this process + children)
    cpu_sys = children1[1] - children0[1]    # system CPU seconds
    cpu_total = cpu_user + cpu_sys           # total CPU work

    tokens = n_batches * batch_size * seq_len
    return {
        "n_batches": n_batches,
        "batch_size": batch_size,
        "seq_len": seq_len,
        "vocab": vocab,
        "n_params": int(sum(p.numel() for p in model.parameters())),
        "torch_threads": int(torch.get_num_threads()),
        "wall_seconds": round(wall, 4),
        "cpu_seconds": round(cpu_total, 4),
        "tokens_per_second": round(tokens / wall, 2),
        "ms_per_batch": round(1000.0 * (sum(batch_times) / len(batch_times)), 3),
        "cpu_seconds_per_1k_tokens": round(1000.0 * cpu_total / tokens, 5),
        "final_ce": round(float(ce.item()), 4),
    }


def run_bench_median(runs=3, **kwargs):
    """Run the bench `runs` times and return the MEDIAN of the key metrics.

    Median is robust to system-load spikes (unlike mean). This is the honest
    way to compare before/after on a shared CPU.
    """
    results = [run_bench(**kwargs) for _ in range(runs)]
    # Median of the timing-sensitive fields.
    med = dict(results[0])  # copy static fields
    for key in ["wall_seconds", "cpu_seconds", "tokens_per_second",
                "ms_per_batch", "cpu_seconds_per_1k_tokens"]:
        vals = sorted(r[key] for r in results)
        med[key] = vals[len(vals) // 2]  # median
    med["runs"] = runs
    med["all_tokens_per_second"] = [r["tokens_per_second"] for r in results]
    return med


def main():
    p = argparse.ArgumentParser(description="Benchmark fractus training on CPU.")
    p.add_argument("--tag", default="baseline",
                   help="Tag for the output file (e.g. baseline, optimized).")
    p.add_argument("--n-batches", type=int, default=80)
    p.add_argument("--runs", type=int, default=3,
                   help="Number of runs; report the median (robust to load spikes).")
    p.add_argument("--compare", action="store_true",
                   help="Print a diff between baseline and optimized JSON.")
    args = p.parse_args()

    if args.compare:
        here = os.path.dirname(os.path.abspath(__file__))
        base_path = os.path.join(here, "bench_baseline.json")
        opt_path = os.path.join(here, "bench_optimized.json")
        if not os.path.exists(base_path):
            print(f"No baseline file at {base_path}. Run with --tag baseline first.")
            sys.exit(1)
        if not os.path.exists(opt_path):
            print(f"No optimized file at {opt_path}. Run with --tag optimized first.")
            sys.exit(1)
        with open(base_path) as f:
            base = json.load(f)
        with open(opt_path) as f:
            opt = json.load(f)
        print("=" * 64)
        print("  L8 BENCHMARK: baseline vs optimized")
        print("=" * 64)
        for key in ["tokens_per_second", "ms_per_batch", "cpu_seconds",
                    "cpu_seconds_per_1k_tokens"]:
            b = base[key]
            o = opt[key]
            if o != 0:
                if key in ("tokens_per_second",):
                    ratio = o / b
                    better = "FASTER" if ratio > 1 else "slower"
                    print(f"  {key:30s}  {b:>10.3f}  ->  {o:>10.3f}   "
                          f"×{ratio:.2f} {better}")
                else:
                    ratio = o / b
                    better = "LOWER (better)" if ratio < 1 else "HIGHER (worse)"
                    print(f"  {key:30s}  {b:>10.3f}  ->  {o:>10.3f}   "
                          f"×{ratio:.2f} {better}")
        print("=" * 64)
        return

    print(f"Running benchmark (tag={args.tag}, {args.n_batches} batches, "
          f"{args.runs} runs, median)...")
    metrics = run_bench_median(runs=args.runs, n_batches=args.n_batches)

    here = os.path.dirname(os.path.abspath(__file__))
    out_path = os.path.join(here, f"bench_{args.tag}.json")
    with open(out_path, "w") as f:
        json.dump(metrics, f, indent=2)

    print()
    print("=" * 64)
    print(f"  BENCHMARK RESULTS ({args.tag})")
    print("=" * 64)
    for k, v in metrics.items():
        print(f"  {k:30s}  {v}")
    print("=" * 64)
    print(f"Written to: {out_path}")


if __name__ == "__main__":
    main()
