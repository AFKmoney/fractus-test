"""Demo L7: full integration of the 3 fractus tasks.

The spec (L7) said: 3 demonstrable demos — text, mathematical reasoning,
causal inference. This demo orchestrates all 3 in a single script, and uses
the honest metrics from fractus.metrics.

What works (already validated in the L2b, L4 demos):
    - Text: TinyFractalLM learns 'hello world', loss 4.73 → 0.65.
    - Causal: NOTEARS recovers a synthetic DAG, SHD = 0.
What does NOT work (diagnosed in L5):
    - Proofs: pure REINFORCE does not learn (error plateaus). We document
      this honestly instead of pretending.

Run:
    python scripts/demo_full.py
"""

import sys
import os
import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fractus.nn import FractalEmbedding, FractalBlockFull
from fractus.causal.notears import notears_penalty
from fractus.metrics.causal import structural_hamming_distance
from fractus.metrics.perplexity import honest_perplexity
from fractus.metrics.compression import measure_compression_ratio
from data.causal.generate_scm import generate_linear_scm


def section(title: str) -> None:
    print()
    print("=" * 70)
    print(f"  {title}")
    print("=" * 70)


# ---------------------------------------------------------------------------
# Task 1: Text generation (FractalBlockFull)
# ---------------------------------------------------------------------------

class TinyFractalLM(nn.Module):
    def __init__(self, vocab, d_model, n_blocks=2):
        super().__init__()
        self.embed = FractalEmbedding(vocab, d_model, n_frequencies=8)
        self.blocks = nn.ModuleList([
            FractalBlockFull(
                d_model=d_model, n_heads=4, d_head=8, n_levels=2,
                n_oscillators=8, coupling_rank=4,
                n_experts=4, top_k=2, kappa=4.0,
            )
            for _ in range(n_blocks)
        ])
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab, bias=False)

    def forward(self, ids):
        x = self.embed(ids)
        aux = torch.tensor(0.0)
        for b in self.blocks:
            x, lb = b(x)
            aux = aux + lb
        x = self.norm(x)
        return self.head(x), aux


def demo_text():
    section("Task 1: Text generation (full FractalBlockFull)")
    torch.manual_seed(42)
    text = "hello world " * 8
    vocab = 128
    ids = torch.tensor([ord(c) - 32 for c in text if 0 <= ord(c) - 32 < vocab])
    seq_len = 16
    n_seqs = len(ids) // seq_len
    ids = ids[:n_seqs * seq_len].view(n_seqs, seq_len)

    model = TinyFractalLM(vocab=vocab, d_model=32, n_blocks=2)
    print(f"Parameters: {sum(p.numel() for p in model.parameters())}")
    print(f"Compression ratio (measured): {measure_compression_ratio(model):.2f}x")

    opt = torch.optim.Adam(model.parameters(), lr=3e-3)
    initial_loss = None
    for epoch in range(40):
        opt.zero_grad()
        logits, aux = model(ids)
        ce = nn.functional.cross_entropy(logits[:, :-1].reshape(-1, vocab), ids[:, 1:].reshape(-1))
        loss = ce + 0.1 * aux
        if initial_loss is None:
            initial_loss = ce.item()
        loss.backward()
        opt.step()
    final_loss = ce.item()

    # Honest perplexity (not a proxy).
    ppl = honest_perplexity(model, ids[:, :-1], ids[:, 1:])

    print(f"\nCE loss: {initial_loss:.4f} -> {final_loss:.4f} ({(1-final_loss/initial_loss)*100:.0f}% drop)")
    print(f"Honest perplexity: {ppl:.2f}  (= exp(CE))")

    # Generation.
    model.eval()
    with torch.no_grad():
        ctx = torch.tensor([[ord(c) - 32 for c in "hello"]])
        for _ in range(15):
            logits, _ = model(ctx)
            nxt = max(0, min(vocab - 1, int(logits[0, -1].argmax().item())))
            ctx = torch.cat([ctx, torch.tensor([[nxt]])], dim=1)
        gen = "".join(chr(int(i) + 32) for i in ctx[0].tolist())
    print(f"Generation: '{gen}'")


# ---------------------------------------------------------------------------
# Task 2: Mathematical reasoning (ProofGenerator + Verifier)
# ---------------------------------------------------------------------------

def demo_proofs():
    section("Task 2: Mathematical reasoning (exact verification)")
    from fractus.reasoning.proof import ProofGenerator, ProofVerifier, ProofReward
    from fractus.reasoning.conjecture import ConjectureDiscoveryLoop

    # 2a. The exact verifier is SOUND: everything it accepts is true.
    verify = ProofVerifier()
    examples = [
        ("7 is prime", lambda: verify.verify_primality(7, True)),
        ("8 is not prime", lambda: verify.verify_primality(8, False)),
        ("Fermat 2^6 mod 7 = 1", lambda: verify.verify_fermat(2, 7)),
        ("Wilson 6! mod 7 = 6", lambda: verify.verify_wilson(7)),
        ("gcd(12,18)=6, lcm=36, 6*36=216=12*18", lambda: verify.verify_gcd(12, 18)),
    ]
    print("Exact verifications (soundness guaranteed):")
    for desc, fn in examples:
        result = fn()
        print(f"  [{'OK' if result else 'KO'}] {desc}")

    # 2b. Conjecture discovery (Popper).
    print("\nConjecture discovery (Popperian falsification):")
    loop = ConjectureDiscoveryLoop(state_dim=32, n_trials=50, seed=42)
    for _ in range(30):
        loop.discover_step()
    print(f"  Conjectures in memory: {len(loop.memory.discovered)}")
    print(f"  Discoveries (survived + novelty): {loop.n_discoveries}")
    print(f"  Discovery rate: {loop.discovery_rate():.1%}")

    # 2c. HONEST VERDICT on REINFORCE.
    print("\nHONEST VERDICT (proof generator):")
    print("  The ProofGenerator + pure REINFORCE does NOT learn the proof task")
    print("  (error plateaus, see demo_proof_reinforce.py). The verifier is")
    print("  sound, but the generator is not enough to exploit it. Future work.")
    print("  → Conjectures, on the other hand, are tested by exact falsification (works).")


# ---------------------------------------------------------------------------
# Task 3: Causal inference (NOTEARS + do-calculus)
# ---------------------------------------------------------------------------

def demo_causal():
    section("Task 3: Causal inference (NOTEARS + do-calculus)")
    torch.manual_seed(42)
    W_true, X = generate_linear_scm(n_vars=5, n_samples=500, edge_prob=0.5, seed=7)
    print(f"Synthetic SCM: {X.shape[0]} samples, {X.shape[1]} variables")
    print(f"True DAG ({int((W_true != 0).sum())} edges):")
    print((W_true != 0).int())

    n_vars = W_true.shape[0]
    W_pred = torch.zeros(n_vars, n_vars, requires_grad=True)
    torch.nn.init.normal_(W_pred, std=0.1)
    opt = torch.optim.Adam([W_pred], lr=0.05)
    for _ in range(500):
        opt.zero_grad()
        X_pred = X @ W_pred
        recon = ((X_pred - X) ** 2).mean()
        h = notears_penalty(W_pred)
        loss = recon + h.abs()
        loss.backward()
        opt.step()

    shd = structural_hamming_distance(W_true, W_pred.detach(), threshold=0.3)
    print(f"\nLearned DAG (threshold 0.3):")
    print((W_pred.detach().abs() > 0.3).int())
    print(f"\nSHD = {shd}  (0 = perfect)")
    print(f"  (linear + upper-triangular toy case — proves the pipeline runs,")
    print(f"   not competence on real data.)")

    # do-calculus: effect of do(X_0 = v) on X_4.
    from fractus.causal.do import do_intervention
    X_do = do_intervention(X, var_idx=0, value=2.0)
    print(f"\ndo(X_0 = 2.0) applied: column 0 fixed to 2.0")
    print(f"  Average effect on X_4: observational {X[:, 4].mean():.3f} → "
          f"interventional {X_do[:, 4].mean():.3f}")


def main():
    print("FRACTUS — Full demo L7 (3 integrated tasks)")
    print("Honest rebuild of the original systems")
    demo_text()
    demo_proofs()
    demo_causal()
    section("END")
    print("Everything above uses HONEST MEASUREMENTS (no hardcoding).")
    print("See docs/SPEC.md.")


if __name__ == "__main__":
    main()
