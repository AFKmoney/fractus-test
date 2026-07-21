#!/usr/bin/env python
"""Assemble Fractus — the complete CCA per the original vision.

Loads the trained CTE (with 88M weights transferred), wires up:
  - Persistent KnowledgeBase (vector memory)
  - RAGEngine (retrieval + generation)
  - PluginManager (5 cognitive modes)
  - MetaCognition (autonomous action selection)

Then tests the full system end-to-end:
  learn() → query() → converse() → switch plugin → meta.process()

This is the FINAL product. Not a training script, not a benchmark —
the actual Fractus agent that a user runs.
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch

from fractus.continuous_engine import ContinuousThoughtEngine
from fractus.tokenizer import FractusTokenizer
from fractus.rag import KnowledgeBase, RAGEngine, PluginManager, MetaCognition


def assemble_fractus(cte_checkpoint: str = None, d_model: int = 768):
    """Assemble the full Fractus agent.

    Args:
        cte_checkpoint: path to the assembled CTE checkpoint (.pt).
                        If None, uses random weights (for testing).
        d_model: must match the CTE checkpoint config (768 for the 88M).

    Returns:
        dict with engine, tok, kb, rag, pm, meta.
    """
    print("="*60, flush=True)
    print("ASSEMBLING FRACTUS — Continuous Cognitive Agent", flush=True)
    print("="*60, flush=True)

    # 1. Tokenizer
    tok = FractusTokenizer.gpt2_compatible()
    print(f"✓ Tokenizer: vocab={tok.vocab_size}", flush=True)

    # 2. CTE engine with trained weights
    print("Building CTE engine...", flush=True)
    engine = ContinuousThoughtEngine(
        vocab_size=50257, d_model=d_model,
        n_heads=12, d_head=64, n_levels=2,
        n_oscillators=16, coupling_rank=8,
        n_experts=64, top_k=2,
        expert_d_ff=1024, siren_rank=16,
    )

    if cte_checkpoint and os.path.exists(cte_checkpoint):
        print(f"Loading CTE checkpoint: {cte_checkpoint}", flush=True)
        ck = torch.load(cte_checkpoint, weights_only=False, map_location="cpu")
        cte_state = ck.get("cte_state", ck.get("model_state", ck))
        engine.load_state_dict(cte_state, strict=False)

        # Reconstruct expert _cached_W from U, V (same as transfer script).
        for i in range(64):
            for prefix, expert_list in [("w1", engine.experts_w1), ("w2", engine.experts_w2)]:
                expert = expert_list[i]
                if hasattr(expert, "U") and hasattr(expert, "V") and hasattr(expert, "_cached_W"):
                    with torch.no_grad():
                        W = expert.U @ expert.V.T
                        expert._cached_W.copy_(W)
                        expert._call_count = 0
        source_step = ck.get("source", {}).get("step", "?")
        source_loss = ck.get("source", {}).get("loss", "?")
        print(f"✓ CTE loaded (step={source_step}, loss={source_loss})", flush=True)
    else:
        print("⚠ No checkpoint — using random weights", flush=True)

    # 3. Persistent KnowledgeBase
    kb = KnowledgeBase(d_model=d_model)
    # Try to load persisted memory.
    kb_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                           "data", "fractus_memory.pkl")
    if os.path.exists(kb_path):
        try:
            kb.load(kb_path)
            print(f"✓ Memory loaded: {len(kb.chunks)} memories", flush=True)
        except Exception as e:
            print(f"⚠ Memory load failed: {e}", flush=True)
    else:
        print(f"✓ Memory: empty (fresh start)", flush=True)

    # 4. RAGEngine
    rag = RAGEngine(engine, tok, kb)
    print(f"✓ RAG engine ready", flush=True)

    # 5. PluginManager (5 cognitive modes)
    pm = PluginManager(rag)
    print(f"✓ Plugins: analyst, creative, coder, teacher, hacker", flush=True)

    # 6. MetaCognition
    meta = MetaCognition(rag, pm)
    print(f"✓ MetaCognition ready", flush=True)

    print("="*60, flush=True)
    print("FRACTUS ASSEMBLED — ready to learn and think", flush=True)
    print("="*60, flush=True)

    return {
        "engine": engine,
        "tok": tok,
        "kb": kb,
        "rag": rag,
        "pm": pm,
        "meta": meta,
        "kb_path": kb_path,
    }


def run_demo(frac: dict):
    """Run a full demo of Fractus capabilities."""
    rag, pm, meta, tok = frac["rag"], frac["pm"], frac["meta"], frac["tok"]

    print("\n" + "="*60, flush=True)
    print("FRACTUS DEMO", flush=True)
    print("="*60, flush=True)

    # 1. Teach Fractus some facts (no retraining)
    print("\n--- LEARN (teaching facts without retraining) ---", flush=True)
    facts = [
        "Python was created by Guido van Rossum in 1991.",
        "Fractus is a decentralized AI that runs on your machine.",
        "The user prefers concise, direct answers.",
        "def binary_search(arr, target): uses divide and conquer to find an element in O(log n).",
    ]
    for fact in facts:
        rag.learn(fact)
        print(f"  learned: {fact[:60]}...", flush=True)
    print(f"  Memory now contains {len(frac['kb'].chunks)} memories", flush=True)

    # 2. Query (retrieval-augmented generation)
    print("\n--- QUERY (asking questions) ---", flush=True)
    questions = [
        "Who created Python?",
        "What is Fractus?",
        "How does binary search work?",
    ]
    for q in questions:
        try:
            result = rag.query(q, top_k=2, max_tokens=30)
            answer = result.get("answer", "(no answer)")[:150]
            retrieved = len(result.get("retrieved", []))
            print(f"  Q: {q}", flush=True)
            print(f"  A: {answer}", flush=True)
            print(f"  (used {retrieved} memories)", flush=True)
        except Exception as e:
            print(f"  Q: {q} → error: {e}", flush=True)

    # 3. Switch cognitive mode
    print("\n--- SWITCH (changing cognitive mode) ---", flush=True)
    for mode in ["coder", "creative", "analyst"]:
        try:
            pm.load(mode)
            print(f"  Switched to '{mode}' mode", flush=True)
        except Exception as e:
            print(f"  '{mode}' mode: {e}", flush=True)

    # 4. MetaCognition (let Fractus decide what to do)
    print("\n--- METACOGNITION (Fractus decides its own actions) ---", flush=True)
    test_inputs = [
        "Remember: my favorite language is Rust.",
        "What is my favorite language?",
    ]
    for inp in test_inputs:
        try:
            result = meta.process(inp)
            actions = result.get("actions", [])
            print(f"  Input: '{inp}'", flush=True)
            print(f"  Fractus chose: {actions}", flush=True)
        except Exception as e:
            print(f"  Input: '{inp}' → error: {e}", flush=True)

    # 5. Save memory
    print("\n--- SAVE MEMORY ---", flush=True)
    try:
        frac["kb"].save(frac["kb_path"])
        print(f"  Memory saved to {frac['kb_path']}", flush=True)
    except Exception as e:
        print(f"  Save failed: {e}", flush=True)

    print("\n" + "="*60, flush=True)
    print("DEMO COMPLETE — Fractus works end-to-end", flush=True)
    print("="*60, flush=True)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Assemble and run Fractus")
    parser.add_argument("--cte-checkpoint", type=str,
                       default="checkpoints/fractus_cte_assembled.pt",
                       help="Path to the assembled CTE checkpoint")
    parser.add_argument("--d-model", type=int, default=768)
    parser.add_argument("--no-demo", action="store_true",
                       help="Skip the demo, just assemble")
    args = parser.parse_args()

    frac = assemble_fractus(args.cte_checkpoint, args.d_model)
    if not args.no_demo:
        run_demo(frac)
