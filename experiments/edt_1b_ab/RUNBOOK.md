# EDT 1B AB-test — GPU Runbook

This documents how to launch the full EDT 1B AB-test (which is not feasible on CPU: 1.6 tok/s measured on the dev Ryzen 5 5500U).

## Prerequisites

- A CUDA GPU with ≥ 24GB VRAM (RTX 3090/4090, A100, etc.). The 1B model uses ~4GB params + activations; AMP bf16 keeps it comfortable.
- A ~21B-token multi-domain corpus at `data/fractus_1b_corpus.pt` (build via `scripts/build_fractus_1b_corpus.py` if absent). The existing `data/communication_corpus.pt` (18.6M tokens) is far below Chinchilla-scale and will give an under-trained, contestable result.

## Launch

```bash
python scripts/ab_edt_1b.py \
    --budget 2000000000 \
    --n-holdout 100000 \
    --seq-len 64 \
    --lr 3e-4 \
    --corpus data/fractus_1b_corpus.pt
```

Expected runtime: ~2 days on a single A100/3090 (per the EDT doc's estimate). Outputs land in `experiments/edt_1b_ab/` (`results.json`, `loss_curves.png`, `samples/`).

## Smoke check (CPU, before spending GPU)

```bash
python scripts/ab_edt_1b.py --budget 1000 --force-cpu   # ~minutes; validates wiring
```

For a CPU validation of the full arm pipeline at reduced config, the unit tests in `tests/test_ablib_1b.py` already cover it (17 tests, including 3 arm smoke tests at `n_layers=2, n_experts=4`).

## Interpreting results

After the run, `experiments/edt_1b_ab/results.json` contains ppl/diversity/accuracy per arm. Apply the pre-registered decision rule (spec §9 of `docs/superpowers/specs/2026-07-27-edt-ab-test-design.md`):

- **EDT accelerates (A vs B):** supported iff `ppl_B ≤ 0.95 · ppl_A`.
- **Specialization matters (B vs C):** supported iff `ppl_C ≤ 0.95 · ppl_B` AND `diversity_C ≤ diversity_B − 0.05`.

If A-vs-B is positive, EDT works at the scale it was designed for (the ×186 acceleration claim confirmed at the decisive scale). If negative, EDT is refuted at every scale tested (13M and 1B) and the project should stop claiming it works. Either outcome is publishable.

## Known accounting notes (inherited from the 13M design)

- **Token overlap (Phase 1 vs Phase 3):** Arm B builds its Phase-1 banks from `train[:n1]`, which is a subset of `train[:n3]` used in Phase 3. Arm C's Phase-1 tokens come from the disjoint `phase1` split. This asymmetry is **inherited verbatim from the 13M experiment** (same `ablib` pattern) and is a known accepted design choice — it slightly advantages Arm A (from-scratch) which sees more distinct train tokens. It is symmetric for B-vs-C. Documented here for reviewers; see `experiments/edt_ab/REPORT.md` Limitation §1 for the 13M precedent.
- **Phase 2a is self-supervised (no corpus tokens):** Phase 2a trains on `torch.randn` (denoising target), so its "budget" is compute steps, not corpus consumption. The step count is computed so the total Phase-2a compute matches the nominal 15% allocation (`n2a / (n_layers * seq_len * batch_size)` steps per layer).

## The trained checkpoint is a living model

The trained 1B from each arm is **not a frozen artifact**. It is reusable, expandable via `scripts/rank_expand.py` (which copies existing trained weights and initializes only the new parameters), and continuable via `load_state_dict` (see `fractus1B/train_1b.py` for the resume pattern). Fractus is designed to grow; the EDT-trained checkpoint is the seed for the next growth step.

The natural follow-up experiment (out of scope here) is **"EDT for expansion"**: take a trained checkpoint, expand its rank/experts/layers, and test whether EDT accelerates integrating the *new* parameters — the practical use case that the living-model premise makes central.
