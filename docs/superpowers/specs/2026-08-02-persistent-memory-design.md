# Persistent Memory — Design spec

**Date:** 2026-08-02
**Author:** Philippe-Antoine Robert (with ZCode)
**Status:** Design — pending user review
**Scope:** Modifies `fractus/memory.py` and `fractus/continuous_engine.py`. New test file. No other components touched.

---

## 1. Motivation

The PersistentMemory module (`fractus/memory.py`, 195 lines) is functional — it can store vectors, recall by cosine similarity, consolidate with LRU eviction, save/load to disk. But it is **not connected to the ContinuousThoughtEngine**. It exists as a standalone bank that nothing calls during the engine's tick loop. This makes it dead code: Fractus's headline USP (true long-term, personal, cross-session memory — *"the engine remembers the user, their preferences, and the context of past interactions"* per white paper §10) is not wired up.

This spec connects the memory to the CTE with two mechanisms:
- **Continuous injection**: at every tick, recalled memories are blended into the thought state (dilute: 95% current thought + 5% memory) so the engine "remembers in the background" without drowning the current signal.
- **Automatic consolidation**: a learnable `salience_head` scores each tick's thought state in `[0,1]`; when the score exceeds a threshold, the thought is consolidated into the memory bank. The engine learns (in future work) what is worth keeping — for now we build the mechanism, the training signal comes later.

## 2. What changes

### 2.1 `fractus/memory.py` (modify)

- `inject(engine, *, blend=0.05, top_k=3)`: lower default blend from the current 0.20 to **0.05** (95/5). The current 0.20 was designed for one-shot reset-time injection; continuous per-tick injection at 0.20 would dominate the signal.
- `consolidate_if_salient(thought_state, salience_score, *, context="", threshold=0.7, min_distance=0.1)`: only calls `consolidate` if `salience_score >= threshold` AND the thought is at least `min_distance` (cosine) from the closest existing memory (de-duplication). Returns `True` if a memory was written, `False` otherwise.
- `recall`, `consolidate`, `save`, `load`, `clear`, `__len__`, `summary`: unchanged.

### 2.2 `fractus/continuous_engine.py` (modify)

- Constructor: add two attributes.
  - `self.salience_head = nn.Linear(d_model, 1)` — scores how memorable the current thought is.
  - `self.memory: PersistentMemory | None = None` — optional attached memory bank.
  - `self.memory_active: bool = True` — toggle to disable injection during training if it perturbs convergence (default True; the trainer can set it False during Phase 3).
- `attach_memory(memory)`, `detach_memory()`: connect/disconnect a `PersistentMemory`.
- `tick(observation)`: after the MoE step and before the output head, insert the memory block:
  1. Compute `salience = torch.sigmoid(self.salience_head(h[:, 0, :]))` — shape `(B, 1)`. For consolidation decisions, take the first batch element's value (the CTE's `tick` is a singleton-thought path, `B=1`).
  2. If `self.memory is not None and self.memory_active`:
     a. `self.memory.consolidate_if_salient(h[0:1, 0, :], salience[0].item())` — capture if salient (first batch element).
     b. `self.memory.inject(self, blend=0.05, top_k=3)` — recall + blend into `thought_state`.
- `tick_chunk(observations)`: the memory block is NOT added to `tick_chunk` (chunk processing is for throughput; memory is a per-tick semantic concern). The trainer calls `tick_chunk` in Phase 3 with `memory_active=False` by default; conversational inference uses `tick` with memory on.

### 2.3 `tests/test_memory.py` (new)

Tests for `PersistentMemory` (in isolation from the engine) and for the CTE+memory integration:

- `test_memory_recall_topk`: store 5 labeled vectors, query with a near-duplicate, assert the right ones come back ranked by cosine.
- `test_memory_consolidate_lru`: fill past `max_memories`, assert the lowest-importance entry is evicted.
- `test_memory_save_load_roundtrip`: save to a temp file, load into a fresh instance, assert vectors/contexts/importance match.
- `test_memory_inject_blend`: build a fake engine-like object with a `thought_state`, inject a known memory, assert the blend ratio (5% memory) holds.
- `test_memory_consolidate_if_salient_threshold`: score below threshold → no write; above threshold → write; above threshold but duplicate (within `min_distance`) → no write.
- `test_cte_has_salience_head`: `build_engine()` returns a CTE with `salience_head` and `memory=None`.
- `test_cte_salience_head_outputs_in_range`: one tick → `salience_head` output is a finite sigmoid value in `[0,1]`.
- `test_cte_attach_memory_consolidates_when_salient`: attach a memory, force `salience_head` to output ~1.0 (by setting its bias high), tick once, assert `len(memory) > 0`.
- `test_cte_attach_memory_skips_when_not_salient`: same but force salience ~0.0, assert `len(memory) == 0`.
- `test_cte_memory_active_toggle`: with `memory_active=False`, even a salient tick does not consolidate.

### 2.4 Out of scope

- **Training the salience head.** It is constructed and forward-called but its weights are at init in this iteration. Future work: a reward signal (does a consolidated memory improve future next-token loss when recalled?) to train it. Mirrors the white paper's admission about cognitive modes being untrained.
- **Memory in `tick_chunk`.** Chunk path stays memory-free (it's a throughput path). Memory lives in the per-tick `tick` path.
- **`min_distance` / `threshold` tuning.** Use sensible defaults (0.1 / 0.7); tuning is empirical future work once the salience head is trained.
- **Changing `inject`'s recall mechanism** (still top-k cosine). A learned retrieval mechanism is future work.

## 3. Data flow (per-tick inference with memory)

```
tick(observation)
  → absorb obs → attention → kuramoto → moe → h
  → salience = sigmoid(salience_head(h))                  [new]
  → if memory attached and memory_active:                  [new]
       memory.consolidate_if_salient(h, salience)          [new, conditional write]
       memory.inject(self, blend=0.05, top_k=3)            [new, recall + blend 95/5]
  → thought_state = h (post-blend)
  → output_head → logits
```

During Phase-3 training, the trainer sets `memory_active=False` so the memory does not perturb the joint training loop. Memory is a runtime/conversational feature, not a training-time one (until the salience head is trained, which is future work).

## 4. Risks

| Risk | Mitigation |
|---|---|
| Continuous 5% blend still perturbs the thought trajectory enough to degrade generation | `memory_active` toggle lets the trainer disable it; the 5% default is conservative. If still harmful at inference, expose `blend` as a runtime arg and default lower (e.g., 0.02). |
| Untrained salience head outputs ~0.5 → consolidates ~half of all ticks → memory bank fills with near-duplicates | `consolidate_if_salient`'s `min_distance=0.1` de-duplicates; the LRU eviction caps the bank. Plus the untrained head is a known temporary state. |
| `inject` moves the thought_state on a different device than the memory vectors (CPU) | `inject` already does `memory_contribution.to(engine.thought_state.device)`; preserved. |
| Adding the salience_head changes the parameter count, breaking the `test_build_engine_is_deterministic_and_13m` bounds [12M, 14M] | One extra `Linear(128, 1)` = 129 params; well within the 13M→14M band. No test update needed. |
| Existing CTE callers (the 13M ablib, the 1B ablib_1b) break because the constructor signature changes | The constructor signature is unchanged — `salience_head` and `memory` are created internally, not passed in. Existing callers are unaffected. |

## 5. Success criteria

1. `tests/test_memory.py` passes (all ~10 tests above).
2. Existing `tests/test_continuous_engine_grad.py` still passes (the salience_head does not break gradient flow to the MoE).
3. Existing `tests/test_ablib.py` still passes (CTE construction + Phase 1/2b/3 unaffected).
4. A demo snippet works: build a CTE, attach a memory, tick a few tokens with high-salience bias, detach, build a fresh CTE, attach the same memory, tick → the memory is recalled and influences the new thought state (proves cross-session persistence end-to-end).
