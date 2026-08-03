# Persistent Memory Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Wire the `PersistentMemory` module into the `ContinuousThoughtEngine` with continuous injection (5% blend per tick) and automatic consolidation via a learnable salience head.

**Architecture:** `PersistentMemory` gains a softer `inject` (blend 0.05) and a salience-gated `consolidate_if_salient`. The CTE gains a `salience_head` and an optional `memory` slot; `tick` calls consolidation + injection after the MoE block. Memory is toggleable (`memory_active`) so Phase-3 training can disable it.

**Tech Stack:** PyTorch 2.9 (CPU), pytest, Windows Store CPython 3.13 (`$PY`).

**Reference spec:** `docs/superpowers/specs/2026-08-02-persistent-memory-design.md`.

**API facts (verified):**
- `PersistentMemory(d_model, max_memories=256, path=None)` — current methods: `recall`, `consolidate`, `inject`, `save`, `load`, `clear`, `__len__`, `summary`.
- `ContinuousThoughtEngine.__init__` — creates `observe`, `attn`, `norm_attn`, `kuramoto`, `norm_kur`, `moe`, `norm_moe`, `confidence_head`, `output_head`, and buffers `thought_state`/`attn_S`/`attn_z`/`kuramoto_phases`.
- `tick(observation) -> (logits, confidence)` — the per-tick forward; `h` is the thought state after MoE, before confidence/output heads.
- `engine.thought_state` is a buffer of shape `(B, 1, d_model)`.

**IMPORTANT — Python interpreter (Windows):**
```
PY="/c/Users/PHIL/AppData/Local/Microsoft/WindowsApps/PythonSoftwareFoundation.Python.3.13_qbz5n2kfra8p0/python.exe"
"$PY" -m pytest tests/test_memory.py -v
```
Shell: Git Bash. Working dir: `C:\Users\PHIL\ZCodeProject\fractus-test`.

---

## File structure

| Path | Responsibility |
|---|---|
| `fractus/memory.py` | + softer `inject`, + `consolidate_if_salient` |
| `fractus/continuous_engine.py` | + `salience_head`, + `memory` slot, + memory block in `tick`, + `attach_memory`/`detach_memory` |
| `tests/test_memory.py` | all new — ~10 tests for memory + CTE integration |

---

## Task 1: `PersistentMemory.consolidate_if_salient` + softer `inject`

**Files:**
- Modify: `fractus/memory.py`
- Test: `tests/test_memory.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_memory.py`:
```python
"""Tests for PersistentMemory (isolated from the engine)."""
import tempfile, os
import torch
import pytest
from fractus.memory import PersistentMemory


def test_memory_recall_topk():
    mem = PersistentMemory(d_model=4, max_memories=10)
    # Store 5 vectors; query should rank the closest first.
    for i in range(5):
        v = torch.zeros(4); v[i] = 1.0
        mem.consolidate(v, context=f"mem{i}", importance=0.5)
    query = torch.zeros(4); query[0] = 1.0  # identical to mem0
    results = mem.recall(query, top_k=3)
    assert len(results) == 3
    assert results[0][0] == "mem0"  # highest cosine
    assert results[0][1] > results[1][1]  # descending


def test_memory_consolidate_lru():
    mem = PersistentMemory(d_model=4, max_memories=3)
    for i in range(5):
        mem.consolidate(torch.randn(4), context=f"mem{i}", importance=float(i) / 10)
    assert len(mem) == 3
    # Lowest importance (mem0, imp=0.0) should have been evicted.
    contexts = mem.contexts
    assert "mem0" not in contexts


def test_memory_save_load_roundtrip():
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "mem.pt")
        mem = PersistentMemory(d_model=4, max_memories=10, path=path)
        mem.consolidate(torch.tensor([1.0, 0, 0, 0]), context="hello", importance=0.8)
        mem.save()
        mem2 = PersistentMemory(d_model=4, max_memories=10, path=path)
        assert len(mem2) == 1
        assert mem2.contexts[0] == "hello"
        assert torch.allclose(mem2.vectors[0], torch.tensor([1.0, 0, 0, 0]))


def test_memory_inject_blend_ratio():
    """inject with blend=0.05 → thought_state is 95% original + 5% memory."""
    mem = PersistentMemory(d_model=4, max_memories=10)
    mem.consolidate(torch.tensor([0.0, 0.0, 0.0, 10.0]), context="big", importance=1.0)

    class FakeEngine:
        def __init__(self):
            self.thought_state = torch.tensor([[[1.0, 1.0, 1.0, 1.0]]])  # (1,1,4)

    eng = FakeEngine()
    before = eng.thought_state.clone()
    mem.inject(eng, blend=0.05, top_k=1)
    after = eng.thought_state
    # The memory vector is [0,0,0,10]; recalled contribution pushes toward it.
    # dim 3 (index 3) should increase, dims 0-2 should stay ~same (memory is 0 there).
    assert after[0, 0, 3] > before[0, 0, 3]  # memory's non-zero dim increased
    assert abs(after[0, 0, 0].item() - before[0, 0, 0].item()) < 0.01  # memory's zero dim unchanged


def test_memory_consolidate_if_salient_threshold():
    mem = PersistentMemory(d_model=4, max_memories=10)
    v = torch.tensor([1.0, 0, 0, 0])
    # Below threshold → no write.
    wrote = mem.consolidate_if_salient(v, salience_score=0.3, context="low")
    assert wrote is False
    assert len(mem) == 0
    # Above threshold → write.
    wrote = mem.consolidate_if_salient(v, salience_score=0.9, context="high")
    assert wrote is True
    assert len(mem) == 1


def test_memory_consolidate_if_salient_dedup():
    mem = PersistentMemory(d_model=4, max_memories=10)
    v = torch.tensor([1.0, 0, 0, 0])
    mem.consolidate_if_salient(v, salience_score=0.9, context="first")
    # Same vector again (cosine ~1.0, within min_distance) → no write.
    wrote = mem.consolidate_if_salient(v, salience_score=0.9, context="dup",
                                       min_distance=0.1)
    assert wrote is False
    assert len(mem) == 1
    # Different vector (cosine < 1 - min_distance) → write.
    v2 = torch.tensor([0.0, 1.0, 0, 0])
    wrote = mem.consolidate_if_salient(v2, salience_score=0.9, context="new",
                                       min_distance=0.1)
    assert wrote is True
    assert len(mem) == 2
```

- [ ] **Step 2: Run to verify they fail**

Run: `"$PY" -m pytest tests/test_memory.py -v`
Expected: FAIL — `consolidate_if_salient` doesn't exist; `inject` default blend is 0.20 not 0.05 (the blend test may pass or fail depending on current default; `consolidate_if_salient` definitely fails).

- [ ] **Step 3: Modify `inject` default blend and add `consolidate_if_salient`**

In `fractus/memory.py`, change `inject`'s signature default from `blend=0.20` (or whatever it currently is) to `blend=0.05`:

Find the `def inject(self, engine, top_k: int = 3):` line and change it to:
```python
    def inject(self, engine, top_k: int = 3, blend: float = 0.05):
```
And update the blend line inside from `0.8 * ... + 0.2 * ...` to use `blend`:
```python
                engine.thought_state[:, 0, :] = (
                    (1.0 - blend) * engine.thought_state[:, 0, :] +
                    blend * memory_contribution.to(engine.thought_state.device)
                )
```

Then add `consolidate_if_salient` after the existing `consolidate` method:
```python
    def consolidate_if_salient(
        self,
        thought_state: torch.Tensor,
        salience_score: float,
        *,
        context: str = "",
        importance: float = 0.5,
        threshold: float = 0.7,
        min_distance: float = 0.1,
    ) -> bool:
        """Consolidate only if salient enough AND not a near-duplicate.

        Returns True if a memory was written, False otherwise.
        """
        if salience_score < threshold:
            return False
        vec = thought_state.flatten().detach().cpu()
        if vec.shape[0] != self.d_model:
            return False
        # De-duplication: skip if too close to an existing memory.
        if self.vectors:
            bank = torch.stack(self.vectors)
            sim = torch.nn.functional.cosine_similarity(
                vec.unsqueeze(0), bank, dim=-1)
            if sim.max().item() > 1.0 - min_distance:
                return False
        self.consolidate(vec, context=context, importance=importance)
        return True
```

- [ ] **Step 4: Run to verify they pass**

Run: `"$PY" -m pytest tests/test_memory.py -v`
Expected: PASS (6 tests).

- [ ] **Step 5: Commit**

```bash
git add fractus/memory.py tests/test_memory.py
git commit -m "feat(memory): softer inject (5%) + consolidate_if_salient with de-dup"
```

---

## Task 2: CTE salience head + memory slot + attach/detach

**Files:**
- Modify: `fractus/continuous_engine.py`
- Test: `tests/test_memory.py`

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_memory.py`:
```python
def test_cte_has_salience_head_and_memory_slot():
    from fractus.continuous_engine import ContinuousThoughtEngine
    eng = ContinuousThoughtEngine(
        vocab_size=50257, d_model=128, n_heads=2, d_head=64, n_levels=2,
        n_oscillators=8, coupling_rank=4, n_experts=4, top_k=2,
        expert_d_ff=128, siren_rank=32,
    )
    assert hasattr(eng, "salience_head")
    assert hasattr(eng, "memory")
    assert eng.memory is None
    assert eng.memory_active is True


def test_cte_attach_detach_memory():
    from fractus.continuous_engine import ContinuousThoughtEngine
    eng = ContinuousThoughtEngine(
        vocab_size=50257, d_model=128, n_heads=2, d_head=64, n_levels=2,
        n_oscillators=8, coupling_rank=4, n_experts=4, top_k=2,
        expert_d_ff=128, siren_rank=32,
    )
    mem = PersistentMemory(d_model=128, max_memories=10)
    eng.attach_memory(mem)
    assert eng.memory is mem
    eng.detach_memory()
    assert eng.memory is None
```

- [ ] **Step 2: Run to verify they fail**

Run: `"$PY" -m pytest tests/test_memory.py -k "cte_has_salience or cte_attach_detach" -v`
Expected: FAIL — no `salience_head` / `memory` / `attach_memory`.

- [ ] **Step 3: Add salience head + memory slot to the CTE constructor**

In `fractus/continuous_engine.py`, in `__init__`, after the `confidence_head` and `output_head` lines (and before the `register_buffer("thought_state", ...)` block), add:

```python
        # 6. Salience head: "is this thought worth remembering?" (for PersistentMemory)
        self.salience_head = nn.Linear(d_model, 1)

        # 7. Optional persistent memory (attached via attach_memory()).
        self.memory = None
        self.memory_active = True
```

- [ ] **Step 4: Add `attach_memory` / `detach_memory` methods**

After `reset_thought`, add:

```python
    def attach_memory(self, memory):
        """Attach a PersistentMemory bank. Memory injection activates on tick."""
        self.memory = memory

    def detach_memory(self):
        """Detach the memory bank."""
        self.memory = None
```

- [ ] **Step 5: Run to verify they pass**

Run: `"$PY" -m pytest tests/test_memory.py -k "cte_has_salience or cte_attach_detach" -v`
Expected: PASS (2 tests).

- [ ] **Step 6: Commit**

```bash
git add fractus/continuous_engine.py tests/test_memory.py
git commit -m "feat(cte): salience head + memory slot + attach/detach"
```

---

## Task 3: CTE memory block in `tick`

**Files:**
- Modify: `fractus/continuous_engine.py` (the `tick` method)
- Test: `tests/test_memory.py`

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_memory.py`:
```python
def test_cte_salience_head_outputs_in_range():
    from fractus.continuous_engine import ContinuousThoughtEngine
    eng = ContinuousThoughtEngine(
        vocab_size=50257, d_model=128, n_heads=2, d_head=64, n_levels=2,
        n_oscillators=8, coupling_rank=4, n_experts=4, top_k=2,
        expert_d_ff=128, siren_rank=32,
    )
    eng.reset_thought(batch_size=1)
    obs = torch.tensor([42], dtype=torch.long)
    eng.tick(obs)
    # salience_head output is captured during tick; we check via a direct call.
    with torch.no_grad():
        score = torch.sigmoid(eng.salience_head(eng.thought_state[:, 0, :]))
    assert 0.0 <= score.item() <= 1.0


def test_cte_consolidates_when_salient():
    from fractus.continuous_engine import ContinuousThoughtEngine
    eng = ContinuousThoughtEngine(
        vocab_size=50257, d_model=128, n_heads=2, d_head=64, n_levels=2,
        n_oscillators=8, coupling_rank=4, n_experts=4, top_k=2,
        expert_d_ff=128, siren_rank=32,
    )
    # Force salience_head to output ~1.0 (set bias high).
    with torch.no_grad():
        eng.salience_head.bias.fill_(10.0)
    mem = PersistentMemory(d_model=128, max_memories=10)
    eng.attach_memory(mem)
    eng.reset_thought(batch_size=1)
    obs = torch.tensor([42], dtype=torch.long)
    eng.tick(obs)
    assert len(mem) > 0, "should have consolidated a salient thought"


def test_cte_skips_consolidate_when_not_salient():
    from fractus.continuous_engine import ContinuousThoughtEngine
    eng = ContinuousThoughtEngine(
        vocab_size=50257, d_model=128, n_heads=2, d_head=64, n_levels=2,
        n_oscillators=8, coupling_rank=4, n_experts=4, top_k=2,
        expert_d_ff=128, siren_rank=32,
    )
    # Force salience_head to output ~0.0 (set bias very negative).
    with torch.no_grad():
        eng.salience_head.bias.fill_(-10.0)
    mem = PersistentMemory(d_model=128, max_memories=10)
    eng.attach_memory(mem)
    eng.reset_thought(batch_size=1)
    obs = torch.tensor([42], dtype=torch.long)
    eng.tick(obs)
    assert len(mem) == 0, "should NOT have consolidated a non-salient thought"


def test_cte_memory_active_toggle():
    from fractus.continuous_engine import ContinuousThoughtEngine
    eng = ContinuousThoughtEngine(
        vocab_size=50257, d_model=128, n_heads=2, d_head=64, n_levels=2,
        n_oscillators=8, coupling_rank=4, n_experts=4, top_k=2,
        expert_d_ff=128, siren_rank=32,
    )
    with torch.no_grad():
        eng.salience_head.bias.fill_(10.0)  # salient
    mem = PersistentMemory(d_model=128, max_memories=10)
    eng.attach_memory(mem)
    eng.memory_active = False  # disabled
    eng.reset_thought(batch_size=1)
    obs = torch.tensor([42], dtype=torch.long)
    eng.tick(obs)
    assert len(mem) == 0, "memory_active=False should prevent consolidation"
```

- [ ] **Step 2: Run to verify they fail**

Run: `"$PY" -m pytest tests/test_memory.py -k "cte_salience_head_outputs or cte_consolidates or cte_skips or cte_memory_active" -v`
Expected: FAIL — the salient test will fail because `tick` does not yet call consolidation; the toggle test will pass trivially (nothing consolidates yet).

- [ ] **Step 3: Add the memory block to `tick`**

In `fractus/continuous_engine.py`, in the `tick` method, find the section after the MoE and before the confidence/output heads (the block that updates `self.thought_state = h.detach()`). Insert the memory block **after** `self.thought_state` is updated but **before** the confidence/output computation:

```python
        # 3b. Memory: salience-gated consolidation + continuous injection.
        if self.memory is not None and self.memory_active:
            salience = torch.sigmoid(self.salience_head(h[:, 0, :]))  # (B, 1)
            self.memory.consolidate_if_salient(
                h[0:1, 0, :], salience[0].item())
            self.memory.inject(self, blend=0.05, top_k=3)
```

Place this right after the line `self.thought_state = h.detach()` (or wherever `thought_state` is finalized) and before the confidence/output computation. Read the actual `tick` method to find the exact insertion point — it's after the MoE residual `h = h + moe_out` and the thought-state update.

- [ ] **Step 4: Run to verify they pass**

Run: `"$PY" -m pytest tests/test_memory.py -v`
Expected: PASS (all ~10 tests).

- [ ] **Step 5: Run existing CTE tests to check for regressions**

Run: `"$PY" -m pytest tests/test_continuous_engine_grad.py tests/test_ablib.py -k "not arm_" -v`
Expected: PASS (the salience head adds 129 params but does not break gradient flow or determinism).

- [ ] **Step 6: Commit**

```bash
git add fractus/continuous_engine.py tests/test_memory.py
git commit -m "feat(cte): memory block in tick — salience-gated consolidation + 5% injection"
```

---

## Task 4: Cross-session persistence demo + final review

**Files:**
- Read-only validation.

- [ ] **Step 1: Run the full memory test suite**

Run: `"$PY" -m pytest tests/test_memory.py tests/test_continuous_engine_grad.py -v`
Expected: all pass.

- [ ] **Step 2: Cross-session persistence smoke test**

Run:
```bash
"$PY" -c "
import sys; sys.path.insert(0,'.')
import torch, tempfile, os
from fractus.continuous_engine import ContinuousThoughtEngine
from fractus.memory import PersistentMemory

with tempfile.TemporaryDirectory() as d:
    path = os.path.join(d, 'mem.pt')
    # Session 1: build engine, attach memory, force salience high, tick.
    eng1 = ContinuousThoughtEngine(vocab_size=50257, d_model=128, n_heads=2, d_head=64, n_levels=2, n_oscillators=8, coupling_rank=4, n_experts=4, top_k=2, expert_d_ff=128, siren_rank=32)
    with torch.no_grad(): eng1.salience_head.bias.fill_(10.0)
    mem1 = PersistentMemory(d_model=128, max_memories=10, path=path)
    eng1.attach_memory(mem1)
    eng1.reset_thought(batch_size=1)
    for t in [1,2,3,4,5]: eng1.tick(torch.tensor([t]))
    mem1.save()
    print(f'Session 1: {len(mem1)} memories saved')

    # Session 2: fresh engine, load the same memory.
    eng2 = ContinuousThoughtEngine(vocab_size=50257, d_model=128, n_heads=2, d_head=64, n_levels=2, n_oscillators=8, coupling_rank=4, n_experts=4, top_k=2, expert_d_ff=128, siren_rank=32)
    mem2 = PersistentMemory(d_model=128, max_memories=10, path=path)
    eng2.attach_memory(mem2)
    print(f'Session 2: loaded {len(mem2)} memories from disk')
    eng2.reset_thought(batch_size=1)
    thought_before = eng2.thought_state.clone()
    eng2.tick(torch.tensor([1]))
    thought_after = eng2.thought_state
    # The thought state should have shifted (memory injected).
    moved = (thought_after - thought_before).abs().sum().item()
    print(f'Thought state moved by {moved:.4f} after 1 tick with memory')
    assert len(mem2) > 0, 'cross-session memory should survive'
    print('CROSS-SESSION PERSISTENCE: OK')
"
```
Expected: prints `Session 1: N memories saved`, `Session 2: loaded N memories from disk`, `CROSS-SESSION PERSISTENCE: OK`. No traceback.

- [ ] **Step 3: Commit (if any cleanup was needed; otherwise skip)**

If the smoke test passed on the first try, no commit is needed — the implementation is already committed in Tasks 1-3. If a fix was required, commit it:
```bash
git add -A && git commit -m "fix(memory): cross-session persistence fix"
```

---

## Self-review notes (done by the planner)

- **Spec coverage:** §2.1 (inject blend 0.05 + consolidate_if_salient) → Task 1. §2.2 (salience_head, memory slot, attach/detach, tick memory block) → Tasks 2-3. §2.3 (tests) → Tasks 1-3 (10 tests). §3 (data flow) → Task 3 Step 3. §5 success criteria: tests pass (Task 4), existing tests unbroken (Task 3 Step 5), demo works (Task 4 Step 2).
- **Placeholder scan:** no TBD/TODO; every code step shows full code. Task 3 Step 3 says "find the exact insertion point" — this is correct guidance (the planner cannot know the exact line after Tasks 1-2 shift line numbers, but the description "after thought_state is finalized, before confidence/output" is unambiguous).
- **Type consistency:** `consolidate_if_salient(thought_state, salience_score, *, context, importance, threshold, min_distance) -> bool` consistent across Task 1 (definition) and Task 3 (call). `attach_memory(memory)` / `detach_memory()` consistent across Task 2 (definition) and tests. `inject(self, engine, top_k=3, blend=0.05)` consistent across Task 1 and Task 3. `salience_head` is `nn.Linear(d_model, 1)` everywhere.
```
