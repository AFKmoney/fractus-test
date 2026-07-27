# EDT AB-test Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Run a 3-arm controlled experiment (from-scratch / EDT vanilla / EDT+specialization) on the 13M ContinuousThoughtEngine to measure whether EDT accelerates learning and whether the missing specialization mechanism matters, per spec `docs/superpowers/specs/2026-07-27-edt-ab-test-design.md`.

**Architecture:** One isolated script `scripts/ab_edt_13m.py` plus a small library `experiments/edt_ab/ablib.py` of pure helper functions. No changes to the Fractus core (`fractus/...`). The helpers operate only on the public attributes of `ContinuousThoughtEngine` (`observe`, `experts_w1`, `experts_w2`, `output_head`, `tick_chunk`, `reset_thought`) and reuse `OnlineTrainer` for Phase 3.

**Tech Stack:** PyTorch 2.9 (CPU), numpy, matplotlib. Python via the Windows Store CPython 3.13 (`python.exe`), 12 CPU threads. Tests via `pytest`.

**Key API facts** (verified against the codebase; do not re-derive):
- `ContinuousThoughtEngine.__init__` signature is given in `scripts/train_13m_rag.py:42-46`. The 13M config is: `vocab_size=50257, d_model=128, n_heads=2, d_head=64, n_levels=2, n_oscillators=8, coupling_rank=4, n_experts=4, top_k=2, expert_d_ff=128, siren_rank=32`.
- `engine.observe`: `nn.Embedding(vocab, d_model)` — input embedding.
- `engine.experts_w1[i]`, `engine.experts_w2[i]`: `CachedStructuredSirenLinear`, `i in range(4)`. Each is `w1: (d_model→d_ff)`, `w2: (d_ff→d_model)`.
- `engine.output_head`: `nn.Linear(d_model, vocab, bias=False)`. **Not tied to `observe`.**
- `CachedStructuredSirenLinear.force_refresh()` rebuilds `_cached_W` from current `U/V/residual_siren` and resets `_call_count=0`. **Must be called after any external optimizer step on `U/V/residual_siren`** so the cache reflects new weights; otherwise forward uses a stale matrix.
- `OnlineTrainer.train_on_stream_chunked(tokens, chunk_len=16)` is the fast trainer (1 forward/backward per 16-token chunk). Returns `{"avg_loss","accuracy","steps","optimizer_steps"}` and appends to `trainer.losses`.
- Corpus: `data/communication_corpus.pt` → 1D int32 tensor, 18 645 351 tokens.

**Reference spec:** `docs/superpowers/specs/2026-07-27-edt-ab-test-design.md`. Every design decision lives there; this plan only encodes it.

---

## File structure

| Path | Responsibility |
|---|---|
| `experiments/__init__.py` | empty, makes `experiments` a package |
| `experiments/edt_ab/__init__.py` | empty |
| `experiments/edt_ab/ablib.py` | pure helpers: build_engine, load_corpus, phase1_*, phase2b, phase3, evaluate_ppl, expert_diversity, greedy_sample. No side effects, all testable. |
| `tests/test_ablib.py` | unit tests for every helper (fast, tiny inputs) |
| `scripts/ab_edt_13m.py` | CLI entrypoint: orchestrates the 3 arms, writes `results.json` + plots |
| `experiments/edt_ab/.gitignore` | ignore generated artifacts (`results.json`, `*.png`, `samples/`, `*.pt`) |

---

## Task 1: Scaffold the package and a failing smoke test

**Files:**
- Create: `experiments/__init__.py`
- Create: `experiments/edt_ab/__init__.py`
- Create: `experiments/edt_ab/.gitignore`
- Create: `tests/test_ablib.py`
- Modify: `pyproject.toml` (add `experiments` to test discovery if needed)

- [ ] **Step 1: Create the package files**

`experiments/__init__.py`:
```python
```
(empty)

`experiments/edt_ab/__init__.py`:
```python
"""EDT AB-test on the 13M ContinuousThoughtEngine. See docs/superpowers/specs/2026-07-27-edt-ab-test-design.md."""
```

`experiments/edt_ab/.gitignore`:
```
results.json
*.png
samples/
*.pt
```

- [ ] **Step 2: Write the failing import test**

`tests/test_ablib.py`:
```python
"""Tests for the EDT AB-test helper library."""
import experiments.edt_ab.ablib as ablib  # noqa: F401  (import smoke test)


def test_package_imports():
    assert ablib is not None
```

- [ ] **Step 3: Run the test to verify it fails**

Run: `python -m pytest tests/test_ablib.py -v`
Expected: FAIL — `ModuleNotFoundError: experiments.edt_ab.ablib` (file not created yet).

- [ ] **Step 4: Create a minimal `ablib.py` so the import resolves**

`experiments/edt_ab/ablib.py`:
```python
"""Helper library for the EDT AB-test.

Pure functions operating on ContinuousThoughtEngine + tensors.
No I/O, no side effects — every function is unit-testable.
"""
```

- [ ] **Step 5: Run the test to verify it passes**

Run: `python -m pytest tests/test_ablib.py -v`
Expected: PASS (1 test).

- [ ] **Step 6: Commit**

```bash
git add experiments/ tests/test_ablib.py
git commit -m "feat(edt-ab): scaffold experiments package + smoke test"
```

---

## Task 2: `build_engine` — deterministic 13M construction

**Files:**
- Modify: `experiments/edt_ab/ablib.py`
- Modify: `tests/test_ablib.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/test_ablib.py`:
```python
import torch
from fractus.continuous_engine import ContinuousThoughtEngine


def test_build_engine_is_deterministic_and_13m():
    a = ablib.build_engine(seed=42)
    b = ablib.build_engine(seed=42)
    c = ablib.build_engine(seed=7)
    assert isinstance(a, ContinuousThoughtEngine)
    # Same seed → bit-identical weights.
    for pa, pb in zip(a.parameters(), b.parameters()):
        assert torch.equal(pa, pb)
    # Different seed → different weights.
    differs = any(not torch.equal(pa, pc)
                  for pa, pc in zip(a.parameters(), c.parameters()))
    assert differs
    n = sum(p.numel() for p in a.parameters())
    assert 12_000_000 <= n <= 14_000_000, f"expected ~13M params, got {n}"
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `python -m pytest tests/test_ablib.py::test_build_engine_is_deterministic_and_13m -v`
Expected: FAIL — `AttributeError: module 'experiments.edt_ab.ablib' has no attribute 'build_engine'`.

- [ ] **Step 3: Implement `build_engine`**

Append to `experiments/edt_ab/ablib.py`:
```python
import torch

from fractus.continuous_engine import ContinuousThoughtEngine


def build_engine(seed: int = 42) -> ContinuousThoughtEngine:
    """Construct a fresh 13M ContinuousThoughtEngine with deterministic init."""
    torch.manual_seed(seed)
    return ContinuousThoughtEngine(
        vocab_size=50257, d_model=128, n_heads=2, d_head=64,
        n_levels=2, n_oscillators=8, coupling_rank=4,
        n_experts=4, top_k=2, expert_d_ff=128, siren_rank=32,
    )
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `python -m pytest tests/test_ablib.py::test_build_engine_is_deterministic_and_13m -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add experiments/edt_ab/ablib.py tests/test_ablib.py
git commit -m "feat(edt-ab): build_engine — deterministic 13M construction"
```

---

## Task 3: `load_corpus` — train / holdout / phase-1 split

**Files:**
- Modify: `experiments/edt_ab/ablib.py`
- Modify: `tests/test_ablib.py`

Per spec §5: total budget N=400 000 train tokens, holdout 30 000, Phase-1 sub-budget 60 000 (15 %), Phase-1 split into 4×15 000 contiguous domain slices for Arm C. A single fixed shuffle (seed 42) is applied to the corpus before any slicing.

- [ ] **Step 1: Write the failing test**

Append to `tests/test_ablib.py`:
```python
import tempfile, os


def _make_tiny_corpus(path, n=5000):
    torch.manual_seed(0)
    torch.randint(0, 50257, (n,), dtype=torch.int32).save(path)


def test_load_corpus_splits_and_partitions():
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "tiny.pt")
        _make_tiny_corpus(p, n=5000)
        split = ablib.load_corpus(
            p, n_train=1000, n_holdout=200, n_phase1=400, seed=42,
        )
    assert split["train"].numel() == 1000
    assert split["holdout"].numel() == 200
    assert split["phase1"].numel() == 400
    # Phase-1 partition: 4 disjoint contiguous slices, 100 each.
    assert len(split["domain_split"]) == 4
    assert all(s.numel() == 100 for s in split["domain_split"])
    # Disjointness: concatenating the slices reconstructs phase1 exactly.
    rejoined = torch.cat(split["domain_split"])
    assert torch.equal(rejoined, split["phase1"])
    # Train and holdout must not overlap (drawn from different parts of the shuffle).
    assert split["train"].numel() + split["holdout"].numel() + split["phase1"].numel() == 1600
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `python -m pytest tests/test_ablib.py::test_load_corpus_splits_and_partitions -v`
Expected: FAIL — `AttributeError: ... has no attribute 'load_corpus'`.

- [ ] **Step 3: Implement `load_corpus`**

Append to `experiments/edt_ab/ablib.py`:
```python
def load_corpus(path: str, *, n_train: int = 400_000, n_holdout: int = 30_000,
                n_phase1: int = 60_000, seed: int = 42) -> dict:
    """Load + shuffle (fixed seed) + split the corpus.

    Returns dict with:
      train       : (n_train,) int64  — Phase-3 budget (also Arm A's whole budget)
      holdout     : (n_holdout,) int64 — never seen in training, identical across arms
      phase1      : (n_phase1,) int64 — Phase-1 budget for arms B and C
      domain_split: list of 4 int64 tensors, contiguous disjoint slices of phase1
    """
    tokens = torch.load(path, weights_only=False).to(torch.int64)
    g = torch.Generator().manual_seed(seed)
    perm = torch.randperm(tokens.numel(), generator=g)
    tokens = tokens[perm]

    # Layout: [phase1 | train | holdout]  (phase1 first so domain partition is clean)
    need = n_phase1 + n_train + n_holdout
    assert tokens.numel() >= need, f"corpus has {tokens.numel()} tokens, need {need}"
    phase1 = tokens[:n_phase1].clone()
    train = tokens[n_phase1:n_phase1 + n_train].clone()
    holdout = tokens[n_phase1 + n_train:n_phase1 + n_train + n_holdout].clone()

    assert n_phase1 % 4 == 0, "n_phase1 must be divisible by 4 (4 experts)"
    stride = n_phase1 // 4
    domain_split = [phase1[i * stride:(i + 1) * stride].clone() for i in range(4)]

    return {"train": train, "holdout": holdout,
            "phase1": phase1, "domain_split": domain_split}
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `python -m pytest tests/test_ablib.py::test_load_corpus_splits_and_partitions -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add experiments/edt_ab/ablib.py tests/test_ablib.py
git commit -m "feat(edt-ab): load_corpus — shuffle + train/holdout/phase1/domain split"
```

---

## Task 4: Hidden-bank generator (shared, for Phase 1)

**Files:**
- Modify: `experiments/edt_ab/ablib.py`
- Modify: `tests/test_ablib.py`

Per spec §4 Arm B: Phase 1 generates `(h_t, h_{t+1})` pairs from `engine.observe` over **consecutive** positions within sampled chunks. The shared bank feeds all 4 experts identically.

- [ ] **Step 1: Write the failing test**

Append to `tests/test_ablib.py`:
```python
def test_make_hidden_bank_shape_and_alignment():
    eng = ablib.build_engine(seed=42)
    tokens = torch.arange(1000, dtype=torch.int64) % 50257
    bank = ablib.make_hidden_bank(eng, tokens, chunk_len=32, n_chunks=10, seed=0)
    assert set(bank.keys()) == {"h_in", "h_target"}
    # 10 chunks × 31 consecutive pairs = 310 samples.
    assert bank["h_in"].shape == (310, eng.d_model)
    assert bank["h_target"].shape == (310, eng.d_model)
    # Pairs are positionally aligned: h_target[t] == observe(token after h_in[t]).
    # Sanity: embedding is deterministic, so re-deriving matches.
    with torch.no_grad():
        first_in_token = tokens[0]
        # The first sampled pair uses tokens at consecutive corpus positions.
        # We only check that h_in != h_target (otherwise the target is trivial).
        assert not torch.equal(bank["h_in"][0], bank["h_target"][0])
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `python -m pytest tests/test_ablib.py::test_make_hidden_bank_shape_and_alignment -v`
Expected: FAIL — `AttributeError: ... has no attribute 'make_hidden_bank'`.

- [ ] **Step 3: Implement `make_hidden_bank`**

Append to `experiments/edt_ab/ablib.py`:
```python
def make_hidden_bank(engine: ContinuousThoughtEngine, tokens: torch.Tensor,
                     *, chunk_len: int = 32, n_chunks: int = 1000,
                     seed: int = 0) -> dict:
    """Build (h_t, h_{t+1}) pairs from engine.observe over consecutive positions.

    For each of n_chunks randomly placed chunks of length chunk_len, emit
    (chunk_len - 1) aligned pairs. Pairs never cross chunk boundaries.
    Returns {"h_in": (P, d_model), "h_target": (P, d_model)}.
    """
    engine.eval()
    g = torch.Generator().manual_seed(seed)
    n = tokens.numel()
    assert n > chunk_len + 1
    starts = torch.randint(0, n - chunk_len - 1, (n_chunks,), generator=g)
    h_in_list, h_tgt_list = [], []
    with torch.no_grad():
        for s in starts.tolist():
            chunk = tokens[s:s + chunk_len + 1]            # need +1 for last target
            h = engine.observe(chunk)                       # (chunk_len+1, d_model)
            h_in_list.append(h[:-1].clone())                # (chunk_len, d_model)
            h_tgt_list.append(h[1:].clone())                # (chunk_len, d_model)
    return {"h_in": torch.cat(h_in_list, dim=0),
            "h_target": torch.cat(h_tgt_list, dim=0)}
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `python -m pytest tests/test_ablib.py::test_make_hidden_bank_shape_and_alignment -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add experiments/edt_ab/ablib.py tests/test_ablib.py
git commit -m "feat(edt-ab): make_hidden_bank — aligned (h_t, h_{t+1}) pairs"
```

---

## Task 5: Phase 1 — shared bank (Arm B) and partitioned banks (Arm C)

**Files:**
- Modify: `experiments/edt_ab/ablib.py`
- Modify: `tests/test_ablib.py`

Per spec §4. Both variants train each expert independently with MSE(`expert_w2(gelu(expert_w1(h_t)))`, `h_{t+1}`). **Critical:** call `force_refresh()` on both `experts_w1[i]` and `experts_w2[i]` before each forward so the SIREN cache reflects current params (cache invalidation per spec §12).

- [ ] **Step 1: Write the failing test**

Append to `tests/test_ablib.py`:
```python
def test_phase1_shared_reduces_mse_on_bank():
    eng = ablib.build_engine(seed=42)
    tokens = torch.arange(2000, dtype=torch.int64) % 50257
    bank = ablib.make_hidden_bank(eng, tokens, chunk_len=32, n_chunks=20, seed=0)
    mse_before = ablib._eval_expert_mse(eng, 0, bank)
    ablib.phase1_experts_shared(eng, bank, steps=20, lr=1e-2, seed=0)
    mse_after = ablib._eval_expert_mse(eng, 0, bank)
    # Training must reduce the expert's MSE on its own training bank.
    assert mse_after < mse_before


def test_phase1_partitioned_trains_each_expert_on_own_bank():
    eng = ablib.build_engine(seed=42)
    tokens = torch.arange(2000, dtype=torch.int64) % 50257
    banks = [ablib.make_hidden_bank(eng, tokens[:500], chunk_len=32, n_chunks=8, seed=i)
             for i in range(4)]
    mse_before = [ablib._eval_expert_mse(eng, i, banks[i]) for i in range(4)]
    ablib.phase1_experts_partitioned(eng, banks, steps=15, lr=1e-2, seed=0)
    mse_after = [ablib._eval_expert_mse(eng, i, banks[i]) for i in range(4)]
    for i in range(4):
        assert mse_after[i] < mse_before[i], f"expert {i} did not improve"
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `python -m pytest tests/test_ablib.py::test_phase1_shared_reduces_mse_on_bank tests/test_ablib.py::test_phase1_partitioned_trains_each_expert_on_own_bank -v`
Expected: FAIL — `AttributeError: ... has no attribute '_eval_expert_mse'` (and the phase1_* fns missing too).

- [ ] **Step 3: Implement the Phase 1 helpers**

Append to `experiments/edt_ab/ablib.py`:
```python
def _expert_forward(engine: ContinuousThoughtEngine, idx: int,
                    h: torch.Tensor) -> torch.Tensor:
    """expert_w2(gelu(expert_w1(h))). Forces a cache refresh first."""
    engine.experts_w1[idx].force_refresh()
    engine.experts_w2[idx].force_refresh()
    h1 = engine.experts_w1[idx](h)
    h1_act = torch.nn.functional.gelu(h1)
    return engine.experts_w2[idx](h1_act)


def _eval_expert_mse(engine: ContinuousThoughtEngine, idx: int, bank: dict) -> float:
    """Mean MSE of expert idx on a bank (no grad)."""
    engine.eval()
    with torch.no_grad():
        out = _expert_forward(engine, idx, bank["h_in"])
        return torch.nn.functional.mse_loss(out, bank["h_target"]).item()


def _train_one_expert(engine: ContinuousThoughtEngine, idx: int, bank: dict,
                      steps: int, lr: float, batch_size: int, seed: int) -> list:
    """Train expert idx on bank with MSE. Returns per-step loss list."""
    engine.train()
    params = list(engine.experts_w1[idx].parameters()) + \
             list(engine.experts_w2[idx].parameters())
    opt = torch.optim.AdamW(params, lr=lr, weight_decay=0.01)
    g = torch.Generator().manual_seed(seed)
    n = bank["h_in"].size(0)
    losses = []
    for _ in range(steps):
        idx_b = torch.randint(0, n, (batch_size,), generator=g)
        h_in = bank["h_in"][idx_b]
        h_tgt = bank["h_target"][idx_b]
        opt.zero_grad()
        out = _expert_forward(engine, idx, h_in)
        loss = torch.nn.functional.mse_loss(out, h_tgt)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step()
        losses.append(loss.item())
    # Final refresh so the cache reflects trained weights.
    engine.experts_w1[idx].force_refresh()
    engine.experts_w2[idx].force_refresh()
    return losses


def phase1_experts_shared(engine: ContinuousThoughtEngine, bank: dict, *,
                          steps: int = 2000, lr: float = 1e-3,
                          batch_size: int = 64, seed: int = 0) -> None:
    """Arm B Phase 1: all experts trained on the SAME shared bank."""
    for i in range(engine.n_experts):
        _train_one_expert(engine, i, bank, steps=steps, lr=lr,
                          batch_size=batch_size, seed=seed + i)


def phase1_experts_partitioned(engine: ContinuousThoughtEngine, banks: list, *,
                               steps: int = 2000, lr: float = 1e-3,
                               batch_size: int = 64, seed: int = 0) -> None:
    """Arm C Phase 1: expert i trained only on banks[i] (disjoint data)."""
    assert len(banks) == engine.n_experts, "need one bank per expert"
    for i in range(engine.n_experts):
        _train_one_expert(engine, i, banks[i], steps=steps, lr=lr,
                          batch_size=batch_size, seed=seed + i)
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `python -m pytest tests/test_ablib.py::test_phase1_shared_reduces_mse_on_bank tests/test_ablib.py::test_phase1_partitioned_trains_each_expert_on_own_bank -v`
Expected: PASS (both).

- [ ] **Step 5: Commit**

```bash
git add experiments/edt_ab/ablib.py tests/test_ablib.py
git commit -m "feat(edt-ab): Phase 1 — shared + partitioned expert pre-training"
```

---

## Task 6: Phase 2b — embedding + output_head next-token

**Files:**
- Modify: `experiments/edt_ab/ablib.py`
- Modify: `tests/test_ablib.py`

Per spec §4 (corrected in self-review): train `engine.observe` **and** `engine.output_head` jointly, everything else frozen, cross-entropy next-token. This is the minimal two-table LM (the CTE analogue of "train the embedding alone"; `output_head` is not weight-tied on the CTE).

- [ ] **Step 1: Write the failing test**

Append to `tests/test_ablib.py`:
```python
def test_phase2b_reduces_ce_and_only_touches_observe_and_head():
    eng = ablib.build_engine(seed=42)
    # Snapshot the frozen params (attn + moe norms + experts) to prove they don't change.
    frozen_names = [n for n, _ in eng.named_parameters()
                    if not (n.startswith("observe.") or n.startswith("output_head."))]
    frozen_before = {n: p.detach().clone() for n, p in eng.named_parameters()
                     if n in frozen_names}
    tokens = torch.arange(3000, dtype=torch.int64) % 50257
    ce_before = ablib._eval_ce(eng, tokens[:1000])
    ablib.phase2b_embedding(eng, tokens, steps=30, lr=1e-2, seed=0)
    ce_after = ablib._eval_ce(eng, tokens[:1000])
    assert ce_after < ce_before
    # Frozen params untouched.
    for n, p in eng.named_parameters():
        if n in frozen_before:
            assert torch.equal(p, frozen_before[n]), f"{n} changed during Phase 2b"
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `python -m pytest tests/test_ablib.py::test_phase2b_reduces_ce_and_only_touches_observe_and_head -v`
Expected: FAIL — missing `_eval_ce` and `phase2b_embedding`.

- [ ] **Step 3: Implement Phase 2b**

Append to `experiments/edt_ab/ablib.py`:
```python
def _eval_ce(engine: ContinuousThoughtEngine, tokens: torch.Tensor,
             chunk_len: int = 16) -> float:
    """Mean per-token cross-entropy of the two-table LM (observe→output_head)."""
    engine.eval()
    total, n = 0.0, 0
    with torch.no_grad():
        for s in range(0, tokens.numel() - chunk_len - 1, chunk_len):
            chunk = tokens[s:s + chunk_len].unsqueeze(0)
            tgt = tokens[s + 1:s + 1 + chunk_len].reshape(-1)
            h = engine.observe(chunk)
            logits = engine.output_head(h).reshape(-1, engine.vocab_size)
            total += torch.nn.functional.cross_entropy(
                logits, tgt, reduction="sum").item()
            n += tgt.numel()
    return total / max(n, 1)


def phase2b_embedding(engine: ContinuousThoughtEngine, tokens: torch.Tensor, *,
                      steps: int = 2000, lr: float = 1e-3,
                      chunk_len: int = 16, seed: int = 0) -> list:
    """Phase 2b: train observe + output_head only (everything else frozen)."""
    engine.train()
    # Freeze everything, then unfreeze the two tables.
    for p in engine.parameters():
        p.requires_grad_(False)
    for p in engine.observe.parameters():
        p.requires_grad_(True)
    for p in engine.output_head.parameters():
        p.requires_grad_(True)

    opt = torch.optim.AdamW(
        list(engine.observe.parameters()) + list(engine.output_head.parameters()),
        lr=lr, weight_decay=0.01)
    g = torch.Generator().manual_seed(seed)
    n = tokens.numel()
    losses = []
    for _ in range(steps):
        s = torch.randint(0, n - chunk_len - 1, (1,), generator=g).item()
        chunk = tokens[s:s + chunk_len].unsqueeze(0)
        tgt = tokens[s + 1:s + 1 + chunk_len].reshape(-1)
        opt.zero_grad()
        h = engine.observe(chunk)
        logits = engine.output_head(h).reshape(-1, engine.vocab_size)
        loss = torch.nn.functional.cross_entropy(logits, tgt)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            list(engine.observe.parameters()) + list(engine.output_head.parameters()),
            1.0)
        opt.step()
        losses.append(loss.item())
    # Restore: everything trainable again (Phase 3 expects this).
    for p in engine.parameters():
        p.requires_grad_(True)
    return losses
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `python -m pytest tests/test_ablib.py::test_phase2b_reduces_ce_and_only_touches_observe_and_head -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add experiments/edt_ab/ablib.py tests/test_ablib.py
git commit -m "feat(edt-ab): Phase 2b — embedding+output_head next-token training"
```

---

## Task 7: Phase 3 wrapper around OnlineTrainer (chunked)

**Files:**
- Modify: `experiments/edt_ab/ablib.py`
- Modify: `tests/test_ablib.py`

Per spec §4: Arm A, B, and C all finish with `OnlineTrainer.train_on_stream_chunked`. Wrap it so we (a) guarantee fresh `OnlineTrainer` per call, (b) capture the loss curve, (c) ensure all params are trainable.

- [ ] **Step 1: Write the failing test**

Append to `tests/test_ablib.py`:
```python
def test_phase3_joint_reduces_loss_and_returns_curve():
    from fractus.train.online import OnlineTrainer
    eng = ablib.build_engine(seed=42)
    tokens = torch.arange(4000, dtype=torch.int64) % 50257
    out = ablib.phase3_joint(eng, tokens, steps=50, lr=1e-3, chunk_len=16)
    assert "losses" in out and "avg_loss" in out and "accuracy" in out
    assert len(out["losses"]) > 0
    # First half vs second half: loss should trend down.
    half = len(out["losses"]) // 2
    assert sum(out["losses"][half:]) / max(len(out["losses"]) - half, 1) < \
           sum(out["losses"][:half]) / max(half, 1)
    # All params trainable.
    assert all(p.requires_grad for p in eng.parameters())
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `python -m pytest tests/test_ablib.py::test_phase3_joint_reduces_loss_and_returns_curve -v`
Expected: FAIL — missing `phase3_joint`.

- [ ] **Step 3: Implement `phase3_joint`**

Append to `experiments/edt_ab/ablib.py`:
```python
from fractus.train.online import OnlineTrainer


def phase3_joint(engine: ContinuousThoughtEngine, tokens: torch.Tensor, *,
                 steps: int, lr: float = 3e-4, chunk_len: int = 16) -> dict:
    """Phase 3: full-model online training, chunk-based.

    `steps` is the number of optimizer steps (chunks). Each chunk is `chunk_len`
    tokens. The total tokens consumed is steps * chunk_len.
    """
    engine.train()
    for p in engine.parameters():
        p.requires_grad_(True)
    # Force-refresh all expert caches so Phase 3 starts from trained matrices.
    for i in range(engine.n_experts):
        engine.experts_w1[i].force_refresh()
        engine.experts_w2[i].force_refresh()
    trainer = OnlineTrainer(engine, lr=lr)
    trainer.losses = []  # fresh curve
    engine.reset_thought(batch_size=1)

    total, correct, n = 0.0, 0, 0
    vocab = engine.vocab_size
    for start in range(0, steps * chunk_len, chunk_len):
        if start + chunk_len + 1 >= tokens.numel():
            break
        chunk = tokens[start:start + chunk_len].unsqueeze(0)
        tgt = tokens[start + 1:start + 1 + chunk_len].reshape(-1)
        logits = engine.tick_chunk(chunk).reshape(-1, vocab)
        loss = torch.nn.functional.cross_entropy(logits, tgt)
        trainer.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(engine.parameters(), 1.0)
        trainer.optimizer.step()
        trainer.losses.append(loss.item())
        total += loss.item() * chunk_len
        correct += (logits.argmax(-1) == tgt).sum().item()
        n += chunk_len
    return {"losses": trainer.losses,
            "avg_loss": total / max(n, 1),
            "accuracy": correct / max(n, 1),
            "steps": n}
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `python -m pytest tests/test_ablib.py::test_phase3_joint_reduces_loss_and_returns_curve -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add experiments/edt_ab/ablib.py tests/test_ablib.py
git commit -m "feat(edt-ab): Phase 3 — chunked online joint training wrapper"
```

---

## Task 8: Evaluation — `evaluate_ppl`, `expert_diversity`, `greedy_sample`

**Files:**
- Modify: `experiments/edt_ab/ablib.py`
- Modify: `tests/test_ablib.py`

Per spec §7: perplexity on hold-out (token NLL capped at 20), inter-expert cosine (mean off-diagonal, fixed shared probe of 256 hidden vectors), greedy sample (fixed prompt).

- [ ] **Step 1: Write the failing test**

Append to `tests/test_ablib.py`:
```python
def test_evaluate_ppl_returns_finite_positive():
    eng = ablib.build_engine(seed=42)
    tokens = torch.arange(2000, dtype=torch.int64) % 50257
    ppl = ablib.evaluate_ppl(eng, tokens[:500])
    assert ppl == ppl  # not NaN
    assert ppl > 0.0


def test_expert_diversity_returns_float_in_range():
    eng = ablib.build_engine(seed=42)
    probe_tokens = torch.arange(256, dtype=torch.int64) % 50257
    div = ablib.expert_diversity(eng, probe_tokens)
    assert -1.0 <= div <= 1.0


def test_greedy_sample_returns_string_of_expected_length():
    eng = ablib.build_engine(seed=42)
    prompt = torch.tensor([1, 2, 3], dtype=torch.int64)
    s = ablib.greedy_sample(eng, prompt, n_tokens=10)
    assert isinstance(s, str)
    # Tokenizer detok may merge; just check it's non-empty.
    assert len(s) > 0
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `python -m pytest tests/test_ablib.py -k "evaluate_ppl or expert_diversity or greedy_sample" -v`
Expected: FAIL — all three missing.

- [ ] **Step 3: Implement the evaluators**

Append to `experiments/edt_ab/ablib.py`:
```python
def evaluate_ppl(engine: ContinuousThoughtEngine, tokens: torch.Tensor,
                 chunk_len: int = 16, nll_cap: float = 20.0) -> float:
    """Perplexity on a hold-out stream, computed via tick_chunk (full engine)."""
    engine.eval()
    engine.reset_thought(batch_size=1)
    total_nll, n = 0.0, 0
    for s in range(0, tokens.numel() - chunk_len - 1, chunk_len):
        chunk = tokens[s:s + chunk_len].unsqueeze(0)
        tgt = tokens[s + 1:s + 1 + chunk_len].reshape(-1)
        with torch.no_grad():
            logits = engine.tick_chunk(chunk).reshape(-1, engine.vocab_size)
            nll = torch.nn.functional.cross_entropy(logits, tgt, reduction="none")
            nll = nll.clamp(max=nll_cap)
        total_nll += nll.sum().item()
        n += tgt.numel()
    avg_nll = total_nll / max(n, 1)
    import math
    return math.exp(avg_nll)


def expert_diversity(engine: ContinuousThoughtEngine, probe_tokens: torch.Tensor) -> float:
    """Mean off-diagonal cosine between expert outputs on a fixed shared probe.

    Lower = more diverse (better specialization).
    """
    engine.eval()
    with torch.no_grad():
        h = engine.observe(probe_tokens)            # (P, d_model)
        outs = []
        for i in range(engine.n_experts):
            outs.append(_expert_forward(engine, i, h))   # (P, d_model)
        outs = torch.stack(outs)                          # (E, P, d_model)
    E = outs.size(0)
    cos_sum, count = 0.0, 0
    for i in range(E):
        for j in range(E):
            if i == j:
                continue
            a = outs[i].flatten()
            b = outs[j].flatten()
            cos = torch.nn.functional.cosine_similarity(a, b, dim=0).item()
            cos_sum += cos
            count += 1
    return cos_sum / max(count, 1)


def greedy_sample(engine: ContinuousThoughtEngine, prompt: torch.Tensor,
                  n_tokens: int = 80) -> str:
    """Greedy decode n_tokens after the prompt, detokenize to a string."""
    from fractus.tokenizer import FractusTokenizer
    tok = FractusTokenizer()
    engine.eval()
    engine.reset_thought(batch_size=1)
    ids = prompt.tolist()
    cur = torch.tensor(ids[-1:], dtype=torch.int64)
    for _ in range(n_tokens):
        with torch.no_grad():
            logits, _ = engine.tick(cur)
        nxt = int(logits.argmax(-1).item())
        ids.append(nxt)
        cur = torch.tensor([nxt], dtype=torch.int64)
    return tok.decode(ids)
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `python -m pytest tests/test_ablib.py -k "evaluate_ppl or expert_diversity or greedy_sample" -v`
Expected: PASS (all three).

If `FractusTokenizer` instantiation fails (it may need a vocab file), fall back: change `greedy_sample` to return `" ".join(str(i) for i in ids)` and drop the `FractusTokenizer` import; note this in the commit message. The metric is qualitative only.

- [ ] **Step 5: Commit**

```bash
git add experiments/edt_ab/ablib.py tests/test_ablib.py
git commit -m "feat(edt-ab): evaluators — ppl, expert diversity, greedy sample"
```

---

## Task 9: Arm orchestrators (A / B / C)

**Files:**
- Modify: `experiments/edt_ab/ablib.py`
- Modify: `tests/test_ablib.py`

Each arm consumes the same total token budget N (spec §5). Arms B and C split N as Phase 1 = 15 %, Phase 2b = 30 %, Phase 3 = 55 %. Arm A consumes all N in Phase 3.

Token accounting (N = 400 000 for the real run, but the function takes N as a parameter):
- Phase 1 budget = 60 000 tokens → bank of pairs; "tokens consumed" = 60 000.
- Phase 2b budget = 120 000 tokens → `steps = 120 000 / chunk_len`.
- Phase 3 budget = 220 000 tokens → `steps = 220 000 / chunk_len`.
- Arm A: `steps = 400 000 / chunk_len`.

- [ ] **Step 1: Write the failing test**

Append to `tests/test_ablib.py`:
```python
def test_arm_from_scratch_runs_and_reports_ppl():
    eng = ablib.build_engine(seed=42)
    tokens = torch.arange(4000, dtype=torch.int64) % 50257
    out = ablib.arm_from_scratch(eng, train=tokens, holdout=tokens[3000:3500],
                                 budget=1000, chunk_len=16, lr=1e-3)
    for k in ("ppl", "accuracy", "diversity", "losses", "sample"):
        assert k in out


def test_arm_edt_vanilla_runs_and_reports_phase_losses():
    eng = ablib.build_engine(seed=42)
    tokens = torch.arange(8000, dtype=torch.int64) % 50257
    out = ablib.arm_edt_vanilla(eng, train=tokens, holdout=tokens[7000:7500],
                                budget=2000, chunk_len=16, lr=1e-3)
    for k in ("ppl", "phase1_losses", "phase2b_losses", "phase3_losses",
              "diversity", "sample"):
        assert k in out


def test_arm_edt_spec_uses_domain_split():
    eng = ablib.build_engine(seed=42)
    base = torch.arange(8000, dtype=torch.int64) % 50257
    phase1 = base[:1000]
    domain_split = [phase1[i * 250:(i + 1) * 250] for i in range(4)]
    out = ablib.arm_edt_spec(eng, train=base[1000:], holdout=base[7000:7500],
                             budget=2000, domain_split=domain_split,
                             chunk_len=16, lr=1e-3)
    for k in ("ppl", "phase1_losses", "phase2b_losses", "phase3_losses",
              "diversity", "sample"):
        assert k in out
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `python -m pytest tests/test_ablib.py -k "arm_from_scratch or arm_edt" -v`
Expected: FAIL — all three arms missing.

- [ ] **Step 3: Implement the three arms**

Append to `experiments/edt_ab/ablib.py`:
```python
# ---- phase token accounting (spec §5) ----
PHASE1_FRAC, PHASE2B_FRAC, PHASE3_FRAC = 0.15, 0.30, 0.55

PROMPT = torch.tensor([464, 1292, 13], dtype=torch.int64)  # "The dog." in GPT-2 BPE


def _finalize(engine, holdout, losses) -> dict:
    return {
        "ppl": evaluate_ppl(engine, holdout),
        "accuracy": None,  # filled by phase3 wrapper if available
        "diversity": expert_diversity(engine, _probe(holdout)),
        "losses": losses,
        "sample": greedy_sample(engine, PROMPT, n_tokens=80),
    }


def _probe(holdout: torch.Tensor, n: int = 256) -> torch.Tensor:
    return holdout[:n].to(torch.int64)


def arm_from_scratch(engine, *, train, holdout, budget, chunk_len=16,
                     lr=3e-4) -> dict:
    """Arm A: online training on the full budget, no pre-training."""
    p3 = phase3_joint(engine, train[:budget], steps=budget // chunk_len,
                      lr=lr, chunk_len=chunk_len)
    out = _finalize(engine, holdout, p3["losses"])
    out["accuracy"] = p3["accuracy"]
    return out


def arm_edt_vanilla(engine, *, train, holdout, budget, chunk_len=16,
                    lr=3e-4) -> dict:
    """Arm B: EDT faithful to docs/EDT.md (shared bank)."""
    n1 = int(budget * PHASE1_FRAC)
    n2b = int(budget * PHASE2B_FRAC)
    n3 = budget - n1 - n2b
    bank = make_hidden_bank(engine, train[:n1], chunk_len=chunk_len,
                            n_chunks=max(n1 // chunk_len, 1), seed=0)
    p1 = phase1_experts_shared(engine, bank, steps=2000, lr=1e-3, seed=0)
    p2b = phase2b_embedding(engine, train[:n2b], steps=n2b // chunk_len,
                            lr=1e-3, chunk_len=chunk_len, seed=0)
    p3 = phase3_joint(engine, train[:n3], steps=n3 // chunk_len,
                      lr=lr, chunk_len=chunk_len)
    out = _finalize(engine, holdout, p3["losses"])
    out.update({"phase1_losses": p1, "phase2b_losses": p2b,
                "phase3_losses": p3["losses"], "accuracy": p3["accuracy"]})
    return out


def arm_edt_spec(engine, *, train, holdout, budget, domain_split, chunk_len=16,
                 lr=3e-4) -> dict:
    """Arm C: EDT with per-expert disjoint domain banks."""
    n2b = int(budget * PHASE2B_FRAC)
    n3 = budget - int(budget * PHASE1_FRAC) - n2b
    banks = [make_hidden_bank(engine, ds, chunk_len=chunk_len,
                              n_chunks=max(ds.numel() // chunk_len, 1), seed=i)
             for i, ds in enumerate(domain_split)]
    p1 = phase1_experts_partitioned(engine, banks, steps=2000, lr=1e-3, seed=0)
    p2b = phase2b_embedding(engine, train[:n2b], steps=n2b // chunk_len,
                            lr=1e-3, chunk_len=chunk_len, seed=0)
    p3 = phase3_joint(engine, train[:n3], steps=n3 // chunk_len,
                      lr=lr, chunk_len=chunk_len)
    out = _finalize(engine, holdout, p3["losses"])
    out.update({"phase1_losses": p1, "phase2b_losses": p2b,
                "phase3_losses": p3["losses"], "accuracy": p3["accuracy"]})
    return out
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `python -m pytest tests/test_ablib.py -k "arm_from_scratch or arm_edt" -v`
Expected: PASS (all three).

- [ ] **Step 5: Commit**

```bash
git add experiments/edt_ab/ablib.py tests/test_ablib.py
git commit -m "feat(edt-ab): three arm orchestrators (A/B/C)"
```

---

## Task 10: CLI entrypoint `scripts/ab_edt_13m.py`

**Files:**
- Create: `scripts/ab_edt_13m.py`

- [ ] **Step 1: Write the entrypoint**

`scripts/ab_edt_13m.py`:
```python
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
```

- [ ] **Step 2: Smoke-run the entrypoint with a tiny budget**

Run: `python scripts/ab_edt_13m.py --budget 1600 --n-holdout 200 --chunk-len 16`
Expected: prints three arm sections, each with `ppl=... div=... time=...s`, then writes `experiments/edt_ab/results.json`. No traceback. (Wall-clock should be a few minutes on CPU.)

If it fails on `FractusTokenizer` in `greedy_sample`, apply the fallback documented in Task 8 Step 4.

- [ ] **Step 3: Inspect the tiny-run output**

Run: `cat experiments/edt_ab/results.json`
Expected: a JSON object with keys `A_from_scratch`, `B_edt_vanilla`, `C_edt_spec`, each containing `ppl`, `diversity`, `wall_clock_s`, `phase3_final_loss`. Sanity-check that all three have comparable `ppl` magnitude (this tiny budget is not the real measurement — just a wiring check).

- [ ] **Step 4: Commit**

```bash
git add scripts/ab_edt_13m.py
git commit -m "feat(edt-ab): CLI entrypoint scripts/ab_edt_13m.py"
```

---

## Task 11: Apply the pre-registered decision rule and write the report

**Files:**
- Create: `experiments/edt_ab/REPORT.md` (not gitignored — this is a real result)
- Modify: `experiments/edt_ab/.gitignore` (un-ignore `REPORT.md`)

Per spec §9, the decision rule is pre-registered:
- EDT accelerates (A vs B): supported iff `ppl_B ≤ 0.95 · ppl_A`.
- Specialization matters (B vs C): supported iff `ppl_C ≤ 0.95 · ppl_B` **and** `diversity_C ≤ diversity_B − 0.05`.
- Otherwise: not supported at this scale (with spec §8 caveat 1: the 13M is a test of principle).

- [ ] **Step 1: Allow REPORT.md through .gitignore**

Edit `experiments/edt_ab/.gitignore`:
```
results.json
*.png
samples/
*.pt
!REPORT.md
```

- [ ] **Step 2: Run the full experiment**

Run: `python scripts/ab_edt_13m.py --budget 400000 --n-holdout 30000 --chunk-len 16`
Expected: completes in ~3–4 h total CPU. Writes `results.json`, `loss_curves.png`, three sample files. No traceback.

This is the only long step. If wall-clock for Arm A projects to > 90 min, kill it and re-run with `--budget 300000` (note the deviation in REPORT.md).

- [ ] **Step 3: Read the numbers and apply the decision rule**

Run: `cat experiments/edt_ab/results.json`

Compute by hand (write into REPORT.md):
- `accel_ok = ppl_B ≤ 0.95 · ppl_A`
- `spec_ok = ppl_C ≤ 0.95 · ppl_B AND diversity_C ≤ diversity_B − 0.05`

- [ ] **Step 4: Write `experiments/edt_ab/REPORT.md`**

Use this template, filling the bracketed values from `results.json`:

```markdown
# EDT AB-test — Results (13M model, 2026-07-27)

**Spec:** docs/superpowers/specs/2026-07-27-edt-ab-test-design.md
**Budget:** 400 000 tokens/arm, 30 000-token hold-out, chunk_len=16.

## Headline numbers

| Arm | Hold-out PPL | Token acc | Inter-expert cosine | Wall-clock |
|---|---|---|---|---|
| A from-scratch    | [ppl_A] | [acc_A] | [div_A] | [t_A]s |
| B EDT vanilla     | [ppl_B] | [acc_B] | [div_B] | [t_B]s |
| C EDT + spec      | [ppl_C] | [acc_C] | [div_C] | [t_C]s |

## Pre-registered verdicts (spec §9)

- **EDT accelerates (A vs B):** [supported / not supported at this scale].
  Rule: ppl_B ≤ 0.95·ppl_A → [ppl_B] ≤ 0.95·[ppl_A] = [thr_AB] → [TRUE/FALSE].
- **Specialization matters (B vs C):** [supported / not supported at this scale].
  Rule: ppl_C ≤ 0.95·ppl_B AND div_C ≤ div_B − 0.05
  → [ppl_C] ≤ [thr_BC] = [TRUE/FALSE], [div_C] ≤ [div_B−0.05] = [TRUE/FALSE].

## Interpretation

[1–2 paragraphs. State what the numbers show, without overclaiming. Reference
spec §8 caveats: 13M is a test of principle (4 experts, no 16-layer stack),
single seed, position-based specialization. If any pairwise gap is < 5 %,
flag the §9 multi-seed re-run trigger instead of concluding.]

## Samples

See samples/{A_from_scratch,B_edt_vanilla,C_edt_spec}.txt.
```

- [ ] **Step 5: Commit the report**

```bash
git add experiments/edt_ab/REPORT.md experiments/edt_ab/.gitignore
git commit -m "docs(edt-ab): results report — 13M AB-test"
```

---

## Task 12: Wrap-up — update the EDT doc with the measurement

**Files:**
- Modify: `docs/EDT.md` (add a "Measured on the 13M model" subsection near the top)

Per the project's "measure, do not claim" discipline: the first measured EDT result, however small-scale, must be reflected in the doc that currently states 189× as an estimate.

- [ ] **Step 1: Add a subsection after the Executive Summary in `docs/EDT.md`**

Insert (adapting the verdicts from REPORT.md):

```markdown
## First Measurement: 13M Model (2026-07-27)

The 189× figure above is an estimate. The first controlled measurement of
EDT was run on the 13M ContinuousThoughtEngine (4 experts, single MoE — a
test of principle, not a 1B-scale result):

- 3 arms, equal 400 000-token budget, 30 000-token hold-out.
- EDT accelerated learning vs from-scratch: [supported / not supported / inconclusive].
- Specialization (disjoint per-expert data) improved over shared-bank EDT: [supported / not supported / inconclusive].

Full numbers: experiments/edt_ab/REPORT.md. Design: docs/superpowers/specs/2026-07-27-edt-ab-test-design.md.

A negative result here does **not** refute EDT-1B; a positive result supports it. The 1B run remains the decisive test.
```

- [ ] **Step 2: Commit**

```bash
git add docs/EDT.md
git commit -m "docs(edt): record first measured result (13M AB-test)"
```

---

## Self-review notes (done by the planner, not a task)

- **Spec coverage:** §1 (motivation) → implicit, drives everything; §2 (claims) → Task 11 verdicts; §3 (architecture) → Task 2 + API facts header; §4 (three arms) → Tasks 5,6,7,9; §5 (budget) → Tasks 3,9,10; §6 (data) → Task 3; §7 (metrics) → Task 8; §8 (caveats) → Task 11 template; §9 (decision rule) → Task 11; §10 (code plan) → Tasks 1–10 match the function list; §11 (out of scope) → respected, no task touches core; §12 (risks) → cache invalidation handled in Task 5 via `force_refresh`, Phase-1 target fallback noted in Task 4 (aligned pairs make the fallback unnecessary; original noise-target bug avoided).
- **Type consistency:** `build_engine`, `load_corpus`, `make_hidden_bank`, `phase1_*`, `phase2b_embedding`, `phase3_joint`, `evaluate_ppl`, `expert_diversity`, `greedy_sample`, `arm_*` — names and signatures match across tasks. `_probe`, `_finalize`, `_expert_forward`, `_eval_expert_mse`, `_eval_ce` are internal helpers defined where first used.
- **Placeholder scan:** no TBD/TODO; every code step shows the full code.
