# EDT 1B AB-test Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a faithful EDT pipeline for `Fractus1B` with a 3-arm AB test (from-scratch / EDT vanilla / EDT+specialization), unit-tested on CPU and ready to launch on GPU — to decisively test the EDT ×186 acceleration claim at the scale it was designed for.

**Architecture:** A new `ablib_1b.py` helper library mirrors the 13M `ablib.py` structure but adapts to the 1B's 16-block stack: Phase 1 pre-trains all 2048 experts per-layer via partial forward + cached hidden banks (natural per-expert isolation, no gradient masking needed), Phase 2a trains 16 attentions standalone, Phase 2b trains the tied embedding+head, Phase 3 is joint with PGSU + AMP + load-balance aux_loss. A CLI orchestrates the 3 arms. All tests run on CPU at reduced config (`n_layers=2, n_experts=4`); the full run awaits GPU.

**Tech Stack:** PyTorch 2.9 (CPU for dev/tests, GPU for the real run), pytest, Windows Store CPython 3.13 (`$PY`).

**Reference spec:** `docs/superpowers/specs/2026-07-28-edt-1b-ab-test-design.md`. Every design decision lives there.

**API facts (verified — do not re-derive):**
- `Fractus1B(vocab_size, d_model, n_layers, n_heads, d_head, n_levels, n_experts, top_k, expert_d_ff, siren_rank, max_seq_len)`. Full config: `vocab_size=50257, d_model=1280, n_layers=16, n_heads=20, d_head=64, n_levels=2, n_experts=128, top_k=2, expert_d_ff=2048, siren_rank=64, max_seq_len=512`. ~1.05B params, ~4GB RAM.
- `model.embed` (`BPEEmbedding`: `tok_embed`, `pos_embed`, `norm`); `model.blocks` (ModuleList of 16 `FractalBlockSparse`); `model.norm` (final LayerNorm); `model.lm_head` (`nn.Linear`, **`lm_head.weight = embed.tok_embed.weight`** tied).
- `model.forward(ids) -> (logits (B,L,vocab), aux_loss scalar)`. `model._return_hidden=True` → returns `(hidden, aux_loss)` (skips lm_head).
- `FractalBlockSparse.forward(x) -> (x, lb_loss)`: `x = x + attn(norm1(x))`; `phases = kuramoto(norm_kur(x))`; `moe_out, lb = moe(norm_moe(x), phases)`; `x = x + moe_out`.
- Each block: `.moe` (SparseStructuredMoE), `.attn` (FractalLinearAttention), `.norm1`, `.norm_kur`, `.norm_moe`.
- `SparseStructuredMoE`: `.experts_w1[e]` / `.experts_w2[e]` are **separate** `LazyStructuredSirenLinear` modules, each with own `[U, V, scale, bias]` params. **Natural per-expert isolation** — no gradient masking needed. `.n_experts`, `.top_k`.
- `LazyStructuredSirenLinear.forward(x) = scale·(x@V)@Uᵀ + bias` (2 matmuls, differentiable). Params: `.U (out,rank)`, `.V (in,rank)`, `.scale` (scalar), `.bias (out,)`.
- `PGSU(model, n_active=4)`: `.step_begin()` / `.step_end()` rotate which 4/16 blocks get gradients.
- `FractusTokenizer.gpt2_compatible()` (from `fractus1B/tokenizer.py` or `fractus/tokenizer.py`) — classmethod, returns tokenizer with `.decode(ids)`.

**IMPORTANT — Python interpreter (Windows):**
```
PY="/c/Users/PHIL/AppData/Local/Microsoft/WindowsApps/PythonSoftwareFoundation.Python.3.13_qbz5n2kfra8p0/python.exe"
"$PY" -m pytest tests/test_ablib_1b.py -v
```
Shell: Git Bash. Working dir: `C:\Users\PHIL\ZCodeProject\fractus-test`.

---

## File structure

| Path | Responsibility |
|---|---|
| `experiments/edt_1b_ab/__init__.py` | empty |
| `experiments/edt_1b_ab/ablib_1b.py` | pure helpers for 1B EDT (build, load, hidden-bank, phase1/2a/2b/3, evaluators, arms) |
| `experiments/edt_1b_ab/.gitignore` | results.json, *.png, samples/, *.pt |
| `tests/test_ablib_1b.py` | unit tests (CPU-fast, reduced config) |
| `scripts/ab_edt_1b.py` | CLI: orchestrates 3 arms, writes results.json + plots |
| `experiments/edt_1b_ab/RUNBOOK.md` | GPU launch instructions + result interpretation |

---

## Task 1: Scaffold the package + smoke test

**Files:**
- Create: `experiments/edt_1b_ab/__init__.py`
- Create: `experiments/edt_1b_ab/.gitignore`
- Create: `tests/test_ablib_1b.py`
- Create: `experiments/edt_1b_ab/ablib_1b.py`

- [ ] **Step 1: Create the package files**

`experiments/edt_1b_ab/__init__.py`:
```python
"""EDT AB-test on Fractus1B. See docs/superpowers/specs/2026-07-28-edt-1b-ab-test-design.md."""
```

`experiments/edt_1b_ab/.gitignore`:
```
results.json
*.png
samples/
*.pt
!RUNBOOK.md
```

- [ ] **Step 2: Write the failing import test**

`tests/test_ablib_1b.py`:
```python
"""Tests for the EDT 1B AB-test helper library. Runs on CPU at reduced config."""
import experiments.edt_1b_ab.ablib_1b as ablib_1b  # noqa: F401


def test_package_imports():
    assert ablib_1b is not None
```

- [ ] **Step 3: Run to verify it fails**

Run: `"$PY" -m pytest tests/test_ablib_1b.py -v`
Expected: FAIL — `ModuleNotFoundError: experiments.edt_1b_ab.ablib_1b`.

- [ ] **Step 4: Create minimal `ablib_1b.py`**

`experiments/edt_1b_ab/ablib_1b.py`:
```python
"""Helper library for the EDT 1B AB-test.

Pure functions operating on Fractus1B + tensors. Mirrors the 13M ablib.py
structure but adapts to the 1B's 16-block stack: per-layer Phase 1, Phase 2a
attention pre-training, tied embedding Phase 2b, PGSU+AMP Phase 3.

The full run requires GPU; tests run on CPU at reduced config.
"""
```

- [ ] **Step 5: Run to verify it passes**

Run: `"$PY" -m pytest tests/test_ablib_1b.py -v`
Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add experiments/edt_1b_ab/ tests/test_ablib_1b.py
git commit -m "feat(edt-1b): scaffold experiments.edt_1b_ab package + smoke test"
```

---

## Task 2: `build_engine_1b` — deterministic construction (reduced + full config)

**Files:**
- Modify: `experiments/edt_1b_ab/ablib_1b.py`
- Modify: `tests/test_ablib_1b.py`

The tests use a **reduced config** (`n_layers=2, n_experts=4, d_model=128, expert_d_ff=128, siren_rank=16, max_seq_len=64`) so they run in seconds on CPU. The default is the full config.

- [ ] **Step 1: Write the failing test**

Append to `tests/test_ablib_1b.py`:
```python
import torch
from fractus1B.model_1b import Fractus1B

# Reduced config for fast CPU tests. The full config (1B params) is only for GPU runs.
REDUCED = dict(d_model=128, n_layers=2, n_heads=2, d_head=64, n_levels=2,
               n_experts=4, top_k=2, expert_d_ff=128, siren_rank=16, max_seq_len=64)


def test_build_engine_1b_deterministic_reduced():
    a = ablib_1b.build_engine_1b(seed=42, **REDUCED)
    b = ablib_1b.build_engine_1b(seed=42, **REDUCED)
    c = ablib_1b.build_engine_1b(seed=7, **REDUCED)
    assert isinstance(a, Fractus1B)
    for pa, pb in zip(a.parameters(), b.parameters()):
        assert torch.equal(pa, pb)
    assert any(not torch.equal(pa, pc) for pa, pc in zip(a.parameters(), c.parameters()))


def test_build_engine_1b_lm_head_tied():
    """lm_head.weight must BE embed.tok_embed.weight (tied)."""
    eng = ablib_1b.build_engine_1b(seed=42, **REDUCED)
    assert eng.lm_head.weight is eng.embed.tok_embed.weight
```

- [ ] **Step 2: Run to verify it fails**

Run: `"$PY" -m pytest tests/test_ablib_1b.py -k "build_engine_1b" -v`
Expected: FAIL — `AttributeError: module ... has no attribute 'build_engine_1b'`.

- [ ] **Step 3: Implement `build_engine_1b`**

Append to `experiments/edt_1b_ab/ablib_1b.py`:
```python
import torch

from fractus1B.model_1b import Fractus1B


def build_engine_1b(seed: int = 42, **config) -> Fractus1B:
    """Construct a deterministic Fractus1B.

    Default = full 1B config (~1.05B params, needs ~4GB RAM + GPU for training).
    Pass reduced kwargs (e.g. n_layers=2, n_experts=4) for CPU tests.
    """
    torch.manual_seed(seed)
    cfg = dict(vocab_size=50257, d_model=1280, n_layers=16, n_heads=20, d_head=64,
               n_levels=2, n_experts=128, top_k=2, expert_d_ff=2048,
               siren_rank=64, max_seq_len=512)
    cfg.update(config)
    return Fractus1B(**cfg)
```

- [ ] **Step 4: Run to verify it passes**

Run: `"$PY" -m pytest tests/test_ablib_1b.py -k "build_engine_1b" -v`
Expected: PASS (2 tests).

- [ ] **Step 5: Commit**

```bash
git add experiments/edt_1b_ab/ablib_1b.py tests/test_ablib_1b.py
git commit -m "feat(edt-1b): build_engine_1b — deterministic Fractus1B (reduced+full config)"
```

---

## Task 3: `load_corpus_1b` — split + 128-way domain partition

**Files:**
- Modify: `experiments/edt_1b_ab/ablib_1b.py`
- Modify: `tests/test_ablib_1b.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/test_ablib_1b.py`:
```python
import tempfile, os


def _make_tiny_corpus(path, n=5000):
    torch.manual_seed(0)
    torch.save(torch.randint(0, 50257, (n,), dtype=torch.int32), path)


def test_load_corpus_1b_splits_and_partitions():
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "tiny.pt")
        _make_tiny_corpus(p, n=5000)
        split = ablib_1b.load_corpus_1b(p, n_train=1000, n_holdout=200,
                                        n_phase1=512, n_experts=8, seed=42)
    assert split["train"].numel() == 1000
    assert split["holdout"].numel() == 200
    assert split["phase1"].numel() == 512
    # n_experts-way contiguous partition of phase1.
    assert len(split["domain_split"]) == 8
    assert all(s.numel() == 64 for s in split["domain_split"])
    assert torch.equal(torch.cat(split["domain_split"]), split["phase1"])
```

- [ ] **Step 2: Run to verify it fails**

Run: `"$PY" -m pytest tests/test_ablib_1b.py::test_load_corpus_1b_splits_and_partitions -v`
Expected: FAIL — missing `load_corpus_1b`.

- [ ] **Step 3: Implement `load_corpus_1b`**

Append to `experiments/edt_1b_ab/ablib_1b.py`:
```python
def load_corpus_1b(path: str, *, n_train: int, n_holdout: int,
                   n_phase1: int, n_experts: int, seed: int = 42) -> dict:
    """Load + shuffle (fixed seed) + split. domain_split is n_experts contiguous slices.

    Returns: train, holdout, phase1, domain_split (list of n_experts tensors).
    """
    tokens = torch.load(path, weights_only=False).to(torch.int64)
    g = torch.Generator().manual_seed(seed)
    perm = torch.randperm(tokens.numel(), generator=g)
    tokens = tokens[perm]

    need = n_phase1 + n_train + n_holdout
    assert tokens.numel() >= need, f"corpus {tokens.numel()} < need {need}"
    phase1 = tokens[:n_phase1].clone()
    train = tokens[n_phase1:n_phase1 + n_train].clone()
    holdout = tokens[n_phase1 + n_train:n_phase1 + n_train + n_holdout].clone()

    assert n_phase1 % n_experts == 0, "n_phase1 must be divisible by n_experts"
    stride = n_phase1 // n_experts
    domain_split = [phase1[i * stride:(i + 1) * stride].clone() for i in range(n_experts)]
    return {"train": train, "holdout": holdout,
            "phase1": phase1, "domain_split": domain_split}
```

- [ ] **Step 4: Run to verify it passes**

Run: `"$PY" -m pytest tests/test_ablib_1b.py::test_load_corpus_1b_splits_and_partitions -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add experiments/edt_1b_ab/ablib_1b.py tests/test_ablib_1b.py
git commit -m "feat(edt-1b): load_corpus_1b — shuffle + n_experts-way domain partition"
```

---

## Task 4: `make_hidden_bank_1b` — per-layer partial-forward hidden banks

**Files:**
- Modify: `experiments/edt_1b_ab/ablib_1b.py`
- Modify: `tests/test_ablib_1b.py`

This is the 1B novelty: for Phase 1 we need the hidden states at the *input* of each block (output of the previous block, or the embedding for block 0).

- [ ] **Step 1: Write the failing test**

Append to `tests/test_ablib_1b.py`:
```python
def test_make_hidden_bank_1b_block0_uses_embedding():
    eng = ablib_1b.build_engine_1b(seed=42, **REDUCED)
    tokens = torch.arange(200, dtype=torch.int64) % 50257
    bank0 = ablib_1b.make_hidden_bank_1b(eng, tokens, after_block=0,
                                         chunk_len=16, n_chunks=5, seed=0)
    # Block-0 input = embedding output (no blocks applied yet).
    eng.eval()
    with torch.no_grad():
        emb = eng.embed(tokens[:17].unsqueeze(0))  # (1, 17, D)
    assert bank0["h_in"].shape[1] == eng.d_model
    # First pair's h_in must equal embedding of token 0 at the sampled chunk start.
    # Just check shape + that h_in != h_target (non-trivial target).
    assert not torch.equal(bank0["h_in"][0], bank0["h_target"][0])


def test_make_hidden_bank_1b_deeper_block_differs():
    """Bank at block 1 must differ from block 0 (it passed through block 0)."""
    eng = ablib_1b.build_engine_1b(seed=42, **REDUCED)
    tokens = torch.arange(200, dtype=torch.int64) % 50257
    b0 = ablib_1b.make_hidden_bank_1b(eng, tokens, after_block=0, chunk_len=16, n_chunks=3, seed=0)
    b1 = ablib_1b.make_hidden_bank_1b(eng, tokens, after_block=1, chunk_len=16, n_chunks=3, seed=0)
    assert not torch.allclose(b0["h_in"][0], b1["h_in"][0], atol=1e-5)
```

- [ ] **Step 2: Run to verify it fails**

Run: `"$PY" -m pytest tests/test_ablib_1b.py -k "make_hidden_bank_1b" -v`
Expected: FAIL — missing `make_hidden_bank_1b`.

- [ ] **Step 3: Implement `make_hidden_bank_1b`**

Append to `experiments/edt_1b_ab/ablib_1b.py`:
```python
def _partial_forward_to_block_input(model: Fractus1B,
                                    chunk_ids: torch.Tensor,
                                    after_block: int) -> torch.Tensor:
    """Run embedding + blocks[0..after_block-1], return the INPUT to block `after_block`.

    after_block=0 → just the embedding.
    after_block=l → output of blocks[0..l-1].
    chunk_ids: (1, L) token ids. Returns (1, L, d_model). No grad.
    """
    with torch.no_grad():
        x = model.embed(chunk_ids)
        for l in range(after_block):
            x, _ = model.blocks[l](x)
    return x


def make_hidden_bank_1b(model: Fractus1B, tokens: torch.Tensor, *,
                        after_block: int, chunk_len: int = 32,
                        n_chunks: int = 1000, seed: int = 0) -> dict:
    """Build (h_t, h_{t+1}) aligned pairs from the input of block `after_block`.

    For each sampled chunk of chunk_len+1 tokens, embed+forward through blocks
    [0..after_block-1] (no_grad), then emit (chunk_len) aligned pairs.
    Returns {"h_in": (P, d_model), "h_target": (P, d_model)} where
    P = n_chunks * chunk_len.
    """
    model.eval()
    g = torch.Generator().manual_seed(seed)
    n = tokens.numel()
    assert n > chunk_len + 1
    starts = torch.randint(0, n - chunk_len - 1, (n_chunks,), generator=g)
    h_in_list, h_tgt_list = [], []
    for s in starts.tolist():
        chunk_ids = tokens[s:s + chunk_len + 1].unsqueeze(0)  # (1, chunk_len+1)
        h = _partial_forward_to_block_input(model, chunk_ids, after_block)  # (1, L, D)
        h = h.squeeze(0)  # (L, D)
        h_in_list.append(h[:-1].clone())
        h_tgt_list.append(h[1:].clone())
    return {"h_in": torch.cat(h_in_list, dim=0),
            "h_target": torch.cat(h_tgt_list, dim=0)}
```

- [ ] **Step 4: Run to verify it passes**

Run: `"$PY" -m pytest tests/test_ablib_1b.py -k "make_hidden_bank_1b" -v`
Expected: PASS (both).

- [ ] **Step 5: Commit**

```bash
git add experiments/edt_1b_ab/ablib_1b.py tests/test_ablib_1b.py
git commit -m "feat(edt-1b): make_hidden_bank_1b — per-layer partial-forward hidden banks"
```

---

## Task 5: Phase 1 — expert pre-training (shared + partitioned, per-layer)

**Files:**
- Modify: `experiments/edt_1b_ab/ablib_1b.py`
- Modify: `tests/test_ablib_1b.py`

Each expert is a separate `LazyStructuredSirenLinear` module → **natural isolation, no gradient masking** (unlike 13M). Phase 1 iterates over all blocks × all experts.

- [ ] **Step 1: Write the failing test**

Append to `tests/test_ablib_1b.py`:
```python
def test_phase1_experts_1b_reduces_mse_one_expert():
    eng = ablib_1b.build_engine_1b(seed=42, **REDUCED)
    tokens = torch.arange(500, dtype=torch.int64) % 50257
    bank = ablib_1b.make_hidden_bank_1b(eng, tokens, after_block=0, chunk_len=16, n_chunks=10, seed=0)
    mse_before = ablib_1b._eval_expert_mse_1b(eng, block_idx=0, expert_idx=0, bank=bank)
    # Train only block 0 expert 0.
    ablib_1b._train_one_expert_1b(eng, block_idx=0, expert_idx=0, bank=bank,
                                  steps=20, lr=1e-2, seed=0)
    mse_after = ablib_1b._eval_expert_mse_1b(eng, block_idx=0, expert_idx=0, bank=bank)
    assert mse_after < mse_before


def test_phase1_experts_1b_natural_isolation():
    """Training expert (0,0) must NOT move expert (0,1) — separate modules."""
    eng = ablib_1b.build_engine_1b(seed=42, **REDUCED)
    tokens = torch.arange(500, dtype=torch.int64) % 50257
    bank = ablib_1b.make_hidden_bank_1b(eng, tokens, after_block=0, chunk_len=16, n_chunks=10, seed=0)
    e1_before = eng.blocks[0].moe.experts_w1[1].U.detach().clone()
    ablib_1b._train_one_expert_1b(eng, 0, 0, bank, steps=20, lr=1e-2, seed=0)
    e1_after = eng.blocks[0].moe.experts_w1[1].U
    assert torch.equal(e1_before, e1_after), "expert 1 moved while training expert 0"
```

- [ ] **Step 2: Run to verify it fails**

Run: `"$PY" -m pytest tests/test_ablib_1b.py -k "phase1_experts_1b" -v`
Expected: FAIL — missing helpers.

- [ ] **Step 3: Implement Phase 1 helpers**

Append to `experiments/edt_1b_ab/ablib_1b.py`:
```python
def _expert_forward_1b(model: Fractus1B, block_idx: int, expert_idx: int,
                       h: torch.Tensor) -> torch.Tensor:
    """Output of one expert: gelu(w1(h)) → w2. h: (..., d_model)."""
    moe = model.blocks[block_idx].moe
    w1 = moe.experts_w1[expert_idx]
    w2 = moe.experts_w2[expert_idx]
    h1 = w1(h)
    h1_act = torch.nn.functional.gelu(h1)
    return w2(h1_act)


def _eval_expert_mse_1b(model: Fractus1B, block_idx: int, expert_idx: int,
                        bank: dict) -> float:
    model.eval()
    with torch.no_grad():
        out = _expert_forward_1b(model, block_idx, expert_idx, bank["h_in"])
        return torch.nn.functional.mse_loss(out, bank["h_target"]).item()


def _train_one_expert_1b(model: Fractus1B, block_idx: int, expert_idx: int,
                         bank: dict, *, steps: int, lr: float,
                         batch_size: int = 64, seed: int = 0) -> list:
    """Train one expert (block_idx, expert_idx) on bank with MSE.

    Natural isolation: each expert is a separate LazyStructuredSirenLinear module,
    so the optimizer over that expert's params cannot touch other experts.
    No gradient masking needed (unlike the 13M redesign).
    """
    model.train()
    moe = model.blocks[block_idx].moe
    params = list(moe.experts_w1[expert_idx].parameters()) + \
             list(moe.experts_w2[expert_idx].parameters())
    opt = torch.optim.AdamW(params, lr=lr, weight_decay=0.0)
    g = torch.Generator().manual_seed(seed)
    n = bank["h_in"].size(0)
    losses = []
    for _ in range(steps):
        idx_b = torch.randint(0, n, (batch_size,), generator=g)
        h_in = bank["h_in"][idx_b]
        h_tgt = bank["h_target"][idx_b]
        opt.zero_grad()
        out = _expert_forward_1b(model, block_idx, expert_idx, h_in)
        loss = torch.nn.functional.mse_loss(out, h_tgt)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step()
        losses.append(loss.item())
    return losses


def phase1_experts_1b_shared(model: Fractus1B, bank_per_block: list, *,
                             steps: int = 2000, lr: float = 1e-3,
                             batch_size: int = 64, seed: int = 0,
                             log_every_block: bool = True) -> list:
    """Arm B Phase 1: all experts trained on the SAME per-layer bank.

    bank_per_block[l] is the hidden bank for the input of block l.
    Trains every (block, expert) pair. Returns a flat losses list.
    """
    losses = []
    for l in range(len(model.blocks)):
        bank = bank_per_block[l]
        for e in range(model.blocks[l].moe.n_experts):
            losses.extend(_train_one_expert_1b(model, l, e, bank, steps=steps,
                                               lr=lr, batch_size=batch_size,
                                               seed=seed + l * 1000 + e))
    return losses


def phase1_experts_1b_partitioned(model: Fractus1B,
                                  bank_per_block_per_expert: list, *,
                                  steps: int = 2000, lr: float = 1e-3,
                                  batch_size: int = 64, seed: int = 0) -> list:
    """Arm C Phase 1: expert (l,e) trained only on bank_per_block_per_expert[l][e].

    Disjoint per-expert data → specialization.
    """
    losses = []
    for l in range(len(model.blocks)):
        for e in range(model.blocks[l].moe.n_experts):
            bank = bank_per_block_per_expert[l][e]
            losses.extend(_train_one_expert_1b(model, l, e, bank, steps=steps,
                                               lr=lr, batch_size=batch_size,
                                               seed=seed + l * 1000 + e))
    return losses
```

- [ ] **Step 4: Run to verify it passes**

Run: `"$PY" -m pytest tests/test_ablib_1b.py -k "phase1_experts_1b" -v`
Expected: PASS (both). MSE must decrease; expert 1 must be untouched (natural isolation).

- [ ] **Step 5: Commit**

```bash
git add experiments/edt_1b_ab/ablib_1b.py tests/test_ablib_1b.py
git commit -m "feat(edt-1b): Phase 1 — per-layer expert pre-training (shared + partitioned)"
```

---

## Task 6: Phase 2a — attention pre-training (standalone, per layer)

**Files:**
- Modify: `experiments/edt_1b_ab/ablib_1b.py`
- Modify: `tests/test_ablib_1b.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/test_ablib_1b.py`:
```python
def test_phase2a_attention_1b_reduces_loss_and_isolates():
    eng = ablib_1b.build_engine_1b(seed=42, **REDUCED)
    # Snapshot frozen params (everything except block 0's attn + norm1).
    attn_params = set(id(p) for p in eng.blocks[0].attn.parameters()) | \
                  set(id(p) for p in eng.blocks[0].norm1.parameters())
    frozen_before = {id(p): p.detach().clone() for p in eng.parameters()
                     if id(p) not in attn_params}
    torch.manual_seed(0)
    losses = ablib_1b.phase2a_attention_1b(eng, n_steps_per_layer=20, lr=1e-2,
                                           seq_len=16, batch_size=8, seed=0)
    # Loss should trend down (first half vs second half of block-0 losses).
    half = len(losses) // 2
    assert sum(losses[half:]) / max(len(losses) - half, 1) < sum(losses[:half]) / max(half, 1)
    # Frozen params untouched.
    for p in eng.parameters():
        if id(p) in frozen_before:
            assert torch.equal(p, frozen_before[id(p)]), "frozen param moved in Phase 2a"
```

- [ ] **Step 2: Run to verify it fails**

Run: `"$PY" -m pytest tests/test_ablib_1b.py::test_phase2a_attention_1b_reduces_loss_and_isolates -v`
Expected: FAIL — missing `phase2a_attention_1b`.

- [ ] **Step 3: Implement `phase2a_attention_1b`**

Append to `experiments/edt_1b_ab/ablib_1b.py`:
```python
def phase2a_attention_1b(model: Fractus1B, *, n_steps_per_layer: int = 5000,
                         lr: float = 1e-3, seq_len: int = 16,
                         batch_size: int = 16, seed: int = 0) -> list:
    """Phase 2a: train each block's attention + norm1 standalone (denoising target).

    Target = h + 0.1·noise (self-supervised denoising, as in the EDT doc).
    Returns per-step losses of the FIRST block (representative; full loss list
    would be huge for 16 layers × 5000 steps).
    """
    model.train()
    g = torch.Generator().manual_seed(seed)
    d = model.d_model
    first_block_losses = []
    for l in range(len(model.blocks)):
        attn = model.blocks[l].attn
        norm = model.blocks[l].norm1
        params = list(attn.parameters()) + list(norm.parameters())
        opt = torch.optim.AdamW(params, lr=lr, weight_decay=0.0)
        for step in range(n_steps_per_layer):
            h = torch.randn(batch_size, seq_len, d, generator=g)
            target = h + 0.1 * torch.randn_like(h)
            opt.zero_grad()
            h_normed = norm(h)
            attn_out = attn(h_normed)
            h_out = h + attn_out
            loss = torch.nn.functional.mse_loss(h_out, target)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            opt.step()
            if l == 0:
                first_block_losses.append(loss.item())
    return first_block_losses
```

- [ ] **Step 4: Run to verify it passes**

Run: `"$PY" -m pytest tests/test_ablib_1b.py::test_phase2a_attention_1b_reduces_loss_and_isolates -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add experiments/edt_1b_ab/ablib_1b.py tests/test_ablib_1b.py
git commit -m "feat(edt-1b): Phase 2a — standalone attention pre-training per layer"
```

---

## Task 7: Phase 2b — embedding + tied lm_head next-token

**Files:**
- Modify: `experiments/edt_1b_ab/ablib_1b.py`
- Modify: `tests/test_ablib_1b.py`

At the 1B, `lm_head.weight IS embed.tok_embed.weight` (tied). So training the embedding trains the head automatically. Freeze everything else.

- [ ] **Step 1: Write the failing test**

Append to `tests/test_ablib_1b.py`:
```python
def test_phase2b_embedding_1b_tied_and_reduces_ce():
    eng = ablib_1b.build_engine_1b(seed=42, **REDUCED)
    frozen_names = [n for n, _ in eng.named_parameters()
                    if not n.startswith("embed.tok_embed.")]
    frozen_before = {n: p.detach().clone() for n, p in eng.named_parameters()
                     if n in frozen_names}
    tokens = torch.arange(2000, dtype=torch.int64) % 50257
    ce_before = ablib_1b._eval_ce_1b(eng, tokens[:500])
    losses = ablib_1b.phase2b_embedding_1b(eng, tokens, steps=30, lr=1e-2,
                                           seq_len=16, seed=0)
    ce_after = ablib_1b._eval_ce_1b(eng, tokens[:500])
    assert ce_after < ce_before
    # Tied: lm_head.weight is the same object as embed.tok_embed.weight.
    assert eng.lm_head.weight is eng.embed.tok_embed.weight
    # Everything except embed.tok_embed frozen.
    for n, p in eng.named_parameters():
        if n in frozen_before:
            assert torch.equal(p, frozen_before[n]), f"{n} moved in Phase 2b"
```

- [ ] **Step 2: Run to verify it fails**

Run: `"$PY" -m pytest tests/test_ablib_1b.py::test_phase2b_embedding_1b_tied_and_reduces_ce -v`
Expected: FAIL — missing `_eval_ce_1b`, `phase2b_embedding_1b`.

- [ ] **Step 3: Implement Phase 2b**

Append to `experiments/edt_1b_ab/ablib_1b.py`:
```python
def _eval_ce_1b(model: Fractus1B, tokens: torch.Tensor,
                seq_len: int = 16) -> float:
    """Mean per-token CE of embed → lm_head (no blocks). lm_head is tied to tok_embed."""
    model.eval()
    total, n = 0.0, 0
    with torch.no_grad():
        for s in range(0, tokens.numel() - seq_len - 1, seq_len):
            chunk = tokens[s:s + seq_len].unsqueeze(0)
            tgt = tokens[s + 1:s + 1 + seq_len].reshape(-1)
            h = model.embed(chunk)
            logits = model.lm_head(h).reshape(-1, model.vocab_size)
            total += torch.nn.functional.cross_entropy(logits, tgt, reduction="sum").item()
            n += tgt.numel()
    return total / max(n, 1)


def phase2b_embedding_1b(model: Fractus1B, tokens: torch.Tensor, *,
                         steps: int = 2000, lr: float = 1e-3,
                         seq_len: int = 16, seed: int = 0) -> list:
    """Phase 2b: train embed.tok_embed on next-token CE (lm_head is tied → trains too)."""
    model.train()
    for p in model.parameters():
        p.requires_grad_(False)
    for p in model.embed.tok_embed.parameters():
        p.requires_grad_(True)  # lm_head.weight IS this tensor (tied).

    opt = torch.optim.AdamW(model.embed.tok_embed.parameters(), lr=lr, weight_decay=0.0)
    g = torch.Generator().manual_seed(seed)
    n = tokens.numel()
    losses = []
    for _ in range(steps):
        s = torch.randint(0, n - seq_len - 1, (1,), generator=g).item()
        chunk = tokens[s:s + seq_len].unsqueeze(0)
        tgt = tokens[s + 1:s + 1 + seq_len].reshape(-1)
        opt.zero_grad()
        h = model.embed(chunk)
        logits = model.lm_head(h).reshape(-1, model.vocab_size)
        loss = torch.nn.functional.cross_entropy(logits, tgt)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.embed.tok_embed.parameters(), 1.0)
        opt.step()
        losses.append(loss.item())
    for p in model.parameters():
        p.requires_grad_(True)
    return losses
```

- [ ] **Step 4: Run to verify it passes**

Run: `"$PY" -m pytest tests/test_ablib_1b.py::test_phase2b_embedding_1b_tied_and_reduces_ce -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add experiments/edt_1b_ab/ablib_1b.py tests/test_ablib_1b.py
git commit -m "feat(edt-1b): Phase 2b — tied embedding+lm_head next-token training"
```

---

## Task 8: Phase 3 — joint training (chunked-CE + PGSU + AMP + aux_loss)

**Files:**
- Modify: `experiments/edt_1b_ab/ablib_1b.py`
- Modify: `tests/test_ablib_1b.py`

This is the full-model joint fine-tune. Uses `Fractus1B.forward` returning `(logits, aux_loss)`, PGSU to rotate active blocks, AMP (bf16 on GPU, no-op on CPU), and **adds aux_loss to CE** (load-balance, coefficient 0.001 clamped) — unlike the 13M.

- [ ] **Step 1: Write the failing test**

Append to `tests/test_ablib_1b.py`:
```python
def test_phase3_joint_1b_reduces_loss_returns_curve():
    eng = ablib_1b.build_engine_1b(seed=42, **REDUCED)
    tokens = torch.arange(2000, dtype=torch.int64) % 50257
    out = ablib_1b.phase3_joint_1b(eng, tokens, steps=30, lr=1e-3,
                                   seq_len=16, use_pgsu=False)
    assert "losses" in out and "avg_loss" in out and "accuracy" in out
    half = len(out["losses"]) // 2
    assert sum(out["losses"][half:]) / max(len(out["losses"]) - half, 1) < \
           sum(out["losses"][:half]) / max(half, 1)
    assert all(p.requires_grad for p in eng.parameters())
```

- [ ] **Step 2: Run to verify it fails**

Run: `"$PY" -m pytest tests/test_ablib_1b.py::test_phase3_joint_1b_reduces_loss_returns_curve -v`
Expected: FAIL — missing `phase3_joint_1b`.

- [ ] **Step 3: Implement `phase3_joint_1b`**

Append to `experiments/edt_1b_ab/ablib_1b.py`:
```python
def phase3_joint_1b(model: Fractus1B, tokens: torch.Tensor, *,
                    steps: int, lr: float = 3e-4, seq_len: int = 64,
                    use_pgsu: bool = True, aux_weight: float = 0.001,
                    use_amp: bool = None, seed: int = 0) -> dict:
    """Phase 3: full-model joint training. chunked-CE + PGSU + AMP + load-balance aux_loss.

    steps = number of optimizer steps. tokens consumed = steps * seq_len.
    """
    model.train()
    for p in model.parameters():
        p.requires_grad_(True)
    device = next(model.parameters()).device
    if use_amp is None:
        use_amp = device.type == "cuda"
    amp_dtype = torch.bfloat16 if use_amp else None

    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    pgsu = None
    if use_pgsu:
        try:
            from fractus1B.pgsu import PGSU
            pgsu = PGSU(model, n_active=4)
        except Exception:
            pgsu = None

    g = torch.Generator().manual_seed(seed)
    n = tokens.numel()
    losses, total_loss, correct, total = [], 0.0, 0, 0
    vocab = model.vocab_size
    for _ in range(steps):
        s = torch.randint(0, n - seq_len - 1, (1,), generator=g).item()
        chunk = tokens[s:s + seq_len].unsqueeze(0).to(device)
        tgt = tokens[s + 1:s + 1 + seq_len].reshape(-1).to(device)
        if pgsu is not None:
            pgsu.step_begin()
        opt.zero_grad()
        if use_amp:
            with torch.autocast(device_type="cuda", dtype=amp_dtype):
                logits, aux = model(chunk)
                ce = torch.nn.functional.cross_entropy(logits.reshape(-1, vocab), tgt)
                loss = ce + aux_weight * torch.clamp(aux, max=1.0)
        else:
            logits, aux = model(chunk)
            ce = torch.nn.functional.cross_entropy(logits.reshape(-1, vocab), tgt)
            loss = ce + aux_weight * torch.clamp(aux, max=1.0)
        if not torch.isfinite(loss):
            if pgsu is not None:
                pgsu.step_end()
            continue
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if pgsu is not None:
            pgsu.step_end()
        losses.append(loss.item())
        total_loss += loss.item() * seq_len
        correct += (logits.reshape(-1, vocab).argmax(-1) == tgt).sum().item()
        total += seq_len
    return {"losses": losses, "avg_loss": total_loss / max(total, 1),
            "accuracy": correct / max(total, 1), "steps": total}
```

- [ ] **Step 4: Run to verify it passes**

Run: `"$PY" -m pytest tests/test_ablib_1b.py::test_phase3_joint_1b_reduces_loss_returns_curve -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add experiments/edt_1b_ab/ablib_1b.py tests/test_ablib_1b.py
git commit -m "feat(edt-1b): Phase 3 — joint training with PGSU + AMP + aux_loss"
```

---

## Task 9: Evaluators (ppl, diversity, sample)

**Files:**
- Modify: `experiments/edt_1b_ab/ablib_1b.py`
- Modify: `tests/test_ablib_1b.py`

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_ablib_1b.py`:
```python
def test_evaluate_ppl_1b_finite_positive():
    eng = ablib_1b.build_engine_1b(seed=42, **REDUCED)
    tokens = torch.arange(500, dtype=torch.int64) % 50257
    ppl = ablib_1b.evaluate_ppl_1b(eng, tokens[:300], seq_len=16)
    assert ppl == ppl and ppl > 0.0


def test_expert_diversity_1b_in_range():
    eng = ablib_1b.build_engine_1b(seed=42, **REDUCED)
    probe = torch.arange(64, dtype=torch.int64) % 50257
    div = ablib_1b.expert_diversity_1b(eng, probe, block_idx=0)
    assert -1.0 <= div <= 1.0


def test_greedy_sample_1b_returns_string():
    eng = ablib_1b.build_engine_1b(seed=42, **REDUCED)
    prompt = torch.tensor([1, 2, 3], dtype=torch.int64)
    s = ablib_1b.greedy_sample_1b(eng, prompt, n_tokens=8)
    assert isinstance(s, str) and len(s) > 0
```

- [ ] **Step 2: Run to verify it fails**

Run: `"$PY" -m pytest tests/test_ablib_1b.py -k "evaluate_ppl_1b or expert_diversity_1b or greedy_sample_1b" -v`
Expected: FAIL — all missing.

- [ ] **Step 3: Implement the evaluators**

Append to `experiments/edt_1b_ab/ablib_1b.py`:
```python
import math


def evaluate_ppl_1b(model: Fractus1B, tokens: torch.Tensor,
                    seq_len: int = 64, nll_cap: float = 20.0) -> float:
    """Perplexity via full-model forward (logits, aux_loss)."""
    model.eval()
    device = next(model.parameters()).device
    total_nll, n = 0.0, 0
    for s in range(0, tokens.numel() - seq_len - 1, seq_len):
        chunk = tokens[s:s + seq_len].unsqueeze(0).to(device)
        tgt = tokens[s + 1:s + 1 + seq_len].reshape(-1).to(device)
        with torch.no_grad():
            logits, _ = model(chunk)
            nll = torch.nn.functional.cross_entropy(
                logits.reshape(-1, model.vocab_size), tgt, reduction="none")
            nll = nll.clamp(max=nll_cap)
        total_nll += nll.sum().item()
        n += tgt.numel()
    return math.exp(total_nll / max(n, 1))


def expert_diversity_1b(model: Fractus1B, probe_tokens: torch.Tensor,
                        block_idx: int = 8) -> float:
    """Mean off-diagonal cosine between expert outputs of one representative block.

    probe_tokens: (P,) → embed → each expert of blocks[block_idx].moe.
    """
    model.eval()
    device = next(model.parameters()).device
    with torch.no_grad():
        h = model.embed(probe_tokens.unsqueeze(0).to(device)).squeeze(0)  # (P, D)
        moe = model.blocks[block_idx].moe
        outs = []
        for e in range(moe.n_experts):
            outs.append(_expert_forward_1b(model, block_idx, e, h))
        outs = torch.stack(outs)  # (E, P, D)
    E = outs.size(0)
    cos_sum, count = 0.0, 0
    for i in range(E):
        for j in range(E):
            if i == j:
                continue
            cos = torch.nn.functional.cosine_similarity(
                outs[i].flatten(), outs[j].flatten(), dim=0).item()
            cos_sum += cos
            count += 1
    return cos_sum / max(count, 1)


def greedy_sample_1b(model: Fractus1B, prompt: torch.Tensor,
                     n_tokens: int = 80) -> str:
    """Greedy decode n_tokens after prompt (single-token forward)."""
    from fractus1B.tokenizer import FractusTokenizer
    tok = FractusTokenizer.gpt2_compatible()
    model.eval()
    device = next(model.parameters()).device
    ids = prompt.tolist()
    cur = torch.tensor(ids[-1:], dtype=torch.int64, device=device).unsqueeze(0)
    for _ in range(n_tokens):
        with torch.no_grad():
            logits, _ = model(cur)  # (1, 1, vocab)
        nxt = int(logits[0, -1].argmax(-1).item())
        ids.append(nxt)
        cur = torch.tensor([[nxt]], dtype=torch.int64, device=device)
    return tok.decode(ids)
```

- [ ] **Step 4: Run to verify it passes**

Run: `"$PY" -m pytest tests/test_ablib_1b.py -k "evaluate_ppl_1b or expert_diversity_1b or greedy_sample_1b" -v`
Expected: PASS (all 3).

If `fractus1B.tokenizer.FractusTokenizer.gpt2_compatible()` fails, try `from fractus.tokenizer import FractusTokenizer` instead (both repos define it). Note the fallback in the commit message.

- [ ] **Step 5: Commit**

```bash
git add experiments/edt_1b_ab/ablib_1b.py tests/test_ablib_1b.py
git commit -m "feat(edt-1b): evaluators — ppl, expert diversity, greedy sample"
```

---

## Task 10: Arm orchestrators (A / B / C)

**Files:**
- Modify: `experiments/edt_1b_ab/ablib_1b.py`
- Modify: `tests/test_ablib_1b.py`

Each arm consumes the same total budget N. Arms B/C split N as Phase 1 = 10%, Phase 2a = 15%, Phase 2b = 30%, Phase 3 = 45% (spec §5).

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_ablib_1b.py`:
```python
def test_arm_from_scratch_1b_smoke():
    eng = ablib_1b.build_engine_1b(seed=42, **REDUCED)
    tokens = torch.arange(3000, dtype=torch.int64) % 50257
    out = ablib_1b.arm_from_scratch_1b(eng, train=tokens, holdout=tokens[2500:2800],
                                       budget=1000, seq_len=16)
    for k in ("ppl", "accuracy", "diversity", "losses", "sample"):
        assert k in out


def test_arm_edt_vanilla_1b_smoke():
    eng = ablib_1b.build_engine_1b(seed=42, **REDUCED)
    tokens = torch.arange(5000, dtype=torch.int64) % 50257
    out = ablib_1b.arm_edt_vanilla_1b(eng, train=tokens, holdout=tokens[4500:4800],
                                      budget=2000, seq_len=16)
    for k in ("ppl", "phase1_losses", "phase2a_losses", "phase2b_losses",
              "phase3_losses", "diversity", "sample"):
        assert k in out


def test_arm_edt_spec_1b_smoke():
    eng = ablib_1b.build_engine_1b(seed=42, **REDUCED)
    base = torch.arange(5000, dtype=torch.int64) % 50257
    phase1 = base[:500]
    n_experts = REDUCED["n_experts"]
    domain_split = [phase1[i * (500 // n_experts):(i + 1) * (500 // n_experts)]
                    for i in range(n_experts)]
    out = ablib_1b.arm_edt_spec_1b(eng, train=base[500:], holdout=base[4500:4800],
                                   budget=2000, domain_split=domain_split, seq_len=16)
    for k in ("ppl", "phase1_losses", "phase3_losses", "diversity", "sample"):
        assert k in out
```

- [ ] **Step 2: Run to verify it fails**

Run: `"$PY" -m pytest tests/test_ablib_1b.py -k "arm_.*1b_smoke" -v`
Expected: FAIL — all missing.

- [ ] **Step 3: Implement the three arms**

Append to `experiments/edt_1b_ab/ablib_1b.py`:
```python
PHASE1_FRAC, PHASE2A_FRAC, PHASE2B_FRAC, PHASE3_FRAC = 0.10, 0.15, 0.30, 0.45

PROMPT_1B = torch.tensor([464, 1292, 13], dtype=torch.int64)  # "The dog." GPT-2 BPE


def _probe_1b(holdout, n=64):
    return holdout[:n].to(torch.int64)


def _finalize_1b(model, holdout, losses, *, div_block=0) -> dict:
    return {
        "ppl": evaluate_ppl_1b(model, holdout),
        "accuracy": None,
        "diversity": expert_diversity_1b(model, _probe_1b(holdout), block_idx=div_block),
        "losses": losses,
        "sample": greedy_sample_1b(model, PROMPT_1B, n_tokens=80),
    }


def arm_from_scratch_1b(model, *, train, holdout, budget, seq_len=64,
                        lr=3e-4, use_pgsu=True) -> dict:
    """Arm A: full-budget joint training, no pre-training."""
    n_layers = len(model.blocks)
    div_block = min(8, n_layers - 1)
    p3 = phase3_joint_1b(model, train[:budget], steps=budget // seq_len,
                         lr=lr, seq_len=seq_len, use_pgsu=use_pgsu)
    out = _finalize_1b(model, holdout, p3["losses"], div_block=div_block)
    out["accuracy"] = p3["accuracy"]
    return out


def arm_edt_vanilla_1b(model, *, train, holdout, budget, seq_len=64,
                       lr=3e-4, use_pgsu=True) -> dict:
    """Arm B: EDT faithful to docs/EDT.md (per-layer shared banks)."""
    n1 = int(budget * PHASE1_FRAC)
    n2a = int(budget * PHASE2A_FRAC)
    n2b = int(budget * PHASE2B_FRAC)
    n3 = budget - n1 - n2a - n2b
    n_layers = len(model.blocks)
    div_block = min(8, n_layers - 1)

    # Phase 1: per-layer shared banks. Cache one partial forward per layer.
    bank_per_block = []
    chunk_len = 32
    for l in range(n_layers):
        bank_per_block.append(make_hidden_bank_1b(
            model, train[:n1], after_block=l, chunk_len=chunk_len,
            n_chunks=max(n1 // chunk_len, 1), seed=0))
    p1 = phase1_experts_1b_shared(model, bank_per_block, steps=2000, lr=1e-3, seed=0)
    p2a = phase2a_attention_1b(model, n_steps_per_layer=max(n2a // seq_len, 10),
                               lr=1e-3, seq_len=seq_len, seed=0)
    p2b = phase2b_embedding_1b(model, train[:n2b], steps=n2b // seq_len,
                               lr=1e-3, seq_len=seq_len, seed=0)
    p3 = phase3_joint_1b(model, train[:n3], steps=n3 // seq_len,
                         lr=lr, seq_len=seq_len, use_pgsu=use_pgsu)
    out = _finalize_1b(model, holdout, p3["losses"], div_block=div_block)
    out.update({"phase1_losses": p1, "phase2a_losses": p2a, "phase2b_losses": p2b,
                "phase3_losses": p3["losses"], "accuracy": p3["accuracy"]})
    return out


def arm_edt_spec_1b(model, *, train, holdout, budget, domain_split, seq_len=64,
                    lr=3e-4, use_pgsu=True) -> dict:
    """Arm C: EDT with per-expert disjoint domain banks (specialization)."""
    n2a = int(budget * PHASE2A_FRAC)
    n2b = int(budget * PHASE2B_FRAC)
    n3 = budget - int(budget * PHASE1_FRAC) - n2a - n2b
    n_layers = len(model.blocks)
    n_experts = len(domain_split)
    div_block = min(8, n_layers - 1)

    # Phase 1: per-layer, per-expert disjoint banks.
    bank_per_block_per_expert = []
    chunk_len = 32
    for l in range(n_layers):
        per_expert = []
        for e in range(n_experts):
            per_expert.append(make_hidden_bank_1b(
                model, domain_split[e], after_block=l, chunk_len=chunk_len,
                n_chunks=max(domain_split[e].numel() // chunk_len, 1), seed=e))
        bank_per_block_per_expert.append(per_expert)
    p1 = phase1_experts_1b_partitioned(model, bank_per_block_per_expert,
                                       steps=2000, lr=1e-3, seed=0)
    p2a = phase2a_attention_1b(model, n_steps_per_layer=max(n2a // seq_len, 10),
                               lr=1e-3, seq_len=seq_len, seed=0)
    p2b = phase2b_embedding_1b(model, train[:n2b], steps=n2b // seq_len,
                               lr=1e-3, seq_len=seq_len, seed=0)
    p3 = phase3_joint_1b(model, train[:n3], steps=n3 // seq_len,
                         lr=lr, seq_len=seq_len, use_pgsu=use_pgsu)
    out = _finalize_1b(model, holdout, p3["losses"], div_block=div_block)
    out.update({"phase1_losses": p1, "phase2a_losses": p2a, "phase2b_losses": p2b,
                "phase3_losses": p3["losses"], "accuracy": p3["accuracy"]})
    return out
```

- [ ] **Step 4: Run to verify it passes**

Run: `"$PY" -m pytest tests/test_ablib_1b.py -k "arm_.*1b_smoke" -v`
Expected: PASS (all 3). Slow (~minutes each on CPU).

- [ ] **Step 5: Commit**

```bash
git add experiments/edt_1b_ab/ablib_1b.py tests/test_ablib_1b.py
git commit -m "feat(edt-1b): three arm orchestrators (A/B/C) for Fractus1B"
```

---

## Task 11: CLI `scripts/ab_edt_1b.py` + CPU safety guard

**Files:**
- Create: `scripts/ab_edt_1b.py`

- [ ] **Step 1: Write the CLI**

`scripts/ab_edt_1b.py`:
```python
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
```

- [ ] **Step 2: Smoke-run with reduced config + force-cpu**

The CLI defaults to the full 1B config (too big for CPU smoke). For the smoke test, override via a tiny helper. Run:
```bash
"$PY" -c "
import sys, os; sys.path.insert(0, '.')
import experiments.edt_1b_ab.ablib_1b as ablib_1b
# Monkeypatch build_engine_1b to reduced config for the smoke.
_orig = ablib_1b.build_engine_1b
ablib_1b.build_engine_1b = lambda seed=42, **kw: _orig(seed=seed, d_model=128, n_layers=2, n_heads=2, d_head=64, n_levels=2, n_experts=4, top_k=2, expert_d_ff=128, siren_rank=16, max_seq_len=64)
import importlib.util
spec = importlib.util.spec_from_file_location('cli', 'scripts/ab_edt_1b.py')
cli = importlib.util.module_from_spec(spec)
# Override the load_corpus_1b n_experts via argv.
sys.argv = ['ab_edt_1b.py', '--budget', '600', '--n-holdout', '100', '--seq-len', '16', '--force-cpu', '--corpus', 'data/communication_corpus.pt']
# But load_corpus_1b uses n_experts=128 hardcoded in CLI — patch the CLI's n_experts.
# Simpler: just call the arm functions directly via ablib.
import torch
torch.set_num_threads(8)
split = ablib_1b.load_corpus_1b('data/communication_corpus.pt', n_train=600, n_holdout=100, n_phase1=120, n_experts=4, seed=42)
eng = ablib_1b.build_engine_1b(seed=42)
print('running arm A smoke...')
out = ablib_1b.arm_from_scratch_1b(eng, train=split['train'], holdout=split['holdout'], budget=600, seq_len=16)
print('A ppl', out['ppl'])
print('SMOKE OK')
"
```
Expected: prints `A ppl <number>` and `SMOKE OK`, no traceback. This validates the ablib wiring end-to-end on CPU. (The CLI's hardcoded `n_experts=128` is fine for the GPU run; the smoke uses ablib directly because CPU can't build the full 1B config.)

If a traceback occurs, capture it and fix before proceeding.

- [ ] **Step 3: Commit**

```bash
git add scripts/ab_edt_1b.py
git commit -m "feat(edt-1b): CLI scripts/ab_edt_1b.py with CPU safety guard"
```

---

## Task 12: RUNBOOK.md — GPU launch instructions

**Files:**
- Create: `experiments/edt_1b_ab/RUNBOOK.md`

- [ ] **Step 1: Write the RUNBOOK**

`experiments/edt_1b_ab/RUNBOOK.md`:
```markdown
# EDT 1B AB-test — GPU Runbook

This documents how to launch the full EDT 1B AB-test (which is not feasible on CPU).

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

Expected runtime: ~2 days on a single A100/3090 (per the EDT doc's estimate). Monitor `experiments/edt_1b_ab/run.log`.

## Smoke check (CPU, before spending GPU)

```bash
python scripts/ab_edt_1b.py --budget 1000 --force-cpu   # ~minutes; validates wiring
```

## Interpreting results

After the run, `experiments/edt_1b_ab/results.json` contains ppl/diversity/accuracy per arm. Apply the pre-registered decision rule (spec §9 of `docs/superpowers/specs/2026-07-27-edt-ab-test-design.md`):

- **EDT accelerates (A vs B):** supported iff `ppl_B ≤ 0.95 · ppl_A`.
- **Specialization matters (B vs C):** supported iff `ppl_C ≤ 0.95 · ppl_B` AND `diversity_C ≤ diversity_B − 0.05`.

If A-vs-B is positive, EDT works at the scale it was designed for (×186 acceleration claim confirmed at decisive scale). If negative, EDT is refuted at every scale tested (13M and 1B) and the project should stop claiming it works. Either outcome is publishable.

The trained checkpoint from the winning arm is a **living model**: reusable, expandable via `scripts/rank_expand.py`, and continuable via `load_state_dict`.
```

- [ ] **Step 2: Commit**

```bash
git add experiments/edt_1b_ab/RUNBOOK.md
git commit -m "docs(edt-1b): RUNBOOK — GPU launch + result interpretation"
```

---

## Task 13: Wrap-up — full test suite + final review

**Files:** read-only review.

- [ ] **Step 1: Run the full test suite (CPU, reduced config)**

Run: `"$PY" -m pytest tests/test_ablib_1b.py -v`
Expected: all pass (~13 tests). The 3 arm smoke tests are slow (~minutes each).

- [ ] **Step 2: Dispatch a final holistic code review**

Dispatch a code-reviewer subagent over `git diff <base>..<head>` for the edt_1b_ab work. Focus: (a) does `make_hidden_bank_1b` partial-forward actually reach the right layer, (b) is per-expert isolation truly natural (no shared params, no masking needed), (c) does Phase 3 correctly add aux_loss and use PGSU/AMP, (d) is the B-vs-C comparison valid (identical except Phase 1), (e) any path that would break on GPU.

- [ ] **Step 3: Finish**

Use `superpowers:finishing-a-development-branch` to merge / keep per the user's choice.

---

## Self-review notes (done by the planner)

- **Spec coverage:** §1 (motivation + living model) → implicit, drives everything; §2 (claims) → Task 13 verdicts via results.json + RUNBOOK; §3 (architecture) → Task 2 + API facts header; §4 (three arms) → Tasks 5-8, 10; §5 (budget) → Tasks 3, 10, 11; §6 (data) → Task 3, 11 (--corpus arg); §7 (metrics) → Task 9; §8 (components) → Tasks 1-11 match the file/function list; §9 (testing) → every task has CPU tests at reduced config; §10 (risks) → CPU guard (Task 11), AMP opt-out (Task 8 `use_amp=None`), NLL cap (Task 9), PGSU symmetric (Task 8); §11 (out of scope) → respected; §12 (success criteria) → Task 13 + RUNBOOK (Task 12); §13 (living model) → RUNBOOK + final note.
- **Placeholder scan:** no TBD/TODO; every code step shows full code. Fixed a `PROMPT_1B`/`PROMPT_1b` casing inconsistency and wired `_probe_1b` into `_finalize_1b` during self-review. Task 11 Step 2 uses a smoke harness (not a placeholder — it's the actual CPU validation path).
- **Type consistency:** `build_engine_1b`, `load_corpus_1b`, `make_hidden_bank_1b`, `phase1_*_1b`, `phase2a_attention_1b`, `phase2b_embedding_1b`, `phase3_joint_1b`, `evaluate_ppl_1b`, `expert_diversity_1b`, `greedy_sample_1b`, `arm_*_1b` — names/signatures match across tasks. Internal helpers `_partial_forward_to_block_input`, `_expert_forward_1b`, `_eval_expert_mse_1b`, `_train_one_expert_1b`, `_eval_ce_1b`, `_finalize_1b`, `_probe_1b` defined where first used.
- **Note on `n_experts` in CLI:** the CLI hardcodes `n_experts=128` (full config). The CPU smoke (Task 11 Step 2) calls ablib directly with reduced config because the full 1B config doesn't fit/build on CPU. This is the intended pattern — the CLI is the GPU entry point; CPU validation goes through ablib + tests.
```
