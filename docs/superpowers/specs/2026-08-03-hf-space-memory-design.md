# HF Space — PersistentMemory + RAG Coexistence — Design spec

**Date:** 2026-08-03
**Status:** Approved
**Scope:** Modify `space/app.py` + `space/static/index.html`. No core changes.

## Goal

Wire the `PersistentMemory` into the existing HF Space (which uses RAG/KnowledgeBase), in coexistence. The demo must show memory visibly (memory panel) and concretely (name-retention test across sessions).

## Design

- **RAG (existing, unchanged)**: `RAGEngine.converse()` handles text retrieval + generation. Produces the visible reply.
- **PersistentMemory (new)**: attached to the CTE via `engine.attach_memory(mem)`. Salience head bias forced to ~3.0 (sigmoid → ~0.95, consolidates most thoughts). Per-tick injection at 5%. Persists to disk like the KB.
- **Visibility**: new `GET /pmemories` endpoint returns PM count + recent context labels. Frontend panel displays them.
- **Retention test**: RAG stores "Je m'appelle X" as text → next session retrieves it via keyword search. PM adds subconscious context via thought-state blend.

## Files

| Path | Change |
|---|---|
| `space/app.py` | Attach PM to engine, force salience bias, add `GET /pmemories`, persist PM |
| `space/static/index.html` | Memory panel + badge |

## Out of scope

- Salience head training (chantier 2).
- Cognitive modes (chantier 3).
- Replacing RAG with PM.
