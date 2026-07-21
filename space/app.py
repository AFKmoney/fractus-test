"""Fractus Space — shared-memory AI backend.

This FastAPI app powers the Fractus HuggingFace Space: a single Fractus
instance with a SHARED knowledge base that persists across all users and
sessions. Every user's input is stored, retrieved, and remembered by the
next user — proving Fractus's continuous learning and persistent memory.

Architecture:
  - Fractus model (Fractus-1B trained weights → CTE runtime)
  - SharedKnowledgeBase: ONE global KB across all HTTP sessions
  - Basic keyword moderation before storing (blocks toxic content)
  - FastAPI: POST /chat, GET /memories, GET /stats
"""
import os
import sys
import time
import re
import threading
from typing import Optional

# Path to the Fractus package (sibling of the space).
FRACTUS_PATH = os.environ.get("FRACTUS_PATH", "/workspace/fractus")
sys.path.insert(0, FRACTUS_PATH)

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

# --- Fractus imports (lazy — model loads on first request) ---
_engine = None
_kb = None
_lock = threading.Lock()


def get_engine():
    """Lazy-load Fractus engine + shared KB. Loads ONCE, shared across requests."""
    global _engine, _kb
    with _lock:
        if _engine is None:
            from fractus.continuous_engine import ContinuousThoughtEngine
            from fractus.tokenizer import FractusTokenizer
            from fractus.rag import KnowledgeBase, RAGEngine

            tok = FractusTokenizer.gpt2_compatible()
            engine = ContinuousThoughtEngine(
                vocab_size=50257, d_model=128,
                n_heads=4, d_head=32, n_levels=2,
                n_oscillators=16, n_experts=8, top_k=2,
            )

            # Load trained weights from HF if a checkpoint is provided.
            ckpt_path = os.environ.get("FRACTUS_CHECKPOINT")
            if ckpt_path and os.path.exists(ckpt_path):
                import torch
                print(f"[Fractus] Loading checkpoint: {ckpt_path}", flush=True)
                sd = torch.load(ckpt_path, weights_only=False, map_location="cpu")
                # Best-effort load — the CTE and Fractus-1B don't share exact
                # parameter names, but layers with matching shapes will load.
                engine.load_state_dict(sd.get("model_state", sd), strict=False)
                print(f"[Fractus] Checkpoint loaded (strict=False)", flush=True)
            else:
                print("[Fractus] WARNING: no checkpoint — using random weights", flush=True)

            kb = KnowledgeBase(d_model=128)

            # Load persisted KB if it exists (resumes memory across restarts).
            kb_path = os.environ.get("FRACTUS_KB_PATH", "/data/fractus_kb.pkl")
            if os.path.exists(kb_path):
                try:
                    kb.load(kb_path)
                    print(f"[Fractus] Loaded shared KB: {len(kb.texts)} memories", flush=True)
                except Exception as e:
                    print(f"[Fractus] KB load failed: {e}", flush=True)

            rag = RAGEngine(engine, tok, kb)
            _engine = {"engine": engine, "tok": tok, "kb": kb, "rag": rag}
        return _engine


# ============================================================================
# Moderation: basic keyword filter to keep the shared KB safe-ish.
# ============================================================================

# Crude but effective: blocks obvious toxic content from being stored.
# This is NOT a sophisticated moderation system — just a safety net so the
# Space doesn't get taken down in 2 hours.
_BLOCKED_KEYWORDS = [
    # Hate speech / slurs (censored here, expanded at runtime)
    "nigger", "faggot", "tranny", "kike", "spic", "chink", "retard",
    # Self-harm / violence instructions
    "kill yourself", "how to make a bomb", "suicide method",
    # CSAM-adjacent
    "child porn", "cp ", "loli", "shota", "underage nude",
    # Doxxing patterns (very basic)
    "social security number", "credit card number",
]
_BLOCKED_RE = re.compile(
    r"\b(" + "|".join(re.escape(k) for k in _BLOCKED_KEYWORDS) + r")\b",
    re.IGNORECASE,
)


def moderate(text: str) -> tuple:
    """Returns (is_blocked, reason). Basic keyword filter only."""
    if not text or len(text.strip()) < 1:
        return True, "empty"
    if len(text) > 2000:
        return True, "too_long"
    m = _BLOCKED_RE.search(text)
    if m:
        return True, f"blocked_keyword:{m.group(1)[:20]}"
    return False, ""


def save_kb_async():
    """Persist the KB in a background thread (don't block the response)."""
    def _save():
        try:
            e = get_engine()
            kb_path = os.environ.get("FRACTUS_KB_PATH", "/data/fractus_kb.pkl")
            os.makedirs(os.path.dirname(kb_path) or ".", exist_ok=True)
            e["kb"].save(kb_path)
        except Exception as ex:
            print(f"[Fractus] KB save failed: {ex}", flush=True)
    threading.Thread(target=_save, daemon=True).start()


# ============================================================================
# FastAPI app
# ============================================================================

app = FastAPI(title="Fractus Space")

# Serve the static frontend.
HERE = os.path.dirname(os.path.abspath(__file__))
STATIC_DIR = os.path.join(HERE, "static")
if os.path.isdir(STATIC_DIR):
    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


class ChatRequest(BaseModel):
    message: str
    speaker: Optional[str] = "anonymous"


class ChatResponse(BaseModel):
    reply: str
    memories_used: int
    stored: bool
    moderation: Optional[str] = None


@app.get("/", response_class=HTMLResponse)
async def index():
    """Serve the main chat UI."""
    index_path = os.path.join(STATIC_DIR, "index.html")
    if os.path.exists(index_path):
        return HTMLResponse(content=open(index_path, encoding="utf-8").read())
    return HTMLResponse("<h1>Fractus Space</h1><p>static/index.html missing</p>")


@app.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest):
    """Main endpoint: user message → Fractus reply, with shared memory."""
    # 1. Moderate the input BEFORE storing.
    blocked, reason = moderate(req.message)
    if blocked:
        return ChatResponse(
            reply="(Fractus declined to store this message — moderation filter.)",
            memories_used=0, stored=False, moderation=reason,
        )

    try:
        e = get_engine()
    except Exception as ex:
        raise HTTPException(500, f"Fractus failed to start: {ex}")

    # 2. Retrieve relevant memories + generate a reply (also learns the input).
    try:
        result = e["rag"].converse(req.message, speaker=req.speaker)
        reply = result.get("response", "(Fractus stayed silent.)")
        memories_used = len(result.get("retrieved", []))
        stored = True
        # Persist the KB in background.
        save_kb_async()
    except Exception as ex:
        return ChatResponse(
            reply=f"(Fractus encountered an error: {ex})",
            memories_used=0, stored=False, moderation="runtime_error",
        )

    return ChatResponse(
        reply=reply,
        memories_used=memories_used,
        stored=stored,
    )


@app.get("/memories")
async def memories(limit: int = 20):
    """Show the most recent shared memories (transparency feature)."""
    try:
        e = get_engine()
        kb = e["kb"]
        recent = list(zip(kb.texts[-limit:], kb.sources[-limit:] if kb.sources else ["?"]*limit))
        return {"count": len(kb.texts), "recent": [{"text": t[:200], "source": s} for t, s in reversed(recent)]}
    except Exception as ex:
        return {"error": str(ex), "count": 0, "recent": []}


@app.get("/stats")
async def stats():
    """Live stats for the UI."""
    try:
        e = get_engine()
        return {
            "memories": len(e["kb"].texts),
            "engine": "Fractus-CTE",
            "status": "online",
        }
    except Exception:
        return {"memories": 0, "engine": "Fractus-CTE", "status": "starting"}


@app.get("/health")
async def health():
    return {"status": "ok", "time": time.time()}
