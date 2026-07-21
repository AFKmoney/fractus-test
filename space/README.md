---
title: Fractus
emoji: 🧠
colorFrom: red
colorTo: pink
sdk: docker
app_port: 7860
pinned: true
license: mit
---

# Fractus — The AI That Remembers Everything

**🧠 Shared-memory AI demo.** Fractus is a decentralized AI trained from scratch
(no GPT, no Claude, no external LLM). It has a **shared knowledge base** that
persists across all visitors and sessions.

> ⚠️ Everything you say here will be remembered by Fractus and seen by the next
> visitor. Don't share personal information.

## How it works

1. You type a message.
2. Fractus **retrieves** relevant memories from everyone who came before.
3. Fractus **generates** a reply using its trained brain.
4. Your message is **stored permanently** in the shared knowledge base.
5. The next visitor benefits from what you taught Fractus.

This proves three core Fractus capabilities live:
- **Persistent memory** — `rag.learn()` survives across sessions
- **Retrieval-augmented generation** — context-aware answers from memory
- **Continuous learning without retraining** — zero gradients, zero backward

## Architecture

```
Browser → FastAPI (this Space) → Fractus runtime
                              → SharedKnowledgeBase (singleton, persisted to disk)
```

The Fractus model is a 88M-param fractal transformer (0.86B effective capacity
via LazyStructuredSiren low-rank weights) trained on a 1.38B-token Chinchilla-
optimal corpus. Source: [github.com/AFKmoney/fractus](https://github.com/AFKmoney/fractus)

## Moderation

Basic keyword filtering is applied before storing user input. Slurs, self-harm
instructions, and CSAM-adjacent content are blocked from the knowledge base.
This is a safety net, not a sophisticated moderation system.

## Run locally

```bash
pip install -r requirements.txt
FRACTUS_PATH=/path/to/fractus FRACTUS_CHECKPOINT=/path/to/ckpt.pt uvicorn app:app --port 7860
```

## Author

**Philippe-Antoine Robert** — 2026
