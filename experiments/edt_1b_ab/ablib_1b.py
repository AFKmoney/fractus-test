"""Helper library for the EDT 1B AB-test.

Pure functions operating on Fractus1B + tensors. Mirrors the 13M ablib.py
structure but adapts to the 1B's 16-block stack: per-layer Phase 1, Phase 2a
attention pre-training, tied embedding Phase 2b, PGSU+AMP Phase 3.

The full run requires GPU; tests run on CPU at reduced config.
"""
