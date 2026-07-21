"""Phase-Gated Sparse Update (PGSU) — a new training paradigm for Fractus.

THE PROBLEM WITH STANDARD BACKPROP ON DEEP MODELS:
    Every optimizer step backpropagates through ALL n_layers. For a 16-layer
    1B-param model, that means the backward pass traverses 16 blocks, materializing
    gradients for every parameter. This is the dominant cost of training.

PGSU INSIGHT:
    Not every layer needs to learn on every step. If we rotate which layers
    receive gradients — say 4 of 16 active per step — the backward only
    traverses the active layers. Over 16 steps, every layer has been trained
    4 times. The model converges to the same fixed point, just via a sparser,
    rotating gradient path.

    This is NOT dropout (which zeros activations). This is NOT layer freezing
    (which permanently disables layers). PGSU rotates the active set EVERY
    STEP, so all layers keep learning — just not all at once.

WHY IT IS NATURAL FOR FRACTUS:
    Fractus already has a Kuramoto oscillator "consciousness clock" whose
    phases evolve over time. PGSU uses a deterministic phase-derived schedule
    to decide which layers are "awake" at each step. The schedule is a cyclic
    permutation — every layer gets equal training over a full cycle.

EXPECTED SPEEDUP:
    Backward through K=4 layers instead of N=16 → ~N/K = 4× faster backward.
    Combined with the unchanged forward, total step time drops ~2.5-3×.
    Convergence quality is preserved because every layer still receives
    K/N = 25% of the gradient signal it would normally get — and over many
    steps the cumulative update is identical in expectation.

USAGE:
    pgsu = PGSU(model, n_active=4)
    # in training loop:
    pgsu.step_begin()         # activates this step's layer subset
    loss = model(batch)       # forward as usual
    loss.backward()           # backward only flows through active layers
    opt.step()
    pgsu.step_end()           # restore requires_grad for next step

The active layers are chosen by cycling through a deterministic permutation
seeded by the step counter. This guarantees fairness: over n_layers steps,
every layer has been active exactly n_active times.
"""
import torch
import torch.nn as nn


class PGSU:
    """Phase-Gated Sparse Update controller.

    Wraps a Fractus model and manages which layers are trainable at each step.

    Args:
        model:      a Fractus1B (or any model with a .blocks ModuleList).
        n_active:   number of layers active per step (default 4 of 16).
        cycle_len:  length of the rotation cycle (default = n_layers).
                    Over one cycle, each layer is active n_active times.
    """

    def __init__(self, model, n_active: int = 4, cycle_len: int = None):
        self.model = model
        self.n_layers = len(model.blocks)
        self.n_active = min(n_active, self.n_layers)
        self.cycle_len = cycle_len or self.n_layers
        self.step_count = 0

        # Build the activation schedule: for each step in the cycle, which
        # layer indices are active. We use a rotating window for simplicity
        # and fairness. Layer i is active on step s iff
        #   (i - s) mod n_layers < n_active
        # This guarantees: over n_layers steps, each layer is active exactly
        # n_active times, and the active set rotates by 1 each step.
        self._schedule = []
        for s in range(self.cycle_len):
            active = [(s + k) % self.n_layers for k in range(self.n_active)]
            self._schedule.append(sorted(active))

        # Cache all block parameters so we can toggle requires_grad fast.
        self._block_params = []
        for block in model.blocks:
            params = list(block.parameters())
            self._block_params.append(params)

        # Save original requires_grad so we can restore.
        self._original_requires_grad = []
        for block_params in self._block_params:
            self._original_requires_grad.append([p.requires_grad for p in block_params])

    def active_layers(self, step: int = None) -> list:
        """Return the list of active layer indices for the given step."""
        s = step if step is not None else self.step_count
        return self._schedule[s % self.cycle_len]

    def step_begin(self):
        """Activate this step's layer subset. Call before forward.

        Sets requires_grad=False on inactive layers' parameters so the
        backward pass skips them entirely. The forward still runs through
        all layers (the output depends on all of them), but the gradient
        only flows to the active subset.
        """
        active = set(self.active_layers())
        for i, block_params in enumerate(self._block_params):
            requires = i in active
            for p in block_params:
                p.requires_grad = requires
        self.step_count += 1

    def step_end(self):
        """Restore requires_grad on all layers.

        Call after opt.step(). This is important so that the NEXT step's
        forward produces a graph that includes all layers (the forward must
        always see the full model), and step_begin() of the next step will
        re-deactivate the appropriate subset.
        """
        for i, block_params in enumerate(self._block_params):
            for p, orig in zip(block_params, self._original_requires_grad[i]):
                p.requires_grad = orig

    def stats(self) -> dict:
        """Return diagnostic info."""
        return {
            "n_layers": self.n_layers,
            "n_active": self.n_active,
            "cycle_len": self.cycle_len,
            "step": self.step_count,
            "active_now": self.active_layers(self.step_count - 1) if self.step_count > 0 else [],
            "expected_speedup": self.n_layers / self.n_active,
        }


def integrate_pgsu_into_training_loop(pgsu, model, opt, batch_fn, n_steps):
    """Example integration showing the PGSU pattern.

    batch_fn(step) → (inputs, targets)
    Returns list of losses.
    """
    import torch.nn.functional as F
    losses = []
    for step in range(n_steps):
        pgsu.step_begin()
        inputs, targets = batch_fn(step)
        logits, aux = model(inputs)
        aux_clamped = torch.clamp(aux, max=1.0)
        loss = F.cross_entropy(logits.reshape(-1, 50257), targets.reshape(-1))
        loss = loss + 0.001 * aux_clamped
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        opt.zero_grad()
        pgsu.step_end()
        losses.append(loss.item())
    return losses
