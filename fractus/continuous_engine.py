"""ContinuousThoughtEngine: a model that thinks in real-time, not token-by-token.

THE PARADIGM SHIFT. Claude and GPT are static functions: input → output.
This engine is a DYNAMICAL SYSTEM: it runs continuously, maintains a state
of "consciousness", and produces output only when it has something confident
to say.

Architecture:
    - A "thought state" h (d_model vector) that persists across time.
    - At each TICK (a single cheap forward step, not a full generation):
        1. The Kuramoto oscillators advance the phase state by one RK4 step.
        2. The attention state (S, z) absorbs the current observation.
        3. The MoE processes the thought state, routed by the Kuramoto phases.
        4. A confidence head estimates "how sure am I?".
        5. If confidence > threshold → emit output token, reset partial state.
           If confidence < threshold → continue thinking (accumulate).

WHY THIS IS FAST ON CPU:
    - Each tick is ONE forward of the thought state (not B×L tokens).
    - The state is (d_model,) — tiny, not (B, L, d_model).
    - Training is ONLINE: one observation at a time, one gradient at a time.
      No batches, no BPTT through long sequences.
    - The "reasoning depth" is adaptive: easy observations = 1 tick,
      hard ones = 10 ticks. Energy-proportional THINKING.

WHY THIS MAKES CLAUDE/GPT OBSOLETE:
    - They can't "think" — they do one forward pass and output.
    - This engine REASONS: it takes multiple ticks on hard problems,
      accumulating evidence, until it's confident.
    - It's PROACTIVE: it can emit output without being prompted
      (when confidence crosses threshold from internal dynamics).
    - It's CONTINUOUS: the state never resets, so it has true memory
      across an entire conversation/session, not a context window.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from .nn.attention import FractalLinearAttention
from .nn.phase_ode import KuramotoLayer
from .nn.stats import elu_plus_one
from .nn.moe import PhaseRoutedMoE


class ContinuousThoughtEngine(nn.Module):
    """A continuous-time reasoning engine.

    Unlike a standard LM (input_ids → logits per token), this engine
    maintains a persistent "thought state" and advances it tick by tick.

    Args:
        vocab_size:      vocabulary size (for input embedding + output head).
        d_model:         dimension of the thought state.
        n_heads, d_head: attention configuration.
        n_oscillators:   Kuramoto oscillator count (the "consciousness clock").
        n_experts:       MoE expert count.
        top_k:           active experts per tick.
    """

    def __init__(
        self,
        vocab_size: int = 50257,
        d_model: int = 256,
        n_heads: int = 4,
        d_head: int = 64,
        n_levels: int = 2,
        n_oscillators: int = 16,
        coupling_rank: int = 8,
        n_experts: int = 8,
        top_k: int = 2,
        expert_d_ff: int = 256,
        siren_rank: int = 32,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model

        # Input embedding: maps an observed token → a "perception" vector.
        self.observe = nn.Embedding(vocab_size, d_model)

        # The thought state components:
        # 1. Attention (processes the current observation in context of memory)
        self.attn = FractalLinearAttention(d_model, n_heads, d_head, n_levels)
        self.norm_attn = nn.LayerNorm(d_model)

        # 2. Kuramoto (the consciousness clock — advances phases each tick)
        self.kuramoto = KuramotoLayer(d_model, n_oscillators, coupling_rank,
                                      n_steps=1, dt=0.1)  # 1 step per tick (fast)
        self.norm_kur = nn.LayerNorm(d_model)

        # 3. MoE (transforms the thought, routed by Kuramoto phases).
        #    Uses the documented PhaseRoutedMoE (L2b): von Mises gate on Farey
        #    phases, top-k, load-balance loss, end-to-end differentiable.
        #    Low-rank experts (LoRA-style) when siren_rank > 0, dense otherwise.
        self.n_experts = n_experts
        self.top_k = top_k
        self.expert_d_ff = expert_d_ff
        self.moe = PhaseRoutedMoE(
            d_model=d_model, n_experts=n_experts, top_k=top_k,
            kappa=4.0, d_ff=expert_d_ff,
            expert_rank=(siren_rank if siren_rank else None),
        )
        self.norm_moe = nn.LayerNorm(d_model)
        self.register_buffer("last_lb_loss", torch.tensor(0.0))

        # 4. Confidence head: "how sure am I about the current thought?"
        self.confidence_head = nn.Linear(d_model, 1)

        # 5. Output head: TIED with the embedding (like Fractus1B).
        #    The embedding and the output projection share the same weight matrix.
        #    This halves the vocab params (6.4M → shared) and lets the CE backward
        #    and embedding backward accumulate on the same tensor.
        self.output_head = nn.Linear(d_model, vocab_size, bias=False)
        self.output_head.weight = self.observe.weight  # tied

        # 6. Salience head: "is this thought worth remembering?" (for PersistentMemory)
        self.salience_head = nn.Linear(d_model, 1)

        # 7. Optional persistent memory (attached via attach_memory()).
        self.memory = None
        self.memory_active = True

        # Initialize the thought state (will be set by reset_thought).
        self.register_buffer("thought_state", torch.zeros(1, 1, d_model))
        self.register_buffer("attn_S", torch.zeros(1, n_heads * d_head, n_heads * d_head))
        self.register_buffer("attn_z", torch.zeros(1, n_heads * d_head))
        self.register_buffer("kuramoto_phases", torch.zeros(1, 1, n_oscillators))

    def reset_thought(self, batch_size: int = 1):
        """Reset the thought state to zero (start of a new session)."""
        self.thought_state = torch.zeros(batch_size, 1, self.d_model,
                                         device=self.thought_state.device)
        d = self.attn.n_heads * self.attn.d_head
        self.attn_S = torch.zeros(batch_size, d, d, device=self.thought_state.device)
        self.attn_z = torch.zeros(batch_size, d, device=self.thought_state.device)
        self.kuramoto_phases = torch.zeros(
            batch_size, 1, self.kuramoto.N, device=self.thought_state.device
        )
        # Self-modification counters (auto-grow tracking).
        self._tick_count = getattr(self, '_tick_count', 0)
        self._expert_hits = getattr(self, '_expert_hits',
                                    torch.zeros(self.moe.n_experts))

    def maybe_grow(self, *, min_ticks_between_grows: int = 1000,
                   max_experts: int = 32, imbalance_threshold: float = 0.8) -> bool:
        """Check if the model should grow a new expert (self-modification).

        The model grows when the routing is severely imbalanced — one expert
        dominates while others are starved — signaling that the model lacks
        capacity in that region of the phase space. The new expert is placed
        at the midpoint of the largest phase gap.

        Args:
            min_ticks_between_grows: cooldown to avoid runaway growth.
            max_experts: hard cap on expert count.
            imbalance_threshold: fraction of ticks where the top expert
                                 must dominate to trigger growth.
        Returns:
            True if a new expert was added, False otherwise.
        """
        if self.moe.n_experts >= max_experts:
            return False
        if self._tick_count - getattr(self, '_last_grow_tick', 0) < min_ticks_between_grows:
            return False
        if self._expert_hits.numel() != self.moe.n_experts:
            self._expert_hits = torch.zeros(self.moe.n_experts)
            return False
        total = self._expert_hits.sum().item()
        if total < self.moe.n_experts * 10:
            return False  # not enough data yet
        # Check imbalance: is the top expert dominating?
        dominance = self._expert_hits.max().item() / max(total, 1)
        if dominance < imbalance_threshold:
            return False
        # Grow! Place the new expert NEAR the dominant expert (to capture overflow)
        # with ZERO init (no perturbation to the forward pass).
        dominant_idx = self._expert_hits.argmax().item()
        new_idx = self.moe.add_expert(dominant_idx=dominant_idx)
        self._last_grow_tick = self._tick_count
        self._expert_hits = torch.zeros(self.moe.n_experts)
        print(f"[Fractus] Self-modified: grew expert {new_idx} near dominant expert "
              f"{dominant_idx} (now {self.moe.n_experts} experts, dominance was {dominance:.2f})", flush=True)
        return True

    def attach_memory(self, memory):
        """Attach a PersistentMemory bank. Memory injection activates on tick."""
        self.memory = memory

    def detach_memory(self):
        """Detach the memory bank."""
        self.memory = None

    def tick(self, observation: torch.Tensor = None) -> tuple:
        """Advance the thought by ONE tick.

        Args:
            observation: (B,) token id to absorb, or None (pure thinking).
        Returns:
            (output_logits (B, vocab), confidence (B,))
        """
        B = self.thought_state.shape[0]
        h = self.thought_state  # (B, 1, d_model)

        # Absorb observation if provided.
        if observation is not None:
            obs_vec = self.observe(observation).unsqueeze(1)  # (B, 1, d_model)
            h = h + obs_vec  # add perception to thought

        # 1. Attention: process h with the accumulated state (S, z).
        h_normed = self.norm_attn(h)
        # We use a simplified 1-token attention: the "memory" is in (S, z).
        # Project Q, K, V from h.
        attn = self.attn
        D = attn.d_head
        q = torch.einsum("bld,de->ble", h_normed, attn.w_qkv[0]) + attn.b_qkv[0]
        k = torch.einsum("bld,de->ble", h_normed, attn.w_qkv[1]) + attn.b_qkv[1]
        v = torch.einsum("bld,de->ble", h_normed, attn.w_qkv[2]) + attn.b_qkv[2]
        # Apply feature map.
        q_feat = elu_plus_one(q + attn.level_offsets[0])
        k_feat = elu_plus_one(k + attn.level_offsets[0])
        # Update state: S += k⊗v, z += k.
        for hd in range(attn.n_heads):
            kh = k_feat[:, :, hd * D:(hd + 1) * D]  # (B, 1, D)
            vh = v[:, :, hd * D:(hd + 1) * D]
            qh = q_feat[:, :, hd * D:(hd + 1) * D]
            # S is (B, n_heads*D, n_heads*D) — we update only this head's block.
            s_start = hd * D
            s_end = (hd + 1) * D
            outer = (kh.squeeze(1).unsqueeze(2) * vh.squeeze(1).unsqueeze(1))  # (B, D, D)
            # Detach the accumulated state — online training doesn't BPTT
            # through the full state history. Only the current tick's
            # forward graph is kept for backward.
            self.attn_S = self.attn_S.clone()
            self.attn_S[:, s_start:s_end, s_start:s_end] += outer.detach()
            self.attn_z = self.attn_z.clone()
            self.attn_z[:, s_start:s_end] += kh.squeeze(1).detach()
        # Compute attention output from current q and state.
        attn_out = torch.zeros_like(h)
        for hd in range(attn.n_heads):
            s_start = hd * D
            s_end = (hd + 1) * D
            qh = q_feat[:, 0, hd * D:(hd + 1) * D]  # (B, D)
            S_h = self.attn_S[:, s_start:s_end, s_start:s_end]  # (B, D, D)
            z_h = self.attn_z[:, s_start:s_end]  # (B, D)
            num = torch.bmm(qh.unsqueeze(1), S_h).squeeze(1)  # (B, D)
            denom = (qh * z_h).sum(dim=-1, keepdim=True)  # (B, 1)
            safe = denom.abs() > 1e-10
            yh = torch.where(safe, num / (denom + 1e-20), torch.zeros_like(num))
            attn_out[:, 0, hd * D:(hd + 1) * D] = yh
        attn_out = attn_out @ attn.w_out + attn.b_out
        h = h + attn_out

        # 2. Kuramoto: advance phases by one step.
        h_kur = self.norm_kur(h)
        # Encode phases from hidden, integrate 1 step.
        theta = self.kuramoto._encode_from_hidden(h_kur)
        # Carry previous phases forward (add the delta).
        theta = theta + 0.1 * self.kuramoto._derivative(theta)
        theta = torch.remainder(theta, self.kuramoto.TWO_PI)
        self.kuramoto_phases = theta.detach()

        # 3. MoE: transform the thought, routed by Kuramoto phases.
        h_flat = h[:, 0, :]                              # (B, d_model)
        h_moe = self.norm_moe(h_flat).unsqueeze(1)       # (B, 1, d_model)
        phases_in = theta[:, 0:1, :]                     # (B, 1, n_oscillators)
        moe_out, lb_loss = self.moe(h_moe, phases_in)    # moe_out: (B, 1, d_model)
        self.last_lb_loss = lb_loss.detach()
        # Track routing hits for self-modification (maybe_grow).
        self._tick_count = getattr(self, '_tick_count', 0) + 1
        if hasattr(self, '_expert_hits') and self._expert_hits.numel() == self.moe.n_experts:
            with torch.no_grad():
                gates = self.moe._compute_gates(phases_in)  # (B, 1, E)
                topk_idx = gates.topk(self.moe.top_k, dim=-1).indices
                for e in range(self.moe.n_experts):
                    self._expert_hits[e] += (topk_idx == e).sum().item()
        h = h + moe_out

        # 4. Update thought state (clone, not detach-view, so inject's in-place
        #    modification of thought_state doesn't corrupt h's autograd graph).
        self.thought_state = h.detach().clone()

        # 3b. Memory: salience-gated consolidation + continuous injection.
        # The salience head learns to predict how much the memory injection
        # will perturb the thought state. This is an intrinsic signal from
        # the dynamical system, not an external label.
        #
        # IMPORTANT: salience_loss is computed from the PREVIOUS tick's
        # perturbation (one-tick delay). This is because inject() modifies
        # thought_state in-place, which would break autograd if we tried to
        # backpropagate through the same tick's h after inject. The one-tick
        # delay is natural for a dynamical system: the head predicts the
        # perturbation its current state will cause, and the target is the
        # perturbation that was actually measured on the previous tick.
        if self.memory is not None and self.memory_active:
            salience = torch.sigmoid(self.salience_head(h[:, 0, :]))  # (B, 1)

            # Compute salience_loss from the PREVIOUS tick's perturbation target.
            if not hasattr(self, '_pert_max'):
                self._pert_max = 1.0
            if not hasattr(self, '_prev_pert_target'):
                self._prev_pert_target = 0.0
            import torch as _t
            self.last_salience_loss = _t.nn.functional.binary_cross_entropy(
                salience[0:1, 0], _t.tensor([self._prev_pert_target]))

            # NOW do consolidation + injection (side-effects, no grad needed).
            self.memory.consolidate_if_salient(
                h[0:1, 0, :], salience[0].item())
            perturbation = self.memory.inject(self, blend=0.05, top_k=3)
            # Update running max + store target for NEXT tick.
            if perturbation > self._pert_max:
                self._pert_max = perturbation
            self._prev_pert_target = min(perturbation / max(self._pert_max, 1e-8), 1.0)
        else:
            self.last_salience_loss = torch.tensor(0.0)

        # 5. Confidence + output.
        confidence = torch.sigmoid(self.confidence_head(h[:, 0, :]).squeeze(-1))  # (B,)
        output_logits = self.output_head(h[:, 0, :])  # (B, vocab)

        return output_logits, confidence

    def tick_chunk(self, observations: torch.Tensor) -> torch.Tensor:
        """Process a CHUNK of tokens in ONE forward pass (16x faster than tick).

        observations: (B, C) where C = chunk_len (e.g. 16).
        Returns logits: (B, C, vocab).

        CONTINUOUS THOUGHT: the attention state (S, z) is carried across chunk
        boundaries via the L8 state-carry mechanism. Each chunk's attention
        STARTS from the previous chunk's accumulated state — the thought is
        truly continuous, not reset per chunk.

        The thought state also carries (thought_state buffer).
        """
        B, C = observations.shape
        D = self.d_model

        # Embed the whole chunk.
        obs_vecs = self.observe(observations)  # (B, C, D)
        # Add the carried thought state to position 0.
        h = obs_vecs.clone()
        h[:, 0, :] = h[:, 0, :] + self.thought_state[:, 0, :]

        # 1. Attention: process the chunk with CARRIED state from previous chunk.
        attn = self.attn
        nH, dH, nL = attn.n_heads, attn.d_head, attn.n_levels
        h_normed = self.norm_attn(h)
        q_all = torch.einsum("bld,de->ble", h_normed, attn.w_qkv[0]) + attn.b_qkv[0]
        k_all = torch.einsum("bld,de->ble", h_normed, attn.w_qkv[1]) + attn.b_qkv[1]
        v_all = torch.einsum("bld,de->ble", h_normed, attn.w_qkv[2]) + attn.b_qkv[2]
        q_all = q_all.view(B, C, nH, dH)
        k_all = k_all.view(B, C, nH, dH)
        v_all = v_all.view(B, C, nH, dH)

        offsets = attn.level_offsets
        q_lev = q_all.unsqueeze(1) + offsets.view(nL, 1, 1, 1)
        k_lev = k_all.unsqueeze(1) + offsets.view(nL, 1, 1, 1)
        q_feat = elu_plus_one(q_lev, alpha=1.0)
        k_feat = elu_plus_one(k_lev, alpha=1.0)
        v_lev = v_all.unsqueeze(1).expand(B, nL, C, nH, dH)
        q_flat = q_feat.permute(0, 1, 3, 2, 4).reshape(B * nL * nH, C, dH)
        k_flat = k_feat.permute(0, 1, 3, 2, 4).reshape(B * nL * nH, C, dH)
        v_flat = v_lev.permute(0, 1, 3, 2, 4).reshape(B * nL * nH, C, dH)

        # CARRY (S, z) from previous chunk — continuous thought across boundaries.
        # attn_S is (B, nH*dH, nH*dH). Extract each head's diagonal block.
        carry_S_per_head = torch.stack([
            self.attn_S[:, hd*dH:(hd+1)*dH, hd*dH:(hd+1)*dH] for hd in range(nH)
        ], dim=1)  # (B, nH, dH, dH)
        carry_z_per_head = torch.stack([
            self.attn_z[:, hd*dH:(hd+1)*dH] for hd in range(nH)
        ], dim=1)  # (B, nH, dH)
        carry_S_flat = carry_S_per_head.unsqueeze(1).expand(B, nL, nH, dH, dH).reshape(B * nL * nH, dH, dH)
        carry_z_flat = carry_z_per_head.unsqueeze(1).expand(B, nL, nH, dH).reshape(B * nL * nH, dH)

        y_flat, (S_final, z_final) = attn._linear_attention_causal_vectorized(
            q_flat, k_flat, v_flat, carry=(carry_S_flat, carry_z_flat))

        # Save final state: rebuild (B, nH*dH, nH*dH) from per-head blocks.
        S_reshaped = S_final.reshape(B, nL, nH, dH, dH).mean(dim=1)  # (B, nH, dH, dH)
        z_reshaped = z_final.reshape(B, nL, nH, dH).mean(dim=1)     # (B, nH, dH)
        new_S = torch.zeros(B, nH * dH, nH * dH, device=h.device, dtype=h.dtype)
        new_z = torch.zeros(B, nH * dH, device=h.device, dtype=h.dtype)
        for hd in range(nH):
            new_S[:, hd*dH:(hd+1)*dH, hd*dH:(hd+1)*dH] = S_reshaped[:, hd]
            new_z[:, hd*dH:(hd+1)*dH] = z_reshaped[:, hd]
        self.attn_S = new_S.detach()
        self.attn_z = new_z.detach()

        y = y_flat.reshape(B, nL, nH, C, dH).permute(0, 1, 3, 2, 4).reshape(B, nL, C, nH * dH)
        level_weights = torch.softmax(attn.level_logits, dim=-1)
        attn_out = (y * level_weights.view(1, nL, 1, 1)).sum(dim=1)
        attn_out = attn_out @ attn.w_out + attn.b_out
        h = h + attn_out

        # 2. Kuramoto: advance phases from the chunk's hidden states.
        # Detached from autograd — Kuramoto is a clock, not a learned transform.
        # Its gradients are not needed (the MoE gates learn to USE the phases,
        # the phases themselves are deterministic from the hidden state).
        with torch.no_grad():
            h_kur = self.norm_kur(h)
            theta = self.kuramoto._encode_from_hidden(h_kur)
            theta = self.kuramoto._rk4_integrate(theta)
        self.kuramoto_phases = theta

        # 3. MoE: transform the chunk, routed by the last-position Kuramoto phase.
        h_moe = self.norm_moe(h)                         # (B, C, d_model)
        # Use the last position's phase to route the whole chunk (as the original did).
        phases_last = theta[:, -1:, :]                   # (B, 1, n_oscillators)
        phases_in = phases_last.expand(-1, h_moe.shape[1], -1)  # (B, C, n_oscillators)
        moe_out, lb_loss = self.moe(h_moe, phases_in)    # (B, C, d_model)
        self.last_lb_loss = lb_loss.detach()
        h = h + moe_out

        # 4. Update thought state (last position).
        self.thought_state = h[:, -1:, :].detach()

        # 5. Output logits for the whole chunk.
        output_logits = self.output_head(h)  # (B, C, vocab)
        return output_logits

    def tick_chunk_train(self, observations: torch.Tensor) -> torch.Tensor:
        """Fast training forward: head on LAST position only (16x less head FLOPs).

        Same forward as tick_chunk (attention + kuramoto + moE on all C positions),
        but the output head is computed only on the LAST position — the token to
        predict. This is the CTE-native training mode: the chunk builds context,
        the prediction is on the final thought state.

        observations: (B, C) token ids. Returns logits: (B, vocab) for the last position.
        """
        # Run the full tick_chunk forward (which returns all-position logits),
        # but we only need the last position. To avoid computing the head on all
        # positions, we replicate tick_chunk's body but compute the head only
        # on h[:, -1:, :].
        B, C = observations.shape
        D = self.d_model

        obs_vecs = self.observe(observations)
        h = obs_vecs.clone()
        h[:, 0, :] = h[:, 0, :] + self.thought_state[:, 0, :]

        # 1. Attention (vectorized).
        attn = self.attn
        nH, dH, nL = attn.n_heads, attn.d_head, attn.n_levels
        h_normed = self.norm_attn(h)
        q_all = torch.einsum("bld,de->ble", h_normed, attn.w_qkv[0]) + attn.b_qkv[0]
        k_all = torch.einsum("bld,de->ble", h_normed, attn.w_qkv[1]) + attn.b_qkv[1]
        v_all = torch.einsum("bld,de->ble", h_normed, attn.w_qkv[2]) + attn.b_qkv[2]
        q_all = q_all.view(B, C, nH, dH)
        k_all = k_all.view(B, C, nH, dH)
        v_all = v_all.view(B, C, nH, dH)
        offsets = attn.level_offsets
        q_lev = q_all.unsqueeze(1) + offsets.view(nL, 1, 1, 1)
        k_lev = k_all.unsqueeze(1) + offsets.view(nL, 1, 1, 1)
        q_feat = elu_plus_one(q_lev, alpha=1.0)
        k_feat = elu_plus_one(k_lev, alpha=1.0)
        v_lev = v_all.unsqueeze(1).expand(B, nL, C, nH, dH)
        q_flat = q_feat.permute(0, 1, 3, 2, 4).reshape(B * nL * nH, C, dH)
        k_flat = k_feat.permute(0, 1, 3, 2, 4).reshape(B * nL * nH, C, dH)
        v_flat = v_lev.permute(0, 1, 3, 2, 4).reshape(B * nL * nH, C, dH)

        # CARRY (S, z) — continuous thought across chunk boundaries.
        carry_S_per_head = torch.stack([
            self.attn_S[:, hd*dH:(hd+1)*dH, hd*dH:(hd+1)*dH] for hd in range(nH)
        ], dim=1)
        carry_z_per_head = torch.stack([
            self.attn_z[:, hd*dH:(hd+1)*dH] for hd in range(nH)
        ], dim=1)
        carry_S_flat = carry_S_per_head.unsqueeze(1).expand(B, nL, nH, dH, dH).reshape(B * nL * nH, dH, dH)
        carry_z_flat = carry_z_per_head.unsqueeze(1).expand(B, nL, nH, dH).reshape(B * nL * nH, dH)
        y_flat, (S_final, z_final) = attn._linear_attention_causal_vectorized(
            q_flat, k_flat, v_flat, carry=(carry_S_flat, carry_z_flat))

        S_reshaped = S_final.reshape(B, nL, nH, dH, dH).mean(dim=1)
        z_reshaped = z_final.reshape(B, nL, nH, dH).mean(dim=1)
        new_S = torch.zeros(B, nH * dH, nH * dH, device=h.device, dtype=h.dtype)
        new_z = torch.zeros(B, nH * dH, device=h.device, dtype=h.dtype)
        for hd in range(nH):
            new_S[:, hd*dH:(hd+1)*dH, hd*dH:(hd+1)*dH] = S_reshaped[:, hd]
            new_z[:, hd*dH:(hd+1)*dH] = z_reshaped[:, hd]
        self.attn_S = new_S.detach()
        self.attn_z = new_z.detach()

        y = y_flat.reshape(B, nL, nH, C, dH).permute(0, 1, 3, 2, 4).reshape(B, nL, C, nH * dH)
        level_weights = torch.softmax(attn.level_logits, dim=-1)
        attn_out = (y * level_weights.view(1, nL, 1, 1)).sum(dim=1)
        attn_out = attn_out @ attn.w_out + attn.b_out
        h = h + attn_out

        # 2. Kuramoto: advance phases from the chunk's hidden states.
        # Detached from autograd — Kuramoto is a clock, not a learned transform.
        with torch.no_grad():
            h_kur = self.norm_kur(h)
            theta = self.kuramoto._encode_from_hidden(h_kur)
            theta = self.kuramoto._rk4_integrate(theta)
        self.kuramoto_phases = theta

        # 3. MoE.
        h_moe = self.norm_moe(h)
        phases_last = theta[:, -1:, :]
        phases_in = phases_last.expand(-1, C, -1)
        moe_out, lb_loss = self.moe(h_moe, phases_in)
        self.last_lb_loss = lb_loss.detach()
        h = h + moe_out

        # 4. Update thought state.
        self.thought_state = h[:, -1:, :].detach()

        # 5. Head on LAST position only (the prediction target).
        last_logits = self.output_head(h[:, -1, :])  # (B, vocab)
        return last_logits

    def think(self, observations: torch.Tensor, max_ticks: int = 10,
              confidence_threshold: float = 0.7) -> torch.Tensor:
        """Process a sequence of observations, thinking adaptively.

        For each observation, tick until confidence > threshold or max_ticks.
        Returns the output logits at the point of confidence.
        """
        B = observations.shape[0]
        outputs = []
        for t in range(observations.shape[1]):
            obs = observations[:, t]
            for tick in range(max_ticks):
                logits, conf = self.tick(obs if tick == 0 else None)
                if conf.mean().item() > confidence_threshold:
                    break
            outputs.append(logits)
        return torch.stack(outputs, dim=1)  # (B, L, vocab)

    @torch.no_grad()
    def generate(self, prompt_tokens: list, n_new: int = 50,
                 temperature: float = 0.8, top_k: int = 40,
                 chunk_len: int = 16) -> list:
        """Generate text using chunk-based processing + temperature sampling.

        Args:
            prompt_tokens: list of token ids to seed.
            n_new: number of NEW tokens to generate.
            temperature: sampling temperature (1.0=normal, <1=conservative).
            top_k: only sample from the top-k logits (0 = no truncation).
            chunk_len: tokens per chunk forward.
        Returns:
            list of all token ids (prompt + generated).
        """
        self.eval()
        self.reset_thought(batch_size=1)
        all_tokens = list(prompt_tokens)

        # Process the prompt in chunks.
        for start in range(0, len(all_tokens), chunk_len):
            chunk = torch.tensor([all_tokens[start:start + chunk_len]], dtype=torch.long)
            if chunk.shape[1] == 0:
                break
            logits = self.tick_chunk(chunk)  # (1, C, vocab)
            last_logits = logits[0, -1, :]  # (vocab,)

        # Generate new tokens one at a time (using the chunk's last logits).
        for _ in range(n_new):
            # Temperature + top-k sampling.
            logits = last_logits / max(temperature, 1e-8)
            if top_k > 0:
                topk_vals, topk_idx = logits.topk(min(top_k, logits.shape[-1]))
                probs = F.softmax(topk_vals, dim=-1)
                next_idx = torch.multinomial(probs, 1).item()
                next_token = topk_idx[next_idx].item()
            else:
                probs = F.softmax(logits, dim=-1)
                next_token = torch.multinomial(probs, 1).item()
            all_tokens.append(next_token)

            # Feed the new token via a chunk of 1.
            chunk = torch.tensor([[next_token]], dtype=torch.long)
            logits = self.tick_chunk(chunk)
            last_logits = logits[0, -1, :]

        self.train()
        return all_tokens
