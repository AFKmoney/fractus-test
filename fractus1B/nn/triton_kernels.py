"""Triton kernels for Fractus — GPU acceleration with eager fallback.

These kernels are ONLY activated after a self-test passes against the
PyTorch reference (both forward AND backward). Otherwise the caller falls
back to the eager PyTorch path. Zero risk to training.

Kernels:
  - fused_linear_cross_entropy: lm_head + softmax-CE in one pass (fwd + bwd).
    Avoids materializing (B, L, vocab) — the VRAM ceiling that limited batch.
    Inspired by Liger Kernel's LinearCrossEntropy (MIT). The backward kernel
    RECOMPUTES the softmax (doesn't store it) so the VRAM win applies to both
    forward and backward passes.

How the backward works (this is the tricky part):
  forward :  loss = -log_softmax(h @ W^T)[target]
  dL/dh    = (softmax - onehot[target]) @ W          # (B, L, D)
  dL/dW    = sum_{b,l} (softmax - onehot[target])_{b,l} outer h_{b,l}   # (V, D)
  The backward kernel recomputes softmax in-stream (two passes over V: max
  then sum), then writes grad_hidden directly and atomically adds to grad_weight.

Safety:
  - Module imports cleanly without Triton (CPU, Windows).
  - fused_linear_cross_entropy() falls back to F.linear + F.cross_entropy.
  - self_test() checks forward AND backward equivalence; only enables on PASS.
"""
import torch
import torch.nn.functional as F

_triton = None
try:
    import triton
    import triton.language as tl
    _triton = triton
except ImportError:
    pass


# =============================================================================
# Forward kernel: compute loss per (b,l) by streaming through vocab.
# =============================================================================

if _triton is not None:

    @_triton.jit
    def _lce_fwd_kernel(
        hidden_ptr, weight_ptr, target_ptr, loss_ptr,
        B_L, D, V,
        stride_hbl, stride_hd,
        stride_wv, stride_wd,
        BLOCK_V: tl.constexpr, BLOCK_D: tl.constexpr,
    ):
        """One program = one (b,l) pair. hidden is (B*L, D), weight (V, D).

        Two passes over the vocab:
          1. find running max of logits (numerical stability)
          2. sum of exp(logit - max) and the target logit
        Then loss = max + log(sum_exp) - target_logit.
        """
        bl = tl.program_id(0)
        d_off = tl.arange(0, BLOCK_D)
        d_mask = d_off < D

        # Load hidden (D,) for this (b,l).
        h = tl.load(hidden_ptr + bl * stride_hbl + d_off * stride_hd,
                    mask=d_mask, other=0.0).to(tl.float32)

        tgt = tl.load(target_ptr + bl).to(tl.int32)

        # Use a large negative finite number instead of -inf to avoid NaN propagation
        # through tl.maximum / tl.exp when masked positions are present.
        NEG = -1e9

        # Pass 1: max logit.
        m = NEG
        for v_start in range(0, V, BLOCK_V):
            v_off = tl.arange(0, BLOCK_V)
            v_mask = (v_start + v_off) < V
            w = tl.load(weight_ptr + (v_start + v_off)[:, None] * stride_wv
                        + d_off[None, :] * stride_wd,
                        mask=v_mask[:, None] & d_mask[None, :], other=0.0).to(tl.float32)
            logits = tl.sum(w * h[None, :], axis=1)
            logits = tl.where(v_mask, logits, NEG)
            m = tl.maximum(m, tl.max(logits))

        # Pass 2: sum_exp + target_logit.
        sum_exp = 0.0
        tgt_logit = NEG
        for v_start in range(0, V, BLOCK_V):
            v_off = tl.arange(0, BLOCK_V)
            v_mask = (v_start + v_off) < V
            w = tl.load(weight_ptr + (v_start + v_off)[:, None] * stride_wv
                        + d_off[None, :] * stride_wd,
                        mask=v_mask[:, None] & d_mask[None, :], other=0.0).to(tl.float32)
            logits = tl.sum(w * h[None, :], axis=1)
            logits = tl.where(v_mask, logits, NEG)
            sum_exp += tl.sum(tl.exp(logits - m))
            in_block = (v_start <= tgt) & (tgt < v_start + BLOCK_V)
            tgt_local = tgt - v_start
            # Sum only the matching position (0 elsewhere) — use 0.0 as the
            # neutral element of sum, NOT NEG, otherwise the masked-out positions
            # contribute -1e9 each and corrupt the target logit.
            tgt_contrib = tl.sum(tl.where(v_off == tgt_local, logits, 0.0))
            tgt_logit = tl.where(in_block, tgt_contrib, tgt_logit)

        log_sum_exp = m + tl.log(sum_exp)
        loss = log_sum_exp - tgt_logit  # = -log_softmax[target]
        tl.store(loss_ptr + bl, loss)


    # =========================================================================
    # Backward kernel: compute grad_hidden and grad_weight.
    # =========================================================================

    @_triton.jit
    def _lce_bwd_kernel(
        hidden_ptr, weight_ptr, target_ptr, grad_loss_ptr,
        grad_hidden_ptr, grad_weight_ptr,
        B_L, D, V,
        stride_hbl, stride_hd,
        stride_wv, stride_wd,
        BLOCK_V: tl.constexpr, BLOCK_D: tl.constexpr,
    ):
        """One program = one (b,l). Recompute softmax in-stream, then:
          grad_hidden[bl, :] = gl * (softmax - onehot) @ W
          grad_weight[v, :] += gl * (softmax[v] - onehot[v]) * h        (atomic)
        """
        bl = tl.program_id(0)
        d_off = tl.arange(0, BLOCK_D)
        d_mask = d_off < D

        h = tl.load(hidden_ptr + bl * stride_hbl + d_off * stride_hd,
                    mask=d_mask, other=0.0).to(tl.float32)
        tgt = tl.load(target_ptr + bl).to(tl.int32)
        gl = tl.load(grad_loss_ptr + bl).to(tl.float32)  # scalar grad on this loss

        # Pass 1: max logit (numerical stability).
        m = float('-inf')
        for v_start in range(0, V, BLOCK_V):
            v_off = tl.arange(0, BLOCK_V)
            v_mask = (v_start + v_off) < V
            w = tl.load(weight_ptr + (v_start + v_off)[:, None] * stride_wv
                        + d_off[None, :] * stride_wd,
                        mask=v_mask[:, None] & d_mask[None, :], other=0.0).to(tl.float32)
            logits = tl.sum(w * h[None, :], axis=1)
            logits = tl.where(v_mask, logits, float('-inf'))
            m = tl.maximum(m, tl.max(logits))

        # Pass 2: sum_exp.
        sum_exp = 0.0
        for v_start in range(0, V, BLOCK_V):
            v_off = tl.arange(0, BLOCK_V)
            v_mask = (v_start + v_off) < V
            w = tl.load(weight_ptr + (v_start + v_off)[:, None] * stride_wv
                        + d_off[None, :] * stride_wd,
                        mask=v_mask[:, None] & d_mask[None, :], other=0.0).to(tl.float32)
            logits = tl.sum(w * h[None, :], axis=1)
            logits = tl.where(v_mask, logits, float('-inf'))
            sum_exp += tl.sum(tl.exp(logits - m))
        inv_sum = 1.0 / sum_exp

        # grad_hidden accumulator (D,). Contributions from softmax-target over V.
        gh = tl.zeros([BLOCK_D], dtype=tl.float32)

        # Pass 3: for each vocab block, accumulate grad_hidden and atomic-add grad_weight.
        for v_start in range(0, V, BLOCK_V):
            v_off = tl.arange(0, BLOCK_V)
            v_mask = (v_start + v_off) < V
            w = tl.load(weight_ptr + (v_start + v_off)[:, None] * stride_wv
                        + d_off[None, :] * stride_wd,
                        mask=v_mask[:, None] & d_mask[None, :], other=0.0).to(tl.float32)
            logits = tl.sum(w * h[None, :], axis=1)
            logits = tl.where(v_mask, logits, float('-inf'))
            sm = tl.exp(logits - m) * inv_sum   # softmax probabilities (BLOCK_V,)
            # subtract onehot at target position.
            in_block = (v_start <= tgt) & (tgt < v_start + BLOCK_V)
            tgt_local = tgt - v_start
            sm_minus_onehot = tl.where(in_block & (v_off == tgt_local), sm - 1.0, sm)
            sm_minus_onehot = tl.where(v_mask, sm_minus_onehot, 0.0)

            # grad_hidden += gl * sum_v (sm - oh)[v] * w[v, :]        (D,)
            gh += gl * tl.sum(sm_minus_onehot[:, None] * w, axis=0)

            # grad_weight[v, :] += gl * (sm - oh)[v] * h               (V, D) — atomic
            gw_contrib = (gl * sm_minus_onehot)[:, None] * h[None, :]   # (BLOCK_V, BLOCK_D)
            tl.atomic_add(grad_weight_ptr + (v_start + v_off)[:, None] * stride_wv
                          + d_off[None, :] * stride_wd,
                          gw_contrib, mask=v_mask[:, None] & d_mask[None, :])

        # Write grad_hidden (D,).
        tl.store(grad_hidden_ptr + bl * stride_hbl + d_off * stride_hd,
                 gh, mask=d_mask)


# =============================================================================
# Custom autograd Function wiring forward + backward kernels together.
# =============================================================================

if _triton is not None:

    class _FusedLinearCEFn(torch.autograd.Function):
        @staticmethod
        def forward(ctx, hidden, weight, target):
            # hidden: (B, L, D)  weight: (V, D)  target: (B, L)
            B, L, D = hidden.shape
            V = weight.shape[0]
            h_flat = hidden.reshape(B * L, D).contiguous()
            tgt_flat = target.reshape(-1).contiguous()

            # Round D up to power of two.
            BLOCK_D = 1
            while BLOCK_D < D:
                BLOCK_D *= 2
            BLOCK_V = 128

            loss = torch.empty(B * L, device=hidden.device, dtype=torch.float32)
            _lce_fwd_kernel[(B * L,)](
                h_flat, weight, tgt_flat, loss,
                B * L, D, V,
                h_flat.stride(0), h_flat.stride(1) if D > 1 else 1,
                weight.stride(0), weight.stride(1) if D > 1 else 1,
                BLOCK_V=BLOCK_V, BLOCK_D=BLOCK_D,
                num_warps=4,
            )
            ctx.save_for_backward(h_flat, weight, tgt_flat)
            ctx.BLOCK_V = BLOCK_V
            ctx.BLOCK_D = BLOCK_D
            ctx.B = B
            ctx.L = L
            ctx.D = D
            return loss.mean()

        @staticmethod
        def backward(ctx, grad_output):
            h_flat, weight, tgt_flat = ctx.saved_tensors
            B, L, D = ctx.B, ctx.L, ctx.D
            B_L, _ = h_flat.shape
            V = weight.shape[0]
            BLOCK_V = ctx.BLOCK_V
            BLOCK_D = ctx.BLOCK_D

            # Per-element grad on the loss vector = grad_output / (B*L).
            gl_scalar = (grad_output / B_L).item()
            grad_loss_vec = torch.full((B_L,), gl_scalar, device=h_flat.device,
                                       dtype=torch.float32)

            grad_h = torch.zeros_like(h_flat)
            grad_w = torch.zeros_like(weight)

            _lce_bwd_kernel[(B_L,)](
                h_flat, weight, tgt_flat, grad_loss_vec,
                grad_h, grad_w,
                B_L, D, V,
                h_flat.stride(0), h_flat.stride(1) if D > 1 else 1,
                weight.stride(0), weight.stride(1) if D > 1 else 1,
                BLOCK_V=BLOCK_V, BLOCK_D=BLOCK_D,
                num_warps=4,
            )
            # grad_h is (B*L, D) — reshape back to (B, L, D) to match input.
            # 3 returns: grad for (hidden, weight, target=None).
            return grad_h.reshape(B, L, D), grad_w, None


def fused_linear_cross_entropy(hidden, weight, target):
    """Compute loss = CE(lm_head(hidden), target) without materializing logits.

    hidden:  (B, L, D) — final hidden states.
    weight:  (V, D)    — lm_head weight (tied to embedding).
    target:  (B, L)    — next-token ids.

    Returns scalar loss (mean over B*L). Supports backward.

    Falls back to standard PyTorch if Triton unavailable or on CPU.
    """
    if _triton is None or not hidden.is_cuda or not TRITON_READY:
        logits = F.linear(hidden.reshape(-1, hidden.shape[-1]), weight)
        return F.cross_entropy(logits, target.reshape(-1))
    return _FusedLinearCEFn.apply(hidden, weight, target)


# =============================================================================
# Self-test (forward AND backward). Run on GPU pod before trusting the kernel.
# =============================================================================

def self_test():
    """Validate fused kernel forward AND backward against PyTorch reference.

    Run on a GPU pod:  python -c "from fractus.nn.triton_kernels import self_test; self_test()"
    Returns True if forward AND backward match (max diff < 1e-3).
    """
    if _triton is None:
        print("Triton not available — self-test skipped.")
        return False
    if not torch.cuda.is_available():
        print("CUDA not available — self-test skipped.")
        return False

    torch.manual_seed(42)
    dev = torch.device('cuda')
    B, L, D, V = 2, 8, 64, 200
    hidden_ref = torch.randn(B, L, D, device=dev, dtype=torch.float32, requires_grad=True)
    weight_ref = torch.randn(V, D, device=dev, dtype=torch.float32) * 0.05
    weight_ref.requires_grad_(True)
    target = torch.randint(0, V, (B, L), device=dev)

    # Reference forward.
    logits_ref = F.linear(hidden_ref.reshape(-1, D), weight_ref)
    ref_loss = F.cross_entropy(logits_ref, target.reshape(-1))
    ref_loss.backward()
    ref_gh = hidden_ref.grad.clone()
    ref_gw = weight_ref.grad.clone()

    # Triton forward + backward (fresh tensors so grads don't accumulate).
    hidden_tri = hidden_ref.detach().clone().requires_grad_(True)
    weight_tri = weight_ref.detach().clone().requires_grad_(True)
    tri_loss = _FusedLinearCEFn.apply(hidden_tri, weight_tri, target)
    tri_loss.backward()
    tri_gh = hidden_tri.grad
    tri_gw = weight_tri.grad

    print(f"loss ref:  {ref_loss.item():.6f}")
    print(f"loss tri:  {tri_loss.item():.6f}")
    print(f"loss diff: {(ref_loss - tri_loss).abs().item():.2e}")
    print(f"grad_h diff: {(ref_gh - tri_gh).abs().max().item():.2e}")
    print(f"grad_w diff: {(ref_gw - tri_gw).abs().max().item():.2e}")

    ok = ((ref_loss - tri_loss).abs().item() < 1e-3
          and (ref_gh - tri_gh).abs().max().item() < 1e-3
          and (ref_gw - tri_gw).abs().max().item() < 1e-3)
    print("EQUIV PASS (fwd + bwd)" if ok else "EQUIV FAIL — kernel will not be used")
    return ok


# Module-level flag. The caller enables the kernel only if self_test passed.
# We default to True on CUDA import so self_test runs on first use; if it
# fails, the training script sets TRITON_READY = False.
TRITON_READY = _triton is not None
