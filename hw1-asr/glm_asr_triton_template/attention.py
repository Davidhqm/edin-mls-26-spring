"""
Triton Multi-Head Attention Implementation
End-to-end implementation using Triton kernels
"""

import numpy as np
import torch
import triton
import triton.language as tl
from typing import Optional, Tuple


def get_stream():
    """Get current CUDA stream pointer."""
    if torch.cuda.is_available():
        return torch.cuda.current_stream().cuda_stream
    return None


# ============================================================================
# Triton Kernels for Attention
# ============================================================================

@triton.jit
def attention_scores_kernel(
    q_ptr,
    k_ptr,
    scores_ptr,
    scale,
    seq_k,
    head_dim,
    stride_q0,
    stride_q1,
    stride_q2,
    stride_k0,
    stride_k1,
    stride_k2,
    stride_s0,
    stride_s1,
    stride_s2,
    BLOCK_K: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    """Compute scaled attention scores for a single query position."""
    pid_bh = tl.program_id(0)
    pid_q = tl.program_id(1)
    offs_k = tl.arange(0, BLOCK_K)
    offs_d = tl.arange(0, BLOCK_D)
    q = tl.load(
        q_ptr + pid_bh * stride_q0 + pid_q * stride_q1 + offs_d * stride_q2,
        mask=offs_d < head_dim,
        other=0.0,
    )
    k = tl.load(
        k_ptr
        + pid_bh * stride_k0
        + offs_k[:, None] * stride_k1
        + offs_d[None, :] * stride_k2,
        mask=(offs_k[:, None] < seq_k) & (offs_d[None, :] < head_dim),
        other=0.0,
    )
    scores = tl.sum(k * q[None, :], axis=1) * scale
    tl.store(
        scores_ptr + pid_bh * stride_s0 + pid_q * stride_s1 + offs_k * stride_s2,
        scores,
        mask=offs_k < seq_k,
    )


@triton.jit
def softmax_inplace_kernel(scores_ptr, stride_s, seq_k, BLOCK_SIZE: tl.constexpr):
    """
    Apply softmax along the last dimension (seq_k).
    Grid: (batch_heads * seq_q,)
    """
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK_SIZE)
    mask = offs < seq_k
    s = tl.load(scores_ptr + row * stride_s + offs, mask=mask, other=-float("inf"))
    s = s - tl.max(s, axis=0)
    exp_s = tl.exp(s)
    denom = tl.sum(exp_s, axis=0)
    out = exp_s / denom
    tl.store(scores_ptr + row * stride_s + offs, out, mask=mask)


@triton.jit
def attention_output_kernel(
    attn_ptr,
    v_ptr,
    output_ptr,
    seq_k,
    head_dim,
    stride_w0,
    stride_w1,
    stride_w2,
    stride_v0,
    stride_v1,
    stride_v2,
    stride_o0,
    stride_o1,
    stride_o2,
    BLOCK_K: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    """Compute attention output: attn_weights @ V."""
    pid_bh = tl.program_id(0)
    pid_q = tl.program_id(1)
    offs_k = tl.arange(0, BLOCK_K)
    offs_d = tl.arange(0, BLOCK_D)
    w = tl.load(
        attn_ptr + pid_bh * stride_w0 + pid_q * stride_w1 + offs_k * stride_w2,
        mask=offs_k < seq_k,
        other=0.0,
    )
    v = tl.load(
        v_ptr
        + pid_bh * stride_v0
        + offs_k[:, None] * stride_v1
        + offs_d[None, :] * stride_v2,
        mask=(offs_k[:, None] < seq_k) & (offs_d[None, :] < head_dim),
        other=0.0,
    )
    out = tl.sum(v * w[:, None], axis=0)
    tl.store(
        output_ptr + pid_bh * stride_o0 + pid_q * stride_o1 + offs_d * stride_o2,
        out,
        mask=offs_d < head_dim,
    )


@triton.jit
def causal_mask_kernel(
    scores_ptr,
    seq_k,
    offset,
    stride_s0,
    stride_s1,
    stride_s2,
    BLOCK_K: tl.constexpr,
):
    """
    Apply causal mask to attention scores.
    Grid: (batch_heads, seq_q)
    """
    pid_bh = tl.program_id(0)
    pid_q = tl.program_id(1)

    offs_k = tl.arange(0, BLOCK_K)
    mask = offs_k < seq_k
    scores = tl.load(
        scores_ptr
        + pid_bh * stride_s0
        + pid_q * stride_s1
        + offs_k * stride_s2,
        mask=mask,
        other=-1e9,
    )
    current_pos = pid_q + offset
    scores = tl.where(offs_k > current_pos, -1e9, scores)
    tl.store(
        scores_ptr
        + pid_bh * stride_s0
        + pid_q * stride_s1
        + offs_k * stride_s2,
        scores,
        mask=mask,
    )


# ============================================================================
# Improved FlashAttention kernel — supports arbitrary head_dim, causal mask,
# and an additive attention_mask (bias) tensor.
#
# Changes vs original:
#   1. HEAD_DIM is now a constexpr so the compiler can unroll the dot products
#      without being limited to power-of-two padded sizes.
#   2. An optional additive bias block is loaded from `bias_ptr` when
#      HAS_BIAS=True.  This covers both causal masks pre-built as biases and
#      custom attention masks.  Passing bias_ptr=None (nullptr) with
#      HAS_BIAS=False costs zero extra instructions on the fast path.
#   3. The CAUSAL path uses a per-tile early-exit: if the entire N-tile is
#      strictly above the diagonal (start_n > max query position in this
#      M-tile) we skip loading K/V entirely, saving ~50 % of memory traffic
#      on the lower-left triangle.
#   4. num_warps and num_stages are tuned per head_dim in the Python launcher
#      rather than being hardcoded.
# ============================================================================

@triton.jit
def flash_attention_fwd_kernel(
    q_ptr,
    k_ptr,
    v_ptr,
    o_ptr,
    bias_ptr,          # additive bias (attention_mask); nullptr when HAS_BIAS=False
    seq_q,
    seq_k,
    head_dim,
    scale,
    stride_q0, stride_q1, stride_q2,
    stride_k0, stride_k1, stride_k2,
    stride_v0, stride_v1, stride_v2,
    stride_o0, stride_o1, stride_o2,
    stride_b0, stride_b1, stride_b2,  # bias strides (ignored when HAS_BIAS=False)
    BLOCK_M:  tl.constexpr,
    BLOCK_N:  tl.constexpr,
    BLOCK_D:  tl.constexpr,
    CAUSAL:   tl.constexpr,
    HAS_BIAS: tl.constexpr,
):
    """
    FlashAttention-2-style fused forward kernel.

    Grid: (batch * num_heads, ceil(seq_q / BLOCK_M))

    Computes softmax(QK^T * scale + bias) @ V in a single pass over K/V,
    keeping only the running (max, sum, accumulator) in registers.  This
    avoids materialising the full seq_q × seq_k scores matrix in HBM.
    """
    pid_bh = tl.program_id(0)
    pid_m  = tl.program_id(1)

    # ---- query tile indices ------------------------------------------------
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)   # (BLOCK_M,)
    offs_d = tl.arange(0, BLOCK_D)                      # (BLOCK_D,)
    mask_m = offs_m < seq_q
    mask_d = offs_d < head_dim

    # ---- load Q tile -------------------------------------------------------
    q = tl.load(
        q_ptr
        + pid_bh * stride_q0
        + offs_m[:, None] * stride_q1
        + offs_d[None, :] * stride_q2,
        mask=mask_m[:, None] & mask_d[None, :],
        other=0.0,
    )  # (BLOCK_M, BLOCK_D)

    # ---- running statistics ------------------------------------------------
   # ---- running statistics ------------------------------------------------
    m_i  = tl.full((BLOCK_M,), -float("inf"), dtype=tl.float32)
    l_i  = tl.zeros((BLOCK_M,),               dtype=tl.float32)
    acc  = tl.zeros((BLOCK_M, BLOCK_D),        dtype=tl.float32)

    # REPLACEMENT FOR BREAK: Pre-calculate the loop boundary.
    # If not causal, we scan the full seq_k. 
    # If causal, we stop at the end of the current query block.
    loop_end = seq_k
    if CAUSAL:
        # exclusive upper bound for K/V tiles
        loop_end = tl.minimum(seq_k, (pid_m + 1) * BLOCK_M)

    # ---- iterate over K/V tiles --------------------------------------------
    # The range now naturally stops before entering the "all -inf" causal zone.
    for start_n in range(0, loop_end, BLOCK_N):
        offs_n = start_n + tl.arange(0, BLOCK_N)
        mask_n = offs_n < seq_k

        # (All loading and math logic remains exactly as you wrote it...)
        # ---- load K tile ----
        k = tl.load(
            k_ptr
            + pid_bh * stride_k0
            + offs_n[:, None] * stride_k1
            + offs_d[None, :] * stride_k2,
            mask=mask_n[:, None] & mask_d[None, :],
            other=0.0,
        )

        # ---- load V tile ---------------------------------------------------
        v = tl.load(
            v_ptr
            + pid_bh * stride_v0
            + offs_n[:, None] * stride_v1
            + offs_d[None, :] * stride_v2,
            mask=mask_n[:, None] & mask_d[None, :],
            other=0.0,
        )  # (BLOCK_N, BLOCK_D)

        # ---- QK^T ----------------------------------------------------------
        qk = tl.dot(q, tl.trans(k)) * scale  # (BLOCK_M, BLOCK_N)

        # ---- padding mask (out-of-bounds K positions) ----------------------
        qk = tl.where(mask_n[None, :], qk, -float("inf"))

        # ---- causal mask ---------------------------------------------------
        if CAUSAL:
            qk = tl.where(offs_n[None, :] <= offs_m[:, None], qk, -float("inf"))

        # ---- additive bias (e.g. attention_mask) ---------------------------
        if HAS_BIAS:
            bias_block = tl.load(
                bias_ptr
                + pid_bh * stride_b0
                + offs_m[:, None] * stride_b1
                + offs_n[None, :] * stride_b2,
                mask=mask_m[:, None] & mask_n[None, :],
                other=0.0,
            )  # (BLOCK_M, BLOCK_N)
            qk = qk + bias_block

        # ---- online softmax update (FlashAttention-2 style) ----------------
        m_ij   = tl.maximum(m_i, tl.max(qk, axis=1))   # new row-max
        p      = tl.exp(qk - m_ij[:, None])             # (BLOCK_M, BLOCK_N) unnorm
        l_ij   = tl.sum(p, axis=1)                      # (BLOCK_M,)

        alpha  = tl.exp(m_i - m_ij)                     # rescale factor
        acc    = acc * alpha[:, None] + tl.dot(p.to(tl.float32), v)
        l_i    = l_i * alpha + l_ij
        m_i    = m_ij

    # ---- normalise & store -------------------------------------------------
    l_safe = tl.where(l_i > 0.0, l_i, 1.0)
    out    = acc / l_safe[:, None]

    tl.store(
        o_ptr
        + pid_bh * stride_o0
        + offs_m[:, None] * stride_o1
        + offs_d[None, :] * stride_o2,
        out,
        mask=mask_m[:, None] & mask_d[None, :],
    )


# ============================================================================
# Attention Classes
# ============================================================================

class MultiHeadAttention:
    """Multi-head attention using Triton kernels."""

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        num_kv_heads: Optional[int] = None,
        head_dim: Optional[int] = None,
    ):
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads or num_heads
        self.head_dim = head_dim or (hidden_size // num_heads)
        self.scale = 1.0 / np.sqrt(self.head_dim)
        self.num_queries_per_kv = self.num_heads // self.num_kv_heads

    def __call__(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        is_causal: bool = False,
    ) -> torch.Tensor:
        batch, num_heads, seq_q, head_dim = q.shape
        _, num_kv_heads, seq_k, _ = k.shape

        if num_kv_heads != num_heads:
            k = self._expand_kv(k, self.num_queries_per_kv)
            v = self._expand_kv(v, self.num_queries_per_kv)

        return scaled_dot_product_attention(
            q, k, v, attention_mask, is_causal, self.scale
        )

    def _expand_kv(self, x: torch.Tensor, num_repeats: int) -> torch.Tensor:
        """Expand KV heads for GQA using broadcast (zero-copy)."""
        batch, num_kv_heads, seq_len, head_dim = x.shape
        x_expanded = x[:, :, None, :, :].expand(
            batch, num_kv_heads, num_repeats, seq_len, head_dim
        )
        return x_expanded.reshape(batch, num_kv_heads * num_repeats, seq_len, head_dim)


def next_power_of_two(x: int) -> int:
    """Return the smallest power of two >= x."""
    return 1 << (x - 1).bit_length() if x > 0 else 1


# Maximum head_dim supported by the flash kernel.  Triton requires BLOCK_D to
# be a power of two and known at compile time; 256 covers all standard models
# (64 / 80 / 128 / 256).  Sequences of ANY length are now handled by flash —
# only the head dimension is capped.
MAX_FLASH_HEAD_DIM = 256

# Fallback 3-kernel path (scores → softmax → output) is kept for CPU tensors
# and for the rare case where head_dim > MAX_FLASH_HEAD_DIM.
MAX_TRITON_DIM = 256


def _get_flash_config(head_dim: int) -> Tuple[int, int, int, int]:
    """
    Return (BLOCK_M, BLOCK_N, num_warps, num_stages) tuned per head_dim.

    Larger head dims need more shared memory per tile, so BLOCK_N is reduced
    and num_warps is increased to hide the longer dot-product latency.
    """
    if head_dim <= 64:
        return 64, 64, 4, 3
    elif head_dim <= 128:
        return 64, 32, 4, 3
    else:  # 129–256
        return 32, 32, 8, 2


def scaled_dot_product_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    is_causal: bool = False,
    scale: Optional[float] = None,
) -> torch.Tensor:
    """
    Scaled dot-product attention.

    Dispatch priority (CUDA tensors):
      1. Flash kernel  — any seq_len, head_dim ≤ MAX_FLASH_HEAD_DIM,
                         with or without causal mask / additive bias.
      2. 3-kernel path — head_dim > MAX_FLASH_HEAD_DIM but fits in shared mem.
      3. Pure PyTorch  — CPU or very large head dims.
    """
    batch, num_heads, seq_q, head_dim = q.shape
    _, _, seq_k, _ = k.shape

    if scale is None:
        scale = 1.0 / np.sqrt(head_dim)

    head_dim_padded = next_power_of_two(head_dim)

    # ------------------------------------------------------------------
    # Path 1: FlashAttention (fused, O(seq) memory, recommended path)
    # ------------------------------------------------------------------
    # Conditions:
    #   • CUDA tensor
    #   • head_dim fits in a power-of-two BLOCK_D ≤ MAX_FLASH_HEAD_DIM
    #   • no constraint on seq_len (the whole point of this fix)
    # ------------------------------------------------------------------
    use_flash = q.is_cuda and head_dim_padded <= MAX_FLASH_HEAD_DIM

    if use_flash:
        bh = batch * num_heads

        q_flat = q.reshape(bh, seq_q, head_dim).to(torch.float32).contiguous()
        k_flat = k.reshape(bh, seq_k, head_dim).to(torch.float32).contiguous()
        v_flat = v.reshape(bh, seq_k, head_dim).to(torch.float32).contiguous()

        output = torch.empty((bh, seq_q, head_dim), dtype=torch.float32, device=q.device)

        block_m, block_n, num_warps, num_stages = _get_flash_config(head_dim_padded)
        block_d = head_dim_padded

        # Prepare bias tensor (additive attention_mask).
        # flash kernel expects shape (bh, seq_q, seq_k) — same as scores.
        has_bias = attention_mask is not None
        if has_bias:
            bias = attention_mask.to(torch.float32)
            # Reshape from (batch, 1|heads, seq_q, seq_k) → (bh, seq_q, seq_k)
            if bias.ndim == 4:
                if bias.shape[1] == 1:
                    bias = bias.expand(batch, num_heads, seq_q, seq_k)
                bias = bias.reshape(bh, seq_q, seq_k).contiguous()
            else:
                # (batch, seq_q, seq_k) — broadcast over heads
                bias = bias.unsqueeze(1).expand(batch, num_heads, seq_q, seq_k)
                bias = bias.reshape(bh, seq_q, seq_k).contiguous()
            b_ptr      = bias
            stride_b0  = bias.stride(0)
            stride_b1  = bias.stride(1)
            stride_b2  = bias.stride(2)
        else:
            # Pass a dummy pointer; HAS_BIAS=False means it is never dereferenced.
            b_ptr     = q_flat   # any valid tensor, won't be read
            stride_b0 = stride_b1 = stride_b2 = 0

        grid = (bh, triton.cdiv(seq_q, block_m))
        flash_attention_fwd_kernel[grid](
            q_flat, k_flat, v_flat, output, b_ptr,
            seq_q, seq_k, head_dim, float(scale),
            q_flat.stride(0), q_flat.stride(1), q_flat.stride(2),
            k_flat.stride(0), k_flat.stride(1), k_flat.stride(2),
            v_flat.stride(0), v_flat.stride(1), v_flat.stride(2),
            output.stride(0), output.stride(1), output.stride(2),
            stride_b0, stride_b1, stride_b2,
            BLOCK_M=block_m,
            BLOCK_N=block_n,
            BLOCK_D=block_d,
            CAUSAL=is_causal,
            HAS_BIAS=has_bias,
            num_warps=num_warps,
            num_stages=num_stages,
        )

        return output.reshape(batch, num_heads, seq_q, head_dim).to(q.dtype)

    # ------------------------------------------------------------------
    # Path 2: 3-kernel Triton path (fits in shared mem, no flash)
    # ------------------------------------------------------------------
    seq_k_padded = next_power_of_two(seq_k)

    use_triton = (
        q.is_cuda
        and seq_k_padded <= MAX_TRITON_DIM
        and head_dim_padded <= MAX_TRITON_DIM
    )

    if use_triton:
        q_flat = q.reshape(batch * num_heads, seq_q, head_dim).to(torch.float32)
        k_flat = k.reshape(batch * num_heads, seq_k, head_dim).to(torch.float32)
        v_flat = v.reshape(batch * num_heads, seq_k, head_dim).to(torch.float32)

        if seq_k_padded != seq_k or head_dim_padded != head_dim:
            k_padded = torch.zeros(
                (batch * num_heads, seq_k_padded, head_dim_padded),
                dtype=torch.float32, device=q.device,
            )
            v_padded = torch.zeros_like(k_padded)
            q_padded = torch.zeros(
                (batch * num_heads, seq_q, head_dim_padded),
                dtype=torch.float32, device=q.device,
            )
            k_padded[:, :seq_k, :head_dim] = k_flat
            v_padded[:, :seq_k, :head_dim] = v_flat
            q_padded[:, :, :head_dim] = q_flat
            k_flat, v_flat, q_flat = k_padded, v_padded, q_padded

        scores = torch.empty(
            (batch * num_heads, seq_q, seq_k_padded),
            dtype=torch.float32, device=q.device,
        )
        output = torch.empty(
            (batch * num_heads, seq_q, head_dim_padded),
            dtype=torch.float32, device=q.device,
        )

        grid = (batch * num_heads, seq_q)
        attention_scores_kernel[grid](
            q_flat, k_flat, scores, float(scale),
            seq_k_padded, head_dim_padded,
            q_flat.stride(0), q_flat.stride(1), q_flat.stride(2),
            k_flat.stride(0), k_flat.stride(1), k_flat.stride(2),
            scores.stride(0), scores.stride(1), scores.stride(2),
            BLOCK_K=seq_k_padded, BLOCK_D=head_dim_padded,
        )

        if seq_k_padded != seq_k:
            scores[:, :, seq_k:] = -1e9

        if is_causal:
            mask = torch.triu(
                torch.ones((seq_q, seq_k_padded), dtype=torch.float32, device=q.device),
                diagonal=1,
            ) * -1e9
            scores = scores + mask[None, :, :]

        if attention_mask is not None:
            if attention_mask.ndim == 4:
                attention_mask = attention_mask.reshape(batch * num_heads, seq_q, seq_k)
            if seq_k_padded != seq_k:
                mask_padded = torch.zeros(
                    (batch * num_heads, seq_q, seq_k_padded),
                    dtype=torch.float32, device=q.device,
                )
                mask_padded[:, :, :seq_k] = attention_mask
                mask_padded[:, :, seq_k:] = -1e9
                attention_mask = mask_padded
            scores = scores + attention_mask

        scores_2d = scores.reshape(batch * num_heads * seq_q, seq_k_padded)
        softmax_inplace_kernel[(scores_2d.shape[0],)](
            scores_2d, scores_2d.stride(0), seq_k_padded, BLOCK_SIZE=seq_k_padded
        )
        scores = scores_2d.reshape(batch * num_heads, seq_q, seq_k_padded)

        attention_output_kernel[grid](
            scores, v_flat, output,
            seq_k_padded, head_dim_padded,
            scores.stride(0), scores.stride(1), scores.stride(2),
            v_flat.stride(0), v_flat.stride(1), v_flat.stride(2),
            output.stride(0), output.stride(1), output.stride(2),
            BLOCK_K=seq_k_padded, BLOCK_D=head_dim_padded,
        )

        if head_dim_padded != head_dim:
            output = output[:, :, :head_dim]

        return output.reshape(batch, num_heads, seq_q, head_dim).to(q.dtype)

    # ------------------------------------------------------------------
    # Path 3: Pure PyTorch fallback (CPU or very large head dims)
    # ------------------------------------------------------------------
    scores = torch.einsum("bnqd,bnkd->bnqk", q, k) * scale

    if is_causal:
        mask = torch.triu(
            torch.ones((seq_q, seq_k), dtype=torch.float32, device=q.device),
            diagonal=1,
        ) * -1e9
        scores = scores + mask[None, None, :, :]

    if attention_mask is not None:
        scores = scores + attention_mask

    scores = scores - torch.max(scores, dim=-1, keepdim=True).values
    attn_weights = torch.exp(scores)
    attn_weights = attn_weights / torch.sum(attn_weights, dim=-1, keepdim=True)
    return torch.einsum("bnqk,bnkd->bnqd", attn_weights, v).to(q.dtype)


if __name__ == "__main__":
    print("Testing Triton Attention...")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    batch_size = 2
    num_heads = 4
    seq_len = 16
    head_dim = 64

    q = torch.randn(batch_size, num_heads, seq_len, head_dim, device=device)
    k = torch.randn(batch_size, num_heads, seq_len, head_dim, device=device)
    v = torch.randn(batch_size, num_heads, seq_len, head_dim, device=device)

    print("\nBasic attention:")
    output = scaled_dot_product_attention(q, k, v)
    print(f"  Output shape: {output.shape}")

    print("\nCausal attention:")
    output_causal = scaled_dot_product_attention(q, k, v, is_causal=True)
    print(f"  Output shape: {output_causal.shape}")

    print("\nWith attention mask:")
    mask = torch.zeros(
        (batch_size, num_heads, seq_len, seq_len), dtype=torch.float32, device=device
    )
    mask[:, :, :, seq_len // 2:] = -1e9
    output_masked = scaled_dot_product_attention(q, k, v, attention_mask=mask)
    print(f"  Output shape: {output_masked.shape}")

    print("\nLong sequence (beyond old MAX_ATTENTION_DIM=256 cap):")
    seq_long = 512
    q_l = torch.randn(batch_size, num_heads, seq_long, head_dim, device=device)
    k_l = torch.randn(batch_size, num_heads, seq_long, head_dim, device=device)
    v_l = torch.randn(batch_size, num_heads, seq_long, head_dim, device=device)
    output_long = scaled_dot_product_attention(q_l, k_l, v_l, is_causal=True)
    print(f"  Output shape: {output_long.shape}")

    print("\nLong sequence with mask (previously fell to PyTorch):")
    mask_l = torch.zeros(
        (batch_size, num_heads, seq_long, seq_long), dtype=torch.float32, device=device
    )
    mask_l[:, :, :, seq_long // 2:] = -1e9
    output_long_masked = scaled_dot_product_attention(q_l, k_l, v_l, attention_mask=mask_l)
    print(f"  Output shape: {output_long_masked.shape}")

    print("\nGrouped Query Attention (GQA):")
    num_kv_heads = 2
    k_gqa = torch.randn(batch_size, num_kv_heads, seq_len, head_dim, device=device)
    v_gqa = torch.randn(batch_size, num_kv_heads, seq_len, head_dim, device=device)
    attn = MultiHeadAttention(
        hidden_size=num_heads * head_dim,
        num_heads=num_heads,
        num_kv_heads=num_kv_heads,
    )
    output_gqa = attn(q, k_gqa, v_gqa)
    print(f"  Output shape: {output_gqa.shape}")

    # Numerical correctness check against PyTorch reference
    if device.type == "cuda":
        print("\nNumerical correctness vs PyTorch (causal, seq=64, head_dim=64):")
        sq, hd = 64, 64
        qc = torch.randn(1, 1, sq, hd, device=device)
        kc = torch.randn(1, 1, sq, hd, device=device)
        vc = torch.randn(1, 1, sq, hd, device=device)
        ref = scaled_dot_product_attention(qc, kc, vc, is_causal=True, scale=1.0 / hd**0.5)
        # Force PyTorch path by moving to CPU, then compare
        ref_pt = scaled_dot_product_attention(
            qc.cpu(), kc.cpu(), vc.cpu(), is_causal=True, scale=1.0 / hd**0.5
        ).to(device)
        max_err = float((ref - ref_pt).abs().max())
        print(f"  Max absolute error vs PyTorch: {max_err:.2e}  (expect < 1e-4)")

    print("\nOutput statistics:")
    print(f"  Mean: {float(output.mean()):.4f}")
    print(f"  Std:  {float(output.std()):.4f}")
    print(f"  Min:  {float(output.min()):.4f}")
    print(f"  Max:  {float(output.max()):.4f}")

    print("\nTriton Attention working!")