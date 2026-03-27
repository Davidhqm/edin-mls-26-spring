"""
Pure CuTile Multi-Head Attention Implementation
End-to-end implementation using only NVIDIA CuTile kernels

*** STUDENT ASSIGNMENT ***
Fill in the TODO sections to implement attention using CuTile kernels
"""

import cuda.tile as ct
import cupy as cp
import numpy as np
from typing import Optional, Tuple


def get_stream():
    """Get current CUDA stream pointer."""
    return cp.cuda.get_current_stream().ptr


# ============================================================================
# CuTile Kernels for Attention
# ============================================================================

@ct.kernel
def attention_scores_kernel(
    q,              # Query: (batch_heads, seq_q, head_dim)
    k,              # Key: (batch_heads, seq_k, head_dim)
    scores,         # Output: (batch_heads, seq_q, seq_k)
    scale: ct.Constant[float],
    seq_k: ct.Constant[int],
    head_dim: ct.Constant[int]
):
    """
    Compute scaled attention scores for a single query position.
    Grid: (batch_heads, seq_q)

    *** TODO: Implement this kernel ***

    Steps:
    1. Load query vector for position pid_q
    2. Load all keys for this batch_head
    3. Compute Q @ K^T using ct.matmul
    4. Scale by the scale factor
    5. Store the result
    """
    pid_bh = ct.bid(0)
    pid_q = ct.bid(1)

    # ============================================================================
    # TODO: Implement attention score computation
    # ============================================================================
    #
    # Step 1: Load query vector: shape (1, head_dim)
    # Hint: q_tile = ct.load(q, index=(pid_bh, pid_q, 0), shape=(1, 1, head_dim))
    #       q_tile = ct.reshape(q_tile, (1, head_dim))
    #
    # Step 2: Load all keys for this batch_head: shape (seq_k, head_dim)
    # Hint: k_tile = ct.load(k, index=(pid_bh, 0, 0), shape=(1, seq_k, head_dim))
    #       k_tile = ct.reshape(k_tile, (seq_k, head_dim))
    #
    # Step 3: Transpose K and compute Q @ K^T
    # Hint: k_t = ct.transpose(k_tile)  # (head_dim, seq_k)
    #       scores_tile = ct.matmul(q_tile, k_t)  # (1, seq_k)
    #
    # Step 4: Scale the scores
    # Hint: scores_tile = scores_tile * scale
    #
    # Step 5: Reshape and store
    # Hint: scores_tile = ct.reshape(scores_tile, (1, 1, seq_k))
    #       ct.store(scores, index=(pid_bh, pid_q, 0), tile=scores_tile)

    # YOUR CODE HERE
    pass  # Remove this and implement


@ct.kernel
def softmax_inplace_kernel(
    scores,         # Input/Output: (batch_heads, seq_q, seq_k)
    seq_k: ct.Constant[int]
):
    """
    Apply softmax along the last dimension (seq_k).
    Grid: (batch_heads, seq_q)

    *** TODO: Implement this kernel ***

    Steps:
    1. Load scores for this position
    2. Compute max for numerical stability
    3. Subtract max and compute exp
    4. Compute sum of exp
    5. Divide by sum to get softmax
    6. Store back
    """
    pid_bh = ct.bid(0)
    pid_q = ct.bid(1)

    # ============================================================================
    # TODO: Implement softmax
    # ============================================================================
    #
    # Step 1: Load scores
    # Hint: s_tile = ct.load(scores, index=(pid_bh, pid_q, 0), shape=(1, 1, seq_k))
    #       s_tile = ct.reshape(s_tile, (seq_k,))
    #
    # Step 2: Numerically stable softmax
    # max_val = ct.max(s_tile)
    # s_tile = s_tile - max_val
    # exp_tile = ct.exp(s_tile)
    # sum_val = ct.sum(exp_tile)
    # softmax_tile = exp_tile / sum_val
    #
    # Step 3: Store back
    # softmax_tile = ct.reshape(softmax_tile, (1, 1, seq_k))
    # ct.store(scores, index=(pid_bh, pid_q, 0), tile=softmax_tile)

    # YOUR CODE HERE
    pass  # Remove this and implement


@ct.kernel
def attention_output_kernel(
    attn_weights,   # Attention weights: (batch_heads, seq_q, seq_k)
    v,              # Values: (batch_heads, seq_k, head_dim)
    output,         # Output: (batch_heads, seq_q, head_dim)
    seq_k: ct.Constant[int],
    head_dim: ct.Constant[int]
):
    """
    Compute attention output: attn_weights @ V
    Grid: (batch_heads, seq_q)

    *** TODO: Implement this kernel ***

    Steps:
    1. Load attention weights for this query position
    2. Load all values for this batch_head
    3. Compute weighted sum: weights @ V
    4. Store the result
    """
    pid_bh = ct.bid(0)
    pid_q = ct.bid(1)

    # ============================================================================
    # TODO: Implement attention output computation
    # ============================================================================
    #
    # Step 1: Load attention weights: shape (1, seq_k)
    # Hint: w_tile = ct.load(attn_weights, index=(pid_bh, pid_q, 0), shape=(1, 1, seq_k))
    #       w_tile = ct.reshape(w_tile, (1, seq_k))
    #
    # Step 2: Load all values: shape (seq_k, head_dim)
    # Hint: v_tile = ct.load(v, index=(pid_bh, 0, 0), shape=(1, seq_k, head_dim))
    #       v_tile = ct.reshape(v_tile, (seq_k, head_dim))
    #
    # Step 3: Compute weighted sum: (1, seq_k) @ (seq_k, head_dim) = (1, head_dim)
    # Hint: out_tile = ct.matmul(w_tile, v_tile)
    #
    # Step 4: Store
    # Hint: out_tile = ct.reshape(out_tile, (1, 1, head_dim))
    #       ct.store(output, index=(pid_bh, pid_q, 0), tile=out_tile)

    # YOUR CODE HERE
    pass  # Remove this and implement


# ============================================================================
# Attention Classes
# ============================================================================

class MultiHeadAttention:
    """Multi-head attention using pure CuTile kernels."""

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        num_kv_heads: Optional[int] = None,
        head_dim: Optional[int] = None
    ):
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads or num_heads
        self.head_dim = head_dim or (hidden_size // num_heads)
        self.scale = 1.0 / np.sqrt(self.head_dim)

        # Number of query heads per KV head (for GQA)
        self.num_queries_per_kv = self.num_heads // self.num_kv_heads

    def __call__(
        self,
        q: cp.ndarray,
        k: cp.ndarray,
        v: cp.ndarray,
        attention_mask: Optional[cp.ndarray] = None,
        is_causal: bool = False
    ) -> cp.ndarray:
        """
        Compute multi-head attention.

        Args:
            q: Query (batch, num_heads, seq_q, head_dim)
            k: Key (batch, num_kv_heads, seq_k, head_dim)
            v: Value (batch, num_kv_heads, seq_k, head_dim)
            attention_mask: Optional mask (batch, 1, seq_q, seq_k)
            is_causal: Whether to apply causal masking

        Returns:
            Output (batch, num_heads, seq_q, head_dim)
        """
        batch, num_heads, seq_q, head_dim = q.shape
        _, num_kv_heads, seq_k, _ = k.shape

        # Expand KV for GQA if needed
        if num_kv_heads != num_heads:
            k = self._expand_kv(k, self.num_queries_per_kv)
            v = self._expand_kv(v, self.num_queries_per_kv)

        return scaled_dot_product_attention(
            q, k, v, attention_mask, is_causal, self.scale
        )

    def _expand_kv(self, x: cp.ndarray, num_repeats: int) -> cp.ndarray:
        """Expand KV heads for GQA."""
        batch, num_kv_heads, seq_len, head_dim = x.shape
        x_expanded = cp.broadcast_to(
            x[:, :, None, :, :],
            (batch, num_kv_heads, num_repeats, seq_len, head_dim)
        )
        return x_expanded.reshape(batch, num_kv_heads * num_repeats, seq_len, head_dim)


def next_power_of_two(x: int) -> int:
    """Return the smallest power of two >= x."""
    return 1 << (x - 1).bit_length() if x > 0 else 1


# Maximum dimension for CuTile attention kernel
MAX_ATTENTION_DIM = 256


def scaled_dot_product_attention(
    q: cp.ndarray,
    k: cp.ndarray,
    v: cp.ndarray,
    attention_mask: Optional[cp.ndarray] = None,
    is_causal: bool = False,
    scale: Optional[float] = None
) -> cp.ndarray:
    """
    Scaled dot-product attention using pure CuTile.

    Args:
        q: Query (batch, num_heads, seq_q, head_dim)
        k: Key (batch, num_heads, seq_k, head_dim)
        v: Value (batch, num_heads, seq_k, head_dim)
        attention_mask: Optional additive mask
        is_causal: Whether to apply causal masking
        scale: Optional custom scale factor

    Returns:
        Attention output (batch, num_heads, seq_q, head_dim)
    """
    batch, num_heads, seq_q, head_dim = q.shape
    _, num_kv_heads, seq_k, _ = k.shape

    if scale is None:
        scale = 1.0 / np.sqrt(head_dim)

    # Check if dimensions fit CuTile requirements
    seq_k_padded = next_power_of_two(seq_k)
    head_dim_padded = next_power_of_two(head_dim)

    use_cutile = (
        seq_k_padded <= MAX_ATTENTION_DIM and
        head_dim_padded <= MAX_ATTENTION_DIM
    )

    if use_cutile:
        # Reshape to (batch*heads, seq, dim)
        q_flat = q.reshape(batch * num_heads, seq_q, head_dim).astype(cp.float32)
        k_flat = k.reshape(batch * num_heads, seq_k, head_dim).astype(cp.float32)
        v_flat = v.reshape(batch * num_heads, seq_k, head_dim).astype(cp.float32)

        # Pad to power of 2 if needed
        if seq_k_padded != seq_k or head_dim_padded != head_dim:
            k_padded = cp.zeros((batch * num_heads, seq_k_padded, head_dim_padded), dtype=cp.float32)
            k_padded[:, :seq_k, :head_dim] = k_flat
            k_flat = k_padded

            v_padded = cp.zeros((batch * num_heads, seq_k_padded, head_dim_padded), dtype=cp.float32)
            v_padded[:, :seq_k, :head_dim] = v_flat
            v_flat = v_padded

            q_padded = cp.zeros((batch * num_heads, seq_q, head_dim_padded), dtype=cp.float32)
            q_padded[:, :, :head_dim] = q_flat
            q_flat = q_padded

        scores = cp.empty((batch * num_heads, seq_q, seq_k_padded), dtype=cp.float32)
        output = cp.empty((batch * num_heads, seq_q, head_dim_padded), dtype=cp.float32)

        # Launch attention_scores_kernel
        ct.launch(
            get_stream(),
            (batch * num_heads, seq_q),
            attention_scores_kernel,
            (q_flat, k_flat, scores, float(scale), seq_k_padded, head_dim_padded)
        )

        # Mask out padded positions
        if seq_k_padded != seq_k:
            scores[:, :, seq_k:] = -1e9

        # Causal mask
        if is_causal:
            mask = cp.triu(cp.ones((seq_q, seq_k_padded), dtype=cp.float32), k=1) * -1e9
            scores = scores + mask[None, :, :]

        # Attention mask
        if attention_mask is not None:
            if attention_mask.ndim == 4:
                attention_mask = attention_mask.reshape(batch * num_heads, seq_q, seq_k)
            if seq_k_padded != seq_k:
                mask_padded = cp.zeros((batch * num_heads, seq_q, seq_k_padded), dtype=cp.float32)
                mask_padded[:, :, :seq_k] = attention_mask
                mask_padded[:, :, seq_k:] = -1e9
                attention_mask = mask_padded
            scores = scores + attention_mask

        # Launch softmax_inplace_kernel
        ct.launch(
            get_stream(),
            (batch * num_heads, seq_q),
            softmax_inplace_kernel,
            (scores, seq_k_padded)
        )

        # Launch attention_output_kernel
        ct.launch(
            get_stream(),
            (batch * num_heads, seq_q),
            attention_output_kernel,
            (scores, v_flat, output, seq_k_padded, head_dim_padded)
        )

        # Extract actual output
        if head_dim_padded != head_dim:
            output = output[:, :, :head_dim]

        return output.reshape(batch, num_heads, seq_q, head_dim).astype(q.dtype)

    else:
        # Fallback to CuPy for large dimensions
        scores = cp.einsum('bnqd,bnkd->bnqk', q, k) * scale

        if is_causal:
            mask = cp.triu(cp.ones((seq_q, seq_k), dtype=cp.float32), k=1) * -1e9
            scores = scores + mask[None, None, :, :]

        if attention_mask is not None:
            scores = scores + attention_mask

        scores = scores - cp.max(scores, axis=-1, keepdims=True)
        exp_scores = cp.exp(scores)
        attn_weights = exp_scores / cp.sum(exp_scores, axis=-1, keepdims=True)

        output = cp.einsum('bnqk,bnkd->bnqd', attn_weights, v)

        return output.astype(q.dtype)


# ============================================================================
# Test
# ============================================================================

if __name__ == "__main__":
    print("Testing Pure CuTile Attention...")

    batch_size = 2
    num_heads = 4
    seq_len = 16
    head_dim = 64

    # Create random Q, K, V
    q = cp.random.randn(batch_size, num_heads, seq_len, head_dim).astype(cp.float32)
    k = cp.random.randn(batch_size, num_heads, seq_len, head_dim).astype(cp.float32)
    v = cp.random.randn(batch_size, num_heads, seq_len, head_dim).astype(cp.float32)

    # Test basic attention
    print("\nBasic attention:")
    output = scaled_dot_product_attention(q, k, v)
    print(f"  Output shape: {output.shape}")

    # Test causal attention
    print("\nCausal attention:")
    output_causal = scaled_dot_product_attention(q, k, v, is_causal=True)
    print(f"  Output shape: {output_causal.shape}")

    # Verify output is reasonable
    print("\nOutput statistics:")
    print(f"  Mean: {float(cp.mean(output)):.4f}")
    print(f"  Std:  {float(cp.std(output)):.4f}")

    print("\nPure CuTile Attention working!")
