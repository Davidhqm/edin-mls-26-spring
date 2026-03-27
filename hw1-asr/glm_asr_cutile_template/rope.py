"""
Pure CuTile Rotary Position Embeddings (RoPE)
End-to-end implementation using only NVIDIA CuTile kernels

*** STUDENT ASSIGNMENT ***
Fill in the TODO sections to implement RoPE using CuTile kernels
"""

import cuda.tile as ct
import cupy as cp
import numpy as np
from typing import Tuple, Optional


def get_stream():
    """Get current CUDA stream pointer."""
    return cp.cuda.get_current_stream().ptr


# ============================================================================
# CuTile Kernels for RoPE
# ============================================================================

@ct.kernel
def compute_freqs_kernel(
    positions,      # Input: (seq_len,)
    inv_freq,       # Input: (rotary_dim // 2,)
    cos_out,        # Output: (seq_len, rotary_dim)
    sin_out,        # Output: (seq_len, rotary_dim)
    seq_len: ct.Constant[int],
    half_dim: ct.Constant[int]
):
    """
    Compute cos and sin for rotary embeddings.

    *** TODO: Implement this kernel ***

    Grid: (seq_len,)
    Each thread block computes cos/sin for one position.
    """
    pid = ct.bid(0)  # position index

    # ============================================================================
    # TODO: Implement frequency computation
    # ============================================================================
    #
    # Step 1: Load position as scalar
    # pos = ct.load(positions, index=(pid,), shape=())
    #
    # Step 2: Load all inverse frequencies
    # inv_freq_tile = ct.load(inv_freq, index=(0,), shape=(half_dim,))
    #
    # Step 3: Compute freqs = position * inv_freq
    # freqs = pos * inv_freq_tile
    #
    # Step 4: Compute cos and sin
    # cos_half = ct.cos(freqs)
    # sin_half = ct.sin(freqs)
    #
    # Step 5: Concatenate to full dimension [cos, cos] and [sin, sin]
    # cos_full = ct.cat((cos_half, cos_half), 0)
    # sin_full = ct.cat((sin_half, sin_half), 0)
    #
    # Step 6: Store
    # cos_full = ct.reshape(cos_full, (1, half_dim * 2))
    # sin_full = ct.reshape(sin_full, (1, half_dim * 2))
    # ct.store(cos_out, index=(pid, 0), tile=cos_full)
    # ct.store(sin_out, index=(pid, 0), tile=sin_full)

    # YOUR CODE HERE
    pass  # Remove this and implement


# ============================================================================
# RoPE Classes
# ============================================================================

class RotaryEmbedding:
    """Rotary Position Embedding using pure CuTile."""

    def __init__(
        self,
        dim: int,
        max_position_embeddings: int = 8192,
        base: float = 10000.0,
        partial_rotary_factor: float = 1.0
    ):
        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        self.partial_rotary_factor = partial_rotary_factor

        # Calculate rotary dimension
        self.rotary_dim = int(dim * partial_rotary_factor)
        self.rotary_dim = self.rotary_dim - (self.rotary_dim % 2)  # Must be even

        # Compute inverse frequencies
        inv_freq = 1.0 / (
            base ** (cp.arange(0, self.rotary_dim, 2, dtype=cp.float32) / self.rotary_dim)
        )
        self.inv_freq = inv_freq

        # Pre-compute cos/sin cache
        self._update_cache(max_position_embeddings)

    def _update_cache(self, seq_len: int):
        """Pre-compute cos and sin using CuTile kernel."""
        self.max_seq_len_cached = seq_len
        half_dim = self.rotary_dim // 2

        positions = cp.arange(seq_len, dtype=cp.float32)
        cos_cache = cp.empty((seq_len, self.rotary_dim), dtype=cp.float32)
        sin_cache = cp.empty((seq_len, self.rotary_dim), dtype=cp.float32)

        ct.launch(
            get_stream(),
            (seq_len,),
            compute_freqs_kernel,
            (positions, self.inv_freq, cos_cache, sin_cache, seq_len, half_dim)
        )

        self.cos_cached = cos_cache
        self.sin_cached = sin_cache

    def __call__(
        self,
        x: cp.ndarray,
        position_ids: Optional[cp.ndarray] = None
    ) -> Tuple[cp.ndarray, cp.ndarray]:
        """Get cos and sin for given positions."""
        seq_len = x.shape[-2]

        if seq_len > self.max_seq_len_cached:
            self._update_cache(seq_len)

        if position_ids is not None:
            cos = self.cos_cached[position_ids].astype(x.dtype)
            sin = self.sin_cached[position_ids].astype(x.dtype)
            if cos.ndim == 3 and cos.shape[0] == 1:
                cos = cos[0]
                sin = sin[0]
        else:
            cos = self.cos_cached[:seq_len].astype(x.dtype)
            sin = self.sin_cached[:seq_len].astype(x.dtype)

        return cos, sin


def _apply_rope_single(
    x: cp.ndarray,
    cos: cp.ndarray,
    sin: cp.ndarray,
    half_dim: int,
    head_dim: int
) -> cp.ndarray:
    """Apply RoPE to a single tensor (Q or K) using CuPy."""
    batch, num_heads, seq_len, _ = x.shape

    cos = cos[:seq_len]
    sin = sin[:seq_len]

    x1 = x[..., :half_dim]
    x2 = x[..., half_dim:half_dim * 2]

    cos_expanded = cos[None, None, :, :]
    sin_expanded = sin[None, None, :, :]

    x1_rot = x1 * cos_expanded - x2 * sin_expanded
    x2_rot = x2 * cos_expanded + x1 * sin_expanded

    if head_dim > half_dim * 2:
        x_pass = x[..., half_dim * 2:]
        return cp.concatenate([x1_rot, x2_rot, x_pass], axis=-1)
    else:
        return cp.concatenate([x1_rot, x2_rot], axis=-1)


def apply_rotary_pos_emb(
    q: cp.ndarray,
    k: cp.ndarray,
    cos: cp.ndarray,
    sin: cp.ndarray,
    rotary_dim: Optional[int] = None
) -> Tuple[cp.ndarray, cp.ndarray]:
    """
    Apply rotary position embeddings.

    Args:
        q: Query (batch, num_q_heads, seq_len, head_dim)
        k: Key (batch, num_kv_heads, seq_len, head_dim)
        cos: Cosine values (seq_len, rotary_dim)
        sin: Sine values (seq_len, rotary_dim)
        rotary_dim: Dimension to apply rotation
    """
    batch, num_q_heads, seq_len, head_dim = q.shape
    _, num_kv_heads, _, _ = k.shape

    if rotary_dim is None:
        rotary_dim = head_dim

    half_dim = rotary_dim // 2

    if cos.shape[1] > half_dim:
        cos = cos[:, :half_dim]
        sin = sin[:, :half_dim]

    cos = cp.ascontiguousarray(cos.astype(cp.float32))
    sin = cp.ascontiguousarray(sin.astype(cp.float32))

    q_out = _apply_rope_single(q, cos, sin, half_dim, head_dim)
    k_out = _apply_rope_single(k, cos, sin, half_dim, head_dim)

    return q_out.astype(q.dtype), k_out.astype(k.dtype)


def apply_partial_rotary_pos_emb(
    q: cp.ndarray,
    k: cp.ndarray,
    cos: cp.ndarray,
    sin: cp.ndarray,
    rotary_dim: int
) -> Tuple[cp.ndarray, cp.ndarray]:
    """Apply rotary embeddings to partial dimensions."""
    return apply_rotary_pos_emb(q, k, cos, sin, rotary_dim)


if __name__ == "__main__":
    print("Testing Pure CuTile RoPE...")

    batch_size = 2
    num_heads = 4
    seq_len = 16
    head_dim = 64

    # Create RoPE
    rope = RotaryEmbedding(dim=head_dim, max_position_embeddings=1024)

    # Create Q, K
    q = cp.random.randn(batch_size, num_heads, seq_len, head_dim).astype(cp.float32)
    k = cp.random.randn(batch_size, num_heads, seq_len, head_dim).astype(cp.float32)

    # Get cos/sin
    cos, sin = rope(q)
    print(f"Cos shape: {cos.shape}")
    print(f"Sin shape: {sin.shape}")

    # Apply rotation
    q_rot, k_rot = apply_rotary_pos_emb(q, k, cos, sin)
    print(f"Q rotated shape: {q_rot.shape}")
    print(f"K rotated shape: {k_rot.shape}")

    print("\nPure CuTile RoPE working!")
