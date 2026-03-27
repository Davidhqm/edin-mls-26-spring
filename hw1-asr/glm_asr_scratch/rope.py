"""
Rotary Position Embeddings (RoPE)
Educational implementation from scratch using PyTorch only

RoPE encodes position by rotating query and key vectors in a way that
makes the dot product between q and k depend only on their relative position.
"""

import torch
import torch.nn as nn
from typing import Tuple, Optional


class RotaryEmbedding(nn.Module):
    """
    Rotary Position Embedding implementation.

    RoPE works by:
    1. Splitting the embedding dimension into pairs
    2. For each pair, treating them as 2D coordinates
    3. Rotating each pair by an angle proportional to position
    4. The rotation angle varies by dimension (lower dims rotate faster)

    This makes the attention score between positions i and j depend on (i-j),
    which is the relative position.
    """

    def __init__(
        self,
        dim: int,
        max_position_embeddings: int = 8192,
        base: float = 10000.0,
        partial_rotary_factor: float = 1.0,
        device: Optional[torch.device] = None
    ):
        """
        Args:
            dim: Dimension of the embeddings (head_dim for attention)
            max_position_embeddings: Maximum sequence length
            base: Base for the geometric series (theta)
            partial_rotary_factor: Fraction of dimensions to apply RoPE to
        """
        super().__init__()
        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        self.partial_rotary_factor = partial_rotary_factor

        # Calculate rotary dimension (might be partial)
        self.rotary_dim = int(dim * partial_rotary_factor)
        # Must be even for rotation
        self.rotary_dim = self.rotary_dim - (self.rotary_dim % 2)

        # Compute inverse frequencies
        # For dimension i, frequency is base^(-2i/d)
        inv_freq = 1.0 / (
            base ** (torch.arange(0, self.rotary_dim, 2, dtype=torch.float32) / self.rotary_dim)
        )
        self.register_buffer("inv_freq", inv_freq, persistent=False)

        # Pre-compute cos and sin for all positions
        self._update_cos_sin_cache(max_position_embeddings, device or torch.device("cpu"))

    def _update_cos_sin_cache(
        self,
        seq_len: int,
        device: torch.device,
        dtype: torch.dtype = torch.float32
    ):
        """Pre-compute cos and sin values for efficiency."""
        self.max_seq_len_cached = seq_len

        # Position indices: [0, 1, 2, ..., seq_len-1]
        t = torch.arange(seq_len, device=device, dtype=torch.float32)

        # Outer product: positions × frequencies
        # Shape: (seq_len, rotary_dim // 2)
        freqs = torch.outer(t, self.inv_freq.to(device))

        # Repeat for pairs: (seq_len, rotary_dim)
        emb = torch.cat((freqs, freqs), dim=-1)

        # Cache cos and sin
        self.register_buffer("cos_cached", emb.cos().to(dtype), persistent=False)
        self.register_buffer("sin_cached", emb.sin().to(dtype), persistent=False)

    def forward(
        self,
        x: torch.Tensor,
        position_ids: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get cos and sin for given positions.

        Args:
            x: Input tensor, used to get sequence length and device
            position_ids: Optional explicit position indices

        Returns:
            Tuple of (cos, sin) tensors for rotation
        """
        seq_len = x.shape[-2]

        # Extend cache if needed
        if seq_len > self.max_seq_len_cached:
            self._update_cos_sin_cache(seq_len, x.device, x.dtype)

        if position_ids is not None:
            # Use explicit positions
            cos = self.cos_cached[position_ids].to(x.dtype)
            sin = self.sin_cached[position_ids].to(x.dtype)
        else:
            # Use sequential positions
            cos = self.cos_cached[:seq_len].to(x.dtype)
            sin = self.sin_cached[:seq_len].to(x.dtype)

        return cos, sin


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    """
    Rotate half the hidden dims of x.

    For a vector [x1, x2, x3, x4], returns [-x3, -x4, x1, x2].
    This implements the rotation operation in 2D subspaces.
    """
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    unsqueeze_dim: int = 1
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Apply rotary position embeddings to query and key tensors.

    The rotation formula is:
    q_rotated = q * cos + rotate_half(q) * sin

    This implements:
    [q1, q2] @ [[cos, -sin], [sin, cos]] = [q1*cos - q2*sin, q1*sin + q2*cos]

    Args:
        q: Query tensor of shape (batch, num_heads, seq_len, head_dim)
        k: Key tensor of shape (batch, num_heads, seq_len, head_dim)
        cos: Cosine values of shape (seq_len, head_dim) or (batch, seq_len, head_dim)
        sin: Sine values of shape (seq_len, head_dim) or (batch, seq_len, head_dim)
        unsqueeze_dim: Dimension to unsqueeze cos/sin for broadcasting

    Returns:
        Tuple of rotated (query, key) tensors
    """
    # cos/sin shape: (seq_len, head_dim) -> need (1, 1, seq_len, head_dim)
    # to broadcast with q/k shape: (batch, num_heads, seq_len, head_dim)
    if cos.dim() == 2:
        # (seq_len, head_dim) -> (1, 1, seq_len, head_dim)
        cos = cos.unsqueeze(0).unsqueeze(0)
        sin = sin.unsqueeze(0).unsqueeze(0)
    elif cos.dim() == 3:
        # (batch, seq_len, head_dim) -> (batch, 1, seq_len, head_dim)
        cos = cos.unsqueeze(unsqueeze_dim)
        sin = sin.unsqueeze(unsqueeze_dim)

    # Apply rotation
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)

    return q_embed, k_embed


def apply_partial_rotary_pos_emb(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    rotary_dim: int,
    unsqueeze_dim: int = 1
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Apply rotary position embeddings to only part of the dimensions.

    Used when partial_rotary_factor < 1.0 (like in the audio encoder).

    Args:
        q: Query tensor
        k: Key tensor
        cos: Cosine values
        sin: Sine values
        rotary_dim: Number of dimensions to apply rotation to
        unsqueeze_dim: Dimension to unsqueeze for broadcasting

    Returns:
        Tuple of rotated (query, key) tensors
    """
    # Split into rotary and non-rotary parts
    q_rot, q_pass = q[..., :rotary_dim], q[..., rotary_dim:]
    k_rot, k_pass = k[..., :rotary_dim], k[..., rotary_dim:]

    # Apply rotation to rotary part
    q_rot, k_rot = apply_rotary_pos_emb(q_rot, k_rot, cos, sin, unsqueeze_dim)

    # Concatenate back
    q = torch.cat([q_rot, q_pass], dim=-1)
    k = torch.cat([k_rot, k_pass], dim=-1)

    return q, k


if __name__ == "__main__":
    # Test RoPE implementation
    batch_size = 2
    seq_len = 10
    num_heads = 4
    head_dim = 64

    # Create RoPE embedding
    rope = RotaryEmbedding(dim=head_dim, max_position_embeddings=1024)

    # Create dummy q, k tensors
    q = torch.randn(batch_size, num_heads, seq_len, head_dim)
    k = torch.randn(batch_size, num_heads, seq_len, head_dim)

    # Get cos, sin
    cos, sin = rope(q)

    # Apply rotation
    q_rot, k_rot = apply_rotary_pos_emb(q, k, cos, sin)

    print(f"Input Q shape: {q.shape}")
    print(f"Output Q shape: {q_rot.shape}")
    print(f"Cos shape: {cos.shape}")

    # Test partial rotary (50% like audio encoder)
    rotary_dim = head_dim // 2
    rope_partial = RotaryEmbedding(dim=head_dim, partial_rotary_factor=0.5)
    cos_p, sin_p = rope_partial(q)
    q_rot_p, k_rot_p = apply_partial_rotary_pos_emb(q, k, cos_p, sin_p, rotary_dim)
    print(f"\nPartial RoPE (50%):")
    print(f"Rotary dim: {rotary_dim}")
    print(f"Output Q shape: {q_rot_p.shape}")
