"""
Attention Mechanisms
Educational implementation from scratch using PyTorch only
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple

from rope import RotaryEmbedding, apply_rotary_pos_emb, apply_partial_rotary_pos_emb


class MultiHeadAttention(nn.Module):
    """
    Multi-Head Attention with optional Grouped Query Attention (GQA).

    In standard MHA: num_kv_heads = num_heads (each head has its own K, V)
    In GQA: num_kv_heads < num_heads (K, V are shared across groups of Q heads)
    In MQA: num_kv_heads = 1 (single K, V shared by all Q heads)

    GQA reduces memory usage while maintaining most of the performance.
    """

    def __init__(
        self,
        hidden_size: int,
        num_attention_heads: int,
        num_key_value_heads: int,
        head_dim: int,
        attention_dropout: float = 0.0,
        bias: bool = False,
        k_bias: bool = None,  # Allow separate config for k_proj bias
        is_causal: bool = False,
        rope_theta: float = 10000.0,
        max_position_embeddings: int = 8192,
        partial_rotary_factor: float = 1.0
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_attention_heads
        self.num_kv_heads = num_key_value_heads
        self.head_dim = head_dim
        self.attention_dropout = attention_dropout
        self.is_causal = is_causal
        self.partial_rotary_factor = partial_rotary_factor

        # Number of Q heads per KV head (for GQA)
        self.num_key_value_groups = self.num_heads // self.num_kv_heads

        # Handle asymmetric bias (some models have no bias for k_proj)
        if k_bias is None:
            k_bias = bias

        # Projection layers
        self.q_proj = nn.Linear(hidden_size, num_attention_heads * head_dim, bias=bias)
        self.k_proj = nn.Linear(hidden_size, num_key_value_heads * head_dim, bias=k_bias)
        self.v_proj = nn.Linear(hidden_size, num_key_value_heads * head_dim, bias=bias)
        self.o_proj = nn.Linear(num_attention_heads * head_dim, hidden_size, bias=bias)

        # Rotary embeddings
        self.rotary_dim = int(head_dim * partial_rotary_factor)
        self.rotary_dim = self.rotary_dim - (self.rotary_dim % 2)  # Must be even

        self.rotary_emb = RotaryEmbedding(
            dim=head_dim,
            max_position_embeddings=max_position_embeddings,
            base=rope_theta,
            partial_rotary_factor=partial_rotary_factor
        )

    def _repeat_kv(self, hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
        """
        Repeat KV heads to match the number of Q heads for GQA.

        From (batch, num_kv_heads, seq_len, head_dim)
        To   (batch, num_heads, seq_len, head_dim)
        """
        if n_rep == 1:
            return hidden_states

        batch, num_kv_heads, seq_len, head_dim = hidden_states.shape
        hidden_states = hidden_states[:, :, None, :, :].expand(
            batch, num_kv_heads, n_rep, seq_len, head_dim
        )
        return hidden_states.reshape(batch, num_kv_heads * n_rep, seq_len, head_dim)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        use_cache: bool = False
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        """
        Forward pass for multi-head attention.

        Args:
            hidden_states: Input tensor of shape (batch, seq_len, hidden_size)
            attention_mask: Mask of shape (batch, 1, seq_len, seq_len)
            position_ids: Position indices for RoPE
            past_key_value: Cached K, V from previous steps
            use_cache: Whether to return cached K, V

        Returns:
            Tuple of (output tensor, optional cached KV)
        """
        batch_size, seq_len, _ = hidden_states.shape

        # Project to Q, K, V
        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        # Reshape: (batch, seq_len, num_heads * head_dim) -> (batch, num_heads, seq_len, head_dim)
        query_states = query_states.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(batch_size, seq_len, self.num_kv_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(batch_size, seq_len, self.num_kv_heads, self.head_dim).transpose(1, 2)

        # Apply rotary position embeddings
        cos, sin = self.rotary_emb(query_states, position_ids)

        if self.partial_rotary_factor < 1.0:
            query_states, key_states = apply_partial_rotary_pos_emb(
                query_states, key_states, cos, sin, self.rotary_dim
            )
        else:
            query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        # Handle KV cache
        if past_key_value is not None:
            key_states = torch.cat([past_key_value[0], key_states], dim=2)
            value_states = torch.cat([past_key_value[1], value_states], dim=2)

        past_key_value = (key_states, value_states) if use_cache else None

        # Repeat KV heads for GQA
        key_states = self._repeat_kv(key_states, self.num_key_value_groups)
        value_states = self._repeat_kv(value_states, self.num_key_value_groups)

        # Compute attention scores
        # (batch, num_heads, seq_len, head_dim) @ (batch, num_heads, head_dim, kv_len)
        # -> (batch, num_heads, seq_len, kv_len)
        attn_weights = torch.matmul(query_states, key_states.transpose(-2, -1))
        attn_weights = attn_weights / math.sqrt(self.head_dim)

        # Apply attention mask
        if attention_mask is not None:
            attn_weights = attn_weights + attention_mask

        # Apply causal mask if needed
        if self.is_causal:
            kv_len = key_states.shape[2]
            causal_mask = torch.triu(
                torch.full((seq_len, kv_len), float('-inf'), device=attn_weights.device),
                diagonal=kv_len - seq_len + 1
            )
            attn_weights = attn_weights + causal_mask

        # Softmax and dropout
        attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
        attn_weights = F.dropout(attn_weights, p=self.attention_dropout, training=self.training)

        # Apply attention to values
        # (batch, num_heads, seq_len, kv_len) @ (batch, num_heads, kv_len, head_dim)
        # -> (batch, num_heads, seq_len, head_dim)
        attn_output = torch.matmul(attn_weights, value_states)

        # Reshape back
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.view(batch_size, seq_len, -1)

        # Output projection
        attn_output = self.o_proj(attn_output)

        return attn_output, past_key_value


class SelfAttention(MultiHeadAttention):
    """
    Self-attention layer (non-causal, for encoder).
    """

    def __init__(self, **kwargs):
        kwargs['is_causal'] = False
        super().__init__(**kwargs)


class CausalSelfAttention(MultiHeadAttention):
    """
    Causal self-attention layer (for decoder).
    """

    def __init__(self, **kwargs):
        kwargs['is_causal'] = True
        super().__init__(**kwargs)


if __name__ == "__main__":
    # Test attention mechanisms
    batch_size = 2
    seq_len = 16
    hidden_size = 256
    num_heads = 4
    num_kv_heads = 2  # GQA
    head_dim = 64

    # Test self-attention (encoder style)
    print("Testing Self-Attention (encoder):")
    self_attn = SelfAttention(
        hidden_size=hidden_size,
        num_attention_heads=num_heads,
        num_key_value_heads=num_kv_heads,
        head_dim=head_dim,
        partial_rotary_factor=0.5  # Like audio encoder
    )

    x = torch.randn(batch_size, seq_len, hidden_size)
    output, _ = self_attn(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")

    # Test causal attention (decoder style)
    print("\nTesting Causal Self-Attention (decoder):")
    causal_attn = CausalSelfAttention(
        hidden_size=hidden_size,
        num_attention_heads=num_heads,
        num_key_value_heads=num_kv_heads,
        head_dim=head_dim
    )

    output, kv_cache = causal_attn(x, use_cache=True)
    print(f"Output shape: {output.shape}")
    print(f"KV cache shapes: K={kv_cache[0].shape}, V={kv_cache[1].shape}")
