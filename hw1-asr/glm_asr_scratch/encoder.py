"""
Audio Encoder (Whisper-style Transformer)
Educational implementation from scratch using PyTorch only
"""

import torch
import torch.nn as nn
from typing import Optional, Tuple

from config import AudioEncoderConfig
from attention import SelfAttention
from layers import RMSNorm, EncoderMLP, Conv1dSubsampler


class AudioEncoderLayer(nn.Module):
    """
    Single layer of the audio encoder.

    Architecture: LayerNorm -> Self-Attention -> Residual -> LayerNorm -> MLP -> Residual
    (Pre-norm architecture)
    """

    def __init__(self, config: AudioEncoderConfig):
        super().__init__()

        # Self attention
        # Note: GLM-ASR encoder has bias for q, v, o but NOT for k
        self.self_attn = SelfAttention(
            hidden_size=config.hidden_size,
            num_attention_heads=config.num_attention_heads,
            num_key_value_heads=config.num_key_value_heads,
            head_dim=config.head_dim,
            attention_dropout=config.attention_dropout,
            bias=True,  # Encoder uses bias for q, v, o
            k_bias=False,  # But not for k
            rope_theta=config.rope_theta,
            max_position_embeddings=config.max_position_embeddings,
            partial_rotary_factor=config.partial_rotary_factor
        )

        # Layer norms (using standard LayerNorm for encoder, not RMSNorm)
        self.input_layernorm = nn.LayerNorm(config.hidden_size)
        self.post_attention_layernorm = nn.LayerNorm(config.hidden_size)

        # MLP (standard, no gating)
        self.mlp = EncoderMLP(
            hidden_size=config.hidden_size,
            intermediate_size=config.intermediate_size,
            activation=config.hidden_act
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass for encoder layer.

        Args:
            hidden_states: Input tensor of shape (batch, seq_len, hidden_size)
            attention_mask: Optional attention mask
            position_ids: Optional position indices for RoPE

        Returns:
            Output tensor of same shape
        """
        # Self-attention with pre-norm and residual
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states, _ = self.self_attn(
            hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids
        )
        hidden_states = residual + hidden_states

        # MLP with pre-norm and residual
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        return hidden_states


class GlmAsrEncoder(nn.Module):
    """
    Full audio encoder for GLM-ASR.

    Takes mel spectrogram features and outputs contextualized audio representations.

    Architecture:
    1. Conv1d subsampling (reduces sequence length, projects to hidden_size)
    2. Positional encoding (learned or RoPE)
    3. N transformer encoder layers
    4. Final layer norm
    """

    def __init__(self, config: AudioEncoderConfig = None):
        super().__init__()
        if config is None:
            config = AudioEncoderConfig()

        self.config = config

        # Convolutional feature projection and subsampling
        # Takes (batch, seq_len, mel_bins) -> (batch, seq_len // 2, hidden_size)
        self.conv1 = nn.Conv1d(config.num_mel_bins, config.hidden_size, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(config.hidden_size, config.hidden_size, kernel_size=3, stride=2, padding=1)

        # Note: No learned positional embeddings - RoPE is applied inside attention

        # Encoder layers
        self.layers = nn.ModuleList([
            AudioEncoderLayer(config) for _ in range(config.num_hidden_layers)
        ])

        # Final layer norm (named 'norm' to match weights)
        self.norm = nn.LayerNorm(config.hidden_size)

        # Activation function for convolutions
        self.activation = nn.GELU()

    def _get_position_ids(self, seq_len: int, device: torch.device) -> torch.Tensor:
        """Generate position IDs for the sequence."""
        return torch.arange(seq_len, device=device).unsqueeze(0)

    def forward(
        self,
        input_features: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass for the audio encoder.

        Args:
            input_features: Mel spectrogram of shape (batch, seq_len, num_mel_bins)
            attention_mask: Optional mask of shape (batch, seq_len)

        Returns:
            Encoded features of shape (batch, seq_len // 2, hidden_size)
        """
        # Conv projection: (batch, seq_len, mel) -> (batch, hidden, seq_len)
        hidden_states = input_features.transpose(1, 2)
        hidden_states = self.activation(self.conv1(hidden_states))
        hidden_states = self.activation(self.conv2(hidden_states))

        # Back to (batch, seq_len, hidden)
        hidden_states = hidden_states.transpose(1, 2)

        # Get sequence length after convolution
        batch_size, seq_len, _ = hidden_states.shape

        # Position IDs for RoPE (no learned embeddings added)
        position_ids = self._get_position_ids(seq_len, hidden_states.device)

        # Prepare attention mask if provided
        if attention_mask is not None:
            # Subsample mask to match reduced sequence length
            attention_mask = attention_mask[:, ::2]  # Simple subsampling
            # Convert to attention mask format: 0 -> 0.0, 1 -> -inf
            attention_mask = (1.0 - attention_mask.unsqueeze(1).unsqueeze(2)) * -10000.0

        # Apply encoder layers
        for layer in self.layers:
            hidden_states = layer(
                hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids
            )

        # Final layer norm
        hidden_states = self.norm(hidden_states)

        return hidden_states


if __name__ == "__main__":
    # Test the encoder
    from config import AudioEncoderConfig

    # Use smaller config for testing
    test_config = AudioEncoderConfig(
        hidden_size=256,
        intermediate_size=512,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=4,
        head_dim=64,
        num_mel_bins=128,
        max_position_embeddings=1500
    )

    encoder = GlmAsrEncoder(test_config)

    # Test input: (batch, seq_len, mel_bins)
    batch_size = 2
    seq_len = 100  # 100 frames of mel spectrogram
    mel_bins = 128

    x = torch.randn(batch_size, seq_len, mel_bins)

    print(f"Input shape: {x.shape}")
    output = encoder(x)
    print(f"Output shape: {output.shape}")
    print(f"Expected: (batch, seq_len // 2, hidden_size) = ({batch_size}, {seq_len // 2}, {test_config.hidden_size})")

    # Count parameters
    total_params = sum(p.numel() for p in encoder.parameters())
    print(f"\nTotal parameters: {total_params:,}")
