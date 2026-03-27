"""
Neural Network Layers
Educational implementation from scratch using PyTorch only
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class RMSNorm(nn.Module):
    """
    Root Mean Square Layer Normalization.

    Unlike LayerNorm which normalizes using mean and variance,
    RMSNorm only uses the RMS (root mean square) for normalization.
    This is simpler and often works just as well.

    Formula: x_norm = x / RMS(x) * weight
    where RMS(x) = sqrt(mean(x^2) + eps)
    """

    def __init__(self, hidden_size: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.eps = eps

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)

        # Compute RMS
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.eps)

        return self.weight * hidden_states.to(input_dtype)


class GELUActivation(nn.Module):
    """
    Gaussian Error Linear Unit activation function.

    GELU(x) = x * Phi(x), where Phi is the CDF of standard normal distribution.

    Approximation: GELU(x) ≈ 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
    """

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.gelu(x)


class SiLUActivation(nn.Module):
    """
    Sigmoid Linear Unit (SiLU) / Swish activation function.

    SiLU(x) = x * sigmoid(x)

    This is a smooth approximation of ReLU that allows small negative values.
    """

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.silu(x)


def get_activation(name: str) -> nn.Module:
    """Get activation function by name."""
    activations = {
        "gelu": GELUActivation(),
        "silu": SiLUActivation(),
        "relu": nn.ReLU(),
        "tanh": nn.Tanh()
    }
    if name not in activations:
        raise ValueError(f"Unknown activation: {name}")
    return activations[name]


class MLP(nn.Module):
    """
    Multi-Layer Perceptron with gating (SwiGLU-style for Llama).

    The standard MLP is: Linear -> Activation -> Linear
    SwiGLU style is: (Linear_gate * Activation(Linear_up)) -> Linear_down

    This gating mechanism has been shown to improve model quality.
    """

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        activation: str = "silu",
        bias: bool = False,
        use_gating: bool = True
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.use_gating = use_gating

        if use_gating:
            # SwiGLU-style: gate and up projections
            self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=bias)
            self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=bias)
        else:
            # Standard MLP
            self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=bias)

        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=bias)
        self.act_fn = get_activation(activation)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.use_gating:
            # SwiGLU: gate * activation(up)
            return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
        else:
            # Standard: activation(up) -> down
            return self.down_proj(self.act_fn(self.up_proj(x)))


class EncoderMLP(nn.Module):
    """
    MLP for encoder (standard, no gating).

    Used in the Whisper-style audio encoder.
    """

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        activation: str = "gelu",
        bias: bool = True
    ):
        super().__init__()
        self.fc1 = nn.Linear(hidden_size, intermediate_size, bias=bias)
        self.fc2 = nn.Linear(intermediate_size, hidden_size, bias=bias)
        self.act_fn = get_activation(activation)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(self.act_fn(self.fc1(x)))


class Conv1dSubsampler(nn.Module):
    """
    Convolutional subsampler for audio features.

    Reduces the sequence length of audio features (typically by 2x)
    while projecting to the encoder hidden size.

    This is similar to what Whisper uses to process mel spectrograms.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_size: int,
        num_layers: int = 2,
        kernel_size: int = 3,
        stride: int = 2
    ):
        super().__init__()

        layers = []
        in_channels = input_dim

        for i in range(num_layers):
            out_channels = hidden_size
            padding = (kernel_size - 1) // 2

            layers.append(
                nn.Conv1d(
                    in_channels,
                    out_channels,
                    kernel_size=kernel_size,
                    stride=stride if i == 0 else 1,
                    padding=padding
                )
            )
            layers.append(nn.GELU())

            in_channels = out_channels

        self.conv = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input of shape (batch, seq_len, input_dim)

        Returns:
            Output of shape (batch, seq_len // stride, hidden_size)
        """
        # Conv1d expects (batch, channels, seq_len)
        x = x.transpose(1, 2)
        x = self.conv(x)
        x = x.transpose(1, 2)
        return x


class MultiModalProjector(nn.Module):
    """
    Projects audio encoder outputs to text decoder input space.

    This bridges the gap between the audio encoder output and
    the text decoder hidden size.

    The actual architecture uses:
    - Input: audio_intermediate_size (5120) - audio features reshaped to combine 4 frames
    - Hidden: text_hidden_size * 2 (4096)
    - Output: text_hidden_size (2048)
    """

    def __init__(
        self,
        audio_intermediate_size: int,
        text_hidden_size: int,
        activation: str = "gelu"
    ):
        super().__init__()
        # First layer: audio_intermediate_size -> text_hidden_size * 2
        self.linear_1 = nn.Linear(audio_intermediate_size, text_hidden_size * 2, bias=True)
        self.act = get_activation(activation)
        # Second layer: text_hidden_size * 2 -> text_hidden_size
        self.linear_2 = nn.Linear(text_hidden_size * 2, text_hidden_size, bias=True)

    def forward(self, audio_features: torch.Tensor) -> torch.Tensor:
        """
        Project audio features to text embedding space.

        Args:
            audio_features: Shape (batch, audio_seq_len, audio_hidden_size)

        Returns:
            Shape (batch, audio_seq_len, text_hidden_size)
        """
        x = self.linear_1(audio_features)
        x = self.act(x)
        x = self.linear_2(x)
        return x


if __name__ == "__main__":
    # Test RMSNorm
    print("Testing RMSNorm:")
    rms_norm = RMSNorm(256)
    x = torch.randn(2, 10, 256)
    y = rms_norm(x)
    print(f"Input shape: {x.shape}, Output shape: {y.shape}")

    # Test MLP (Llama style with gating)
    print("\nTesting MLP (with gating):")
    mlp = MLP(hidden_size=256, intermediate_size=512, activation="silu", use_gating=True)
    y = mlp(x)
    print(f"Output shape: {y.shape}")

    # Test Encoder MLP
    print("\nTesting Encoder MLP (no gating):")
    enc_mlp = EncoderMLP(hidden_size=256, intermediate_size=512, activation="gelu")
    y = enc_mlp(x)
    print(f"Output shape: {y.shape}")

    # Test Conv1d Subsampler
    print("\nTesting Conv1d Subsampler:")
    subsampler = Conv1dSubsampler(input_dim=128, hidden_size=256)
    mel_input = torch.randn(2, 100, 128)  # (batch, frames, mel_bins)
    y = subsampler(mel_input)
    print(f"Input shape: {mel_input.shape}, Output shape: {y.shape}")

    # Test MultiModal Projector
    print("\nTesting MultiModal Projector:")
    projector = MultiModalProjector(audio_intermediate_size=256, text_hidden_size=512)
    y = projector(x)
    print(f"Input shape: {x.shape}, Output shape: {y.shape}")
