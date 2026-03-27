"""
GLM-ASR Configuration Classes
Educational implementation from scratch using PyTorch only
"""

from dataclasses import dataclass
from typing import List, Optional


@dataclass
class AudioEncoderConfig:
    """Configuration for the Whisper-style audio encoder."""
    hidden_size: int = 1280
    intermediate_size: int = 5120
    num_hidden_layers: int = 32
    num_attention_heads: int = 20
    num_key_value_heads: int = 20
    head_dim: int = 64
    num_mel_bins: int = 128
    max_position_embeddings: int = 1500
    hidden_act: str = "gelu"
    attention_dropout: float = 0.0
    initializer_range: float = 0.02
    partial_rotary_factor: float = 0.5
    rope_theta: float = 10000.0


@dataclass
class TextDecoderConfig:
    """Configuration for the Llama-style text decoder."""
    hidden_size: int = 2048
    intermediate_size: int = 6144
    num_hidden_layers: int = 28
    num_attention_heads: int = 16
    num_key_value_heads: int = 4  # GQA: 4 KV heads for 16 Q heads
    head_dim: int = 128
    vocab_size: int = 59264
    max_position_embeddings: int = 8192
    hidden_act: str = "silu"
    rms_norm_eps: float = 1e-5
    attention_dropout: float = 0.0
    attention_bias: bool = False
    mlp_bias: bool = False
    rope_theta: float = 10000.0
    eos_token_ids: List[int] = None

    def __post_init__(self):
        if self.eos_token_ids is None:
            self.eos_token_ids = [59246, 59253, 59255]


@dataclass
class GlmAsrConfig:
    """Full model configuration."""
    audio_config: AudioEncoderConfig = None
    text_config: TextDecoderConfig = None
    audio_token_id: int = 59260
    projector_hidden_act: str = "gelu"
    hidden_size: int = 2048  # Output size of projector (matches text decoder)

    def __post_init__(self):
        if self.audio_config is None:
            self.audio_config = AudioEncoderConfig()
        if self.text_config is None:
            self.text_config = TextDecoderConfig()


@dataclass
class AudioProcessorConfig:
    """Configuration for audio preprocessing."""
    sampling_rate: int = 16000
    n_fft: int = 400
    hop_length: int = 160
    chunk_length: int = 30
    n_samples: int = 480000  # 30 seconds * 16000
    feature_size: int = 128  # num_mel_bins
    nb_max_frames: int = 3000
    padding_value: float = 0.0
