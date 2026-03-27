"""
Weight Loader for GLM-ASR
Educational implementation using NVIDIA CuTile for tile-based GPU programming

Loads pre-trained weights from safetensors format into the model.
"""

import cupy as cp
import numpy as np
from pathlib import Path
from typing import Dict, Optional, Union

def create_config_from_hf(hf_config):
    """Create GlmAsrConfig from HuggingFace config."""
    from model import GlmAsrConfig

    ac = hf_config.audio_config
    tc = hf_config.text_config

    return GlmAsrConfig(
        # Audio encoder
        audio_hidden_size=ac.hidden_size,
        audio_num_heads=ac.num_attention_heads,
        audio_num_layers=ac.num_hidden_layers,
        audio_intermediate_size=ac.intermediate_size,
        audio_max_position_embeddings=getattr(ac, 'max_position_embeddings', 1500),

        # Text decoder
        text_hidden_size=tc.hidden_size,
        text_num_heads=tc.num_attention_heads,
        text_num_kv_heads=tc.num_key_value_heads,
        text_num_layers=tc.num_hidden_layers,
        text_intermediate_size=tc.intermediate_size,
        text_vocab_size=tc.vocab_size,
        text_max_position_embeddings=tc.max_position_embeddings,
        text_rope_base=getattr(tc, 'rope_theta', 10000.0),

        # Projector - uses intermediate size 4096, with 4x frame pooling
        projector_hidden_size=4096,  # Intermediate size (linear_1 output)
        projector_pool_factor=4,  # Concatenate 4 consecutive audio frames

        # Generation - get from text_config if not in main config
        pad_token_id=getattr(tc, 'pad_token_id', 0) if getattr(tc, 'pad_token_id', None) is not None else 0,
        bos_token_id=getattr(tc, 'bos_token_id', 1) if getattr(tc, 'bos_token_id', None) is not None else 1,
        # eos_token_id can be a list - pass all of them for proper stopping
        eos_token_id=getattr(tc, 'eos_token_id', 2),
    )


def load_linear_weight(cutile_linear, hf_weight, hf_bias=None):
    """Load weight (and optional bias) into CuTile Linear layer."""
    cutile_linear.weight = cp.asarray(hf_weight.cpu().numpy(), dtype=cp.float32)
    if hf_bias is not None and cutile_linear.has_bias:
        cutile_linear.bias_param = cp.asarray(hf_bias.cpu().numpy(), dtype=cp.float32)


def load_conv1d_weight_from_hf(cutile_conv, hf_weight, hf_bias=None):
    """Load weight into CuTile Conv1d layer from HF format."""
    weight_np = hf_weight.cpu().numpy()
    out_channels, in_channels, kernel_size = weight_np.shape
    cutile_conv.weight = cp.asarray(
        weight_np.reshape(out_channels, in_channels * kernel_size),
        dtype=cp.float32
    )
    # Also update weight_padded if using CuTile
    if cutile_conv.use_cutile and (
        cutile_conv.col_size_padded != cutile_conv.col_size or
        cutile_conv.out_channels_padded != out_channels
    ):
        cutile_conv.weight_padded = cp.zeros(
            (cutile_conv.out_channels_padded, cutile_conv.col_size_padded),
            dtype=cp.float32
        )
        cutile_conv.weight_padded[:out_channels, :cutile_conv.col_size] = cutile_conv.weight
    else:
        cutile_conv.weight_padded = cutile_conv.weight
    if hf_bias is not None and cutile_conv.has_bias:
        cutile_conv.bias = cp.asarray(hf_bias.cpu().numpy(), dtype=cp.float32)


def load_layernorm_weight_from_hf(cutile_ln, hf_weight, hf_bias):
    """Load LayerNorm weights."""
    cutile_ln.weight = cp.asarray(hf_weight.cpu().numpy(), dtype=cp.float32)
    cutile_ln.bias = cp.asarray(hf_bias.cpu().numpy(), dtype=cp.float32)


def load_rmsnorm_weight_from_hf(cutile_rms, hf_weight):
    """Load RMSNorm weight."""
    cutile_rms.weight = cp.asarray(hf_weight.cpu().numpy(), dtype=cp.float32)


def load_embedding_weight_from_hf(cutile_emb, hf_weight):
    """Load Embedding weight."""
    cutile_emb.weight = cp.asarray(hf_weight.cpu().numpy(), dtype=cp.float32)


def load_weights_from_hf_model(model, hf_model) -> None:
    """
    Load weights from HuggingFace GLM-ASR model into pure CuTile model.

    Args:
        model: Pure CuTile GlmAsrModel
        hf_model: HuggingFace GlmAsrForConditionalGeneration model
    """
    hf_state = hf_model.state_dict()

    print("Loading audio encoder weights...")

    # Audio encoder conv layers
    load_conv1d_weight_from_hf(
        model.audio_encoder.conv1,
        hf_state['audio_tower.conv1.weight'],
        hf_state['audio_tower.conv1.bias']
    )
    load_conv1d_weight_from_hf(
        model.audio_encoder.conv2,
        hf_state['audio_tower.conv2.weight'],
        hf_state['audio_tower.conv2.bias']
    )

    # Audio encoder positional embeddings (if learnable)
    if 'audio_tower.embed_positions.weight' in hf_state:
        model.audio_encoder.embed_positions = cp.asarray(
            hf_state['audio_tower.embed_positions.weight'].cpu().numpy(),
            dtype=cp.float32
        )

    # Audio encoder transformer layers
    for i, layer in enumerate(model.audio_encoder.layers):
        prefix = f'audio_tower.layers.{i}'

        # Input layernorm
        load_layernorm_weight_from_hf(
            layer.self_attn_layer_norm,
            hf_state[f'{prefix}.input_layernorm.weight'],
            hf_state[f'{prefix}.input_layernorm.bias']
        )

        # Attention projections
        load_linear_weight(
            layer.q_proj,
            hf_state[f'{prefix}.self_attn.q_proj.weight'],
            hf_state.get(f'{prefix}.self_attn.q_proj.bias')
        )
        load_linear_weight(
            layer.k_proj,
            hf_state[f'{prefix}.self_attn.k_proj.weight'],
            hf_state.get(f'{prefix}.self_attn.k_proj.bias')
        )
        load_linear_weight(
            layer.v_proj,
            hf_state[f'{prefix}.self_attn.v_proj.weight'],
            hf_state.get(f'{prefix}.self_attn.v_proj.bias')
        )
        load_linear_weight(
            layer.out_proj,
            hf_state[f'{prefix}.self_attn.o_proj.weight'],
            hf_state.get(f'{prefix}.self_attn.o_proj.bias')
        )

        # Post attention layernorm
        load_layernorm_weight_from_hf(
            layer.final_layer_norm,
            hf_state[f'{prefix}.post_attention_layernorm.weight'],
            hf_state[f'{prefix}.post_attention_layernorm.bias']
        )

        # MLP
        load_linear_weight(
            layer.fc1,
            hf_state[f'{prefix}.mlp.fc1.weight'],
            hf_state[f'{prefix}.mlp.fc1.bias']
        )
        load_linear_weight(
            layer.fc2,
            hf_state[f'{prefix}.mlp.fc2.weight'],
            hf_state[f'{prefix}.mlp.fc2.bias']
        )

    # Audio encoder final layernorm
    load_layernorm_weight_from_hf(
        model.audio_encoder.layer_norm,
        hf_state['audio_tower.norm.weight'],
        hf_state['audio_tower.norm.bias']
    )

    print("Loading multi-modal projector weights...")

    # Multi-modal projector
    load_linear_weight(
        model.multi_modal_projector.linear_1,
        hf_state['multi_modal_projector.linear_1.weight'],
        hf_state['multi_modal_projector.linear_1.bias']
    )
    load_linear_weight(
        model.multi_modal_projector.linear_2,
        hf_state['multi_modal_projector.linear_2.weight'],
        hf_state['multi_modal_projector.linear_2.bias']
    )

    print("Loading text decoder weights...")

    # Text decoder embeddings
    load_embedding_weight_from_hf(
        model.text_decoder.embed_tokens,
        hf_state['language_model.model.embed_tokens.weight']
    )

    # Text decoder transformer layers
    for i, layer in enumerate(model.text_decoder.layers):
        prefix = f'language_model.model.layers.{i}'

        # Input layernorm (RMSNorm)
        load_rmsnorm_weight_from_hf(
            layer.input_layernorm,
            hf_state[f'{prefix}.input_layernorm.weight']
        )

        # Attention projections (no bias in Llama-style)
        load_linear_weight(
            layer.q_proj,
            hf_state[f'{prefix}.self_attn.q_proj.weight']
        )
        load_linear_weight(
            layer.k_proj,
            hf_state[f'{prefix}.self_attn.k_proj.weight']
        )
        load_linear_weight(
            layer.v_proj,
            hf_state[f'{prefix}.self_attn.v_proj.weight']
        )
        load_linear_weight(
            layer.o_proj,
            hf_state[f'{prefix}.self_attn.o_proj.weight']
        )

        # Post attention layernorm (RMSNorm)
        load_rmsnorm_weight_from_hf(
            layer.post_attention_layernorm,
            hf_state[f'{prefix}.post_attention_layernorm.weight']
        )

        # MLP (SwiGLU: gate_proj, up_proj, down_proj)
        load_linear_weight(
            layer.mlp.gate_proj,
            hf_state[f'{prefix}.mlp.gate_proj.weight']
        )
        load_linear_weight(
            layer.mlp.up_proj,
            hf_state[f'{prefix}.mlp.up_proj.weight']
        )
        load_linear_weight(
            layer.mlp.down_proj,
            hf_state[f'{prefix}.mlp.down_proj.weight']
        )

    # Text decoder final norm
    load_rmsnorm_weight_from_hf(
        model.text_decoder.norm,
        hf_state['language_model.model.norm.weight']
    )

    # LM head
    load_linear_weight(
        model.lm_head,
        hf_state['language_model.lm_head.weight']
    )

    print("Weight loading complete!")


def load_model_from_hf(model_name: str = "zai-org/GLM-ASR-Nano-2512"):
    """
    Load GLM-ASR model from HuggingFace and create CuTile version.

    Returns:
        tuple: (cutile_model, hf_processor)
    """
    from transformers import AutoProcessor, GlmAsrForConditionalGeneration, AutoConfig
    from model import GlmAsrModel
    import torch

    print(f"Loading HuggingFace model: {model_name}")

    # Load config
    hf_config = AutoConfig.from_pretrained(model_name)
    cutile_config = create_config_from_hf(hf_config)

    print(f"Creating CuTile model with config:")
    print(f"  Audio: hidden={cutile_config.audio_hidden_size}, heads={cutile_config.audio_num_heads}, layers={cutile_config.audio_num_layers}")
    print(f"  Text: hidden={cutile_config.text_hidden_size}, heads={cutile_config.text_num_heads}, kv_heads={cutile_config.text_num_kv_heads}, layers={cutile_config.text_num_layers}")

    # Create CuTile model
    cutile_model = GlmAsrModel(cutile_config)

    # Load HF model
    print("Loading HuggingFace weights...")
    hf_model = GlmAsrForConditionalGeneration.from_pretrained(
        model_name,
        torch_dtype=torch.float32,
        device_map="cpu"
    )

    # Load processor
    processor = AutoProcessor.from_pretrained(model_name)

    # Transfer weights
    load_weights_from_hf_model(cutile_model, hf_model)

    # Free HF model memory
    del hf_model
    import gc
    gc.collect()

    return cutile_model, processor
