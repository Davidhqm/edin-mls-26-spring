"""
Weight Loader for GLM-ASR
Educational implementation using Triton for tile-based GPU programming

Loads pre-trained weights from safetensors format into the model.
"""

from pathlib import Path
from typing import Dict, Optional, Union

import torch


def create_config_from_hf(hf_config):
    """Create GlmAsrConfig from HuggingFace config."""
    from model import GlmAsrConfig

    ac = hf_config.audio_config
    tc = hf_config.text_config

    return GlmAsrConfig(
        audio_hidden_size=ac.hidden_size,
        audio_num_heads=ac.num_attention_heads,
        audio_num_layers=ac.num_hidden_layers,
        audio_intermediate_size=ac.intermediate_size,
        audio_max_position_embeddings=getattr(ac, "max_position_embeddings", 1500),
        text_hidden_size=tc.hidden_size,
        text_num_heads=tc.num_attention_heads,
        text_num_kv_heads=tc.num_key_value_heads,
        text_num_layers=tc.num_hidden_layers,
        text_intermediate_size=tc.intermediate_size,
        text_vocab_size=tc.vocab_size,
        text_max_position_embeddings=tc.max_position_embeddings,
        text_rope_base=getattr(tc, "rope_theta", 10000.0),
        projector_hidden_size=4096,
        projector_pool_factor=4,
        pad_token_id=getattr(tc, "pad_token_id", 0)
        if getattr(tc, "pad_token_id", None) is not None
        else 0,
        bos_token_id=getattr(tc, "bos_token_id", 1)
        if getattr(tc, "bos_token_id", None) is not None
        else 1,
        eos_token_id=getattr(tc, "eos_token_id", 2),
    )


def load_linear_weight(triton_linear, hf_weight, hf_bias=None):
    """Load weight (and optional bias) into Triton Linear layer."""
    triton_linear.weight = hf_weight.detach().to(torch.float32).clone()
    if hf_bias is not None and triton_linear.has_bias:
        triton_linear.bias_param = hf_bias.detach().to(torch.float32).clone()


def load_conv1d_weight_from_hf(triton_conv, hf_weight, hf_bias=None):
    """Load weight into Triton Conv1d layer from HF format."""
    weight = hf_weight.detach().to(torch.float32)
    out_channels, in_channels, kernel_size = weight.shape
    triton_conv.weight = weight.reshape(out_channels, in_channels * kernel_size).clone()

    if triton_conv.use_triton and (
        triton_conv.col_size_padded != triton_conv.col_size
        or triton_conv.out_channels_padded != out_channels
    ):
        triton_conv.weight_padded = torch.zeros(
            (triton_conv.out_channels_padded, triton_conv.col_size_padded),
            dtype=torch.float32,
        )
        triton_conv.weight_padded[:out_channels, : triton_conv.col_size] = triton_conv.weight
    else:
        triton_conv.weight_padded = triton_conv.weight

    if hf_bias is not None and triton_conv.has_bias:
        triton_conv.bias = hf_bias.detach().to(torch.float32).clone()


def load_layernorm_weight_from_hf(triton_ln, hf_weight, hf_bias):
    """Load LayerNorm weights."""
    triton_ln.weight = hf_weight.detach().to(torch.float32).clone()
    triton_ln.bias = hf_bias.detach().to(torch.float32).clone()


def load_rmsnorm_weight_from_hf(triton_rms, hf_weight):
    """Load RMSNorm weight."""
    triton_rms.weight = hf_weight.detach().to(torch.float32).clone()


def load_embedding_weight_from_hf(triton_emb, hf_weight):
    """Load Embedding weight."""
    triton_emb.weight = hf_weight.detach().to(torch.float32).clone()


def load_weights_from_hf_model(model, hf_model) -> None:
    """
    Load weights from HuggingFace GLM-ASR model into Triton model.
    """
    hf_state = hf_model.state_dict()

    print("Loading audio encoder weights...")

    load_conv1d_weight_from_hf(
        model.audio_encoder.conv1,
        hf_state["audio_tower.conv1.weight"],
        hf_state["audio_tower.conv1.bias"],
    )
    load_conv1d_weight_from_hf(
        model.audio_encoder.conv2,
        hf_state["audio_tower.conv2.weight"],
        hf_state["audio_tower.conv2.bias"],
    )

    if "audio_tower.embed_positions.weight" in hf_state:
        model.audio_encoder.embed_positions = (
            hf_state["audio_tower.embed_positions.weight"]
            .detach()
            .to(torch.float32)
            .clone()
        )

    for i, layer in enumerate(model.audio_encoder.layers):
        prefix = f"audio_tower.layers.{i}"

        load_layernorm_weight_from_hf(
            layer.self_attn_layer_norm,
            hf_state[f"{prefix}.input_layernorm.weight"],
            hf_state[f"{prefix}.input_layernorm.bias"],
        )

        load_linear_weight(
            layer.q_proj,
            hf_state[f"{prefix}.self_attn.q_proj.weight"],
            hf_state.get(f"{prefix}.self_attn.q_proj.bias"),
        )
        load_linear_weight(
            layer.k_proj,
            hf_state[f"{prefix}.self_attn.k_proj.weight"],
            hf_state.get(f"{prefix}.self_attn.k_proj.bias"),
        )
        load_linear_weight(
            layer.v_proj,
            hf_state[f"{prefix}.self_attn.v_proj.weight"],
            hf_state.get(f"{prefix}.self_attn.v_proj.bias"),
        )
        load_linear_weight(
            layer.out_proj,
            hf_state[f"{prefix}.self_attn.o_proj.weight"],
            hf_state.get(f"{prefix}.self_attn.o_proj.bias"),
        )

        load_layernorm_weight_from_hf(
            layer.final_layer_norm,
            hf_state[f"{prefix}.post_attention_layernorm.weight"],
            hf_state[f"{prefix}.post_attention_layernorm.bias"],
        )

        load_linear_weight(
            layer.fc1,
            hf_state[f"{prefix}.mlp.fc1.weight"],
            hf_state[f"{prefix}.mlp.fc1.bias"],
        )
        load_linear_weight(
            layer.fc2,
            hf_state[f"{prefix}.mlp.fc2.weight"],
            hf_state[f"{prefix}.mlp.fc2.bias"],
        )

    load_layernorm_weight_from_hf(
        model.audio_encoder.layer_norm,
        hf_state["audio_tower.norm.weight"],
        hf_state["audio_tower.norm.bias"],
    )

    print("Loading multi-modal projector weights...")

    load_linear_weight(
        model.multi_modal_projector.linear_1,
        hf_state["multi_modal_projector.linear_1.weight"],
        hf_state["multi_modal_projector.linear_1.bias"],
    )
    load_linear_weight(
        model.multi_modal_projector.linear_2,
        hf_state["multi_modal_projector.linear_2.weight"],
        hf_state["multi_modal_projector.linear_2.bias"],
    )

    print("Loading text decoder weights...")

    load_embedding_weight_from_hf(
        model.text_decoder.embed_tokens,
        hf_state["language_model.model.embed_tokens.weight"],
    )

    for i, layer in enumerate(model.text_decoder.layers):
        prefix = f"language_model.model.layers.{i}"

        load_rmsnorm_weight_from_hf(
            layer.input_layernorm,
            hf_state[f"{prefix}.input_layernorm.weight"],
        )

        load_linear_weight(
            layer.q_proj,
            hf_state[f"{prefix}.self_attn.q_proj.weight"],
        )
        load_linear_weight(
            layer.k_proj,
            hf_state[f"{prefix}.self_attn.k_proj.weight"],
        )
        load_linear_weight(
            layer.v_proj,
            hf_state[f"{prefix}.self_attn.v_proj.weight"],
        )
        load_linear_weight(
            layer.o_proj,
            hf_state[f"{prefix}.self_attn.o_proj.weight"],
        )

        load_rmsnorm_weight_from_hf(
            layer.post_attention_layernorm,
            hf_state[f"{prefix}.post_attention_layernorm.weight"],
        )

        load_linear_weight(
            layer.mlp.gate_proj,
            hf_state[f"{prefix}.mlp.gate_proj.weight"],
        )
        load_linear_weight(
            layer.mlp.up_proj,
            hf_state[f"{prefix}.mlp.up_proj.weight"],
        )
        load_linear_weight(
            layer.mlp.down_proj,
            hf_state[f"{prefix}.mlp.down_proj.weight"],
        )

    load_rmsnorm_weight_from_hf(
        model.text_decoder.norm,
        hf_state["language_model.model.norm.weight"],
    )

    load_linear_weight(
        model.lm_head,
        hf_state["language_model.lm_head.weight"],
    )

    print("Weight loading complete!")


def load_model_from_hf(model_name: str = "zai-org/GLM-ASR-Nano-2512"):
    """
    Load GLM-ASR model from HuggingFace and create Triton version.
    """
    from transformers import AutoProcessor, GlmAsrForConditionalGeneration, AutoConfig
    from model import GlmAsrModel

    print(f"Loading HuggingFace model: {model_name}")

    hf_config = AutoConfig.from_pretrained(model_name)
    triton_config = create_config_from_hf(hf_config)

    print("Creating Triton model with config:")
    print(
        f"  Audio: hidden={triton_config.audio_hidden_size}, heads={triton_config.audio_num_heads}, layers={triton_config.audio_num_layers}"
    )
    print(
        f"  Text: hidden={triton_config.text_hidden_size}, heads={triton_config.text_num_heads}, kv_heads={triton_config.text_num_kv_heads}, layers={triton_config.text_num_layers}"
    )

    triton_model = GlmAsrModel(triton_config)

    print("Loading HuggingFace weights...")
    hf_model = GlmAsrForConditionalGeneration.from_pretrained(
        model_name, torch_dtype=torch.float32, device_map="cpu"
    )

    processor = AutoProcessor.from_pretrained(model_name)

    load_weights_from_hf_model(triton_model, hf_model)

    del hf_model
    import gc

    gc.collect()

    return triton_model, processor
