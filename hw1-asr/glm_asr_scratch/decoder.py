"""
Text Decoder (Llama-style Transformer)
Educational implementation from scratch using PyTorch only
"""

import torch
import torch.nn as nn
from typing import Optional, Tuple, List

from config import TextDecoderConfig
from attention import CausalSelfAttention
from layers import RMSNorm, MLP


class DecoderLayer(nn.Module):
    """
    Single layer of the text decoder.

    Architecture: RMSNorm -> Causal Self-Attention -> Residual -> RMSNorm -> MLP -> Residual
    (Pre-norm architecture with RMSNorm, like Llama)
    """

    def __init__(self, config: TextDecoderConfig, layer_idx: int):
        super().__init__()
        self.layer_idx = layer_idx

        # Causal self-attention
        self.self_attn = CausalSelfAttention(
            hidden_size=config.hidden_size,
            num_attention_heads=config.num_attention_heads,
            num_key_value_heads=config.num_key_value_heads,
            head_dim=config.head_dim,
            attention_dropout=config.attention_dropout,
            bias=config.attention_bias,
            rope_theta=config.rope_theta,
            max_position_embeddings=config.max_position_embeddings
        )

        # RMSNorm instead of LayerNorm (Llama style)
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        # MLP with gating (SwiGLU style)
        self.mlp = MLP(
            hidden_size=config.hidden_size,
            intermediate_size=config.intermediate_size,
            activation=config.hidden_act,
            bias=config.mlp_bias,
            use_gating=True  # SwiGLU-style
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        use_cache: bool = False
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        """
        Forward pass for decoder layer.

        Args:
            hidden_states: Input tensor of shape (batch, seq_len, hidden_size)
            attention_mask: Optional attention mask
            position_ids: Position indices for RoPE
            past_key_value: Cached K, V from previous steps
            use_cache: Whether to return cached K, V

        Returns:
            Tuple of (output tensor, optional cached KV)
        """
        # Self-attention with pre-norm and residual
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states, present_key_value = self.self_attn(
            hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            use_cache=use_cache
        )
        hidden_states = residual + hidden_states

        # MLP with pre-norm and residual
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        return hidden_states, present_key_value


class LlamaModel(nn.Module):
    """
    Llama-style transformer decoder (without LM head).

    This is the base model that:
    1. Embeds input tokens
    2. Applies N decoder layers with causal attention
    3. Returns final hidden states
    """

    def __init__(self, config: TextDecoderConfig = None):
        super().__init__()
        if config is None:
            config = TextDecoderConfig()

        self.config = config

        # Token embedding
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)

        # Decoder layers
        self.layers = nn.ModuleList([
            DecoderLayer(config, layer_idx=i) for i in range(config.num_hidden_layers)
        ])

        # Final layer norm
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        past_key_values: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None,
        use_cache: bool = False
    ) -> Tuple[torch.Tensor, Optional[List[Tuple[torch.Tensor, torch.Tensor]]]]:
        """
        Forward pass for the decoder.

        Args:
            input_ids: Token IDs of shape (batch, seq_len)
            attention_mask: Attention mask of shape (batch, seq_len)
            position_ids: Position IDs for RoPE
            inputs_embeds: Optional pre-computed embeddings (for multimodal)
            past_key_values: List of cached KV pairs per layer
            use_cache: Whether to return cached KV pairs

        Returns:
            Tuple of (hidden states, optional KV cache)
        """
        # Get embeddings
        if inputs_embeds is None:
            hidden_states = self.embed_tokens(input_ids)
        else:
            hidden_states = inputs_embeds

        batch_size, seq_len, _ = hidden_states.shape

        # Generate position_ids if not provided
        if position_ids is None:
            if past_key_values is not None and len(past_key_values) > 0:
                # During generation, start from cached length
                past_length = past_key_values[0][0].shape[2]
                position_ids = torch.arange(
                    past_length, past_length + seq_len,
                    device=hidden_states.device
                ).unsqueeze(0)
            else:
                position_ids = torch.arange(seq_len, device=hidden_states.device).unsqueeze(0)

        # Prepare attention mask for causal attention
        if attention_mask is not None:
            # Convert padding mask to causal mask format
            # attention_mask: (batch, total_len) where 1 = attend, 0 = mask
            # We need: (batch, 1, seq_len, total_len) where 0 = attend, -inf = mask
            if past_key_values is not None and len(past_key_values) > 0:
                past_length = past_key_values[0][0].shape[2]
            else:
                past_length = 0

            total_len = past_length + seq_len
            attention_mask = attention_mask.view(batch_size, 1, 1, total_len)
            attention_mask = (1.0 - attention_mask) * -10000.0

        # Initialize cache list
        present_key_values = [] if use_cache else None

        # Apply decoder layers
        for i, layer in enumerate(self.layers):
            past_kv = past_key_values[i] if past_key_values is not None else None

            hidden_states, present_kv = layer(
                hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_value=past_kv,
                use_cache=use_cache
            )

            if use_cache:
                present_key_values.append(present_kv)

        # Final layer norm
        hidden_states = self.norm(hidden_states)

        return hidden_states, present_key_values


class LlamaForCausalLM(nn.Module):
    """
    Llama model with a language modeling head for next-token prediction.
    """

    def __init__(self, config: TextDecoderConfig = None):
        super().__init__()
        if config is None:
            config = TextDecoderConfig()

        self.config = config
        self.model = LlamaModel(config)

        # Language modeling head (projects hidden to vocab)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # Tie weights between embedding and lm_head (optional, common practice)
        # self.lm_head.weight = self.model.embed_tokens.weight

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        past_key_values: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None,
        use_cache: bool = False,
        labels: Optional[torch.LongTensor] = None
    ) -> dict:
        """
        Forward pass for causal language modeling.

        Args:
            input_ids: Token IDs of shape (batch, seq_len)
            attention_mask: Attention mask
            position_ids: Position IDs
            inputs_embeds: Pre-computed embeddings
            past_key_values: KV cache
            use_cache: Whether to return KV cache
            labels: Labels for computing loss

        Returns:
            Dictionary with logits, loss (if labels provided), and optional cache
        """
        hidden_states, present_key_values = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            inputs_embeds=inputs_embeds,
            past_key_values=past_key_values,
            use_cache=use_cache
        )

        # Get logits
        logits = self.lm_head(hidden_states)

        # Compute loss if labels provided
        loss = None
        if labels is not None:
            # Shift for next-token prediction
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()

            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(
                shift_logits.view(-1, self.config.vocab_size),
                shift_labels.view(-1)
            )

        return {
            "logits": logits,
            "loss": loss,
            "past_key_values": present_key_values
        }


if __name__ == "__main__":
    # Test the decoder
    from config import TextDecoderConfig

    # Use smaller config for testing
    test_config = TextDecoderConfig(
        hidden_size=256,
        intermediate_size=512,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,  # GQA
        head_dim=64,
        vocab_size=1000,
        max_position_embeddings=512
    )

    model = LlamaForCausalLM(test_config)

    # Test input
    batch_size = 2
    seq_len = 16

    input_ids = torch.randint(0, 1000, (batch_size, seq_len))
    attention_mask = torch.ones(batch_size, seq_len)

    print(f"Input shape: {input_ids.shape}")

    # Forward pass without cache
    outputs = model(input_ids, attention_mask=attention_mask)
    print(f"Logits shape: {outputs['logits'].shape}")
    print(f"Expected: ({batch_size}, {seq_len}, {test_config.vocab_size})")

    # Test with KV cache (autoregressive generation)
    print("\nTesting with KV cache:")
    outputs = model(input_ids, attention_mask=attention_mask, use_cache=True)
    past_kv = outputs['past_key_values']
    print(f"KV cache layers: {len(past_kv)}")
    print(f"KV shapes: K={past_kv[0][0].shape}, V={past_kv[0][1].shape}")

    # Generate one more token
    next_token = torch.randint(0, 1000, (batch_size, 1))
    new_mask = torch.ones(batch_size, seq_len + 1)
    outputs = model(next_token, attention_mask=new_mask, past_key_values=past_kv, use_cache=True)
    print(f"Next token logits shape: {outputs['logits'].shape}")

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f"\nTotal parameters: {total_params:,}")
