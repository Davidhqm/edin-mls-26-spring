"""
GLM-ASR Full Model
Educational implementation from scratch using PyTorch only

Combines:
- Audio Encoder (Whisper-style)
- Multi-modal Projector
- Text Decoder (Llama-style)
"""

import torch
import torch.nn as nn
from typing import Optional, Tuple, List

from config import GlmAsrConfig, AudioEncoderConfig, TextDecoderConfig
from encoder import GlmAsrEncoder
from decoder import LlamaForCausalLM
from layers import MultiModalProjector


class GlmAsrForConditionalGeneration(nn.Module):
    """
    GLM-ASR model for automatic speech recognition.

    This is an encoder-decoder model where:
    1. Audio encoder processes mel spectrograms into audio embeddings
    2. Projector transforms audio embeddings to text embedding space
    3. Audio embeddings replace <audio> tokens in the input
    4. Text decoder generates transcription autoregressively

    Architecture is similar to LLaVA/AudioFlamingo: vision/audio encoder + projector + LLM
    """

    def __init__(self, config: GlmAsrConfig = None):
        super().__init__()
        if config is None:
            config = GlmAsrConfig()

        self.config = config

        # Audio encoder (Whisper-style)
        self.audio_encoder = GlmAsrEncoder(config.audio_config)

        # Multi-modal projector
        # The audio features are reshaped from (batch, seq, hidden_size) to
        # (batch, seq // factor, intermediate_size) before projection
        self.audio_reshape_factor = config.audio_config.intermediate_size // config.audio_config.hidden_size
        self.multi_modal_projector = MultiModalProjector(
            audio_intermediate_size=config.audio_config.intermediate_size,
            text_hidden_size=config.text_config.hidden_size,
            activation=config.projector_hidden_act
        )

        # Text decoder (Llama-style)
        self.language_model = LlamaForCausalLM(config.text_config)

        # Audio token ID (placeholder in input_ids where audio features go)
        self.audio_token_id = config.audio_token_id

    def _merge_audio_features(
        self,
        input_ids: torch.LongTensor,
        input_features: torch.FloatTensor,
        audio_features: torch.FloatTensor
    ) -> torch.FloatTensor:
        """
        Merge audio features into input embeddings at audio token positions.

        The input_ids contain special <audio> tokens (audio_token_id).
        We replace these token embeddings with projected audio features.

        Args:
            input_ids: Token IDs with audio placeholders (batch, seq_len)
            input_features: Mel spectrogram features (batch, audio_len, mel_bins)
            audio_features: Encoded and projected audio (batch, audio_seq_len, hidden_size)

        Returns:
            Combined embeddings (batch, total_seq_len, hidden_size)
        """
        batch_size = input_ids.shape[0]

        # Get text embeddings
        text_embeds = self.language_model.model.embed_tokens(input_ids)

        # Find audio token positions
        audio_mask = input_ids == self.audio_token_id

        # For each sample in batch, replace audio tokens with audio features
        # This is a simplified version - assumes contiguous audio tokens
        merged_embeds = []

        for b in range(batch_size):
            # Find where audio tokens are
            audio_positions = torch.where(audio_mask[b])[0]

            if len(audio_positions) == 0:
                # No audio tokens, use text only
                merged_embeds.append(text_embeds[b])
            else:
                # Get parts before, during, and after audio
                first_audio_pos = audio_positions[0].item()
                last_audio_pos = audio_positions[-1].item()

                before_audio = text_embeds[b, :first_audio_pos]
                after_audio = text_embeds[b, last_audio_pos + 1:]

                # Concatenate: text before + audio features + text after
                merged = torch.cat([
                    before_audio,
                    audio_features[b],
                    after_audio
                ], dim=0)

                merged_embeds.append(merged)

        # Pad to same length and stack
        max_len = max(e.shape[0] for e in merged_embeds)
        padded_embeds = []

        for e in merged_embeds:
            if e.shape[0] < max_len:
                padding = torch.zeros(
                    max_len - e.shape[0],
                    e.shape[1],
                    device=e.device,
                    dtype=e.dtype
                )
                e = torch.cat([e, padding], dim=0)
            padded_embeds.append(e)

        return torch.stack(padded_embeds, dim=0)

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        input_features: Optional[torch.FloatTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: bool = False
    ) -> dict:
        """
        Forward pass for the full model.

        Args:
            input_ids: Token IDs with audio placeholders
            input_features: Mel spectrogram features
            attention_mask: Attention mask
            position_ids: Position IDs
            past_key_values: KV cache for generation
            inputs_embeds: Pre-computed embeddings (alternative to input_ids)
            labels: Labels for computing loss
            use_cache: Whether to return KV cache

        Returns:
            Dictionary with logits, loss, and optional cache
        """
        # If we have input_features, encode audio and merge with text
        if input_features is not None and inputs_embeds is None:
            # Encode audio
            audio_hidden_states = self.audio_encoder(input_features)

            # Reshape audio features: combine 'factor' consecutive frames
            # (batch, seq, hidden_size) -> (batch, seq // factor, intermediate_size)
            batch_size, seq_len, hidden_size = audio_hidden_states.shape

            # Ensure sequence length is divisible by reshape factor (truncate if needed)
            factor = self.audio_reshape_factor
            truncated_len = (seq_len // factor) * factor
            if truncated_len < seq_len:
                audio_hidden_states = audio_hidden_states[:, :truncated_len, :]

            audio_hidden_states = audio_hidden_states.reshape(
                batch_size, -1, self.config.audio_config.intermediate_size
            )

            # Project to text space
            audio_features = self.multi_modal_projector(audio_hidden_states)

            # Merge with text embeddings
            if input_ids is not None:
                inputs_embeds = self._merge_audio_features(
                    input_ids, input_features, audio_features
                )

                # Update attention mask for new sequence length
                if attention_mask is not None:
                    # Compute new sequence length
                    audio_seq_len = audio_features.shape[1]
                    num_audio_tokens = (input_ids == self.audio_token_id).sum(dim=1)

                    # New mask length = original - audio_tokens + audio_seq_len
                    new_mask_len = attention_mask.shape[1] - num_audio_tokens[0].item() + audio_seq_len
                    attention_mask = torch.ones(
                        attention_mask.shape[0], new_mask_len,
                        device=attention_mask.device,
                        dtype=attention_mask.dtype
                    )

                # Don't use input_ids now
                input_ids = None

        # Forward through language model
        outputs = self.language_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            inputs_embeds=inputs_embeds,
            past_key_values=past_key_values,
            use_cache=use_cache,
            labels=labels
        )

        return outputs

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.LongTensor,
        input_features: Optional[torch.FloatTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        max_new_tokens: int = 500,
        do_sample: bool = False,
        temperature: float = 1.0,
        top_k: int = 50,
        top_p: float = 1.0,
        eos_token_ids: Optional[List[int]] = None
    ) -> torch.LongTensor:
        """
        Generate text autoregressively.

        Args:
            input_ids: Initial token IDs (prompt with audio placeholders)
            input_features: Mel spectrogram features
            attention_mask: Attention mask
            max_new_tokens: Maximum tokens to generate
            do_sample: Whether to sample (vs greedy)
            temperature: Sampling temperature
            top_k: Top-k sampling
            top_p: Nucleus sampling threshold
            eos_token_ids: Token IDs that stop generation

        Returns:
            Generated token IDs including the prompt
        """
        if eos_token_ids is None:
            eos_token_ids = self.config.text_config.eos_token_ids

        batch_size = input_ids.shape[0]
        device = input_ids.device

        # First forward pass to get initial hidden states and cache
        outputs = self.forward(
            input_ids=input_ids,
            input_features=input_features,
            attention_mask=attention_mask,
            use_cache=True
        )

        logits = outputs["logits"]
        past_key_values = outputs["past_key_values"]

        # Get actual sequence length from logits (may differ from input_ids due to audio merging)
        actual_seq_len = logits.shape[1]

        # Create attention mask for the actual sequence length
        attention_mask = torch.ones((batch_size, actual_seq_len), device=device, dtype=torch.long)

        # Get the last token logits
        next_token_logits = logits[:, -1, :]

        # Generate tokens one by one
        generated_ids = input_ids.clone()

        for _ in range(max_new_tokens):
            # Apply temperature
            if do_sample and temperature != 1.0:
                next_token_logits = next_token_logits / temperature

            # Apply top-k filtering
            if do_sample and top_k > 0:
                indices_to_remove = next_token_logits < torch.topk(next_token_logits, top_k)[0][..., -1, None]
                next_token_logits[indices_to_remove] = float('-inf')

            # Apply top-p (nucleus) filtering
            if do_sample and top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
                cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)

                # Remove tokens with cumulative probability above threshold
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = 0

                indices_to_remove = sorted_indices_to_remove.scatter(
                    dim=-1, index=sorted_indices, src=sorted_indices_to_remove
                )
                next_token_logits[indices_to_remove] = float('-inf')

            # Sample or greedy
            if do_sample:
                probs = torch.softmax(next_token_logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
            else:
                next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)

            # Append to generated sequence
            generated_ids = torch.cat([generated_ids, next_token], dim=-1)

            # Check for EOS
            if all(next_token[i, 0].item() in eos_token_ids for i in range(batch_size)):
                break

            # Update attention mask for the new token
            attention_mask = torch.cat([
                attention_mask,
                torch.ones((batch_size, 1), device=device, dtype=attention_mask.dtype)
            ], dim=-1)

            # Forward pass for next token (no audio features, just the new token)
            outputs = self.forward(
                input_ids=next_token,
                attention_mask=attention_mask,
                past_key_values=past_key_values,
                use_cache=True
            )

            next_token_logits = outputs["logits"][:, -1, :]
            past_key_values = outputs["past_key_values"]

        return generated_ids


if __name__ == "__main__":
    # Test the full model
    from config import GlmAsrConfig, AudioEncoderConfig, TextDecoderConfig

    # Use smaller configs for testing
    audio_config = AudioEncoderConfig(
        hidden_size=256,
        intermediate_size=512,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=4,
        head_dim=64,
        num_mel_bins=128
    )

    text_config = TextDecoderConfig(
        hidden_size=256,
        intermediate_size=512,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=64,
        vocab_size=1000
    )

    config = GlmAsrConfig(
        audio_config=audio_config,
        text_config=text_config,
        audio_token_id=999  # Special token for audio
    )

    model = GlmAsrForConditionalGeneration(config)

    # Test inputs
    batch_size = 2
    audio_len = 100  # mel spectrogram frames
    mel_bins = 128
    text_len = 10

    # Create input with audio placeholder tokens
    input_ids = torch.randint(0, 998, (batch_size, text_len))
    input_ids[:, 2:5] = 999  # Audio tokens at positions 2, 3, 4

    input_features = torch.randn(batch_size, audio_len, mel_bins)
    attention_mask = torch.ones(batch_size, text_len)

    print(f"Input IDs shape: {input_ids.shape}")
    print(f"Input features shape: {input_features.shape}")

    # Forward pass
    outputs = model(
        input_ids=input_ids,
        input_features=input_features,
        attention_mask=attention_mask
    )

    print(f"Output logits shape: {outputs['logits'].shape}")

    # Test generation
    print("\nTesting generation:")
    generated = model.generate(
        input_ids=input_ids,
        input_features=input_features,
        attention_mask=attention_mask,
        max_new_tokens=10,
        do_sample=False
    )
    print(f"Generated shape: {generated.shape}")

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f"\nTotal parameters: {total_params:,}")
