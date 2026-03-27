"""
GLM-ASR Inference using PyTorch Only (No Transformers)

This script replicates the functionality of test.py but uses our
from-scratch implementation instead of the transformers library.

Usage:
    python torch_glm.py                    # Use librispeech test audio
    python torch_glm.py path/to/audio.wav  # Use custom audio file
"""

import os
import sys
import json
import torch
import numpy as np
from pathlib import Path
from typing import Optional, Union, List

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import GlmAsrConfig, AudioEncoderConfig, TextDecoderConfig, AudioProcessorConfig
from model import GlmAsrForConditionalGeneration
from audio_features import WhisperFeatureExtractor
from tokenizer import Tokenizer
from weight_loader import load_weights_into_model


class GlmAsrProcessor:
    """
    Processor for GLM-ASR that handles audio preprocessing and tokenization.
    Equivalent to AutoProcessor from transformers.
    """

    def __init__(
        self,
        feature_extractor: WhisperFeatureExtractor,
        tokenizer: Tokenizer,
        audio_token: str = "<|audio|>",
        audio_token_id: int = 59260,
        default_prompt: str = "Please transcribe this audio into text"
    ):
        self.feature_extractor = feature_extractor
        self.tokenizer = tokenizer
        self.audio_token = audio_token
        self.audio_token_id = audio_token_id
        self.default_prompt = default_prompt

        # Special token IDs for chat template
        self.user_token_id = 59253        # <|user|>
        self.assistant_token_id = 59254   # <|assistant|>
        self.begin_audio_token_id = 59261 # <|begin_of_audio|>
        self.end_audio_token_id = 59262   # <|end_of_audio|>
        self.newline_token_id = 10        # \n

    @property
    def sampling_rate(self) -> int:
        return self.feature_extractor.sampling_rate

    def apply_transcription_request(
        self,
        audio: Union[np.ndarray, str, List],
        prompt: Optional[str] = None
    ) -> dict:
        """
        Prepare inputs for transcription, matching transformers' behavior.

        The input format follows the chat template:
        <|begin_of_audio|><|pad|>...<|end_of_audio|><|user|>{prompt}<|assistant|>

        Args:
            audio: Audio waveform (numpy array), file path, or list of audio
            prompt: Custom transcription prompt

        Returns:
            Dictionary with input_ids, input_features, attention_mask
        """
        if prompt is None:
            prompt = self.default_prompt

        # Handle list of audio
        if isinstance(audio, list):
            # Process batch - for simplicity, just use first item
            audio = audio[0]

        # Load audio if path
        if isinstance(audio, str):
            audio = self._load_audio(audio)

        # Extract mel spectrogram features
        features = self.feature_extractor(
            audio,
            sampling_rate=self.sampling_rate,
            padding="max_length"
        )

        input_features = features['input_features']

        # Calculate number of audio tokens after processing.
        # Match demo/triton template: use mel_frames // 4.
        mel_frames = input_features.shape[1]
        num_audio_tokens = mel_frames // 4

        # Ensure we have at least some tokens
        num_audio_tokens = max(1, num_audio_tokens)

        # Tokenize the prompt (without adding BOS/EOS since we're building the chat template manually)
        prompt_token_ids = self.tokenizer.encode(prompt, add_special_tokens=False)

        # Build input_ids following the chat template:
        # <|begin_of_audio|><|pad|>...<|end_of_audio|><|user|>{prompt}<|assistant|>
        input_ids_list = [
            self.begin_audio_token_id,    # <|begin_of_audio|>
        ]

        # Add audio placeholder tokens
        input_ids_list.extend([self.audio_token_id] * num_audio_tokens)

        input_ids_list.extend([
            self.end_audio_token_id,      # <|end_of_audio|>
            self.user_token_id,           # <|user|>
        ])

        # Add prompt tokens
        input_ids_list.extend(prompt_token_ids)

        input_ids_list.extend([
            self.assistant_token_id,      # <|assistant|>
        ])

        # Convert to tensor
        input_ids = torch.tensor([input_ids_list], dtype=torch.long)

        # Create attention mask
        attention_mask = torch.ones_like(input_ids)

        return {
            'input_ids': input_ids,
            'input_features': input_features,
            'attention_mask': attention_mask
        }

    def _load_audio(self, file_path: str) -> np.ndarray:
        """Load audio file and resample to target sample rate."""
        try:
            from scipy.io import wavfile
            from scipy import signal

            sr, audio = wavfile.read(file_path)

            # Convert to float
            if audio.dtype == np.int16:
                audio = audio.astype(np.float32) / 32768.0
            elif audio.dtype == np.int32:
                audio = audio.astype(np.float32) / 2147483648.0

            # Convert stereo to mono
            if len(audio.shape) > 1:
                audio = audio.mean(axis=1)

            # Resample if needed
            if sr != self.sampling_rate:
                num_samples = int(len(audio) * self.sampling_rate / sr)
                audio = signal.resample(audio, num_samples)

            return audio.astype(np.float32)

        except ImportError:
            raise ImportError("scipy is required for audio loading. Install with: pip install scipy")

    def batch_decode(
        self,
        token_ids: Union[torch.Tensor, List[List[int]]],
        skip_special_tokens: bool = True
    ) -> List[str]:
        """Decode token IDs to text strings."""
        if isinstance(token_ids, torch.Tensor):
            token_ids = token_ids.tolist()
        return self.tokenizer.batch_decode(token_ids, skip_special_tokens)


def load_model_and_processor(
    model_path: Optional[str] = None,
    device: str = "auto",
    dtype: str = "auto"
) -> tuple:
    """
    Load GLM-ASR model and processor.

    Args:
        model_path: Path to model directory. If None, uses HuggingFace cache.
        device: Device to use ("auto", "cuda", "cpu")
        dtype: Data type ("auto", "float32", "float16", "bfloat16")

    Returns:
        Tuple of (model, processor)
    """
    # Find model path
    if model_path is None:
        # Look in HuggingFace cache
        cache_dir = Path.home() / ".cache" / "huggingface" / "hub" / "models--zai-org--GLM-ASR-Nano-2512"
        if cache_dir.exists():
            # Find the snapshot
            snapshots = list((cache_dir / "snapshots").iterdir())
            if snapshots:
                model_path = str(snapshots[0])

    if model_path is None or not Path(model_path).exists():
        raise FileNotFoundError(
            "Model not found. Please download it first:\n"
            "  huggingface-cli download zai-org/GLM-ASR-Nano-2512"
        )

    model_path = Path(model_path)
    print(f"Loading model from: {model_path}")

    # Load config
    with open(model_path / "config.json", 'r') as f:
        config_dict = json.load(f)

    # Create config objects
    audio_config = AudioEncoderConfig(
        hidden_size=config_dict['audio_config']['hidden_size'],
        intermediate_size=config_dict['audio_config']['intermediate_size'],
        num_hidden_layers=config_dict['audio_config']['num_hidden_layers'],
        num_attention_heads=config_dict['audio_config']['num_attention_heads'],
        num_key_value_heads=config_dict['audio_config']['num_key_value_heads'],
        head_dim=config_dict['audio_config']['head_dim'],
        num_mel_bins=config_dict['audio_config']['num_mel_bins'],
        max_position_embeddings=config_dict['audio_config']['max_position_embeddings'],
        hidden_act=config_dict['audio_config']['hidden_act'],
        partial_rotary_factor=config_dict['audio_config']['partial_rotary_factor'],
        rope_theta=config_dict['audio_config']['rope_parameters']['rope_theta']
    )

    text_config = TextDecoderConfig(
        hidden_size=config_dict['text_config']['hidden_size'],
        intermediate_size=config_dict['text_config']['intermediate_size'],
        num_hidden_layers=config_dict['text_config']['num_hidden_layers'],
        num_attention_heads=config_dict['text_config']['num_attention_heads'],
        num_key_value_heads=config_dict['text_config']['num_key_value_heads'],
        head_dim=config_dict['text_config']['head_dim'],
        vocab_size=config_dict['text_config']['vocab_size'],
        max_position_embeddings=config_dict['text_config']['max_position_embeddings'],
        hidden_act=config_dict['text_config']['hidden_act'],
        rms_norm_eps=config_dict['text_config']['rms_norm_eps'],
        attention_bias=config_dict['text_config']['attention_bias'],
        mlp_bias=config_dict['text_config']['mlp_bias'],
        rope_theta=config_dict['text_config']['rope_parameters']['rope_theta'],
        eos_token_ids=config_dict['text_config']['eos_token_id']
    )

    config = GlmAsrConfig(
        audio_config=audio_config,
        text_config=text_config,
        audio_token_id=config_dict['audio_token_id'],
        projector_hidden_act=config_dict['projector_hidden_act']
    )

    # Determine device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    # Determine dtype
    if dtype == "auto":
        torch_dtype = torch.bfloat16 if device == "cuda" else torch.float32
    elif dtype == "bfloat16":
        torch_dtype = torch.bfloat16
    elif dtype == "float16":
        torch_dtype = torch.float16
    else:
        torch_dtype = torch.float32

    print(f"Using device: {device}, dtype: {torch_dtype}")

    # Create model
    print("Creating model...")
    model = GlmAsrForConditionalGeneration(config)

    # Load weights
    print("Loading weights...")
    weights_path = model_path / "model.safetensors"
    load_weights_into_model(model, str(weights_path), verbose=False)

    # Move to device
    model = model.to(device=device, dtype=torch_dtype)
    model.eval()

    # Create processor
    with open(model_path / "processor_config.json", 'r') as f:
        proc_config = json.load(f)

    audio_proc_config = AudioProcessorConfig(
        sampling_rate=proc_config['feature_extractor']['sampling_rate'],
        n_fft=proc_config['feature_extractor']['n_fft'],
        hop_length=proc_config['feature_extractor']['hop_length'],
        chunk_length=proc_config['feature_extractor']['chunk_length'],
        n_samples=proc_config['feature_extractor']['n_samples'],
        feature_size=proc_config['feature_extractor']['feature_size'],
        nb_max_frames=proc_config['feature_extractor']['nb_max_frames']
    )

    feature_extractor = WhisperFeatureExtractor(audio_proc_config)
    tokenizer = Tokenizer.from_pretrained(str(model_path))

    processor = GlmAsrProcessor(
        feature_extractor=feature_extractor,
        tokenizer=tokenizer,
        audio_token_id=config.audio_token_id,
        default_prompt=proc_config.get('default_transcription_prompt', 'Please transcribe this audio into text')
    )

    return model, processor


def transcribe(
    model: GlmAsrForConditionalGeneration,
    processor: GlmAsrProcessor,
    audio: Union[np.ndarray, str],
    max_new_tokens: int = 500,
    do_sample: bool = False
) -> str:
    """
    Transcribe audio to text.

    Args:
        model: Loaded GLM-ASR model
        processor: Loaded processor
        audio: Audio waveform or file path
        max_new_tokens: Maximum tokens to generate
        do_sample: Whether to use sampling

    Returns:
        Transcribed text
    """
    # Get device and dtype from model
    device = next(model.parameters()).device
    dtype = next(model.parameters()).dtype

    # Prepare inputs
    inputs = processor.apply_transcription_request(audio)

    # Move to device
    inputs = {
        'input_ids': inputs['input_ids'].to(device),
        'input_features': inputs['input_features'].to(device, dtype=dtype),
        'attention_mask': inputs['attention_mask'].to(device)
    }

    input_len = inputs['input_ids'].shape[1]

    # Generate
    with torch.no_grad():
        outputs = model.generate(
            input_ids=inputs['input_ids'],
            input_features=inputs['input_features'],
            attention_mask=inputs['attention_mask'],
            max_new_tokens=max_new_tokens,
            do_sample=do_sample
        )

    # Decode only the new tokens
    generated_ids = outputs[:, input_len:]
    decoded = processor.batch_decode(generated_ids, skip_special_tokens=True)

    return decoded


def load_librispeech_sample(sampling_rate: int = 16000) -> np.ndarray:
    """
    Load a sample from the librispeech dataset.

    This replicates the behavior from test.py:
        ds = load_dataset("hf-internal-testing/librispeech_asr_dummy", "clean", split="validation")
        ds = ds.cast_column("audio", Audio(sampling_rate=16000))
        audio_array = ds[0]["audio"]["array"]
    """
    try:
        from datasets import load_dataset, Audio

        print("Loading librispeech sample from HuggingFace datasets...")
        ds = load_dataset("hf-internal-testing/librispeech_asr_dummy", "clean", split="validation")
        ds = ds.cast_column("audio", Audio(sampling_rate=sampling_rate))
        audio_array = ds[0]["audio"]["array"]

        print(f"Loaded audio: {len(audio_array)} samples @ {sampling_rate}Hz")
        print(f"Duration: {len(audio_array) / sampling_rate:.2f}s")

        # Expected transcription for reference
        expected_text = ds[0].get("text", "")
        if expected_text:
            print(f"Expected transcription: {expected_text}")

        return audio_array.astype(np.float32)

    except ImportError:
        raise ImportError(
            "The 'datasets' library is required to load librispeech samples.\n"
            "Install with: pip install datasets\n"
            "Or provide a path to an audio file as argument."
        )


def main():
    """Main function - equivalent to test.py"""
    print("=" * 60)
    print("GLM-ASR Transcription (PyTorch Only - No Transformers)")
    print("=" * 60)

    # Load model and processor
    processor = None
    model = None

    try:
        model, processor = load_model_and_processor()
    except FileNotFoundError as e:
        print(f"Error: {e}")
        return 1

    # Get audio
    if len(sys.argv) > 1:
        # Use provided audio file
        audio_path = sys.argv[1]
        print(f"\nLoading audio from: {audio_path}")
        audio_array = processor._load_audio(audio_path)
        print(f"Audio: {len(audio_array)} samples @ {processor.sampling_rate}Hz")
        print(f"Duration: {len(audio_array) / processor.sampling_rate:.2f}s")
    else:
        # Load from librispeech dataset
        print("\nNo audio file provided, using librispeech sample...")
        try:
            audio_array = load_librispeech_sample(processor.sampling_rate)
        except ImportError as e:
            print(f"Error: {e}")
            return 1

    # Prepare inputs (equivalent to processor.apply_transcription_request)
    print("\nPreparing inputs...")
    inputs = processor.apply_transcription_request(audio_array)

    # Move to model device/dtype
    device = next(model.parameters()).device
    dtype = next(model.parameters()).dtype

    inputs = {
        'input_ids': inputs['input_ids'].to(device),
        'input_features': inputs['input_features'].to(device, dtype=dtype),
        'attention_mask': inputs['attention_mask'].to(device)
    }

    print(f"Input features shape: {inputs['input_features'].shape}")
    print(f"Input IDs shape: {inputs['input_ids'].shape}")

    # Generate (equivalent to model.generate)
    print("\nGenerating transcription...")
    input_len = inputs['input_ids'].shape[1]

    with torch.no_grad():
        outputs = model.generate(
            input_ids=inputs['input_ids'],
            input_features=inputs['input_features'],
            attention_mask=inputs['attention_mask'],
            do_sample=False,
            max_new_tokens=500
        )

    # Decode (equivalent to processor.batch_decode)
    decoded_outputs = processor.batch_decode(
        outputs[:, input_len:],
        skip_special_tokens=True
    )

    print("\n" + "=" * 60)
    print("Transcription Result:")
    print("=" * 60)
    print(decoded_outputs)

    return 0


if __name__ == "__main__":
    sys.exit(main())
