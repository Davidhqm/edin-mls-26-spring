# GLM-ASR from Scratch

An educational PyTorch implementation of the GLM-ASR-Nano-2512 speech recognition model, built without the transformers library.

## Overview

This implementation demonstrates the core components of a modern speech recognition model:

1. **Audio Feature Extraction** - Whisper-style mel spectrogram extraction
2. **Audio Encoder** - Transformer encoder with RoPE embeddings
3. **Multi-modal Projector** - Bridges audio and text embedding spaces
4. **Text Decoder** - Llama-style causal transformer for text generation

## Architecture

```
Audio Input (16kHz waveform)
    │
    ▼
┌─────────────────────────┐
│  Mel Spectrogram        │  (128 mel bins)
│  Feature Extraction     │
└─────────────────────────┘
    │
    ▼
┌─────────────────────────┐
│  Conv1D Subsampling     │  (2x reduction)
└─────────────────────────┘
    │
    ▼
┌─────────────────────────┐
│  Audio Encoder          │  32 transformer layers
│  (Whisper-style)        │  1280 hidden size
│  - Self Attention       │  20 attention heads
│  - RoPE (partial 50%)   │
│  - GELU MLP             │
└─────────────────────────┘
    │
    ▼
┌─────────────────────────┐
│  Multi-Modal Projector  │  1280 → 2048
│  (2-layer MLP + GELU)   │
└─────────────────────────┘
    │
    ▼
┌─────────────────────────┐
│  Text Decoder           │  28 transformer layers
│  (Llama-style)          │  2048 hidden size
│  - Causal Self Attention│  16 attention heads (GQA: 4 KV)
│  - RoPE                 │
│  - SwiGLU MLP           │
│  - RMSNorm              │
└─────────────────────────┘
    │
    ▼
┌─────────────────────────┐
│  LM Head                │  2048 → 59264 vocab
└─────────────────────────┘
    │
    ▼
Text Output (transcription)
```

## Files

| File | Description |
|------|-------------|
| `config.py` | Configuration dataclasses for all model components |
| `audio_features.py` | Mel spectrogram extraction (Whisper-style) |
| `rope.py` | Rotary Position Embeddings implementation |
| `attention.py` | Multi-head attention with GQA support |
| `layers.py` | MLP, RMSNorm, and other layer implementations |
| `encoder.py` | Audio encoder (Whisper-style transformer) |
| `decoder.py` | Text decoder (Llama-style transformer) |
| `model.py` | Full GLM-ASR model combining all components |
| `tokenizer.py` | BPE tokenizer that loads from tokenizer.json |
| `weight_loader.py` | Safetensors weight loading |
| `torch_glm.py` | Main inference script (PyTorch-only, no transformers) |

## Dependencies

- PyTorch (core deep learning)
- NumPy (audio processing)
- SciPy (optional, for audio file loading)

No transformers, tokenizers, or other HuggingFace libraries required!

## Usage

### Quick Test (Random Weights)

```bash
cd glm_asr_scratch
python torch_glm.py
```

### With Pretrained Weights

1. Download the model from HuggingFace:
   ```
   zai-org/GLM-ASR-Nano-2512
   ├── config.json
   ├── model.safetensors (4.52 GB)
   ├── processor_config.json
   └── tokenizer.json
   ```

2. Run inference:
   ```bash
   python torch_glm.py /path/to/model
   ```

### Python API

```python
from torch_glm import load_model_and_processor, transcribe

# Load model + processor
model, processor = load_model_and_processor(model_path="path/to/model")

# Transcribe audio
text = transcribe(model, processor, "audio.wav")
print(text[0] if isinstance(text, list) else text)
```

## Key Concepts

### Rotary Position Embeddings (RoPE)

RoPE encodes position by rotating query and key vectors:
- Each pair of dimensions is treated as 2D coordinates
- Rotation angle depends on position and dimension
- Makes attention scores depend on relative position

```python
# Rotation formula
q_rot = q * cos(θ) + rotate_half(q) * sin(θ)
```

### Grouped Query Attention (GQA)

Reduces memory by sharing key-value heads across query heads:
- Standard MHA: 16 Q heads, 16 KV heads
- GQA in this model: 16 Q heads, 4 KV heads
- Each KV head serves 4 Q heads

### SwiGLU MLP

Gated linear unit used in modern LLMs:
```python
output = down_proj(silu(gate_proj(x)) * up_proj(x))
```

### RMSNorm

Simplified layer normalization:
```python
x_norm = x / sqrt(mean(x²) + eps) * weight
```

## Model Sizes

| Component | Parameters |
|-----------|------------|
| Audio Encoder | ~450M |
| Projector | ~5M |
| Text Decoder | ~1B |
| **Total** | **~1.5B** |

## Educational Notes

This implementation prioritizes clarity over optimization:
- Explicit loops instead of fused operations
- Verbose comments explaining each step
- No attention optimizations
- Easy to trace through with a debugger

For production use, consider:
- Using the official transformers implementation
- Attention optimizations for faster inference
- Quantization for reduced memory
- KV cache optimizations

## References

- [GLM-ASR HuggingFace](https://huggingface.co/zai-org/GLM-ASR-Nano-2512)
- [RoPE Paper](https://arxiv.org/abs/2104.09864)
- [Llama Paper](https://arxiv.org/abs/2302.13971)
- [Whisper Paper](https://arxiv.org/abs/2212.04356)
