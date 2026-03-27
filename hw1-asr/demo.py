"""
GLM-ASR Student Demo
Test your implementation with audio transcription.
"""

import streamlit as st
import numpy as np
import time
import sys
import os
from pathlib import Path

BASE_DIR = Path(__file__).parent

st.set_page_config(
    page_title="GLM-ASR Student Demo",
    page_icon="🎤",
    layout="centered"
)

def get_cached_model(model_key, loader_fn):
    """Cache a single model bundle per session to avoid repeated loads."""
    current_key = st.session_state.get("current_model_key")
    current_bundle = st.session_state.get("current_model_bundle")

    if current_key == model_key and current_bundle is not None:
        return current_bundle, True

    if current_bundle is not None and current_key != model_key:
        st.session_state.current_model_bundle = None
        st.session_state.current_model_key = None
        release_model_bundle(current_bundle)
        clear_other_model_caches(except_key=model_key)

    bundle = loader_fn()
    st.session_state.current_model_key = model_key
    st.session_state.current_model_bundle = bundle
    return bundle, False


def release_model_bundle(bundle):
    """Best-effort release of GPU/CPU memory for a model bundle."""
    try:
        del bundle
    except Exception:
        pass

    try:
        import gc
        gc.collect()
    except Exception:
        pass

    try:
        import cupy as cp
        cp.get_default_memory_pool().free_all_blocks()
        cp.get_default_pinned_memory_pool().free_all_blocks()
    except Exception:
        pass

    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
    except Exception:
        pass


def clear_other_model_caches(except_key=None):
    """Clear Streamlit cached resources for other models."""
    for key, (_, loader_fn) in MODEL_LOADERS.items():
        if key == except_key:
            continue
        clear_fn = getattr(loader_fn, "clear", None)
        if callable(clear_fn):
            clear_fn()


def force_reload_models():
    """Force clear Streamlit cache and module cache for model reload."""
    current_key = st.session_state.get("current_model_key")
    current_bundle = st.session_state.get("current_model_bundle")

    if current_bundle is not None:
        release_model_bundle(current_bundle)

    if current_key in MODEL_LOADERS:
        _, loader_fn = MODEL_LOADERS[current_key]
        clear_fn = getattr(loader_fn, "clear", None)
        if callable(clear_fn):
            clear_fn()

    try:
        st.cache_resource.clear()
    except Exception:
        pass

    for mod_name in ['weight_loader', 'model', 'layers', 'attention', 'rope', 'conv', 'decode_attention']:
        if mod_name in sys.modules:
            del sys.modules[mod_name]

    st.session_state.current_model_bundle = None
    st.session_state.current_model_key = None


def _prepare_model_path(folder_path):
    """Ensure target folder is first in sys.path and remove other model paths."""
    for p in list(sys.path):
        if os.path.isdir(p) and os.path.basename(p).startswith("glm_asr_"):
            try:
                sys.path.remove(p)
            except ValueError:
                pass
    if folder_path not in sys.path:
        sys.path.insert(0, folder_path)
    else:
        sys.path.remove(folder_path)
        sys.path.insert(0, folder_path)


def load_cutile_model_generic(folder_name, backend_config=None):
    """Load CuTile model from specified folder."""
    import importlib

    start = time.perf_counter()

    cutile_path = str(BASE_DIR / folder_name)
    _prepare_model_path(cutile_path)

    # Clear cached modules
    mods_to_clear = [m for m in sys.modules.keys()
                     if m in ['weight_loader', 'model', 'layers', 'attention', 'rope', 'conv', 'decode_attention']]
    for m in mods_to_clear:
        del sys.modules[m]

    # Apply backend configuration if specified
    if backend_config:
        layers = importlib.import_module("layers")
        if 'BACKEND' in backend_config:
            layers.Linear.BACKEND = backend_config['BACKEND']
        if 'USE_CUBLAS_FP16' in backend_config:
            layers.MLP.USE_CUBLAS_FP16 = backend_config['USE_CUBLAS_FP16']
        if 'FUSED' in backend_config:
            layers.MLP.FUSED = backend_config['FUSED']

    weight_loader = importlib.import_module("weight_loader")
    model, processor = weight_loader.load_model_from_hf("zai-org/GLM-ASR-Nano-2512")

    # Pre-load tokenizer
    from tokenizers import Tokenizer
    from huggingface_hub import snapshot_download
    model_dir = snapshot_download("zai-org/GLM-ASR-Nano-2512")
    tokenizer = Tokenizer.from_file(str(Path(model_dir) / "tokenizer.json"))

    load_time_ms = (time.perf_counter() - start) * 1000

    return model, processor, tokenizer, load_time_ms


@st.cache_resource
def load_cutile_model():
    """Load CuTile template model."""
    return load_cutile_model_generic("glm_asr_cutile_template")


@st.cache_resource
def load_cutile_example_model():
    """Load CuTile Example (Initial CuPy) model."""
    config = {
        'BACKEND': 'cublas',
        'FUSED': False
    }
    return load_cutile_model_generic("glm_asr_cutile_example", config)


def load_triton_model_generic(folder_name, backend_config=None):
    """Load Triton model from specified folder."""
    import importlib

    start = time.perf_counter()

    triton_path = str(BASE_DIR / folder_name)
    _prepare_model_path(triton_path)

    mods_to_clear = [m for m in sys.modules.keys()
                     if m in ['weight_loader', 'model', 'layers', 'attention', 'rope', 'conv', 'decode_attention']]
    for m in mods_to_clear:
        del sys.modules[m]

    if backend_config:
        layers = importlib.import_module("layers")
        if 'BACKEND' in backend_config:
            layers.Linear.BACKEND = backend_config['BACKEND']
        if 'FUSED' in backend_config:
            layers.MLP.FUSED = backend_config['FUSED']

    weight_loader = importlib.import_module("weight_loader")
    model, processor = weight_loader.load_model_from_hf("zai-org/GLM-ASR-Nano-2512")

    from tokenizers import Tokenizer
    from huggingface_hub import snapshot_download
    model_dir = snapshot_download("zai-org/GLM-ASR-Nano-2512")
    tokenizer = Tokenizer.from_file(str(Path(model_dir) / "tokenizer.json"))

    load_time_ms = (time.perf_counter() - start) * 1000

    return model, processor, tokenizer, load_time_ms


@st.cache_resource
def load_triton_example_model():
    """Load Triton Example (Baseline) model."""
    config = {
        'BACKEND': 'cublas',
        'FUSED': False
    }
    return load_triton_model_generic("glm_asr_triton_example", config)


@st.cache_resource
def load_triton_template_model():
    """Load Triton Template model."""
    config = {
        'BACKEND': 'cublas',
        'FUSED': False
    }
    return load_triton_model_generic("glm_asr_triton_template", config)


def load_torch_model(folder_name, dtype="auto"):
    """Load a Torch-based model from the specified folder."""
    import importlib

    start = time.perf_counter()

    folder_path = str(BASE_DIR / folder_name)
    if folder_path not in sys.path:
        sys.path.insert(0, folder_path)

    # Clear cached modules
    mods_to_clear = [m for m in sys.modules.keys()
                     if m in ['weight_loader', 'model', 'layers', 'attention', 'rope', 'decoder', 'encoder', 'audio_features', 'config', 'tokenizer', 'torch_glm', 'inference']]
    for m in mods_to_clear:
        del sys.modules[m]

    # Download model from HuggingFace
    from huggingface_hub import snapshot_download
    model_dir = snapshot_download("zai-org/GLM-ASR-Nano-2512")

    torch_glm = importlib.import_module("torch_glm")
    model, processor = torch_glm.load_model_and_processor(model_path=model_dir, dtype=dtype)
    tokenizer = None

    load_time_ms = (time.perf_counter() - start) * 1000

    return model, processor, tokenizer, load_time_ms


@st.cache_resource
def load_scratch_model():
    """Load PyTorch scratch model."""
    return load_torch_model("glm_asr_scratch")


MODEL_CHOICES = [
    ("Triton Example (Baseline)", "triton_example", load_triton_example_model),
    ("Triton Template", "triton_template", load_triton_template_model),
    ("CuTile Example (Baseline)", "cutile_example", load_cutile_example_model),
    ("CuTile Template", "cutile_template", load_cutile_model),
    ("Scratch (PyTorch)", "scratch", load_scratch_model),
]
MODEL_LABELS = [label for label, _, _ in MODEL_CHOICES]
MODEL_BY_LABEL = {label: (key, loader) for label, key, loader in MODEL_CHOICES}
MODEL_LOADERS = {key: (label, loader) for label, key, loader in MODEL_CHOICES}


def transcribe_cutile(audio_array, model, processor, tokenizer):
    """Run CuTile inference."""
    import cupy as cp

    TOKEN_IDS = {
        "begin_of_audio": 59261, "end_of_audio": 59262,
        "user": 59253, "assistant": 59254, "audio": 59260,
    }

    # Prepare inputs - use feature_extractor directly for compatibility
    features = processor.feature_extractor(audio_array, sampling_rate=16000, return_tensors="pt")
    input_features = features.input_features.numpy()
    mel_time = input_features.shape[2]
    num_audio_tokens = mel_time // 4

    prompt_tokens = [TOKEN_IDS["begin_of_audio"]] + [TOKEN_IDS["audio"]] * num_audio_tokens + [
        TOKEN_IDS["end_of_audio"], TOKEN_IDS["user"]]
    instruction_tokens = tokenizer.encode("Please transcribe this audio into text").ids
    prompt_tokens.extend(instruction_tokens)
    prompt_tokens.append(TOKEN_IDS["assistant"])

    input_features_cp = cp.asarray(input_features, dtype=cp.float32)
    input_ids_cp = cp.asarray(np.array([prompt_tokens], dtype=np.int64), dtype=cp.int64)

    cp.cuda.Stream.null.synchronize()

    start = time.perf_counter()
    generated_ids = model.generate(
        input_features_cp,
        input_ids=input_ids_cp,
        input_features_mask=None,
        max_new_tokens=200,
        temperature=1.0,
        top_k=1
    )
    cp.cuda.Stream.null.synchronize()
    elapsed_ms = (time.perf_counter() - start) * 1000

    generated_np = cp.asnumpy(generated_ids)
    raw_transcription = tokenizer.decode(generated_np[0].tolist(), skip_special_tokens=False)

    transcription = raw_transcription
    if "<|assistant|>" in transcription:
        transcription = transcription.split("<|assistant|>")[-1]
    for token in ["<|endoftext|>", "<|user|>", "<|begin_of_audio|>", "<|end_of_audio|>", "<|pad|>"]:
        transcription = transcription.replace(token, "")
    transcription = transcription.strip()

    return transcription, elapsed_ms


def transcribe_triton(audio_array, model, processor, tokenizer):
    """Run Triton inference."""
    import torch

    TOKEN_IDS = {
        "begin_of_audio": 59261, "end_of_audio": 59262,
        "user": 59253, "assistant": 59254, "audio": 59260,
    }

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    features = processor.feature_extractor(audio_array, sampling_rate=16000, return_tensors="pt")
    input_features = features.input_features.to(device=device, dtype=torch.float32)
    mel_time = input_features.shape[2]
    num_audio_tokens = mel_time // 4

    prompt_tokens = [TOKEN_IDS["begin_of_audio"]] + [TOKEN_IDS["audio"]] * num_audio_tokens + [
        TOKEN_IDS["end_of_audio"], TOKEN_IDS["user"]]
    instruction_tokens = tokenizer.encode("Please transcribe this audio into text").ids
    prompt_tokens.extend(instruction_tokens)
    prompt_tokens.append(TOKEN_IDS["assistant"])

    input_ids = torch.tensor([prompt_tokens], dtype=torch.int64, device=device)

    if torch.cuda.is_available():
        torch.cuda.synchronize()
    start = time.perf_counter()
    generated_ids = model.generate(
        input_features,
        input_ids=input_ids,
        input_features_mask=None,
        max_new_tokens=200,
        temperature=1.0,
        top_k=1
    )
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    elapsed_ms = (time.perf_counter() - start) * 1000

    generated_np = generated_ids.detach().cpu().numpy()
    raw_transcription = tokenizer.decode(generated_np[0].tolist(), skip_special_tokens=False)

    transcription = raw_transcription
    if "<|assistant|>" in transcription:
        transcription = transcription.split("<|assistant|>")[-1]
    for token in ["<|endoftext|>", "<|user|>", "<|begin_of_audio|>", "<|end_of_audio|>", "<|pad|>"]:
        transcription = transcription.replace(token, "")
    transcription = transcription.strip()

    return transcription, elapsed_ms


def transcribe_scratch(audio_array, model, processor, tokenizer, folder_name="glm_asr_scratch"):
    """Run Torch-based inference (scratch)."""
    import torch
    import importlib

    folder_path = str(BASE_DIR / folder_name)
    if folder_path not in sys.path:
        sys.path.insert(0, folder_path)

    torch_glm = importlib.import_module("torch_glm")

    torch.cuda.synchronize() if torch.cuda.is_available() else None

    start = time.perf_counter()
    transcription = torch_glm.transcribe(model, processor, audio_array, max_new_tokens=200)
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    elapsed_ms = (time.perf_counter() - start) * 1000

    if isinstance(transcription, list):
        transcription = transcription[0]

    prompt = getattr(processor, "default_prompt", None)
    if prompt and prompt in transcription:
        transcription = transcription.split(prompt, 1)[-1].strip()
    for prompt_variant in [
        "Please transcribe this audio into text",
        "Transcribe this audio into text."
    ]:
        if transcription.startswith(prompt_variant):
            transcription = transcription[len(prompt_variant):].strip()

    return transcription, elapsed_ms


def load_audio_from_bytes(audio_bytes):
    """Load audio from bytes with robust format handling."""
    import io
    import soundfile as sf
    from scipy import signal
    import subprocess
    import tempfile
    import os

    audio_array = None
    sr = None

    # Method 1: Try soundfile directly
    try:
        audio_array, sr = sf.read(io.BytesIO(audio_bytes))
    except Exception:
        pass

    # Method 2: Try ffmpeg with temp file
    if audio_array is None:
        try:
            with tempfile.NamedTemporaryFile(suffix='.webm', delete=False) as f:
                f.write(audio_bytes)
                tmp_input = f.name

            tmp_output = tmp_input.replace('.webm', '.wav')

            cmd = [
                'ffmpeg', '-y', '-i', tmp_input,
                '-ar', '16000', '-ac', '1', '-f', 'wav', tmp_output
            ]
            result = subprocess.run(cmd, capture_output=True)

            if result.returncode == 0 and os.path.exists(tmp_output):
                audio_array, sr = sf.read(tmp_output)
                sr = 16000

            if os.path.exists(tmp_input):
                os.unlink(tmp_input)
            if os.path.exists(tmp_output):
                os.unlink(tmp_output)
        except Exception:
            pass

    # Method 3: Try ffmpeg with pipe
    if audio_array is None:
        try:
            cmd = ['ffmpeg', '-i', 'pipe:0', '-ar', '16000', '-ac', '1', '-f', 'wav', 'pipe:1']
            process = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            out, _ = process.communicate(input=audio_bytes, timeout=30)
            if len(out) > 44:
                audio_array, sr = sf.read(io.BytesIO(out))
                sr = 16000
        except Exception:
            pass

    if audio_array is None:
        raise RuntimeError("Failed to decode audio. Supported formats: WAV, MP3, FLAC, WebM")

    audio_array = audio_array.astype(np.float32)
    if len(audio_array.shape) > 1:
        audio_array = audio_array.mean(axis=1)
    if sr != 16000:
        num_samples = int(len(audio_array) * 16000 / sr)
        audio_array = signal.resample(audio_array, num_samples)

    return audio_array.astype(np.float32)


# ============================================================================
# UI
# ============================================================================

st.title("🎤 GLM-ASR Student Demo")
st.caption("Test your implementation")

# Version selection
version = st.radio("Select version:", MODEL_LABELS, horizontal=True)
if st.button("🔄 Force reload model", use_container_width=True):
    force_reload_models()
    try:
        st.rerun()
    except Exception:
        st.experimental_rerun()

# Load model
with st.spinner(f"Loading {version} model..."):
    try:
        model_key, loader_fn = MODEL_BY_LABEL[version]
        (model, processor, tokenizer, load_time_ms), reused = get_cached_model(model_key, loader_fn)

        st.session_state.model_load_time_ms = load_time_ms
        if reused:
            st.success("Model loaded | Reusing cached instance")
        else:
            st.success(f"Model loaded | Initial load: {load_time_ms:.0f} ms")
    except Exception as e:
        st.error(f"Failed to load model: {e}")
        import traceback
        st.code(traceback.format_exc())
        st.stop()

st.divider()

# Audio input
input_type = st.radio("Input:", ["🎙️ Record", "📁 Upload", "🎯 Sample"], horizontal=True)

audio_array = None

if input_type == "🎙️ Record":
    recorded = st.audio_input("Record audio")
    if recorded:
        try:
            audio_bytes = recorded.read()
            st.caption(f"Received {len(audio_bytes)} bytes")
            audio_array = load_audio_from_bytes(audio_bytes)
            recorded.seek(0)
            st.audio(recorded)
            st.info(f"Duration: {len(audio_array)/16000:.2f}s")
        except Exception as e:
            st.error(f"Audio decode error: {e}")
            import traceback
            st.code(traceback.format_exc())

elif input_type == "📁 Upload":
    uploaded = st.file_uploader("Upload audio", type=["wav", "mp3", "flac", "webm"])
    if uploaded:
        try:
            audio_array = load_audio_from_bytes(uploaded.read())
            uploaded.seek(0)
            st.audio(uploaded)
            st.info(f"Duration: {len(audio_array)/16000:.2f}s")
        except Exception as e:
            st.error(str(e))

else:  # Sample
    @st.cache_data
    def get_sample():
        import soundfile as sf
        import io
        from datasets import load_dataset
        ds = load_dataset("hf-internal-testing/librispeech_asr_dummy", "clean", split="validation")
        audio_bytes = ds.data.column('audio')[0].as_py()['bytes']
        arr, sr = sf.read(io.BytesIO(audio_bytes))
        if sr != 16000:
            from scipy import signal
            arr = signal.resample(arr, int(len(arr) * 16000 / sr))
        return arr.astype(np.float32), ds.data.column('text')[0].as_py()

    try:
        audio_array, expected = get_sample()
        st.info(f"Duration: {len(audio_array)/16000:.2f}s")
        st.caption(f"Expected: {expected}")
    except Exception as e:
        st.error(str(e))

st.divider()

# Inference
if audio_array is not None:
    if st.button("▶️ Transcribe", type="primary", use_container_width=True):
        with st.spinner("Transcribing..."):
            try:
                if version in ["CuTile Template", "CuTile Example (Baseline)"]:
                    result, elapsed_ms = transcribe_cutile(audio_array, model, processor, tokenizer)
                elif version == "Triton Example (Baseline)":
                    result, elapsed_ms = transcribe_triton(audio_array, model, processor, tokenizer)
                else:
                    result, elapsed_ms = transcribe_scratch(audio_array, model, processor, tokenizer)

                st.subheader("Result")
                st.write(result)

                st.subheader("Timing")
                col1, col2, col3 = st.columns(3)
                col1.metric("Inference", f"{elapsed_ms:.1f} ms")
                col2.metric("Audio Duration", f"{len(audio_array)/16000:.2f} s")
                col3.metric("Model Load", f"{st.session_state.model_load_time_ms:.0f} ms")

            except Exception as e:
                st.error(f"Error: {e}")
                import traceback
                st.code(traceback.format_exc())
else:
    st.warning("Select or record audio first")
