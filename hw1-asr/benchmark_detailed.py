#!/usr/bin/env python3
"""
Detailed Benchmark Script with Operator-level Profiling
Measures execution time for each operator/layer in the model.

Usage:
    python benchmark_detailed.py <folder_name>
    python benchmark_detailed.py glm_asr_cutile_template --nsys
    python benchmark_detailed.py glm_asr_triton_example
"""

import argparse
import time
import sys
import os
import numpy as np
from collections import defaultdict
from contextlib import contextmanager

# Profiling data storage
PROFILE_DATA = defaultdict(list)
PROFILE_ENABLED = False


class CUDATimer:
    """CUDA event-based timer for accurate GPU timing."""

    def __init__(self):
        import cupy as cp
        self.cp = cp
        self.start_event = cp.cuda.Event()
        self.end_event = cp.cuda.Event()

    def start(self):
        self.start_event.record()

    def stop(self):
        self.end_event.record()
        self.end_event.synchronize()
        # CuPy uses get_elapsed_time instead of elapsed_time
        return self.cp.cuda.get_elapsed_time(self.start_event, self.end_event)


class TorchTimer:
    """Torch event-based timer for accurate GPU timing."""

    def __init__(self):
        import torch
        self.torch = torch
        if torch.cuda.is_available():
            self.start_event = torch.cuda.Event(enable_timing=True)
            self.end_event = torch.cuda.Event(enable_timing=True)
        else:
            self.start_event = None
            self.end_event = None
            self._start_time = None

    def start(self):
        if self.start_event is not None:
            self.start_event.record()
        else:
            self._start_time = time.perf_counter()

    def stop(self):
        if self.start_event is not None:
            self.end_event.record()
            self.end_event.synchronize()
            return self.start_event.elapsed_time(self.end_event)
        elapsed = (time.perf_counter() - self._start_time) * 1000
        return elapsed


@contextmanager
def profile_region(name):
    """Context manager for profiling a code region."""
    if not PROFILE_ENABLED:
        yield
        return

    import cupy as cp
    cp.cuda.Device().synchronize()
    start = time.perf_counter()
    yield
    cp.cuda.Device().synchronize()
    elapsed = (time.perf_counter() - start) * 1000
    PROFILE_DATA[name].append(elapsed)


def patch_module_for_profiling(module, prefix=""):
    """Patch a module's forward methods to add profiling."""
    import cupy as cp

    original_forwards = {}

    for name, child in module.named_children():
        full_name = f"{prefix}.{name}" if prefix else name

        # Store original forward
        if hasattr(child, 'forward'):
            original_forward = child.forward
            original_forwards[full_name] = original_forward

            # Create profiled forward
            def make_profiled_forward(orig_fwd, prof_name):
                def profiled_forward(*args, **kwargs):
                    with profile_region(prof_name):
                        return orig_fwd(*args, **kwargs)
                return profiled_forward

            child.forward = make_profiled_forward(original_forward, full_name)

        # Recursively patch children (limit depth to avoid too much overhead)
        if len(full_name.split('.')) < 3:
            patch_module_for_profiling(child, full_name)

    return original_forwards


def profile_operators_cupy(model, input_features, input_ids, input_features_mask, num_runs=3):
    """Profile individual operators using CuPy timing."""
    import cupy as cp

    global PROFILE_ENABLED, PROFILE_DATA
    PROFILE_DATA.clear()
    PROFILE_ENABLED = True

    # Determine generate function
    if hasattr(model, 'generate_v8b'):
        generate_fn = model.generate_v8b
    elif hasattr(model, 'generate_v8'):
        generate_fn = model.generate_v8
    elif hasattr(model, 'generate_v6'):
        generate_fn = model.generate_v6
    else:
        generate_fn = model.generate

    print(f"\nProfiling with {num_runs} runs...")

    for run in range(num_runs):
        # Profile major components manually
        cp.cuda.Device().synchronize()

        # We'll profile by wrapping key operations
        try:
            output = generate_fn(
                input_features,
                input_ids=input_ids,
                input_features_mask=input_features_mask,
                max_new_tokens=50,
                temperature=1.0,
                top_k=1
            )
        except TypeError:
            output = generate_fn(
                input_features,
                input_ids=input_ids,
                max_new_tokens=50,
                temperature=1.0,
                top_k=1
            )

        cp.cuda.Device().synchronize()
        print(f"  Run {run+1}/{num_runs} complete")

    PROFILE_ENABLED = False
    return output


def detailed_profile(model, input_features, input_ids, input_features_mask, num_runs=3):
    """Detailed profiling of model components."""
    import cupy as cp

    results = {}
    timer = CUDATimer()

    print("\n" + "="*70)
    print("DETAILED OPERATOR PROFILING")
    print("="*70)

    # 1. Profile Audio Encoder
    print("\n[1/4] Profiling Audio Encoder...")
    encoder_times = []
    for _ in range(num_runs):
        cp.cuda.Device().synchronize()
        timer.start()
        with cp.cuda.Device():
            audio_features = model.audio_encoder(input_features)
        elapsed = timer.stop()
        encoder_times.append(elapsed)
    results['audio_encoder'] = {
        'mean': np.mean(encoder_times),
        'std': np.std(encoder_times),
        'min': np.min(encoder_times),
        'max': np.max(encoder_times)
    }
    print(f"  Audio Encoder: {results['audio_encoder']['mean']:.2f}ms (+/- {results['audio_encoder']['std']:.2f}ms)")

    # 2. Profile Multi-modal Projector
    print("\n[2/4] Profiling Multi-modal Projector...")
    projector_times = []
    for _ in range(num_runs):
        cp.cuda.Device().synchronize()
        timer.start()
        projected = model.multi_modal_projector(audio_features)
        elapsed = timer.stop()
        projector_times.append(elapsed)
    results['projector'] = {
        'mean': np.mean(projector_times),
        'std': np.std(projector_times),
        'min': np.min(projector_times),
        'max': np.max(projector_times)
    }
    print(f"  Projector: {results['projector']['mean']:.2f}ms (+/- {results['projector']['std']:.2f}ms)")

    # 3. Profile Text Decoder (prefill phase)
    print("\n[3/4] Profiling Text Decoder (Prefill)...")

    # Build input embeddings
    embed_tokens = model.text_decoder.embed_tokens
    text_embeds = embed_tokens(input_ids)

    # Find audio token positions
    audio_token_id = 59260
    audio_mask = (input_ids == audio_token_id)

    # Create combined embeddings
    combined_embeds = text_embeds.copy()
    if cp.any(audio_mask):
        audio_positions = cp.where(audio_mask[0])[0]
        num_audio_tokens = len(audio_positions)
        if num_audio_tokens <= projected.shape[1]:
            combined_embeds[0, audio_positions[:projected.shape[1]]] = projected[0, :num_audio_tokens]

    prefill_times = []
    for _ in range(num_runs):
        cp.cuda.Device().synchronize()
        timer.start()
        # Call with inputs_embeds argument
        hidden_states = model.text_decoder(inputs_embeds=combined_embeds)
        elapsed = timer.stop()
        prefill_times.append(elapsed)
    results['decoder_prefill'] = {
        'mean': np.mean(prefill_times),
        'std': np.std(prefill_times),
        'min': np.min(prefill_times),
        'max': np.max(prefill_times)
    }
    print(f"  Decoder Prefill: {results['decoder_prefill']['mean']:.2f}ms (+/- {results['decoder_prefill']['std']:.2f}ms)")

    # 4. Profile Decode Steps (autoregressive)
    print("\n[4/4] Profiling Decode Steps...")
    decode_times = []
    num_decode_steps = 10

    # Get logits and sample first token
    logits = model.lm_head(hidden_states[:, -1:, :])
    next_token = cp.argmax(logits[:, -1, :], axis=-1, keepdims=True)

    for step in range(num_decode_steps):
        cp.cuda.Device().synchronize()
        timer.start()

        # Single decode step
        next_embed = embed_tokens(next_token)
        step_hidden = model.text_decoder(inputs_embeds=next_embed)
        step_logits = model.lm_head(step_hidden)
        next_token = cp.argmax(step_logits[:, -1, :], axis=-1, keepdims=True)

        elapsed = timer.stop()
        decode_times.append(elapsed)

    results['decode_step'] = {
        'mean': np.mean(decode_times),
        'std': np.std(decode_times),
        'min': np.min(decode_times),
        'max': np.max(decode_times)
    }
    print(f"  Decode Step (avg): {results['decode_step']['mean']:.2f}ms (+/- {results['decode_step']['std']:.2f}ms)")

    # 5. Profile individual layers in decoder
    print("\n[5] Profiling Individual Decoder Layers...")
    layer_times = []

    try:
        test_input = combined_embeds
        seq_len = test_input.shape[1]

        # Try to get layers - different model versions have different structures
        if hasattr(model.text_decoder, 'layers'):
            layers = model.text_decoder.layers[:5]
        else:
            layers = []

        for i, layer in enumerate(layers):
            times = []
            for _ in range(num_runs):
                cp.cuda.Device().synchronize()
                timer.start()
                # Try calling with position_ids if needed
                try:
                    test_output = layer(test_input)
                except TypeError:
                    position_ids = cp.arange(seq_len, dtype=cp.int64).reshape(1, -1)
                    test_output = layer(test_input, position_ids=position_ids)
                elapsed = timer.stop()
                times.append(elapsed)

            layer_times.append({
                'name': f'layer_{i}',
                'mean': np.mean(times),
                'std': np.std(times)
            })
            print(f"  Layer {i}: {np.mean(times):.2f}ms (+/- {np.std(times):.2f}ms)")
            test_input = test_output
    except Exception as e:
        print(f"  Layer profiling skipped: {e}")

    results['layers'] = layer_times

    return results


def detailed_profile_torch(model, input_features, input_ids, input_features_mask, num_runs=3):
    """Detailed profiling of model components (Torch)."""
    import torch

    results = {}
    timer = TorchTimer()

    print("\n" + "="*70)
    print("DETAILED OPERATOR PROFILING (TORCH)")
    print("="*70)

    print("\n[1/4] Profiling Audio Encoder...")
    encoder_times = []
    for _ in range(num_runs):
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        timer.start()
        audio_features = model.audio_encoder(input_features)
        elapsed = timer.stop()
        encoder_times.append(elapsed)
    results['audio_encoder'] = {
        'mean': np.mean(encoder_times),
        'std': np.std(encoder_times),
        'min': np.min(encoder_times),
        'max': np.max(encoder_times)
    }
    print(f"  Audio Encoder: {results['audio_encoder']['mean']:.2f}ms (+/- {results['audio_encoder']['std']:.2f}ms)")

    print("\n[2/4] Profiling Multi-modal Projector...")
    projector_times = []
    for _ in range(num_runs):
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        timer.start()
        projected = model.multi_modal_projector(audio_features)
        elapsed = timer.stop()
        projector_times.append(elapsed)
    results['projector'] = {
        'mean': np.mean(projector_times),
        'std': np.std(projector_times),
        'min': np.min(projector_times),
        'max': np.max(projector_times)
    }
    print(f"  Projector: {results['projector']['mean']:.2f}ms (+/- {results['projector']['std']:.2f}ms)")

    print("\n[3/4] Profiling Text Decoder (Prefill)...")
    embed_tokens = model.text_decoder.embed_tokens
    text_embeds = embed_tokens(input_ids)

    audio_token_id = 59260
    audio_mask = (input_ids == audio_token_id)

    combined_embeds = text_embeds.clone()
    if torch.any(audio_mask):
        audio_positions = torch.where(audio_mask[0])[0]
        num_audio_tokens = int(audio_positions.numel())
        if num_audio_tokens <= projected.shape[1]:
            combined_embeds[0, audio_positions[:projected.shape[1]]] = projected[0, :num_audio_tokens]

    prefill_times = []
    for _ in range(num_runs):
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        timer.start()
        hidden_states = model.text_decoder(inputs_embeds=combined_embeds)
        elapsed = timer.stop()
        prefill_times.append(elapsed)
    results['decoder_prefill'] = {
        'mean': np.mean(prefill_times),
        'std': np.std(prefill_times),
        'min': np.min(prefill_times),
        'max': np.max(prefill_times)
    }
    print(f"  Decoder Prefill: {results['decoder_prefill']['mean']:.2f}ms (+/- {results['decoder_prefill']['std']:.2f}ms)")

    print("\n[4/4] Profiling Decode Steps...")
    decode_times = []
    num_decode_steps = 10

    logits = model.lm_head(hidden_states[:, -1:, :])
    next_token = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)

    for _ in range(num_decode_steps):
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        timer.start()
        next_embed = embed_tokens(next_token)
        step_hidden = model.text_decoder(inputs_embeds=next_embed)
        step_logits = model.lm_head(step_hidden)
        next_token = torch.argmax(step_logits[:, -1, :], dim=-1, keepdim=True)
        elapsed = timer.stop()
        decode_times.append(elapsed)

    results['decode_step'] = {
        'mean': np.mean(decode_times),
        'std': np.std(decode_times),
        'min': np.min(decode_times),
        'max': np.max(decode_times)
    }
    print(f"  Decode Step (avg): {results['decode_step']['mean']:.2f}ms (+/- {results['decode_step']['std']:.2f}ms)")

    print("\n[5] Profiling Individual Decoder Layers...")
    layer_times = []

    try:
        test_input = combined_embeds
        seq_len = test_input.shape[1]

        if hasattr(model.text_decoder, 'layers'):
            layers = model.text_decoder.layers[:5]
        else:
            layers = []

        for i, layer in enumerate(layers):
            times = []
            for _ in range(num_runs):
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                timer.start()
                try:
                    test_output = layer(test_input)
                except TypeError:
                    position_ids = torch.arange(seq_len, dtype=torch.int64, device=test_input.device).reshape(1, -1)
                    test_output = layer(test_input, position_ids=position_ids)
                elapsed = timer.stop()
                times.append(elapsed)

            layer_times.append({
                'name': f'layer_{i}',
                'mean': np.mean(times),
                'std': np.std(times)
            })
            print(f"  Layer {i}: {np.mean(times):.2f}ms (+/- {np.std(times):.2f}ms)")
            test_input = test_output
    except Exception as e:
        print(f"  Layer profiling skipped: {e}")

    results['layers'] = layer_times

    return results


def profile_attention_ops(model, seq_len=256, num_runs=5):
    """Profile attention operations specifically."""
    import cupy as cp

    print("\n" + "="*70)
    print("ATTENTION OPERATION PROFILING")
    print("="*70)

    timer = CUDATimer()
    results = {}

    # Get attention config
    hidden_size = 2048
    num_heads = 16
    head_dim = hidden_size // num_heads

    # Create test tensors
    batch_size = 1
    q = cp.random.randn(batch_size, num_heads, seq_len, head_dim, dtype=cp.float32)
    k = cp.random.randn(batch_size, num_heads, seq_len, head_dim, dtype=cp.float32)
    v = cp.random.randn(batch_size, num_heads, seq_len, head_dim, dtype=cp.float32)

    # 1. Standard attention (QK^T, softmax, V)
    print(f"\nSequence length: {seq_len}")

    print("\n[1] Standard Attention (einsum)...")
    standard_times = []
    for _ in range(num_runs):
        cp.cuda.Device().synchronize()
        timer.start()

        scores = cp.einsum('bhqd,bhkd->bhqk', q, k) / cp.sqrt(cp.float32(head_dim))
        attn_weights = cp.exp(scores - cp.max(scores, axis=-1, keepdims=True))
        attn_weights = attn_weights / cp.sum(attn_weights, axis=-1, keepdims=True)
        output = cp.einsum('bhqk,bhkd->bhqd', attn_weights, v)

        elapsed = timer.stop()
        standard_times.append(elapsed)

    results['standard_attention'] = np.mean(standard_times)
    print(f"  Standard: {np.mean(standard_times):.2f}ms (+/- {np.std(standard_times):.2f}ms)")

    # 2. cuBLAS matmul attention
    print("\n[2] cuBLAS Matmul Attention...")
    cublas_times = []

    # Reshape for matmul
    q_2d = q.reshape(batch_size * num_heads, seq_len, head_dim)
    k_2d = k.reshape(batch_size * num_heads, seq_len, head_dim)
    v_2d = v.reshape(batch_size * num_heads, seq_len, head_dim)

    for _ in range(num_runs):
        cp.cuda.Device().synchronize()
        timer.start()

        scores = cp.matmul(q_2d, k_2d.transpose(0, 2, 1)) / cp.sqrt(cp.float32(head_dim))
        attn_weights = cp.exp(scores - cp.max(scores, axis=-1, keepdims=True))
        attn_weights = attn_weights / cp.sum(attn_weights, axis=-1, keepdims=True)
        output = cp.matmul(attn_weights, v_2d)

        elapsed = timer.stop()
        cublas_times.append(elapsed)

    results['cublas_attention'] = np.mean(cublas_times)
    print(f"  cuBLAS: {np.mean(cublas_times):.2f}ms (+/- {np.std(cublas_times):.2f}ms)")

    return results


def profile_attention_ops_torch(seq_len=256, num_runs=5):
    """Profile attention operations specifically (Torch)."""
    import torch

    print("\n" + "="*70)
    print("ATTENTION OPERATION PROFILING (TORCH)")
    print("="*70)

    timer = TorchTimer()
    results = {}

    hidden_size = 2048
    num_heads = 16
    head_dim = hidden_size // num_heads

    batch_size = 1
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    q = torch.randn(batch_size, num_heads, seq_len, head_dim, device=device)
    k = torch.randn(batch_size, num_heads, seq_len, head_dim, device=device)
    v = torch.randn(batch_size, num_heads, seq_len, head_dim, device=device)

    print(f"\nSequence length: {seq_len}")

    print("\n[1] Standard Attention (einsum)...")
    standard_times = []
    for _ in range(num_runs):
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        timer.start()

        scores = torch.einsum('bhqd,bhkd->bhqk', q, k) / torch.sqrt(torch.tensor(head_dim, dtype=torch.float32, device=device))
        attn_weights = torch.exp(scores - torch.max(scores, dim=-1, keepdim=True).values)
        attn_weights = attn_weights / torch.sum(attn_weights, dim=-1, keepdim=True)
        output = torch.einsum('bhqk,bhkd->bhqd', attn_weights, v)

        elapsed = timer.stop()
        standard_times.append(elapsed)

    results['standard_attention'] = np.mean(standard_times)
    print(f"  Standard: {np.mean(standard_times):.2f}ms (+/- {np.std(standard_times):.2f}ms)")

    print("\n[2] Torch Matmul Attention...")
    matmul_times = []

    q_2d = q.reshape(batch_size * num_heads, seq_len, head_dim)
    k_2d = k.reshape(batch_size * num_heads, seq_len, head_dim)
    v_2d = v.reshape(batch_size * num_heads, seq_len, head_dim)

    for _ in range(num_runs):
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        timer.start()

        scores = torch.matmul(q_2d, k_2d.transpose(1, 2)) / torch.sqrt(torch.tensor(head_dim, dtype=torch.float32, device=device))
        attn_weights = torch.exp(scores - torch.max(scores, dim=-1, keepdim=True).values)
        attn_weights = attn_weights / torch.sum(attn_weights, dim=-1, keepdim=True)
        output = torch.matmul(attn_weights, v_2d)

        elapsed = timer.stop()
        matmul_times.append(elapsed)

    results['matmul_attention'] = np.mean(matmul_times)
    print(f"  Torch matmul: {np.mean(matmul_times):.2f}ms (+/- {np.std(matmul_times):.2f}ms)")

    return results


def profile_linear_ops(hidden_size=2048, intermediate_size=5632, batch_size=1, seq_len=256, num_runs=5):
    """Profile linear/GEMM operations."""
    import cupy as cp

    print("\n" + "="*70)
    print("LINEAR/GEMM OPERATION PROFILING")
    print("="*70)

    timer = CUDATimer()
    results = {}

    # Create test tensors
    x = cp.random.randn(batch_size, seq_len, hidden_size, dtype=cp.float32)
    w_proj = cp.random.randn(hidden_size, intermediate_size, dtype=cp.float32)
    w_down = cp.random.randn(intermediate_size, hidden_size, dtype=cp.float32)

    print(f"\nInput shape: ({batch_size}, {seq_len}, {hidden_size})")
    print(f"Projection: {hidden_size} -> {intermediate_size}")

    # 1. CuPy matmul
    print("\n[1] CuPy matmul...")
    matmul_times = []
    for _ in range(num_runs):
        cp.cuda.Device().synchronize()
        timer.start()
        y = cp.matmul(x, w_proj)
        elapsed = timer.stop()
        matmul_times.append(elapsed)

    results['cupy_matmul'] = np.mean(matmul_times)
    print(f"  CuPy matmul: {np.mean(matmul_times):.2f}ms (+/- {np.std(matmul_times):.2f}ms)")

    # 2. CuPy einsum
    print("\n[2] CuPy einsum...")
    einsum_times = []
    for _ in range(num_runs):
        cp.cuda.Device().synchronize()
        timer.start()
        y = cp.einsum('bsh,ho->bso', x, w_proj)
        elapsed = timer.stop()
        einsum_times.append(elapsed)

    results['cupy_einsum'] = np.mean(einsum_times)
    print(f"  CuPy einsum: {np.mean(einsum_times):.2f}ms (+/- {np.std(einsum_times):.2f}ms)")

    # 3. cuBLAS GEMM (via reshape + matmul)
    print("\n[3] cuBLAS GEMM (batched)...")
    x_2d = x.reshape(-1, hidden_size)
    gemm_times = []
    for _ in range(num_runs):
        cp.cuda.Device().synchronize()
        timer.start()
        y = cp.matmul(x_2d, w_proj)
        elapsed = timer.stop()
        gemm_times.append(elapsed)

    results['cublas_gemm'] = np.mean(gemm_times)
    print(f"  cuBLAS GEMM: {np.mean(gemm_times):.2f}ms (+/- {np.std(gemm_times):.2f}ms)")

    # 4. Full MLP forward (gate + up + down)
    print("\n[4] Full MLP (SwiGLU style)...")
    w_gate = cp.random.randn(hidden_size, intermediate_size, dtype=cp.float32)
    w_up = cp.random.randn(hidden_size, intermediate_size, dtype=cp.float32)

    mlp_times = []
    for _ in range(num_runs):
        cp.cuda.Device().synchronize()
        timer.start()

        gate = cp.matmul(x, w_gate)
        up = cp.matmul(x, w_up)
        # SwiGLU activation
        gate_act = gate * (1 / (1 + cp.exp(-gate)))  # silu
        hidden = gate_act * up
        output = cp.matmul(hidden, w_down)

        elapsed = timer.stop()
        mlp_times.append(elapsed)

    results['full_mlp'] = np.mean(mlp_times)
    print(f"  Full MLP: {np.mean(mlp_times):.2f}ms (+/- {np.std(mlp_times):.2f}ms)")

    # Calculate FLOPS
    flops_proj = 2 * batch_size * seq_len * hidden_size * intermediate_size
    gflops = flops_proj / (np.mean(matmul_times) / 1000) / 1e9
    print(f"\n  Estimated GFLOPS (projection): {gflops:.1f}")

    return results


def profile_linear_ops_torch(hidden_size=2048, intermediate_size=5632, batch_size=1, seq_len=256, num_runs=5):
    """Profile linear/GEMM operations (Torch)."""
    import torch

    print("\n" + "="*70)
    print("LINEAR/GEMM OPERATION PROFILING (TORCH)")
    print("="*70)

    timer = TorchTimer()
    results = {}

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    x = torch.randn(batch_size, seq_len, hidden_size, device=device)
    w_proj = torch.randn(hidden_size, intermediate_size, device=device)
    w_down = torch.randn(intermediate_size, hidden_size, device=device)

    print(f"\nInput shape: ({batch_size}, {seq_len}, {hidden_size})")
    print(f"Projection: {hidden_size} -> {intermediate_size}")

    print("\n[1] Torch matmul...")
    matmul_times = []
    for _ in range(num_runs):
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        timer.start()
        y = torch.matmul(x, w_proj)
        elapsed = timer.stop()
        matmul_times.append(elapsed)

    results['torch_matmul'] = np.mean(matmul_times)
    print(f"  Torch matmul: {np.mean(matmul_times):.2f}ms (+/- {np.std(matmul_times):.2f}ms)")

    print("\n[2] Torch einsum...")
    einsum_times = []
    for _ in range(num_runs):
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        timer.start()
        y = torch.einsum('bsh,ho->bso', x, w_proj)
        elapsed = timer.stop()
        einsum_times.append(elapsed)

    results['torch_einsum'] = np.mean(einsum_times)
    print(f"  Torch einsum: {np.mean(einsum_times):.2f}ms (+/- {np.std(einsum_times):.2f}ms)")

    print("\n[3] Torch GEMM (batched)...")
    x_2d = x.reshape(-1, hidden_size)
    gemm_times = []
    for _ in range(num_runs):
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        timer.start()
        y = torch.matmul(x_2d, w_proj)
        elapsed = timer.stop()
        gemm_times.append(elapsed)

    results['torch_gemm'] = np.mean(gemm_times)
    print(f"  Torch GEMM: {np.mean(gemm_times):.2f}ms (+/- {np.std(gemm_times):.2f}ms)")

    print("\n[4] Full MLP (SwiGLU style)...")
    w_gate = torch.randn(hidden_size, intermediate_size, device=device)
    w_up = torch.randn(hidden_size, intermediate_size, device=device)

    mlp_times = []
    for _ in range(num_runs):
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        timer.start()

        gate = torch.matmul(x, w_gate)
        up = torch.matmul(x, w_up)
        gate_act = gate * (1 / (1 + torch.exp(-gate)))
        hidden = gate_act * up
        output = torch.matmul(hidden, w_down)

        elapsed = timer.stop()
        mlp_times.append(elapsed)

    results['full_mlp'] = np.mean(mlp_times)
    print(f"  Full MLP: {np.mean(mlp_times):.2f}ms (+/- {np.std(mlp_times):.2f}ms)")

    flops_proj = 2 * batch_size * seq_len * hidden_size * intermediate_size
    gflops = flops_proj / (np.mean(matmul_times) / 1000) / 1e9
    print(f"\n  Estimated GFLOPS (projection): {gflops:.1f}")

    return results


def print_summary(component_results, attention_results, linear_results):
    """Print a summary table of all profiling results."""
    print("\n" + "="*70)
    print("PERFORMANCE SUMMARY")
    print("="*70)

    print("\n{:<35} {:>12} {:>12}".format("Component", "Time (ms)", "% of Total"))
    print("-"*60)

    # Calculate total time
    total = 0
    if component_results:
        for key in ['audio_encoder', 'projector', 'decoder_prefill']:
            if key in component_results:
                total += component_results[key]['mean']
        # Add estimated decode time (50 steps)
        if 'decode_step' in component_results:
            total += component_results['decode_step']['mean'] * 50

    if component_results:
        for key, label in [
            ('audio_encoder', 'Audio Encoder'),
            ('projector', 'Multi-modal Projector'),
            ('decoder_prefill', 'Decoder (Prefill)'),
        ]:
            if key in component_results:
                t = component_results[key]['mean']
                pct = (t / total * 100) if total > 0 else 0
                print(f"{label:<35} {t:>10.2f}ms {pct:>10.1f}%")

        if 'decode_step' in component_results:
            t = component_results['decode_step']['mean'] * 50
            pct = (t / total * 100) if total > 0 else 0
            print(f"{'Decoder (50 decode steps)':<35} {t:>10.2f}ms {pct:>10.1f}%")

    print("-"*60)
    print(f"{'TOTAL (estimated for 50 tokens)':<35} {total:>10.2f}ms")

    # Attention comparison
    if attention_results:
        print("\n" + "-"*60)
        print("Attention Methods Comparison:")
        print("-"*60)
        if 'standard_attention' in attention_results:
            print(f"  {'Standard (einsum)':<25} {attention_results['standard_attention']:>10.2f}ms")
        if 'cublas_attention' in attention_results:
            print(f"  {'cuBLAS matmul':<25} {attention_results['cublas_attention']:>10.2f}ms")
        if 'matmul_attention' in attention_results:
            print(f"  {'Torch matmul':<25} {attention_results['matmul_attention']:>10.2f}ms")

    # Linear comparison
    if linear_results:
        print("\n" + "-"*60)
        print("Linear/GEMM Methods Comparison:")
        print("-"*60)
        if 'cupy_matmul' in linear_results:
            print(f"  {'CuPy matmul':<25} {linear_results['cupy_matmul']:>10.2f}ms")
        if 'cupy_einsum' in linear_results:
            print(f"  {'CuPy einsum':<25} {linear_results['cupy_einsum']:>10.2f}ms")
        if 'cublas_gemm' in linear_results:
            print(f"  {'cuBLAS GEMM':<25} {linear_results['cublas_gemm']:>10.2f}ms")
        if 'torch_matmul' in linear_results:
            print(f"  {'Torch matmul':<25} {linear_results['torch_matmul']:>10.2f}ms")
        if 'torch_einsum' in linear_results:
            print(f"  {'Torch einsum':<25} {linear_results['torch_einsum']:>10.2f}ms")
        if 'torch_gemm' in linear_results:
            print(f"  {'Torch GEMM':<25} {linear_results['torch_gemm']:>10.2f}ms")
        if 'full_mlp' in linear_results:
            print(f"  {'Full MLP (SwiGLU)':<25} {linear_results['full_mlp']:>10.2f}ms")


def run_nsys_profile(folder, audio_path=None, runs=1):
    """Run Nsight Systems profiling."""
    import subprocess

    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_name = f"profile_{folder}"

    cmd = [
        "nsys", "profile",
        "--trace=cuda,nvtx",
        "--output", output_name,
        "--force-overwrite", "true",
        "python", os.path.join(script_dir, "benchmark_student.py"),
        folder, "--warmup", "1", "--runs", str(runs)
    ]

    if audio_path:
        cmd.extend(["--audio", audio_path])

    print(f"Running: {' '.join(cmd)}")
    subprocess.run(cmd, cwd=script_dir)
    print(f"\nProfile saved to: {output_name}.nsys-rep")
    print("Open with: nsys-ui " + output_name + ".nsys-rep")


def main():
    parser = argparse.ArgumentParser(description='Detailed operator profiling')
    parser.add_argument('folder', type=str, nargs='?', default='glm_asr_cutile_example',
                       help='Folder name to benchmark')
    parser.add_argument('--audio', type=str, help='Path to test audio file')
    parser.add_argument('--runs', type=int, default=3, help='Number of profiling runs')
    parser.add_argument('--nsys', action='store_true', help='Run Nsight Systems profiling')
    parser.add_argument('--seq-len', type=int, default=256, help='Sequence length for micro-benchmarks')
    args = parser.parse_args()

    print("="*70)
    print("GLM-ASR Detailed Operator Profiling")
    print("="*70)

    # Run nsys if requested
    if args.nsys:
        run_nsys_profile(args.folder, args.audio, args.runs)
        return 0

    use_torch_backend = 'triton' in args.folder.lower()

    # Full profiling

    # Load test audio
    script_dir = os.path.dirname(os.path.abspath(__file__))

    print("\nLoading test audio...")
    audio_path = args.audio or os.path.join(script_dir, "test_audio.wav")

    import wave
    import struct

    with wave.open(audio_path, 'rb') as wav:
        sr = wav.getframerate()
        n_frames = wav.getnframes()
        n_channels = wav.getnchannels()
        sample_width = wav.getsampwidth()
        raw_data = wav.readframes(n_frames)

        if sample_width == 2:
            audio_array = np.array(struct.unpack(f'<{n_frames * n_channels}h', raw_data), dtype=np.float32)
            audio_array = audio_array / 32768.0
        else:
            audio_array = np.zeros(n_frames, dtype=np.float32)

        if n_channels > 1:
            audio_array = audio_array.reshape(-1, n_channels).mean(axis=1)

    print(f"Audio: {len(audio_array)/sr:.2f}s @ {sr}Hz")

    # Load model
    folder_path = os.path.join(script_dir, args.folder)
    sys.path.insert(0, folder_path)

    # Clear cached modules
    for mod_name in list(sys.modules.keys()):
        if mod_name in ['weight_loader', 'model', 'layers', 'attention', 'rope', 'conv']:
            del sys.modules[mod_name]

    print(f"\nLoading model from {args.folder}...")
    from weight_loader import load_model_from_hf
    model, processor = load_model_from_hf("zai-org/GLM-ASR-Nano-2512")

    if use_torch_backend:
        import torch
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if hasattr(processor, 'apply_transcription_request'):
            inputs = processor.apply_transcription_request(audio_array)
            input_features = inputs.input_features.to(device=device, dtype=torch.float32)
            input_ids = inputs.input_ids.to(device=device, dtype=torch.int64)
            input_features_mask = None
            if hasattr(inputs, 'input_features_mask') and inputs.input_features_mask is not None:
                input_features_mask = inputs.input_features_mask.to(device=device, dtype=torch.float32)
        else:
            features = processor(audio_array, sampling_rate=16000, return_tensors="pt", padding="max_length")
            input_features = features['input_features'].to(device=device, dtype=torch.float32)
            input_ids = torch.tensor([[59253, 10, 59261] + [59260] * 100 + [59262, 59253, 10, 9249, 70891, 419, 7122, 1119, 1467, 59254, 10]],
                                     dtype=torch.int64, device=device)
            input_features_mask = None

        print(f"Input features shape: {input_features.shape}")
        print(f"Input IDs shape: {input_ids.shape}")

        component_results = detailed_profile_torch(model, input_features, input_ids, input_features_mask, num_runs=args.runs)
        attention_results = profile_attention_ops_torch(seq_len=args.seq_len, num_runs=args.runs)
        linear_results = profile_linear_ops_torch(seq_len=args.seq_len, num_runs=args.runs)
    else:
        import cupy as cp
        if hasattr(processor, 'apply_transcription_request'):
            inputs = processor.apply_transcription_request(audio_array)
            input_features = cp.asarray(inputs.input_features.numpy(), dtype=cp.float32)
            input_ids = cp.asarray(inputs.input_ids.numpy(), dtype=cp.int64)
            input_features_mask = None
            if hasattr(inputs, 'input_features_mask') and inputs.input_features_mask is not None:
                input_features_mask = cp.asarray(inputs.input_features_mask.numpy(), dtype=cp.float32)
        else:
            features = processor(audio_array, sampling_rate=16000, return_tensors="pt", padding="max_length")
            input_features = cp.asarray(features['input_features'].numpy(), dtype=cp.float32)
            input_ids = cp.array([[59253, 10, 59261] + [59260] * 100 + [59262, 59253, 10, 9249, 70891, 419, 7122, 1119, 1467, 59254, 10]], dtype=cp.int64)
            input_features_mask = None

        print(f"Input features shape: {input_features.shape}")
        print(f"Input IDs shape: {input_ids.shape}")

        component_results = detailed_profile(model, input_features, input_ids, input_features_mask, num_runs=args.runs)
        attention_results = profile_attention_ops(model, seq_len=args.seq_len, num_runs=args.runs)
        linear_results = profile_linear_ops(seq_len=args.seq_len, num_runs=args.runs)

    # Print summary
    print_summary(component_results, attention_results, linear_results)

    sys.path.remove(folder_path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
