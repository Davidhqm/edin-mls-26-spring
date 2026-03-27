"""
Weight Loading from Safetensors
Educational implementation from scratch using PyTorch only

Safetensors is a simple, safe format for storing tensors:
- Header (JSON with tensor metadata) at the beginning
- Raw tensor data follows the header
"""

import json
import struct
from pathlib import Path
from typing import Dict, Any, Optional
import torch


def load_safetensors(file_path: str) -> Dict[str, torch.Tensor]:
    """
    Load tensors from a safetensors file.

    Safetensors format:
    1. 8 bytes: Header size (little-endian uint64)
    2. Header: JSON string containing tensor metadata
    3. Data: Raw tensor bytes

    Args:
        file_path: Path to .safetensors file

    Returns:
        Dictionary mapping tensor names to torch tensors
    """
    with open(file_path, 'rb') as f:
        # Read header size (8 bytes, little-endian uint64)
        header_size_bytes = f.read(8)
        header_size = struct.unpack('<Q', header_size_bytes)[0]

        # Read header JSON
        header_bytes = f.read(header_size)
        header = json.loads(header_bytes.decode('utf-8'))

        # Calculate data start position
        data_start = 8 + header_size

        tensors = {}

        for tensor_name, tensor_info in header.items():
            if tensor_name == "__metadata__":
                continue

            dtype_str = tensor_info['dtype']
            shape = tensor_info['shape']
            data_offsets = tensor_info['data_offsets']
            start_offset, end_offset = data_offsets

            # Map dtype string to torch dtype
            dtype_map = {
                'F32': torch.float32,
                'F16': torch.float16,
                'BF16': torch.bfloat16,
                'F64': torch.float64,
                'I64': torch.int64,
                'I32': torch.int32,
                'I16': torch.int16,
                'I8': torch.int8,
                'U8': torch.uint8,
                'BOOL': torch.bool,
            }

            torch_dtype = dtype_map.get(dtype_str, torch.float32)

            # Read tensor data
            f.seek(data_start + start_offset)
            tensor_bytes = f.read(end_offset - start_offset)

            # Convert to tensor
            tensor = torch.frombuffer(
                bytearray(tensor_bytes),
                dtype=torch_dtype
            ).reshape(shape)

            tensors[tensor_name] = tensor.clone()

    return tensors


def get_safetensors_metadata(file_path: str) -> Dict[str, Any]:
    """
    Get metadata from safetensors file without loading all tensors.

    Returns tensor names, shapes, and dtypes.
    """
    with open(file_path, 'rb') as f:
        header_size = struct.unpack('<Q', f.read(8))[0]
        header = json.loads(f.read(header_size).decode('utf-8'))

    metadata = {}
    for name, info in header.items():
        if name != "__metadata__":
            metadata[name] = {
                'shape': info['shape'],
                'dtype': info['dtype']
            }

    return metadata


def create_weight_mapping() -> Dict[str, str]:
    """
    Create mapping from HuggingFace weight names to our model weight names.

    This handles the different naming conventions between the official
    implementation and our educational implementation.
    """
    mapping = {}

    # Audio encoder mappings
    mapping['audio_tower.conv1.weight'] = 'audio_encoder.conv1.weight'
    mapping['audio_tower.conv1.bias'] = 'audio_encoder.conv1.bias'
    mapping['audio_tower.conv2.weight'] = 'audio_encoder.conv2.weight'
    mapping['audio_tower.conv2.bias'] = 'audio_encoder.conv2.bias'
    mapping['audio_tower.norm.weight'] = 'audio_encoder.norm.weight'
    mapping['audio_tower.norm.bias'] = 'audio_encoder.norm.bias'

    # Audio encoder layers - will be filled dynamically
    # Pattern: audio_tower.layers.{i}.{component}

    # Projector mappings
    mapping['multi_modal_projector.linear_1.weight'] = 'multi_modal_projector.linear_1.weight'
    mapping['multi_modal_projector.linear_1.bias'] = 'multi_modal_projector.linear_1.bias'
    mapping['multi_modal_projector.linear_2.weight'] = 'multi_modal_projector.linear_2.weight'
    mapping['multi_modal_projector.linear_2.bias'] = 'multi_modal_projector.linear_2.bias'

    # Language model mappings
    mapping['language_model.model.embed_tokens.weight'] = 'language_model.model.embed_tokens.weight'
    mapping['language_model.model.norm.weight'] = 'language_model.model.norm.weight'
    mapping['language_model.lm_head.weight'] = 'language_model.lm_head.weight'

    # Language model layers - will be filled dynamically
    # Pattern: language_model.model.layers.{i}.{component}

    return mapping


def map_weight_name(hf_name: str) -> str:
    """
    Map a HuggingFace weight name to our model's weight name.

    Handles both direct mappings and pattern-based mappings for layers.
    """
    # Direct mapping
    static_mapping = create_weight_mapping()
    if hf_name in static_mapping:
        return static_mapping[hf_name]

    # Audio encoder layer pattern
    if hf_name.startswith('audio_tower.layers.'):
        return hf_name.replace('audio_tower.', 'audio_encoder.')

    # Language model layer pattern (already matches)
    if hf_name.startswith('language_model.'):
        return hf_name

    # Default: return as-is
    return hf_name


def load_weights_into_model(
    model: torch.nn.Module,
    weights_path: str,
    strict: bool = False,
    verbose: bool = True
) -> Dict[str, Any]:
    """
    Load weights from safetensors file into model.

    Args:
        model: PyTorch model to load weights into
        weights_path: Path to .safetensors file
        strict: If True, raise error on missing/unexpected keys
        verbose: If True, print loading progress

    Returns:
        Dictionary with 'missing_keys' and 'unexpected_keys'
    """
    if verbose:
        print(f"Loading weights from {weights_path}")

    # Load safetensors
    state_dict = load_safetensors(weights_path)

    if verbose:
        print(f"Loaded {len(state_dict)} tensors from file")

    # Get model's expected state dict keys
    model_state_dict = model.state_dict()
    model_keys = set(model_state_dict.keys())

    # Map and load weights
    loaded_keys = set()
    missing_keys = []
    unexpected_keys = []

    for hf_name, tensor in state_dict.items():
        our_name = map_weight_name(hf_name)

        if our_name in model_keys:
            # Check shape compatibility
            expected_shape = model_state_dict[our_name].shape
            if tensor.shape == expected_shape:
                model_state_dict[our_name] = tensor
                loaded_keys.add(our_name)
                if verbose:
                    print(f"  Loaded: {hf_name} -> {our_name} {tensor.shape}")
            else:
                print(f"  Shape mismatch: {our_name}")
                print(f"    Expected: {expected_shape}, Got: {tensor.shape}")
                unexpected_keys.append(hf_name)
        else:
            unexpected_keys.append(hf_name)
            if verbose and len(unexpected_keys) <= 10:
                print(f"  Unexpected: {hf_name}")

    # Find missing keys
    missing_keys = list(model_keys - loaded_keys)

    if verbose:
        print(f"\nLoaded {len(loaded_keys)} / {len(model_keys)} parameters")
        if missing_keys:
            print(f"Missing {len(missing_keys)} keys")
            for key in missing_keys[:10]:
                print(f"  - {key}")
        if unexpected_keys:
            print(f"Unexpected {len(unexpected_keys)} keys")

    # Load the mapped state dict
    model.load_state_dict(model_state_dict, strict=False)

    if strict and (missing_keys or unexpected_keys):
        raise RuntimeError(
            f"Strict loading failed. "
            f"Missing: {len(missing_keys)}, Unexpected: {len(unexpected_keys)}"
        )

    return {
        'missing_keys': missing_keys,
        'unexpected_keys': unexpected_keys,
        'loaded_keys': list(loaded_keys)
    }


def print_model_weights_info(model: torch.nn.Module):
    """Print information about model's weight tensors."""
    print("\nModel weight tensors:")
    total_params = 0

    for name, param in model.named_parameters():
        params = param.numel()
        total_params += params
        print(f"  {name}: {param.shape} ({params:,} params)")

    print(f"\nTotal parameters: {total_params:,}")
    print(f"Total size (float32): {total_params * 4 / 1024 / 1024 / 1024:.2f} GB")
    print(f"Total size (bfloat16): {total_params * 2 / 1024 / 1024 / 1024:.2f} GB")


if __name__ == "__main__":
    # Test safetensors loading
    import tempfile
    import os

    print("Testing safetensors loading...")

    # Create a test safetensors file
    test_tensors = {
        'test_weight': torch.randn(10, 20),
        'test_bias': torch.randn(10)
    }

    # Write test file
    def write_safetensors(tensors: Dict[str, torch.Tensor], file_path: str):
        """Write tensors to safetensors format."""
        header = {}
        data_parts = []
        current_offset = 0

        dtype_map = {
            torch.float32: 'F32',
            torch.float16: 'F16',
            torch.bfloat16: 'BF16',
            torch.int64: 'I64',
            torch.int32: 'I32',
        }

        for name, tensor in tensors.items():
            tensor = tensor.contiguous()
            tensor_bytes = tensor.numpy().tobytes()
            data_parts.append(tensor_bytes)

            header[name] = {
                'dtype': dtype_map.get(tensor.dtype, 'F32'),
                'shape': list(tensor.shape),
                'data_offsets': [current_offset, current_offset + len(tensor_bytes)]
            }
            current_offset += len(tensor_bytes)

        header_bytes = json.dumps(header).encode('utf-8')

        with open(file_path, 'wb') as f:
            f.write(struct.pack('<Q', len(header_bytes)))
            f.write(header_bytes)
            for data in data_parts:
                f.write(data)

    with tempfile.NamedTemporaryFile(suffix='.safetensors', delete=False) as f:
        temp_path = f.name

    try:
        write_safetensors(test_tensors, temp_path)

        # Load and verify
        loaded = load_safetensors(temp_path)

        print(f"Original tensors: {list(test_tensors.keys())}")
        print(f"Loaded tensors: {list(loaded.keys())}")

        for name in test_tensors:
            if name in loaded:
                original = test_tensors[name]
                loaded_t = loaded[name]
                match = torch.allclose(original, loaded_t, atol=1e-6)
                print(f"  {name}: shape={loaded_t.shape}, match={match}")

        print("\nSafetensors loading test passed!")

    finally:
        os.unlink(temp_path)
