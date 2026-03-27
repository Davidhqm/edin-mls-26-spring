"""
Pure CuTile Neural Network Layers
End-to-end implementation using only NVIDIA CuTile kernels

*** STUDENT ASSIGNMENT ***
Fill in the TODO sections to implement core layers using CuTile kernels
"""

import cuda.tile as ct
import cupy as cp
import numpy as np
from typing import Optional, Tuple
import math


# ============================================================================
# Helper Functions
# ============================================================================

def get_stream():
    """Get current CUDA stream pointer."""
    return cp.cuda.get_current_stream().ptr


def pad_to_multiple(size: int, multiple: int) -> int:
    """Pad size to be a multiple of the given value."""
    return ((size + multiple - 1) // multiple) * multiple


# ============================================================================
# CuTile Kernels - TODO: Implement these
# ============================================================================

@ct.kernel
def rmsnorm_kernel(
    x,              # Input: (batch_size, hidden_size)
    weight,         # Weight: (hidden_size,)
    output,         # Output: (batch_size, hidden_size)
    eps: ct.Constant[float],
    hidden_size: ct.Constant[int]
):
    """
    RMSNorm: x / RMS(x) * weight
    where RMS(x) = sqrt(mean(x^2) + eps)

    *** TODO: Implement this kernel ***

    Grid: (batch_size,)
    Each thread block processes one row of the input.
    """
    pid = ct.bid(0)

    # ============================================================================
    # TODO: Implement RMSNorm kernel
    # ============================================================================
    #
    # Step 1: Load input row and weight
    # x_tile = ct.load(x, index=(pid, 0), shape=(1, hidden_size))
    # x_tile = ct.reshape(x_tile, (hidden_size,))
    # w_tile = ct.load(weight, index=(0,), shape=(hidden_size,))
    #
    # Step 2: Compute variance = mean(x^2)
    # variance = ct.sum(x_tile * x_tile) / hidden_size
    #
    # Step 3: Normalize: x / sqrt(variance + eps)
    # x_norm = x_tile / ct.sqrt(variance + eps)
    #
    # Step 4: Apply weight and store
    # result = x_norm * w_tile
    # result = ct.reshape(result, (1, hidden_size))
    # ct.store(output, index=(pid, 0), tile=result)

    # YOUR CODE HERE
    pass  # Remove this and implement


@ct.kernel
def layernorm_kernel(
    x,              # Input: (batch_size, hidden_size)
    weight,         # Weight: (hidden_size,)
    bias,           # Bias: (hidden_size,)
    output,         # Output: (batch_size, hidden_size)
    eps: ct.Constant[float],
    hidden_size: ct.Constant[int]
):
    """
    LayerNorm: (x - mean) / sqrt(var + eps) * weight + bias

    *** TODO: Implement this kernel ***

    Grid: (batch_size,)
    """
    pid = ct.bid(0)

    # ============================================================================
    # TODO: Implement LayerNorm kernel
    # ============================================================================
    #
    # Step 1: Load input, weight, and bias
    #
    # Step 2: Compute mean
    # mean = ct.sum(x_tile) / hidden_size
    #
    # Step 3: Center the data
    # x_centered = x_tile - mean
    #
    # Step 4: Compute variance = mean((x - mean)^2)
    # variance = ct.sum(x_centered * x_centered) / hidden_size
    #
    # Step 5: Normalize and apply affine transform
    # x_norm = x_centered / ct.sqrt(variance + eps)
    # result = x_norm * w_tile + b_tile

    # YOUR CODE HERE
    pass  # Remove this and implement


@ct.kernel
def gelu_kernel(x, output, tile_size: ct.Constant[int]):
    """
    GELU using tanh approximation.
    GELU(x) = 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))

    *** TODO: Implement this kernel ***

    Grid: (num_tiles,)
    """
    pid = ct.bid(0)

    # ============================================================================
    # TODO: Implement GELU kernel
    # ============================================================================
    #
    # x_tile = ct.load(x, index=(pid,), shape=(tile_size,))
    #
    # sqrt_2_over_pi = 0.7978845608028654
    # x3 = x_tile * x_tile * x_tile
    # inner = sqrt_2_over_pi * (x_tile + 0.044715 * x3)
    # result = x_tile * 0.5 * (1.0 + ct.tanh(inner))
    #
    # ct.store(output, index=(pid,), tile=result)

    # YOUR CODE HERE
    pass  # Remove this and implement


@ct.kernel
def silu_kernel(x, output, tile_size: ct.Constant[int]):
    """
    SiLU/Swish: x * sigmoid(x)

    *** TODO: Implement this kernel ***

    Grid: (num_tiles,)
    """
    pid = ct.bid(0)

    # ============================================================================
    # TODO: Implement SiLU kernel
    # ============================================================================
    #
    # x_tile = ct.load(x, index=(pid,), shape=(tile_size,))
    #
    # sigmoid = 1.0 / (1.0 + ct.exp(-x_tile))
    # result = x_tile * sigmoid
    #
    # ct.store(output, index=(pid,), tile=result)

    # YOUR CODE HERE
    pass  # Remove this and implement


@ct.kernel(occupancy=2)
def linear_kernel_tf32(
    x,              # Input: (M, K) - float32
    weight_t,       # Weight transposed: (K, N) - pre-transposed, float32
    output,         # Output: (M, N) - float32
    M: ct.Constant[int],
    N: ct.Constant[int],
    K: ct.Constant[int]
):
    """
    TF32 tensor core matmul: output = x @ weight_t
    Uses TF32 format for tensor core acceleration.
    64x64 tiles with 32-wide K dimension for optimal performance.

    *** TODO: Implement this kernel ***

    Grid: (M // TILE_M, N // TILE_N)
    """
    TILE_M, TILE_N, TILE_K = 64, 64, 32
    pid_m = ct.bid(0)
    pid_n = ct.bid(1)

    # ============================================================================
    # TODO: Implement tiled matrix multiplication with TF32
    # ============================================================================
    #
    # Step 1: Initialize accumulator
    # acc = ct.zeros((TILE_M, TILE_N), dtype=ct.float32)
    #
    # Step 2: Loop over K tiles
    # num_k_tiles = ct.cdiv(K, TILE_K)
    #
    # for k_idx in range(num_k_tiles):
    #     # Load tiles
    #     x_tile = ct.load(x, index=(pid_m, k_idx), shape=(TILE_M, TILE_K), latency=3)
    #     w_tile = ct.load(weight_t, index=(k_idx, pid_n), shape=(TILE_K, TILE_N), latency=3)
    #
    #     # Convert to TF32 for tensor core
    #     x_tf32 = ct.astype(x_tile, ct.tfloat32)
    #     w_tf32 = ct.astype(w_tile, ct.tfloat32)
    #
    #     # Matrix multiply-accumulate
    #     acc = ct.mma(x_tf32, w_tf32, acc)
    #
    # Step 3: Store result
    # ct.store(output, index=(pid_m, pid_n), tile=acc)

    # YOUR CODE HERE
    pass  # Remove this and implement


@ct.kernel
def softmax_kernel(
    x,              # Input: (batch, seq_len)
    output,         # Output: (batch, seq_len)
    seq_len: ct.Constant[int]
):
    """
    Numerically stable softmax over last dimension.

    *** TODO: Implement this kernel ***

    Grid: (batch,)
    """
    pid = ct.bid(0)

    # ============================================================================
    # TODO: Implement softmax kernel
    # ============================================================================
    #
    # Similar to softmax_inplace_kernel in attention.py

    # YOUR CODE HERE
    pass  # Remove this and implement


# ============================================================================
# Layer Classes
# ============================================================================

def _is_power_of_two(x: int) -> bool:
    """Check if x is a power of two."""
    return x > 0 and (x & (x - 1)) == 0


class RMSNorm:
    """Root Mean Square Normalization using CuTile with CuPy fallback."""

    def __init__(self, hidden_size: int, eps: float = 1e-6):
        self.hidden_size = hidden_size
        self.eps = eps
        self.weight = cp.ones(hidden_size, dtype=cp.float32)
        # Use CuPy fallback for non-power-of-2 hidden sizes
        self.use_cutile = _is_power_of_two(hidden_size)

    def __call__(self, x: cp.ndarray) -> cp.ndarray:
        original_shape = x.shape

        if self.use_cutile:
            batch_size = int(np.prod(x.shape[:-1]))
            x_flat = x.reshape(batch_size, self.hidden_size).astype(cp.float32)
            output = cp.empty_like(x_flat)

            ct.launch(
                get_stream(),
                (batch_size,),
                rmsnorm_kernel,
                (x_flat, self.weight, output, self.eps, self.hidden_size)
            )

            return output.reshape(original_shape)
        else:
            # CuPy fallback for non-power-of-2
            x_float = x.astype(cp.float32)
            variance = cp.mean(x_float ** 2, axis=-1, keepdims=True)
            x_normed = x_float * cp.rsqrt(variance + self.eps)
            return (self.weight * x_normed).astype(x.dtype)


class LayerNorm:
    """Layer Normalization using CuTile with CuPy fallback."""

    def __init__(self, hidden_size: int, eps: float = 1e-5):
        self.hidden_size = hidden_size
        self.eps = eps
        self.weight = cp.ones(hidden_size, dtype=cp.float32)
        self.bias = cp.zeros(hidden_size, dtype=cp.float32)
        self.use_cutile = _is_power_of_two(hidden_size)

    def __call__(self, x: cp.ndarray) -> cp.ndarray:
        original_shape = x.shape

        if self.use_cutile:
            batch_size = int(np.prod(x.shape[:-1]))
            x_flat = x.reshape(batch_size, self.hidden_size).astype(cp.float32)
            output = cp.empty_like(x_flat)

            ct.launch(
                get_stream(),
                (batch_size,),
                layernorm_kernel,
                (x_flat, self.weight, self.bias, output, self.eps, self.hidden_size)
            )

            return output.reshape(original_shape)
        else:
            # CuPy fallback
            x_float = x.astype(cp.float32)
            mean = cp.mean(x_float, axis=-1, keepdims=True)
            variance = cp.var(x_float, axis=-1, keepdims=True)
            x_normed = (x_float - mean) / cp.sqrt(variance + self.eps)
            return (self.weight * x_normed + self.bias).astype(x.dtype)


def gelu(x: cp.ndarray) -> cp.ndarray:
    """GELU activation using CuTile."""
    original_shape = x.shape
    total = int(np.prod(x.shape))
    tile_size = 256
    padded = pad_to_multiple(total, tile_size)

    x_flat = x.reshape(-1).astype(cp.float32)
    if padded > total:
        x_flat = cp.pad(x_flat, (0, padded - total))

    output = cp.empty(padded, dtype=cp.float32)
    num_blocks = padded // tile_size

    ct.launch(get_stream(), (num_blocks,), gelu_kernel, (x_flat, output, tile_size))

    return output[:total].reshape(original_shape)


def silu(x: cp.ndarray) -> cp.ndarray:
    """SiLU activation using CuTile."""
    original_shape = x.shape
    total = int(np.prod(x.shape))
    tile_size = 256
    padded = pad_to_multiple(total, tile_size)

    x_flat = x.reshape(-1).astype(cp.float32)
    if padded > total:
        x_flat = cp.pad(x_flat, (0, padded - total))

    output = cp.empty(padded, dtype=cp.float32)
    num_blocks = padded // tile_size

    ct.launch(get_stream(), (num_blocks,), silu_kernel, (x_flat, output, tile_size))

    return output[:total].reshape(original_shape)


def get_activation(name: str):
    """Get activation function by name."""
    activations = {"gelu": gelu, "silu": silu}
    if name not in activations:
        raise ValueError(f"Unknown activation: {name}")
    return activations[name]


class Linear:
    """Linear layer with switchable backend (cuBLAS or CuTile TF32)."""

    TILE_M = 64
    TILE_N = 64
    TILE_K = 32
    BACKEND = 'cublas'  # 'cublas' or 'cutile_tf32'

    def __init__(self, in_features: int, out_features: int, bias: bool = True):
        self.in_features = in_features
        self.out_features = out_features
        self.has_bias = bias

        self.weight = cp.zeros((out_features, in_features), dtype=cp.float32)
        self.bias_param = cp.zeros(out_features, dtype=cp.float32) if bias else None

        self._weight_t_padded = None
        self._K_padded = None
        self._N_padded = None

    def _ensure_weight_prepared(self):
        """Cache transposed and padded weight for TF32 kernel."""
        if self._weight_t_padded is None:
            K = self.in_features
            N = self.out_features
            self._K_padded = pad_to_multiple(K, self.TILE_K)
            self._N_padded = pad_to_multiple(N, self.TILE_N)

            weight_t = cp.ascontiguousarray(self.weight.T)
            if self._K_padded > K or self._N_padded > N:
                self._weight_t_padded = cp.zeros((self._K_padded, self._N_padded), dtype=cp.float32)
                self._weight_t_padded[:K, :N] = weight_t
            else:
                self._weight_t_padded = weight_t

    def __call__(self, x: cp.ndarray) -> cp.ndarray:
        if Linear.BACKEND == 'cublas':
            return self._forward_cublas(x)
        else:
            return self._forward_cutile_tf32(x)

    def _forward_cublas(self, x: cp.ndarray) -> cp.ndarray:
        """cuBLAS backend."""
        original_shape = x.shape
        batch_dims = original_shape[:-1]

        M = int(np.prod(batch_dims))
        x_2d = x.reshape(M, self.in_features).astype(cp.float32)

        output = x_2d @ self.weight.T

        if self.has_bias and self.bias_param is not None:
            output = output + self.bias_param

        return output.reshape(*batch_dims, self.out_features)

    def _forward_cutile_tf32(self, x: cp.ndarray) -> cp.ndarray:
        """CuTile TF32 tensor core backend."""
        original_shape = x.shape
        batch_dims = original_shape[:-1]

        M = int(np.prod(batch_dims))
        K = self.in_features
        N = self.out_features

        x_2d = x.reshape(M, K).astype(cp.float32)

        self._ensure_weight_prepared()

        M_padded = pad_to_multiple(M, self.TILE_M)

        if M_padded > M or self._K_padded > K:
            x_padded = cp.zeros((M_padded, self._K_padded), dtype=cp.float32)
            x_padded[:M, :K] = x_2d
        else:
            x_padded = cp.ascontiguousarray(x_2d)

        output = cp.zeros((M_padded, self._N_padded), dtype=cp.float32)

        grid_m = M_padded // self.TILE_M
        grid_n = self._N_padded // self.TILE_N

        ct.launch(
            get_stream(),
            (grid_m, grid_n),
            linear_kernel_tf32,
            (x_padded, self._weight_t_padded, output, M_padded, self._N_padded, self._K_padded)
        )

        output = output[:M, :N]

        if self.has_bias and self.bias_param is not None:
            output = output + self.bias_param

        return output.reshape(*batch_dims, self.out_features)


class Embedding:
    """Embedding layer (uses CuPy indexing)."""

    def __init__(self, num_embeddings: int, embedding_dim: int):
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.weight = cp.zeros((num_embeddings, embedding_dim), dtype=cp.float32)

    def __call__(self, input_ids: cp.ndarray) -> cp.ndarray:
        return self.weight[input_ids]


def softmax(x: cp.ndarray, axis: int = -1) -> cp.ndarray:
    """Softmax using CuTile kernel."""
    if axis != -1 and axis != len(x.shape) - 1:
        x = cp.moveaxis(x, axis, -1)

    original_shape = x.shape
    batch_size = int(np.prod(x.shape[:-1]))
    seq_len = x.shape[-1]

    x_flat = x.reshape(batch_size, seq_len).astype(cp.float32)
    output = cp.empty_like(x_flat)

    ct.launch(
        get_stream(),
        (batch_size,),
        softmax_kernel,
        (x_flat, output, seq_len)
    )

    result = output.reshape(original_shape)

    if axis != -1 and axis != len(original_shape) - 1:
        result = cp.moveaxis(result, -1, axis)

    return result


class MLP:
    """MLP with SwiGLU gating using CuTile."""

    FUSED = False  # Fused mode disabled for student implementation
    TILE_M, TILE_N, TILE_K = 64, 64, 32

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        activation: str = "silu",
        bias: bool = False,
        use_gating: bool = True
    ):
        self.use_gating = use_gating
        self.act_fn = get_activation(activation)
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size

        if use_gating:
            self.gate_proj = Linear(hidden_size, intermediate_size, bias=bias)
            self.up_proj = Linear(hidden_size, intermediate_size, bias=bias)
        else:
            self.up_proj = Linear(hidden_size, intermediate_size, bias=bias)

        self.down_proj = Linear(intermediate_size, hidden_size, bias=bias)

    def __call__(self, x: cp.ndarray) -> cp.ndarray:
        if self.use_gating:
            return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
        else:
            return self.down_proj(self.act_fn(self.up_proj(x)))


class EncoderMLP:
    """Encoder MLP (no gating) using CuTile."""

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        activation: str = "gelu",
        bias: bool = True
    ):
        self.fc1 = Linear(hidden_size, intermediate_size, bias=bias)
        self.fc2 = Linear(intermediate_size, hidden_size, bias=bias)
        self.act_fn = get_activation(activation)

    def __call__(self, x: cp.ndarray) -> cp.ndarray:
        return self.fc2(self.act_fn(self.fc1(x)))


if __name__ == "__main__":
    print("Testing Pure CuTile Layers...")

    # Test RMSNorm
    print("\n=== RMSNorm ===")
    norm = RMSNorm(256)
    x = cp.random.randn(2, 16, 256).astype(cp.float32)
    y = norm(x)
    print(f"Input: {x.shape} -> Output: {y.shape}")

    # Test LayerNorm
    print("\n=== LayerNorm ===")
    ln = LayerNorm(256)
    y = ln(x)
    print(f"Input: {x.shape} -> Output: {y.shape}")

    # Test GELU
    print("\n=== GELU ===")
    y = gelu(x)
    print(f"Input: {x.shape} -> Output: {y.shape}")

    # Test SiLU
    print("\n=== SiLU ===")
    y = silu(x)
    print(f"Input: {x.shape} -> Output: {y.shape}")

    # Test Linear
    print("\n=== Linear ===")
    linear = Linear(256, 512)
    y = linear(x)
    print(f"Input: {x.shape} -> Output: {y.shape}")

    # Test MLP
    print("\n=== MLP ===")
    mlp = MLP(256, 512, activation="silu", use_gating=True)
    y = mlp(x)
    print(f"Input: {x.shape} -> Output: {y.shape}")

    print("\n All Pure CuTile layers working!")
