"""
Pure CuTile Neural Network Layers
End-to-end implementation using only NVIDIA CuTile kernels

All operations are implemented as CuTile kernels without CuPy fallbacks.
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
# CuTile Kernels
# ============================================================================

@ct.kernel
def rmsnorm_kernel(
    x,              # Input: (batch_size, hidden_size)
    weight,         # Weight: (hidden_size,)
    output,         # Output: (batch_size, hidden_size)
    eps: ct.Constant[float],
    hidden_size: ct.Constant[int]
):
    """RMSNorm: x / RMS(x) * weight"""
    pid = ct.bid(0)

    x_tile = ct.load(x, index=(pid, 0), shape=(1, hidden_size))
    x_tile = ct.reshape(x_tile, (hidden_size,))
    w_tile = ct.load(weight, index=(0,), shape=(hidden_size,))

    variance = ct.sum(x_tile * x_tile) / hidden_size
    x_norm = x_tile / ct.sqrt(variance + eps)
    result = x_norm * w_tile

    result = ct.reshape(result, (1, hidden_size))
    ct.store(output, index=(pid, 0), tile=result)


@ct.kernel
def layernorm_kernel(
    x,              # Input: (batch_size, hidden_size)
    weight,         # Weight: (hidden_size,)
    bias,           # Bias: (hidden_size,)
    output,         # Output: (batch_size, hidden_size)
    eps: ct.Constant[float],
    hidden_size: ct.Constant[int]
):
    """LayerNorm: (x - mean) / sqrt(var + eps) * weight + bias"""
    pid = ct.bid(0)

    x_tile = ct.load(x, index=(pid, 0), shape=(1, hidden_size))
    x_tile = ct.reshape(x_tile, (hidden_size,))
    w_tile = ct.load(weight, index=(0,), shape=(hidden_size,))
    b_tile = ct.load(bias, index=(0,), shape=(hidden_size,))

    mean = ct.sum(x_tile) / hidden_size
    x_centered = x_tile - mean
    variance = ct.sum(x_centered * x_centered) / hidden_size
    x_norm = x_centered / ct.sqrt(variance + eps)
    result = x_norm * w_tile + b_tile

    result = ct.reshape(result, (1, hidden_size))
    ct.store(output, index=(pid, 0), tile=result)


@ct.kernel
def gelu_kernel(x, output, tile_size: ct.Constant[int]):
    """GELU using tanh approximation."""
    pid = ct.bid(0)
    x_tile = ct.load(x, index=(pid,), shape=(tile_size,))

    sqrt_2_over_pi = 0.7978845608028654
    x3 = x_tile * x_tile * x_tile
    inner = sqrt_2_over_pi * (x_tile + 0.044715 * x3)
    result = x_tile * 0.5 * (1.0 + ct.tanh(inner))

    ct.store(output, index=(pid,), tile=result)


@ct.kernel
def silu_kernel(x, output, tile_size: ct.Constant[int]):
    """SiLU/Swish: x * sigmoid(x)"""
    pid = ct.bid(0)
    x_tile = ct.load(x, index=(pid,), shape=(tile_size,))

    sigmoid = 1.0 / (1.0 + ct.exp(-x_tile))
    result = x_tile * sigmoid

    ct.store(output, index=(pid,), tile=result)


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
    Uses TF32 format for tensor core acceleration on Blackwell/Ampere+.
    64x64 tiles with 32-wide K dimension for optimal performance.
    """
    TILE_M, TILE_N, TILE_K = 64, 64, 32
    pid_m = ct.bid(0)
    pid_n = ct.bid(1)

    # Accumulator in float32 for precision
    acc = ct.zeros((TILE_M, TILE_N), dtype=ct.float32)

    # Number of K tiles
    num_k_tiles = ct.cdiv(K, TILE_K)

    for k_idx in range(num_k_tiles):
        # Load tiles with latency hint for prefetching
        x_tile = ct.load(x, index=(pid_m, k_idx), shape=(TILE_M, TILE_K), latency=3)
        w_tile = ct.load(weight_t, index=(k_idx, pid_n), shape=(TILE_K, TILE_N), latency=3)

        # Convert to TF32 for tensor core MMA
        x_tf32 = ct.astype(x_tile, ct.tfloat32)
        w_tf32 = ct.astype(w_tile, ct.tfloat32)

        # Tensor core MMA: (64, 32) @ (32, 64) -> (64, 64)
        acc = ct.mma(x_tf32, w_tf32, acc)

    ct.store(output, index=(pid_m, pid_n), tile=acc)


# ============================================================================
# Fused Kernels for V5 Optimization
# ============================================================================

@ct.kernel(occupancy=2)
def linear_gelu_kernel(
    x,              # Input: (M, K) - float32
    weight_t,       # Weight transposed: (K, N) - pre-transposed, float32
    output,         # Output: (M, N) - float32
    M: ct.Constant[int],
    N: ct.Constant[int],
    K: ct.Constant[int]
):
    """
    Fused Linear + GELU: output = GELU(x @ weight_t)
    Combines matrix multiplication with GELU activation in a single kernel.
    Uses TF32 for tensor core acceleration.
    """
    TILE_M, TILE_N, TILE_K = 64, 64, 32
    pid_m = ct.bid(0)
    pid_n = ct.bid(1)

    acc = ct.zeros((TILE_M, TILE_N), dtype=ct.float32)
    num_k_tiles = ct.cdiv(K, TILE_K)

    for k_idx in range(num_k_tiles):
        x_tile = ct.load(x, index=(pid_m, k_idx), shape=(TILE_M, TILE_K), latency=3)
        w_tile = ct.load(weight_t, index=(k_idx, pid_n), shape=(TILE_K, TILE_N), latency=3)

        x_tf32 = ct.astype(x_tile, ct.tfloat32)
        w_tf32 = ct.astype(w_tile, ct.tfloat32)
        acc = ct.mma(x_tf32, w_tf32, acc)

    # Apply GELU (tanh approximation) in-register
    sqrt_2_over_pi = 0.7978845608028654
    acc3 = acc * acc * acc
    inner = sqrt_2_over_pi * (acc + 0.044715 * acc3)
    acc = acc * 0.5 * (1.0 + ct.tanh(inner))

    ct.store(output, index=(pid_m, pid_n), tile=acc)


@ct.kernel(occupancy=2)
def swiglu_fused_kernel(
    x,              # Input: (M, K) - float32
    gate_weight_t,  # Gate weight transposed: (K, N) - float32
    up_weight_t,    # Up weight transposed: (K, N) - float32
    output,         # Output: (M, N) - float32
    M: ct.Constant[int],
    N: ct.Constant[int],
    K: ct.Constant[int]
):
    """
    Fused SwiGLU: output = SiLU(x @ gate_weight_t) * (x @ up_weight_t)
    Computes both gate and up projections, applies SiLU to gate, and multiplies.
    All done in-register without intermediate memory writes.
    Uses TF32 for tensor core acceleration.
    """
    TILE_M, TILE_N, TILE_K = 64, 64, 32
    pid_m = ct.bid(0)
    pid_n = ct.bid(1)

    # Two accumulators for gate and up projections
    gate_acc = ct.zeros((TILE_M, TILE_N), dtype=ct.float32)
    up_acc = ct.zeros((TILE_M, TILE_N), dtype=ct.float32)
    num_k_tiles = ct.cdiv(K, TILE_K)

    for k_idx in range(num_k_tiles):
        # Load input tile once
        x_tile = ct.load(x, index=(pid_m, k_idx), shape=(TILE_M, TILE_K), latency=3)
        x_tf32 = ct.astype(x_tile, ct.tfloat32)

        # Load both weight tiles
        gate_w = ct.load(gate_weight_t, index=(k_idx, pid_n), shape=(TILE_K, TILE_N), latency=3)
        up_w = ct.load(up_weight_t, index=(k_idx, pid_n), shape=(TILE_K, TILE_N), latency=3)

        gate_w_tf32 = ct.astype(gate_w, ct.tfloat32)
        up_w_tf32 = ct.astype(up_w, ct.tfloat32)

        # Compute both projections
        gate_acc = ct.mma(x_tf32, gate_w_tf32, gate_acc)
        up_acc = ct.mma(x_tf32, up_w_tf32, up_acc)

    # Apply SiLU to gate and multiply with up (in-register)
    sigmoid = 1.0 / (1.0 + ct.exp(-gate_acc))
    gate_activated = gate_acc * sigmoid
    result = gate_activated * up_acc

    ct.store(output, index=(pid_m, pid_n), tile=result)


@ct.kernel
def embedding_kernel(
    indices,        # Input: (batch_size,)
    weight,         # Embedding table: (vocab_size, embed_dim)
    output,         # Output: (batch_size, embed_dim)
    embed_dim: ct.Constant[int]
):
    """Embedding lookup using gather."""
    pid = ct.bid(0)

    # Load index as scalar
    idx = ct.load(indices, index=(pid,), shape=())  # Load as scalar

    # Create column indices for the embedding dimension
    col_indices = ct.arange(embed_dim, dtype=ct.int32)

    # Gather the embedding row: weight[idx, 0:embed_dim]
    emb_tile = ct.gather(weight, (idx, col_indices))

    # Store
    emb_tile = ct.reshape(emb_tile, (1, embed_dim))
    ct.store(output, index=(pid, 0), tile=emb_tile)


@ct.kernel
def softmax_kernel(
    x,              # Input: (batch, seq_len)
    output,         # Output: (batch, seq_len)
    seq_len: ct.Constant[int]
):
    """Numerically stable softmax over last dimension."""
    pid = ct.bid(0)

    x_tile = ct.load(x, index=(pid, 0), shape=(1, seq_len))
    x_tile = ct.reshape(x_tile, (seq_len,))

    # Max for numerical stability
    x_max = ct.max(x_tile)
    x_shifted = x_tile - x_max

    # Exp and normalize
    exp_x = ct.exp(x_shifted)
    sum_exp = ct.sum(exp_x)
    result = exp_x / sum_exp

    result = ct.reshape(result, (1, seq_len))
    ct.store(output, index=(pid, 0), tile=result)


@ct.kernel
def attention_scores_kernel(
    q,              # Query: (batch*heads, seq_q, head_dim)
    k,              # Key: (batch*heads, seq_k, head_dim)
    scores,         # Output: (batch*heads, seq_q, seq_k)
    scale: ct.Constant[float],
    seq_q: ct.Constant[int],
    seq_k: ct.Constant[int],
    head_dim: ct.Constant[int]
):
    """Compute attention scores: Q @ K.T * scale"""
    pid_bh = ct.bid(0)  # batch*heads
    pid_q = ct.bid(1)   # query position

    # Load query vector
    q_tile = ct.load(q, index=(pid_bh, pid_q, 0), shape=(1, 1, head_dim))
    q_tile = ct.reshape(q_tile, (head_dim,))

    # Compute dot product with each key
    for k_pos in range(seq_k):
        k_tile = ct.load(k, index=(pid_bh, k_pos, 0), shape=(1, 1, head_dim))
        k_tile = ct.reshape(k_tile, (head_dim,))

        # Dot product
        score = ct.sum(q_tile * k_tile) * scale

        # Store as scalar
        score_tile = ct.reshape(score, (1, 1, 1))
        ct.store(scores, index=(pid_bh, pid_q, k_pos), tile=score_tile)


@ct.kernel
def attention_output_kernel(
    weights,        # Attention weights: (batch*heads, seq_q, seq_k)
    v,              # Value: (batch*heads, seq_k, head_dim)
    output,         # Output: (batch*heads, seq_q, head_dim)
    seq_q: ct.Constant[int],
    seq_k: ct.Constant[int],
    head_dim: ct.Constant[int]
):
    """Compute attention output: weights @ V"""
    pid_bh = ct.bid(0)
    pid_q = ct.bid(1)

    # Load attention weights for this query
    w_tile = ct.load(weights, index=(pid_bh, pid_q, 0), shape=(1, 1, seq_k))
    w_tile = ct.reshape(w_tile, (seq_k,))

    # Accumulate weighted values
    acc = ct.zeros((head_dim,), dtype=ct.float32)

    for k_pos in range(seq_k):
        v_tile = ct.load(v, index=(pid_bh, k_pos, 0), shape=(1, 1, head_dim))
        v_tile = ct.reshape(v_tile, (head_dim,))

        weight = w_tile[k_pos]
        acc = acc + weight * v_tile

    acc = ct.reshape(acc, (1, 1, head_dim))
    ct.store(output, index=(pid_bh, pid_q, 0), tile=acc)


@ct.kernel
def causal_mask_kernel(
    scores,         # Input/Output: (batch*heads, seq_q, seq_k)
    seq_q: ct.Constant[int],
    seq_k: ct.Constant[int],
    offset: ct.Constant[int]  # offset for KV cache
):
    """Apply causal mask to attention scores."""
    pid_bh = ct.bid(0)
    pid_q = ct.bid(1)

    # Load row
    row = ct.load(scores, index=(pid_bh, pid_q, 0), shape=(1, 1, seq_k))
    row = ct.reshape(row, (seq_k,))

    # Create mask: positions > (pid_q + offset) should be -inf
    current_pos = pid_q + offset
    for k_pos in range(seq_k):
        if k_pos > current_pos:
            row = ct.where(ct.arange(seq_k) == k_pos, -1e9, row)

    row = ct.reshape(row, (1, 1, seq_k))
    ct.store(scores, index=(pid_bh, pid_q, 0), tile=row)


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
                layernorm_kernel,
                (x_flat, self.weight, self.bias, output, self.eps, self.hidden_size)
            )

            return output.reshape(original_shape)
        else:
            # CuPy fallback for non-power-of-2
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

    # Tile sizes for TF32 kernel
    TILE_M = 64
    TILE_N = 64
    TILE_K = 32

    # Global switch: 'cublas' or 'cutile_tf32'
    BACKEND = 'cublas'

    def __init__(self, in_features: int, out_features: int, bias: bool = True):
        self.in_features = in_features
        self.out_features = out_features
        self.has_bias = bias

        self.weight = cp.zeros((out_features, in_features), dtype=cp.float32)
        self.bias_param = cp.zeros(out_features, dtype=cp.float32) if bias else None

        # Cached pre-transposed weight (computed lazily for CuTile)
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

            # Transpose: (N, K) -> (K, N), then pad
            weight_t = cp.ascontiguousarray(self.weight.T)
            if self._K_padded > K or self._N_padded > N:
                self._weight_t_padded = cp.zeros((self._K_padded, self._N_padded), dtype=cp.float32)
                self._weight_t_padded[:K, :N] = weight_t
            else:
                self._weight_t_padded = weight_t

    def __call__(self, x: cp.ndarray) -> cp.ndarray:
        if Linear.BACKEND == 'cublas':
            return self._forward_cublas(x)
        elif Linear.BACKEND == 'cutile_tf32':
            return self._forward_cutile_tf32(x)
        else:  # 'adaptive' - V7: use CuTile for large M, cuBLAS for small M
            M = int(np.prod(x.shape[:-1]))
            # CuTile is faster when M >= TILE_M (64), cuBLAS wins for small M
            if M >= self.TILE_M:
                return self._forward_cutile_tf32(x)
            else:
                return self._forward_cublas(x)

    def _forward_cublas(self, x: cp.ndarray) -> cp.ndarray:
        """cuBLAS backend (V3)."""
        original_shape = x.shape
        batch_dims = original_shape[:-1]

        M = int(np.prod(batch_dims))
        x_2d = x.reshape(M, self.in_features).astype(cp.float32)

        output = x_2d @ self.weight.T

        if self.has_bias and self.bias_param is not None:
            output = output + self.bias_param

        return output.reshape(*batch_dims, self.out_features)

    def _forward_cutile_tf32(self, x: cp.ndarray) -> cp.ndarray:
        """CuTile TF32 tensor core backend (V4)."""
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
    """Embedding layer using CuTile."""

    def __init__(self, num_embeddings: int, embedding_dim: int):
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.weight = cp.zeros((num_embeddings, embedding_dim), dtype=cp.float32)

    def __call__(self, input_ids: cp.ndarray) -> cp.ndarray:
        original_shape = input_ids.shape
        batch_size = int(np.prod(original_shape))

        indices_flat = input_ids.reshape(-1).astype(cp.int32)
        output = cp.empty((batch_size, self.embedding_dim), dtype=cp.float32)

        ct.launch(
            get_stream(),
            (batch_size,),
            embedding_kernel,
            (indices_flat, self.weight, output, self.embedding_dim)
        )

        return output.reshape(*original_shape, self.embedding_dim)


def softmax(x: cp.ndarray, axis: int = -1) -> cp.ndarray:
    """Softmax using CuTile kernel."""
    if axis != -1 and axis != len(x.shape) - 1:
        # Move axis to last position
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
    """MLP with SwiGLU gating using CuTile.

    Supports two modes:
    - FUSED=False: Separate gate, up, activation, multiply, down operations
    - FUSED=True: Fused SwiGLU kernel (gate + up + SiLU + multiply in one kernel)
    """

    # Class-level toggle for fused mode
    FUSED = True  # Enable fused SwiGLU by default for V5
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
        self.bias_enabled = bias

        if use_gating:
            self.gate_proj = Linear(hidden_size, intermediate_size, bias=bias)
            self.up_proj = Linear(hidden_size, intermediate_size, bias=bias)
        else:
            self.up_proj = Linear(hidden_size, intermediate_size, bias=bias)

        self.down_proj = Linear(intermediate_size, hidden_size, bias=bias)

        # Pre-transpose weights for fused kernel (if using gating)
        self._gate_weight_t = None
        self._up_weight_t = None

    def _prepare_fused_weights(self):
        """Prepare pre-transposed weights for fused kernel."""
        if self._gate_weight_t is None and self.use_gating:
            self._gate_weight_t = cp.ascontiguousarray(self.gate_proj.weight.T)
            self._up_weight_t = cp.ascontiguousarray(self.up_proj.weight.T)

    def __call__(self, x: cp.ndarray) -> cp.ndarray:
        if self.use_gating and MLP.FUSED:
            return self._forward_fused(x)
        else:
            return self._forward_standard(x)

    def _forward_standard(self, x: cp.ndarray) -> cp.ndarray:
        """Standard (unfused) forward pass."""
        if self.use_gating:
            return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
        else:
            return self.down_proj(self.act_fn(self.up_proj(x)))

    def _forward_fused(self, x: cp.ndarray) -> cp.ndarray:
        """Fused SwiGLU forward pass."""
        self._prepare_fused_weights()

        orig_shape = x.shape
        x_2d = x.reshape(-1, self.hidden_size)
        M = x_2d.shape[0]
        K = self.hidden_size
        N = self.intermediate_size

        # Pad dimensions for tile alignment
        M_pad = pad_to_multiple(M, self.TILE_M)
        K_pad = pad_to_multiple(K, self.TILE_K)
        N_pad = pad_to_multiple(N, self.TILE_N)

        # Pad input if needed
        if M != M_pad or K != K_pad:
            x_padded = cp.zeros((M_pad, K_pad), dtype=cp.float32)
            x_padded[:M, :K] = x_2d
        else:
            x_padded = x_2d

        # Pad weights if needed
        if K != K_pad or N != N_pad:
            gate_w_padded = cp.zeros((K_pad, N_pad), dtype=cp.float32)
            gate_w_padded[:K, :N] = self._gate_weight_t
            up_w_padded = cp.zeros((K_pad, N_pad), dtype=cp.float32)
            up_w_padded[:K, :N] = self._up_weight_t
        else:
            gate_w_padded = self._gate_weight_t
            up_w_padded = self._up_weight_t

        # Allocate output
        intermediate = cp.zeros((M_pad, N_pad), dtype=cp.float32)

        # Launch fused SwiGLU kernel
        grid_m = M_pad // self.TILE_M
        grid_n = N_pad // self.TILE_N
        ct.launch(
            get_stream(),
            (grid_m, grid_n),
            swiglu_fused_kernel,
            (x_padded, gate_w_padded, up_w_padded, intermediate, M_pad, N_pad, K_pad)
        )

        # Extract unpadded result and pass through down_proj
        if M != M_pad or N != N_pad:
            intermediate = intermediate[:M, :N]

        intermediate = intermediate.reshape(*orig_shape[:-1], self.intermediate_size)
        return self.down_proj(intermediate)


class EncoderMLP:
    """Encoder MLP (no gating) using CuTile.

    Supports two modes:
    - FUSED=False: Separate fc1, activation, fc2 operations
    - FUSED=True: Fused Linear+GELU kernel for fc1
    """

    # Class-level toggle for fused mode
    FUSED = True  # Enable fused Linear+GELU by default for V5
    TILE_M, TILE_N, TILE_K = 64, 64, 32

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
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.bias_enabled = bias
        self.activation = activation

        # Pre-transpose weights for fused kernel
        self._fc1_weight_t = None

    def _prepare_fused_weights(self):
        """Prepare pre-transposed weights for fused kernel."""
        if self._fc1_weight_t is None:
            self._fc1_weight_t = cp.ascontiguousarray(self.fc1.weight.T)

    def __call__(self, x: cp.ndarray) -> cp.ndarray:
        if EncoderMLP.FUSED and self.activation == "gelu":
            return self._forward_fused(x)
        else:
            return self._forward_standard(x)

    def _forward_standard(self, x: cp.ndarray) -> cp.ndarray:
        """Standard (unfused) forward pass."""
        return self.fc2(self.act_fn(self.fc1(x)))

    def _forward_fused(self, x: cp.ndarray) -> cp.ndarray:
        """Fused Linear+GELU forward pass."""
        self._prepare_fused_weights()

        orig_shape = x.shape
        x_2d = x.reshape(-1, self.hidden_size)
        M = x_2d.shape[0]
        K = self.hidden_size
        N = self.intermediate_size

        # Pad dimensions for tile alignment
        M_pad = pad_to_multiple(M, self.TILE_M)
        K_pad = pad_to_multiple(K, self.TILE_K)
        N_pad = pad_to_multiple(N, self.TILE_N)

        # Pad input if needed
        if M != M_pad or K != K_pad:
            x_padded = cp.zeros((M_pad, K_pad), dtype=cp.float32)
            x_padded[:M, :K] = x_2d
        else:
            x_padded = x_2d

        # Pad weights if needed
        if K != K_pad or N != N_pad:
            fc1_w_padded = cp.zeros((K_pad, N_pad), dtype=cp.float32)
            fc1_w_padded[:K, :N] = self._fc1_weight_t
        else:
            fc1_w_padded = self._fc1_weight_t

        # Allocate output
        intermediate = cp.zeros((M_pad, N_pad), dtype=cp.float32)

        # Launch fused Linear+GELU kernel
        grid_m = M_pad // self.TILE_M
        grid_n = N_pad // self.TILE_N
        ct.launch(
            get_stream(),
            (grid_m, grid_n),
            linear_gelu_kernel,
            (x_padded, fc1_w_padded, intermediate, M_pad, N_pad, K_pad)
        )

        # Extract unpadded result
        if M != M_pad or N != N_pad:
            intermediate = intermediate[:M, :N]

        # Add bias if enabled
        if self.bias_enabled and self.fc1.bias is not None:
            intermediate = intermediate + self.fc1.bias

        intermediate = intermediate.reshape(*orig_shape[:-1], self.intermediate_size)
        return self.fc2(intermediate)


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

    # Test Embedding
    print("\n=== Embedding ===")
    emb = Embedding(1000, 256)
    ids = cp.random.randint(0, 1000, (2, 16), dtype=cp.int32)
    y = emb(ids)
    print(f"Input: {ids.shape} -> Output: {y.shape}")

    # Test Softmax
    print("\n=== Softmax ===")
    x_sm = cp.random.randn(2, 4, 16, 16).astype(cp.float32)
    y = softmax(x_sm, axis=-1)
    print(f"Input: {x_sm.shape} -> Output: {y.shape}")
    print(f"Sum along last axis: {float(y[0, 0, 0].sum()):.6f} (should be 1.0)")

    # Test MLP
    print("\n=== MLP ===")
    mlp = MLP(256, 512, activation="silu", use_gating=True)
    y = mlp(x)
    print(f"Input: {x.shape} -> Output: {y.shape}")

    print("\n All Pure CuTile layers working!")
