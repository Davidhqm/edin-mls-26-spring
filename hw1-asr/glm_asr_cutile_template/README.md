# GLM-ASR CuTile Implementation - Student Assignment

## Overview

This is an educational implementation of GLM-ASR using NVIDIA CuTile for GPU kernel programming. Your task is to implement the core computational kernels marked with `TODO`.

## What is CuTile?

CuTile is NVIDIA's Python-native GPU programming framework that allows you to write high-performance CUDA kernels using Python syntax. Key concepts:

- **`@ct.kernel`**: Decorator to define a GPU kernel
- **`ct.bid(i)`**: Get block ID for dimension i
- **`ct.load(tensor, index, shape)`**: Load a tile from global memory
- **`ct.store(tensor, index, tile)`**: Store a tile to global memory
- **`ct.matmul(a, b)`**: Matrix multiplication
- **`ct.mma(a, b, acc)`**: Matrix multiply-accumulate (for tensor cores)
- **`ct.launch(stream, grid, kernel, args)`**: Launch a kernel

## Your Assignment

Implement the following CuTile kernels:

### 1. `attention.py` - Attention Kernels
- **`attention_scores_kernel`**: Compute Q @ K^T scaled
- **`softmax_inplace_kernel`**: Numerically stable softmax
- **`attention_output_kernel`**: Compute attention_weights @ V

### 2. `layers.py` - Layer Kernels
- **`rmsnorm_kernel`**: RMS normalization
- **`layernorm_kernel`**: Layer normalization
- **`gelu_kernel`**: GELU activation
- **`silu_kernel`**: SiLU/Swish activation
- **`linear_kernel_tf32`**: TF32 tensor core matrix multiplication
- **`softmax_kernel`**: Softmax for general use

### 3. `rope.py` - Rotary Position Embedding Kernels
- **`compute_freqs_kernel`**: Compute cos/sin frequencies
- **`apply_rope_kernel`**: Apply rotation to Q and K

## Key CuTile Patterns

### Loading Data
```python
# Load a 2D tile
tile = ct.load(tensor, index=(row, col), shape=(TILE_H, TILE_W))

# Reshape for computation
tile = ct.reshape(tile, (TILE_H * TILE_W,))
```

### Reduction Operations
```python
# Sum along a dimension
total = ct.sum(tile)

# Max for numerical stability
max_val = ct.max(tile)
```

### Matrix Operations
```python
# Matrix multiplication
C = ct.matmul(A, B)  # A @ B

# Tensor core MMA (faster for compatible sizes)
C = ct.mma(A_tf32, B_tf32, C)  # C += A @ B
```

### Type Conversion for Tensor Cores
```python
# Convert to TF32 for tensor core acceleration
a_tf32 = ct.astype(a, ct.tfloat32)
```

## Key Formulas

### Attention
```
scores = Q @ K^T / sqrt(d_k)
attention = softmax(scores) @ V
```

### Numerically Stable Softmax
```
max_x = max(x)
softmax(x) = exp(x - max_x) / sum(exp(x - max_x))
```

### RMSNorm
```
rms = sqrt(mean(x^2) + eps)
output = x / rms * weight
```

### RoPE Rotation
```
[x1, x2] -> [x1*cos - x2*sin, x2*cos + x1*sin]
```

## Testing Your Implementation

```bash
python attention.py  # Test attention kernels
python layers.py     # Test layer kernels
python rope.py       # Test RoPE kernels
```

## Files Structure

| File | Status | Description |
|------|--------|-------------|
| `attention.py` | **TODO** | Attention kernels |
| `layers.py` | **TODO** | Layer kernels (RMSNorm, GELU, Linear) |
| `rope.py` | **TODO** | RoPE kernels |
| `conv.py` | Complete | Convolution layers |
| `model.py` | Complete | Full model using your kernels |
| `weight_loader.py` | Complete | Weight loading utilities |

## Tips

1. **Power of 2**: CuTile works best with power-of-2 tile sizes
2. **Memory coalescing**: Access memory in contiguous patterns
3. **TF32**: Use `ct.tfloat32` for tensor core acceleration
4. **Latency hints**: Use `latency=3` for memory prefetching
5. **Grid dimensions**: Calculate grid size as `(M // TILE_M, N // TILE_N)`

## CuTile vs CuPy Comparison

| Operation | CuPy | CuTile |
|-----------|------|--------|
| Softmax | `cp.exp(x) / cp.sum(...)` | Explicit kernel with reductions |
| MatMul | `a @ b` | `ct.matmul(a, b)` or `ct.mma(...)` |
| Element-wise | `x * y` | Explicit tile loading/storing |

The benefit of CuTile is fine-grained control over memory access patterns and the ability to fuse operations that would require multiple kernel launches in CuPy.

## Reference

- [CuTile Documentation](https://docs.nvidia.com/cuda/cutile/)
- [Tensor Core Programming](https://developer.nvidia.com/blog/programming-tensor-cores-cuda-9/)

Good luck!
