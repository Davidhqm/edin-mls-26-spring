# GLM-ASR Triton Implementation - Student Assignment

## Overview

This is an educational implementation of GLM-ASR using Triton for GPU kernel programming. Your task is to implement the core computational kernels marked with `TODO`.

## What is Triton?

Triton is a Python-native GPU programming framework that lets you write high-performance kernels with Python syntax. Key concepts:

- **`@triton.jit`**: Decorator to define a GPU kernel
- **`tl.program_id(axis)`**: Get program ID for a grid axis
- **`tl.load(ptr, ...)`**: Load values from global memory
- **`tl.store(ptr, ...)`**: Store values to global memory
- **`tl.dot(a, b)`**: Block-level dot product
- **`tl.where(cond, a, b)`**: Element-wise select

## Your Assignment

Implement the following Triton kernels:

### 1. `attention.py` - Attention Kernels
- **`attention_scores_kernel`**: Compute Q @ K^T scaled
- **`softmax_inplace_kernel`**: Numerically stable softmax
- **`attention_output_kernel`**: Compute attention_weights @ V

### 2. `layers.py` - Layer Kernels
- **`rmsnorm_kernel`**: RMS normalization
- **`layernorm_kernel`**: Layer normalization
- **`gelu_kernel`**: GELU activation
- **`silu_kernel`**: SiLU/Swish activation
- **`linear_kernel_tf32`**: TF32-style matrix multiplication
- **`softmax_kernel`**: Softmax for general use

### 3. `rope.py` - Rotary Position Embedding Kernels
- **`compute_freqs_kernel`**: Compute cos/sin frequencies

## Key Triton Patterns

### Loading Data
```python
offs = tl.arange(0, BLOCK)
mask = offs < n_elements
x = tl.load(x_ptr + offs, mask=mask, other=0.0)
```

### Reduction Operations
```python
max_val = tl.max(x, axis=0)
sum_val = tl.sum(x, axis=0)
```

### Matrix Operations
```python
acc = tl.dot(a, b)
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

1. **Power of 2**: Triton works best with power-of-2 tile sizes
2. **Memory coalescing**: Access memory in contiguous patterns
3. **Keep types explicit**: Use `tl.float32` for accumulation
4. **Grid dimensions**: Use `triton.cdiv` to compute grid sizes

## Triton vs Torch Comparison

| Operation | Torch | Triton |
|-----------|-------|--------|
| Softmax | `torch.softmax(x, dim=-1)` | Explicit kernel with reductions |
| MatMul | `torch.matmul(a, b)` | `tl.dot` inside a kernel |
| Element-wise | `x * y` | Explicit load/compute/store |

The benefit of Triton is fine-grained control over memory access patterns and the ability to fuse operations that would require multiple kernel launches in PyTorch.

## Reference

- [Triton Documentation](https://triton-lang.org/)
- [Triton Tutorials](https://triton-lang.org/main/getting-started/tutorials/index.html)

Good luck!
