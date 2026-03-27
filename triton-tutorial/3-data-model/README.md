# Triton Data Model

This tutorial covers how Triton handles data types, constants, and mixed precision.

## 1. Compile-Time Constants (`tl.constexpr`)

In the kernel signature:
```python
def mixed_precision_scale(X, Y, N, scale, BLOCK: tl.constexpr):
```

**Why use `tl.constexpr`?**
- Triton needs some values (like block sizes) at compile time.
- This enables loop unrolling, register allocation, and shape specialization.

**Rule**: Any value used in `tl.arange(...)`, block shapes, or other compile-time constructs should be a `tl.constexpr`.

## 2. Mixed Precision (FP16 / FP32)

Deep learning often uses FP16 for memory bandwidth, but computations in FP32 for accuracy.

### Loading
```python
x = tl.load(X + offs, mask=mask, other=0.0)
```

- If `X` is `float16`, Triton loads `float16`.

### Compute in FP32
```python
x_fp32 = x.to(tl.float32)
y_fp32 = x_fp32 * scale
```

- Explicit casting makes the compute happen in FP32.

### Store back to FP16
```python
y_fp16 = y_fp32.to(tl.float16)
tl.store(Y + offs, y_fp16, mask=mask)
```

## 3. Supported Types

Triton supports common numeric types:
- `int32`, `int64`
- `float16`, `bfloat16`, `float32`
- `bool`

(Exact support depends on GPU and backend.)
