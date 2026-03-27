# Triton Secret Notes

Tips, gotchas, and best practices that are useful when writing Triton kernels.

---

## 1. `tl.constexpr` is Required for Block Shapes

Any value used in `tl.arange`, block shapes, or static loops must be a `tl.constexpr`.

```python
@triton.jit
def kernel(X, Y, N, BLOCK: tl.constexpr):
    offs = tl.arange(0, BLOCK)  # OK
```

---

## 2. Always Use Masks for Out-of-Bounds

When `N` or matrix sizes are not multiples of block sizes, mask accesses:

```python
mask = offs < N
x = tl.load(X + offs, mask=mask, other=0.0)
tl.store(Y + offs, x, mask=mask)
```

---

## 3. Be Explicit About Type Casting

Triton does not always implicitly cast between FP16 and FP32.
Cast explicitly when you want higher-precision math:

```python
x_fp32 = x.to(tl.float32)
y_fp16 = y_fp32.to(tl.float16)
```

---

## 4. Strides Are in Elements, Not Bytes

Torch provides strides in **elements**, not bytes. Use them directly in pointer arithmetic:

```python
ptrs = X + offs_m * stride_m + offs_n * stride_n
```

---

## 5. Tune `num_warps` and `num_stages`

Kernel launch parameters matter:

```python
kernel[grid](..., num_warps=4, num_stages=2)
```

- **`num_warps`** controls parallelism inside a program.
- **`num_stages`** controls pipelining depth for memory ops.

---

## 6. Use `triton.autotune` for Performance

Autotune lets Triton pick the fastest configuration at runtime:

```python
@triton.autotune(
    configs=[triton.Config({"BLOCK": 128}, num_warps=4),
             triton.Config({"BLOCK": 256}, num_warps=8)],
    key=["N"],
)
@triton.jit
def kernel(...):
    ...
```

---

## 7. Be Careful with Reductions

Reductions require extra care with accumulation precision and masks. Use FP32 accumulation when possible.

---

## 8. Debugging Tips

- Use `tl.device_print` if your Triton version supports it (prints are slow and can overwhelm output).
- Start with small shapes and block sizes.
- Validate against PyTorch or NumPy references.
