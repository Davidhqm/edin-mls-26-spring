# cuTile Secret Notes

Tips, gotchas, and best practices that aren't immediately obvious from the documentation.

---

## 1. Tile Dimensions Must Be Powers of 2

Every dimension of a `Tile` **must** be a power of 2 (1, 2, 4, 8, 16, 32, 64, ...).

```python
# OK
tile = ct.load(arr, index=(pid,), shape=(32,))
tile = ct.load(arr, index=(py, px), shape=(16, 64))

# BAD — will fail at compile time
tile = ct.load(arr, index=(pid,), shape=(48,))   # 48 is not power of 2
tile = ct.load(arr, index=(py, px), shape=(12, 12))
```

If your data size is not a multiple of the tile size, cuTile handles the edges automatically — out-of-bounds loads return padding values, and out-of-bounds stores are silently ignored.

---

## 2. ct.Constant is Critical for Performance

Without `ct.Constant`, the compiler cannot:
- Unroll loops
- Allocate registers efficiently
- Determine tile shapes at compile time

```python
# GOOD — compiler can optimize
@ct.kernel
def my_kernel(data, out, tile_size: ct.Constant[int]):
    tile = ct.load(data, index=(ct.bid(0),), shape=(tile_size,))

# BAD — tile_size is unknown at compile time, shapes won't resolve
@ct.kernel
def my_kernel(data, out, tile_size: int):
    tile = ct.load(data, index=(ct.bid(0),), shape=(tile_size,))
```

**Rule**: Any value used in `shape=`, `ct.zeros()`, `ct.full()`, `range()`, or other compile-time constructs **must** be `ct.Constant`.

---

## 3. Type Promotion Rules

cuTile follows NumPy-like type promotion, but some conversions are **not** implicit:

| Operation | Result | Notes |
|-----------|--------|-------|
| `float16 * float32` | `float32` | Implicit promotion (safe) |
| `int32 + float32` | `float32` | Implicit promotion (safe) |
| `float32 → float16` | **Error** | Requires explicit `ct.astype()` |
| `float32 → int32` | **Error** | Requires explicit `ct.astype()` |

```python
# This will FAIL:
result_fp32 = fp16_tile * fp32_scalar
ct.store(fp16_output, ..., tile=result_fp32)  # TileTypeError!

# Fix: explicit cast
result_fp16 = ct.astype(result_fp32, ct.float16)
ct.store(fp16_output, ..., tile=result_fp16)
```

**Restricted float types** (`float8_e4m3fn`, `float8_e5m2`) always require explicit casts.

---

## 4. Memory Access Order

The `order` parameter in `ct.load()` / `ct.store()` controls the memory access pattern:

- `order='C'` (default): Row-major — elements within a row are contiguous
- `order='F'`: Column-major — elements within a column are contiguous
- `order=(1, 0)`: Explicit axis permutation

When your algorithm accesses data column-by-column (e.g., loading columns of a matrix), using `order='F'` can dramatically improve memory bandwidth by enabling coalesced access.

```python
# Default row-major load
tile = ct.load(matrix, index=(pid_y, pid_x), shape=(16, 16), order='C')

# Column-major load — better when accessing column slices
tile = ct.load(matrix, index=(pid_y, pid_x), shape=(16, 16), order='F')
```

---

## 5. PaddingMode for Edge Tiles

When your data size isn't a multiple of the tile size, the last tiles will access out-of-bounds memory. `ct.load` handles this via `padding_mode`:

```python
ct.load(arr, index=(pid,), shape=(tile_size,),
        padding_mode=ct.PaddingMode.ZERO)       # Out-of-bounds → 0
```

Available modes:
- `UNDETERMINED` (default): Padding value is unspecified
- `ZERO`: Pad with 0
- `NAN`: Pad with NaN
- `NEG_INF` / `POS_INF`: Pad with ±infinity
- `NEG_ZERO`: Pad with -0.0

For `ct.store`, out-of-bounds writes are **always silently ignored** — no padding mode needed.

---

## 6. ct.mma vs @ Operator

Both perform matrix multiplication, but they differ:

| Feature | `@` operator (`ct.matmul`) | `ct.mma(a, b, acc)` |
|---------|---------------------------|---------------------|
| Operation | `a @ b` | `a @ b + acc` (fused) |
| Tensor Cores | Depends on compiler | Explicitly targets hardware MMA |
| Accumulator | Creates new tile | Reuses existing accumulator |
| Best for | One-off multiplications | Inner loops (GEMM, attention) |

```python
# Standard matmul
result = a_tile @ b_tile

# Fused multiply-accumulate (preferred in loops)
acc = ct.zeros((tile_m, tile_n), dtype=ct.float32)
for k in range(num_k_tiles):
    a = ct.load(A, index=(pid_m, k), shape=(tile_m, tile_k))
    b = ct.load(B, index=(k, pid_n), shape=(tile_k, tile_n))
    acc = ct.mma(a, b, acc)
```

---

## 7. Debugging with ct.printf

You can print values from inside kernels:

```python
@ct.kernel
def debug_kernel(data, tile_size: ct.Constant[int]):
    pid = ct.bid(0)
    ct.printf("Block %d starting\n", pid)

    tile = ct.load(data, index=(pid,), shape=(tile_size,))
    # Note: printing tile contents directly may not work;
    # use .item() for scalar extraction if supported
```

**Warning**: `ct.printf` serializes output across all blocks. Use it sparingly and only for debugging — it significantly affects performance and can produce interleaved output.

---

## 8. Kernel Compiler Options

The `@ct.kernel` decorator accepts tuning parameters:

```python
@ct.kernel(num_ctas=4, occupancy=8, opt_level=3)
def optimized_kernel(...):
    ...
```

| Parameter | Range | Effect |
|-----------|-------|--------|
| `num_ctas` | 1–16 (power of 2) | CTAs in a CGA (cooperative group) |
| `occupancy` | 1–32 | Expected active CTAs per SM |
| `opt_level` | 0–3 | Compiler optimization level |

For architecture-specific tuning:

```python
from cuda.tile import ByTarget

@ct.kernel(num_ctas=ByTarget(sm_100=8, sm_120=4, default=2))
def arch_tuned_kernel(...):
    ...
```

---

## 9. Common Error Messages

| Error | Cause | Fix |
|-------|-------|-----|
| `TileSyntaxError` | Unsupported Python construct in kernel | Simplify code, avoid classes/generators/etc. |
| `TileTypeError` | Type mismatch (e.g., fp32 → fp16) | Add `ct.astype()` for explicit conversion |
| `TileValueError` | Invalid constant value | Check `ct.Constant` parameters |
| `TileCompilerTimeoutError` | Kernel too complex | Reduce tile sizes or simplify logic |

---

## 10. Reduction Patterns

cuTile supports axis-wise reductions that are essential for algorithms like softmax:

```python
# Row-wise max: (M, N) -> (M, 1)
row_max = ct.max(tile, axis=1, keepdims=True)

# Row-wise sum: (M, N) -> (M, 1)
row_sum = ct.sum(tile, axis=1, keepdims=True)

# Global reduction: (M, N) -> scalar tile
total = ct.sum(tile)
```

`keepdims=True` preserves the reduced dimension as size 1, enabling broadcasting:

```python
# Subtract row-wise max for numerical stability (softmax trick)
shifted = tile - ct.max(tile, axis=1, keepdims=True)
```

---

## 11. Atomic Operations

For parallel reductions or histograms, use atomic operations:

```python
ct.atomic_add(output_array, indices=(row_idx, col_idx), update=value,
              memory_order=ct.MemoryOrder.RELAXED,
              memory_scope=ct.MemoryScope.DEVICE)
```

Available atomics: `atomic_add`, `atomic_max`, `atomic_min`, `atomic_and`, `atomic_or`, `atomic_xor`, `atomic_cas`, `atomic_xchg`.

---

## 12. gather / scatter for Irregular Access

When you need non-contiguous memory access (e.g., permutations, embeddings):

```python
# Gather: load elements at arbitrary indices
indices_tuple = (row_indices, col_indices)  # tuple of int tiles
values = ct.gather(array, indices_tuple, padding_value=0)

# Scatter: store elements at arbitrary indices
ct.scatter(array, indices_tuple, values)
```

Both support optional bounds checking (`check_bounds=True` by default).

---

## 13. Tile Shape Manipulation

```python
# Reshape (total elements must match, -1 for auto)
flat = ct.reshape(tile_2d, (-1,))
reshaped = ct.reshape(flat, (8, 32))

# Add a dimension
expanded = ct.expand_dims(tile_1d, axis=0)  # (N,) -> (1, N)

# Concatenate tiles along an axis
combined = ct.cat([tile_a, tile_b], axis=0)

# Broadcast to a larger shape
broadcasted = ct.broadcast_to(col_vector, (16, 32))
```

---

## 14. Performance Checklist

Before profiling with NSight Compute, check these common issues:

- [ ] **Tile sizes are powers of 2** and appropriate for the problem
- [ ] **All loop bounds and shapes use `ct.Constant`**
- [ ] **Inner loops use `ct.mma`** instead of `@` for matmul accumulation
- [ ] **Memory access is coalesced** (consider `order` parameter)
- [ ] **Reductions use `keepdims=True`** when the result is used in broadcasting
- [ ] **No unnecessary type conversions** in hot loops
- [ ] **Kernel parameters** (`num_ctas`, `occupancy`) are tuned for the target GPU
