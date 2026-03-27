# cuTile Data Model

This tutorial covers how `cuTile` handles data types, constants, and mixed precision.

## 1. Constants (`ct.Constant`)

In the function signature:
```python
def mixed_precision_scale(..., tile_size: ct.Constant[int]):
```
You notice `tile_size` is annotated with `ct.Constant[int]`. 

**Why?**
- GPU compilers need to know loop bounds and array sizes at *compile time* to perform optimizations like loop unrolling and register allocation.
- By marking an argument as `ct.Constant`, you tell `cuTile`: "I promise this value will be known when I call `ct.launch` and it won't change."
- This allows `cuTile` to specialize the kernel for that specific size (e.g., 32).

## 2. Mixed Precision (FP16 / FP32)

Deep Learning and high-performance computing often use 16-bit floating point (Half Precision) to save memory and increase speed.

### Loading
```python
    # input_ptr is float16
    input_tile = ct.load(input_ptr, ...) 
```
When you load from a `float16` CuPy array, `cuTile` loads it into registers.

### Computation (Promotion)
```python
    # input_tile is FP16, scale_factor is FP32
    result_tile = input_tile * scale_factor
```
Just like in NumPy, operating on mixed types usually promotes to the higher precision. Here, the multiplication happens in FP32 (accumulating in higher precision registers). This is critical for accuracy.

### Storing (Casting)
```python
    # output_ptr is float16
    ct.store(output_ptr, ..., tile=result_tile)
```
When storing an FP32 tile into an FP16 array, `cuTile` automatically handles the down-casting.

## 3. Supported Types

`cuTile` generally supports standard NumPy/CuPy types:
- `int8`, `int32`, `int64`
- `float16`, `float32`, `float64`
- `bool`
