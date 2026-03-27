# Triton Execution Model

This tutorial explains how Triton organizes and executes code on the GPU.

## The Grid and The Program

Triton launches **programs** arranged in a 1D, 2D, or 3D **grid**.
Each program typically handles a block (tile) of data.

### 2D Grid Example (`grid_2d.py`)

In this example, we map a 2D grid of programs to a 2D array (like an image).

```python
grid_x = triton.cdiv(width, block_n)
grid_y = triton.cdiv(height, block_m)
grid = (grid_x, grid_y)
```

If our image is 128x128 and our block size is 16x16:
- We need 8 programs in width.
- We need 8 programs in height.
- Total programs = 64.

### Program Coordinates

Inside the kernel, we find where we are in the grid:

```python
pid_x = tl.program_id(0)  # column index
pid_y = tl.program_id(1)  # row index
```

### Mapping to Data

When working with 2D data, we map:
- `pid_x` -> columns (width)
- `pid_y` -> rows (height)

In row-major tensors, `(row, col)` corresponds to `(y, x)`:

```python
ptrs = output + offs_y * stride_y + offs_x * stride_x
```

## Key Takeaways

1. **Programs** are Triton's execution unit (similar to CUDA blocks).
2. **`tl.program_id(axis)`** gives the coordinate in the launch grid.
3. **`tl.arange` + masking** defines which elements each program handles.
