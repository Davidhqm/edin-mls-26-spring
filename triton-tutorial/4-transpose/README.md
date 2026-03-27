# Matrix Transpose with Triton

This example demonstrates how to perform a 2D matrix transpose using Triton's tiled programming model.

## The Algorithm

A matrix transpose swaps rows and columns: `output[j][i] = input[i][j]`.

With tiled computation, this becomes:

1. **Load** a tile at grid position `(row=pid_y, col=pid_x)` from the input
2. **Transpose** the tile's contents with `tl.trans(tile)`
3. **Store** at the **swapped** grid position `(row=pid_x, col=pid_y)` in the output

## Code Walkthrough

### Grid Setup

For a matrix of shape `(height, width)` with block size `(BLOCK_M, BLOCK_N)`:

```python
grid_x = triton.cdiv(width, BLOCK_N)    # tiles across columns
grid_y = triton.cdiv(height, BLOCK_M)   # tiles across rows
grid = (grid_x, grid_y)
```

### The Kernel

```python
@triton.jit
def transpose_kernel(...):
    pid_x = tl.program_id(0)  # column tile index
    pid_y = tl.program_id(1)  # row tile index

    # Load a tile
    x = tl.load(...)

    # Transpose tile contents
    x_T = tl.trans(x)

    # Store to output at swapped coordinates
    tl.store(...)
```

### Key Points

- **`tl.trans(tile)`** swaps the last two dimensions of a tile.
- **Coordinate Swapping**: The load index is `(pid_y, pid_x)` but the store index is `(pid_x, pid_y)`.
- **Non-square Matrices**: The output shape must be `(width, height)`.

## Files

| File | Description |
|------|-------------|
| `grid_2d.py` | Matrix transpose with square tests |

## How to Run

```bash
python grid_2d.py
```
