# cuTile Execution Model

This tutorial explains how `cuTile` organizes and executes code on the GPU.

## The Grid and The Tile

Unlike traditional CUDA which speaks in terms of "Grids", "Blocks", and "Threads", `cuTile` raises the abstraction level.

1.  **The Grid**: The entire problem space. It's a 1D, 2D, or 3D arrangement of work units.
2.  **The Tile**: The fundamental unit of computation. Instead of writing code for a single scalar thread, you write code for a *tile* of data.

### 2D Grid Example (`execution_2d.py`)

In this example, we map a 2D grid of tiles to a 2D array (like an image).

```python
    # Calculate grid dimensions
    grid_x = ct.cdiv(width, tile_w)
    grid_y = ct.cdiv(height, tile_h)
    
    grid = (grid_x, grid_y, 1)
```

If our image is 128x128 and our tile size is 16x16:
- We need $128/16 = 8$ tiles in width.
- We need $128/16 = 8$ tiles in height.
- Total tiles = $8 \times 8 = 64$.
- These 64 tiles run in parallel on the GPU.

### Kernel Coordinates

Inside the kernel, we need to know "which tile am I?".

```python
    pid_x = ct.bid(0)  # Column index of the tile
    pid_y = ct.bid(1)  # Row index of the tile
```

- `ct.bid(0)` gives the index in the first dimension of the `grid` tuple.
- `ct.bid(1)` gives the index in the second dimension.

### Mapping to Data

When working with 2D data (matrices, images), it is standard to map:
- `grid_x` (dimension 0) $\to$ columns (width)
- `grid_y` (dimension 1) $\to$ rows (height)

In `numpy`/`cupy` arrays, the shape is `(rows, cols)` or `(y, x)`.
So when addressing memory:
```python
    # index=(row_idx, col_idx)
    ct.store(output, index=(pid_y, pid_x), ...)
```

## Key Takeaways

1.  **SIMT vs SIMD**: Traditional CUDA is SIMT (Single Instruction, Multiple Threads). `cuTile` feels more like SIMD (Single Instruction, Multiple Data) where the "Data" is a Tile.
2.  **Coordinates**: Use `ct.bid(n)` to get your coordinate in the grid.
3.  **Parallelism**: All tiles in the grid are independent and can run in parallel.

```