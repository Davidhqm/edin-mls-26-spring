# Matrix Transpose with cuTile

This example demonstrates how to perform a 2D matrix transpose using cuTile's tiled programming model.

## The Algorithm

A matrix transpose swaps rows and columns: `output[j][i] = input[i][j]`.

With tiled computation, this becomes:

1. **Load** a tile at grid position `(row=pid_y, col=pid_x)` from the input
2. **Transpose** the tile's contents with `ct.transpose(tile)`
3. **Store** at the **swapped** grid position `(row=pid_x, col=pid_y)` in the output

## Code Walkthrough

### Grid Setup

For a matrix of shape `(height, width)` with tile size `(tile_h, tile_w)`:

```python
grid_x = ct.cdiv(width, tile_w)    # tiles across columns
grid_y = ct.cdiv(height, tile_h)   # tiles across rows
grid = (grid_x, grid_y, 1)
```

### The Kernel

```python
@ct.kernel
def transpose_kernel(input_arr, output_arr,
                     tile_w: ct.Constant[int],
                     tile_h: ct.Constant[int]):
    pid_x = ct.bid(0)  # column tile index
    pid_y = ct.bid(1)  # row tile index

    # Load from input (height, width) at (row, col)
    tile = ct.load(input_arr, index=(pid_y, pid_x), shape=(tile_h, tile_w))

    # Transpose tile contents: (tile_h, tile_w) -> (tile_w, tile_h)
    tile_T = ct.transpose(tile)

    # Store to output (width, height) at swapped position
    ct.store(output_arr, index=(pid_x, pid_y), tile=tile_T)
```

### Key Points

- **`ct.transpose(tile)`** swaps the last two dimensions of a tile. For a 2D tile `(H, W)`, the result is `(W, H)`.
- **Coordinate Swapping**: The load index is `(pid_y, pid_x)` but the store index is `(pid_x, pid_y)`. This is what moves data from `input[i][j]` to `output[j][i]`.
- **Non-square Matrices**: The output shape must be `(width, height)` — the transposed dimensions.

## Files

| File | Description |
|------|-------------|
| `transpose_2d.py` | Matrix transpose with square and rectangular tests |
| `grid_2d.py` | Alternate implementation |

## How to Run

```bash
python transpose_2d.py
```
