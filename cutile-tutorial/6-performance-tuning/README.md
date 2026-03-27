# Performance Tuning

This tutorial explores how parameters like **Tile Size** affect kernel performance.

## The Trade-off

Choosing the "right" tile size is an art that balances several hardware constraints:

1.  **Parallelism (Occupancy)**:
    -   **Small Tiles**: Create many blocks. Good for filling the GPU if the problem is large. However, too small means higher scheduling overhead.
    -   **Large Tiles**: Fewer blocks. If there are fewer blocks than GPU Streaming Multiprocessors (SMs), the GPU will be underutilized.

2.  **Register Pressure**:
    -   Each tile consumes registers.
    -   **Large Tiles** typically require more registers per block. If a block needs too many registers, the GPU cannot run many blocks simultaneously on the same SM, reducing parallelism (Occupancy).

3.  **Memory Coalescing**:
    -   Larger tiles often allow for wider, more efficient memory transactions (e.g., loading 128 bytes at once).

## Autotuning

Since it's hard to predict the perfect size mathematically (due to complex compiler optimizations and hardware specifics), the industry standard is **Autotuning**:
- Write the kernel with a generic `tile_size`.
- Run a benchmark script (like `autotune_benchmark.py`).
- Pick the winner.

### In this Example

We benchmark `math_kernel` with sizes `[32, 64, ... 1024]`.
- You will likely see a "Sweet Spot".
- Very small sizes (32) might be slow due to overhead.
- Very large sizes (1024) might be slow due to register pressure or lack of wave parallelism.
