# SPDX-FileCopyrightText: Copyright (c) <2025> NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""
Example demonstrating the cuTile Execution Model with 1D grids.
Shows how to use block IDs (bid) with the sigmoid function.

Sigmoid: σ(x) = 1 / (1 + exp(-x))
"""

import cupy as cp
import numpy as np
import cuda.tile as ct


@ct.kernel
def sigmoid_kernel(input, output, tile_size: ct.Constant[int]):
    # Get the block ID (tile index in the 1D grid)
    # bid(0) returns the index of the current tile being processed
    pid = ct.bid(0)

    # Load a tile of data from global memory
    # index=(pid,) specifies which tile to load (tile-based indexing)
    # shape=(tile_size,) specifies the shape of the tile
    x_tile = ct.load(input, index=(pid,), shape=(tile_size,))

    # Compute sigmoid: σ(x) = 1 / (1 + exp(-x))
    # cuTile provides ct.exp() for element-wise exponential

    exp_neg_x = ct.exp(-x_tile)
    sigmoid_tile = 1.0 / (1.0 + exp_neg_x)

    # Store the result back to global memory
    ct.store(output, index=(pid,), tile=sigmoid_tile)


def test_sigmoid_1d():
    # Vector size
    N = 1024

    # Size of each tile (number of elements processed per block)
    TILE_SIZE = 32

    # Calculate grid dimension (number of tiles needed)
    num_tiles = ct.cdiv(N, TILE_SIZE)
    grid = (num_tiles, 1, 1)

    print(f"Vector Size: {N}")
    print(f"Tile Size: {TILE_SIZE}")
    print(f"Number of Tiles: {num_tiles}")

    # Create input data: values from -5 to 5
    data_in = cp.linspace(-5, 5, N, dtype=cp.float32)
    data_out = cp.zeros_like(data_in)

    # Launch kernel
    ct.launch(cp.cuda.get_current_stream(),
              grid,
              sigmoid_kernel,
              (data_in, data_out, TILE_SIZE))

    # Verify against NumPy reference
    h_in = cp.asnumpy(data_in)
    h_out = cp.asnumpy(data_out)

    # NumPy sigmoid for reference
    expected = 1.0 / (1.0 + np.exp(-h_in))

    print("\nVerifying output...")
    print(f"Input range: [{h_in.min():.2f}, {h_in.max():.2f}]")
    print(f"Output range: [{h_out.min():.4f}, {h_out.max():.4f}]")

    # Check specific values
    print(f"\nSample values:")
    print(f"  sigmoid(-5.0) = {h_out[0]:.6f} (expected: {expected[0]:.6f})")
    print(f"  sigmoid(0.0)  = {h_out[N//2]:.6f} (expected: {expected[N//2]:.6f})")
    print(f"  sigmoid(5.0)  = {h_out[-1]:.6f} (expected: {expected[-1]:.6f})")

    if np.allclose(h_out, expected, rtol=1e-5, atol=1e-5):
        print("\n✓ 1D Sigmoid Test Passed!")
    else:
        print("\n✗ 1D Sigmoid Test Failed!")
        max_diff = np.max(np.abs(h_out - expected))
        print(f"Max difference: {max_diff}")


if __name__ == "__main__":
    test_sigmoid_1d()
