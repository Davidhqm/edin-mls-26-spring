# SPDX-FileCopyrightText: Copyright (c) <2025> NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""
Example demonstrating the cuTile Execution Model with 2D grids.
Shows how to use 2D block IDs (bid) for grid mapping.
"""

import cupy as cp
import numpy as np
import cuda.tile as ct


@ct.kernel
def transpose_cutile(input, output, tile_size_x: ct.Constant[int], tile_size_y: ct.Constant[int]):
    pid_x = ct.bid(0)
    pid_y = ct.bid(1)

    input_slice = ct.load(input, index=(pid_y, pid_x), shape=(tile_size_y, tile_size_x))
    input_slice_transposed = ct.transpose(input_slice)

    ct.store(output, index=(pid_x, pid_y), tile=input_slice_transposed)


def test_grid_map_2d():
    # Dimensions of our "image" or 2D array
    height = 128
    width = 128

    # Size of each tile
    tile_h = 16
    tile_w = 16

    # Calculate grid dimensions (number of tiles in each direction)
    grid_x = ct.cdiv(width, tile_w)
    grid_y = ct.cdiv(height, tile_h)

    grid = (grid_x, grid_y, 1)

    print(f"Array Size: {height} x {width}")
    print(f"Tile Size: {tile_h} x {tile_w}")
    print(f"Grid Layout: {grid_x} x {grid_y} tiles")
    print(f"Total Tiles: {grid_x * grid_y}")

    # Allocate memory on device
    # Input shape: (height, width), Output shape: (width, height) for transpose
    input_arr = cp.random.randint(0, 100, (height, width), dtype=cp.int32)
    output = cp.zeros((width, height), dtype=cp.int32)

    # Launch kernel
    ct.launch(cp.cuda.get_current_stream(),
              grid,
              transpose_cutile,
              (input_arr, output, tile_w, tile_h))

    # Verify against numpy transpose
    input_host = cp.asnumpy(input_arr)
    out_host = cp.asnumpy(output)
    expected = input_host.T

    print("\nVerifying output...")
    print(f"Input shape: {input_host.shape}")
    print(f"Output shape: {out_host.shape}")
    print(f"Expected shape: {expected.shape}")

    # Check if the entire array matches
    if np.array_equal(out_host, expected):
        print("\n✓ 2D Transpose Test Passed!")
    else:
        # Show some differences for debugging
        diff_count = np.sum(out_host != expected)
        print(f"\n✗ 2D Transpose Test Failed! ({diff_count} elements differ)")
        print(f"Sample - Input[0,1]: {input_host[0,1]}, Output[1,0]: {out_host[1,0]}, Expected: {expected[1,0]}")


if __name__ == "__main__":
    test_grid_map_2d()
