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
def grid_map_2d(output, tile_size_x: ct.Constant[int], tile_size_y: ct.Constant[int]):
    # Get the 2D block IDs (tile coordinates)
    # bid(0) corresponds to the x-dimension of the grid
    # bid(1) corresponds to the y-dimension of the grid
    pid_x = ct.bid(0)
    pid_y = ct.bid(1)

    # We want to fill the output tile with a value that represents its coordinate.
    # For visualization, let's store: pid_x * 1000 + pid_y
    # This makes it easy to read: 2003 means x=2, y=3
    val = pid_x * 1000 + pid_y

    # Create a tile filled with 'val' using ct.full and store it.
    # ct.full creates a tile of the specified shape filled with the given value.
    # Note: numpy/cupy use (row, col) which is (y, x)
    val_tile = ct.full((tile_size_y, tile_size_x), val, ct.int32)

    # Store to global memory
    # index=(pid_y, pid_x) matches the (row, col) layout
    ct.store(output, index=(pid_y, pid_x), tile=val_tile)


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

    # Allocate output memory on device
    # Note: shape is (height, width) i.e., (y, x)
    output = cp.zeros((height, width), dtype=cp.int32)

    # Launch kernel
    ct.launch(cp.cuda.get_current_stream(),
              grid,
              grid_map_2d,
              (output, tile_w, tile_h))

    # Verify
    out_host = cp.asnumpy(output)

    print("\nVerifying output...")
    # Check a few spots
    # Top-left tile (0,0) -> Expect 0
    # Bottom-right tile (7, 7) -> Expect 7007

    val_0_0 = out_host[0, 0]
    expected_0_0 = 0

    # Tile (7,7) starts at y=7*16=112, x=7*16=112
    val_7_7 = out_host[112, 112]
    expected_7_7 = 7007

    print(f"Tile(0,0) value: {val_0_0} (Expected: {expected_0_0})")
    print(f"Tile(7,7) value: {val_7_7} (Expected: {expected_7_7})")

    if val_0_0 == expected_0_0 and val_7_7 == expected_7_7:
        print("\n✓ 2D Grid Mapping Test Passed!")
    else:
        print("\n✗ 2D Grid Mapping Test Failed!")


if __name__ == "__main__":
    test_grid_map_2d()
