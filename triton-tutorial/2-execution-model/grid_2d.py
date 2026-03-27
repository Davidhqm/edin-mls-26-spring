"""
Example demonstrating the Triton execution model with 2D grids.
Shows how to use 2D program IDs for grid mapping.
"""

import torch
import triton
import triton.language as tl


@triton.jit
def grid_map_2d(output, H, W, stride_y, stride_x, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr):
    # pid(0) -> x dimension (columns), pid(1) -> y dimension (rows)
    pid_x = tl.program_id(0)
    pid_y = tl.program_id(1)

    offs_x = pid_x * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_y = pid_y * BLOCK_M + tl.arange(0, BLOCK_M)

    offs_y = offs_y[:, None]
    offs_x = offs_x[None, :]

    mask = (offs_y < H) & (offs_x < W)
    val = pid_x * 1000 + pid_y

    ptrs = output + offs_y * stride_y + offs_x * stride_x
    tl.store(ptrs, val, mask=mask)


def test_grid_map_2d():
    height = 128
    width = 128
    block_m = 16
    block_n = 16

    grid_x = triton.cdiv(width, block_n)
    grid_y = triton.cdiv(height, block_m)
    grid = (grid_x, grid_y)

    print(f"Array Size: {height} x {width}")
    print(f"Block Size: {block_m} x {block_n}")
    print(f"Grid Layout: {grid_x} x {grid_y} programs")
    print(f"Total Programs: {grid_x * grid_y}")

    output = torch.zeros((height, width), device="cuda", dtype=torch.int32)
    stride_y, stride_x = output.stride()

    grid_map_2d[grid](output, height, width, stride_y, stride_x, BLOCK_M=block_m, BLOCK_N=block_n)

    out_host = output.cpu().numpy()

    print("\nVerifying output...")
    val_0_0 = out_host[0, 0]
    expected_0_0 = 0

    val_7_7 = out_host[112, 112]
    expected_7_7 = 7007

    print(f"Tile(0,0) value: {val_0_0} (Expected: {expected_0_0})")
    print(f"Tile(7,7) value: {val_7_7} (Expected: {expected_7_7})")

    if val_0_0 == expected_0_0 and val_7_7 == expected_7_7:
        print("\n[PASS] 2D Grid Mapping Test Passed!")
    else:
        print("\n[FAIL] 2D Grid Mapping Test Failed!")


if __name__ == "__main__":
    test_grid_map_2d()
