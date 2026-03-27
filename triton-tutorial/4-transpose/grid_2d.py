"""
Example demonstrating a tiled matrix transpose with Triton.
"""

import torch
import triton
import triton.language as tl


@triton.jit
def transpose_kernel(X, Y, H, W, stride_xm, stride_xn, stride_ym, stride_yn,
                     BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr):
    pid_x = tl.program_id(0)  # column tile index
    pid_y = tl.program_id(1)  # row tile index

    offs_m = pid_y * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_x * BLOCK_N + tl.arange(0, BLOCK_N)

    offs_m_2d = offs_m[:, None]
    offs_n_2d = offs_n[None, :]

    x_ptrs = X + offs_m_2d * stride_xm + offs_n_2d * stride_xn
    mask = (offs_m_2d < H) & (offs_n_2d < W)
    x = tl.load(x_ptrs, mask=mask, other=0)

    x_T = tl.trans(x)

    # Store to output at swapped coordinates (n, m)
    offs_n_T = offs_n[:, None]
    offs_m_T = offs_m[None, :]
    y_ptrs = Y + offs_n_T * stride_ym + offs_m_T * stride_yn
    mask_T = (offs_n_T < W) & (offs_m_T < H)
    tl.store(y_ptrs, x_T, mask=mask_T)


def test_transpose_2d():
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

    x = torch.randint(0, 100, (height, width), device="cuda", dtype=torch.int32)
    y = torch.zeros((width, height), device="cuda", dtype=torch.int32)

    stride_xm, stride_xn = x.stride()
    stride_ym, stride_yn = y.stride()

    transpose_kernel[grid](x, y, height, width, stride_xm, stride_xn, stride_ym, stride_yn,
                           BLOCK_M=block_m, BLOCK_N=block_n)

    x_host = x.cpu().numpy()
    y_host = y.cpu().numpy()
    expected = x_host.T

    print("\nVerifying output...")
    if (y_host == expected).all():
        print("\n[PASS] 2D Transpose Test Passed!")
    else:
        diff_count = (y_host != expected).sum()
        print(f"\n[FAIL] 2D Transpose Test Failed! ({diff_count} elements differ)")
        print(f"Sample - Input[0,1]: {x_host[0,1]}, Output[1,0]: {y_host[1,0]}, Expected: {expected[1,0]}")


if __name__ == "__main__":
    test_transpose_2d()
