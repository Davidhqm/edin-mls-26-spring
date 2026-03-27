"""
Example demonstrating a simplified tiled attention mechanism.
Out = exp(Q @ K.T / sqrt(d)) @ V
"""

import math
import torch
import triton
import triton.language as tl


@triton.jit
def simple_attention(Q, K, V, Out, M,
                     stride_qm, stride_qd,
                     stride_km, stride_kd,
                     stride_vm, stride_vd,
                     stride_om, stride_od,
                     SCALE,
                     SEQ_LEN_K: tl.constexpr,
                     HEAD_DIM: tl.constexpr,
                     BLOCK_M: tl.constexpr,
                     BLOCK_N: tl.constexpr):
    pid_m = tl.program_id(0)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, HEAD_DIM)

    q_ptrs = Q + offs_m[:, None] * stride_qm + offs_d[None, :] * stride_qd
    q = tl.load(q_ptrs, mask=(offs_m[:, None] < M) & (offs_d[None, :] < HEAD_DIM), other=0.0)

    acc = tl.zeros((BLOCK_M, HEAD_DIM), dtype=tl.float32)

    for k_start in range(0, SEQ_LEN_K, BLOCK_N):
        offs_n = k_start + tl.arange(0, BLOCK_N)

        k_ptrs = K + offs_n[:, None] * stride_km + offs_d[None, :] * stride_kd
        v_ptrs = V + offs_n[:, None] * stride_vm + offs_d[None, :] * stride_vd

        k = tl.load(k_ptrs, mask=(offs_n[:, None] < SEQ_LEN_K) & (offs_d[None, :] < HEAD_DIM), other=0.0)
        v = tl.load(v_ptrs, mask=(offs_n[:, None] < SEQ_LEN_K) & (offs_d[None, :] < HEAD_DIM), other=0.0)

        scores = tl.dot(q, tl.trans(k), input_precision="ieee")
        scores = scores * SCALE
        scores = tl.exp(scores)

        acc += tl.dot(scores, v, input_precision="ieee")

    out_ptrs = Out + offs_m[:, None] * stride_om + offs_d[None, :] * stride_od
    mask_out = (offs_m[:, None] < M) & (offs_d[None, :] < HEAD_DIM)
    tl.store(out_ptrs, acc, mask=mask_out)


def test_attention():
    M = 128  # Number of Queries
    N = 128  # Number of Keys/Values
    D = 64   # Head Dimension

    BLOCK_M = 32
    BLOCK_N = 32

    print(f"Attention Problem: Q({M}x{D}) @ K({N}x{D}).T @ V({N}x{D})")

    q = torch.randn((M, D), device="cuda", dtype=torch.float32)
    k = torch.randn((N, D), device="cuda", dtype=torch.float32)
    v = torch.randn((N, D), device="cuda", dtype=torch.float32)
    out = torch.zeros((M, D), device="cuda", dtype=torch.float32)

    stride_qm, stride_qd = q.stride()
    stride_km, stride_kd = k.stride()
    stride_vm, stride_vd = v.stride()
    stride_om, stride_od = out.stride()

    scale = 1.0 / math.sqrt(D)

    grid = (triton.cdiv(M, BLOCK_M),)

    simple_attention[grid](
        q, k, v, out, M,
        stride_qm, stride_qd,
        stride_km, stride_kd,
        stride_vm, stride_vd,
        stride_om, stride_od,
        scale,
        SEQ_LEN_K=N,
        HEAD_DIM=D,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        num_warps=4,
        num_stages=2,
    )

    # Reference on CPU
    q_cpu = q.cpu()
    k_cpu = k.cpu()
    v_cpu = v.cpu()
    expected = torch.exp((q_cpu @ k_cpu.T) * scale) @ v_cpu

    out_cpu = out.cpu()

    print("Checking accuracy...")
    torch.testing.assert_close(out_cpu, expected, rtol=1e-3, atol=1e-3)
    print("[PASS] Tiled Attention Passed!")


if __name__ == "__main__":
    test_attention()
