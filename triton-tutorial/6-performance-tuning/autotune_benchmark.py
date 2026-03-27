"""
Example demonstrating Performance Tuning.
Benchmarks the same kernel with different block sizes to find the optimal configuration.
"""

import torch
import triton
import triton.language as tl


@triton.jit
def math_kernel(X, Y, N, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < N
    r = tl.load(X + offs, mask=mask, other=0.0)

    res = r * r
    res = res + r
    res = res * 0.5
    res = res * res

    tl.store(Y + offs, res, mask=mask)


def benchmark_block_size(block_size, N, n_warmup=10, n_iter=100):
    x = torch.rand(N, device="cuda", dtype=torch.float32)
    y = torch.empty_like(x)

    grid = (triton.cdiv(N, block_size),)

    for _ in range(n_warmup):
        math_kernel[grid](x, y, N, BLOCK=block_size, num_warps=4)

    torch.cuda.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    start.record()
    for _ in range(n_iter):
        math_kernel[grid](x, y, N, BLOCK=block_size, num_warps=4)
    end.record()
    end.synchronize()

    total_time_ms = start.elapsed_time(end)
    avg_time_ms = total_time_ms / n_iter
    return avg_time_ms


def main():
    print("== Block Size Autotuning Benchmark ==")

    N = 1024 * 1024 * 32  # 32M elements (~128MB)
    print(f"Vector Size: {N:,} elements")

    candidate_sizes = [32, 64, 128, 256, 512, 1024]

    best_time = float("inf")
    best_size = -1
    results = []

    for size in candidate_sizes:
        print(f"Benchmarking Block Size: {size}...", end="", flush=True)
        try:
            t_ms = benchmark_block_size(size, N)
            print(f" {t_ms:.4f} ms")
            results.append((size, t_ms))

            if t_ms < best_time:
                best_time = t_ms
                best_size = size
        except Exception as e:
            print(f" Failed! ({e})")
            results.append((size, float("inf")))

    print("\n== Results ==")
    print(f"{'Block Size':<15} | {'Time (ms)':<15} | {'Speedup':<15}")
    print("-" * 50)

    base_time = results[0][1]
    for size, t in results:
        speedup = base_time / t if t > 0 else 0
        print(f"{size:<15} | {t:<15.4f} | {speedup:<15.2f}x")

    print(f"\nBest Configuration: Block Size = {best_size}")


if __name__ == "__main__":
    main()
