# SPDX-FileCopyrightText: Copyright (c) <2025> NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""
Example demonstrating Performance Tuning.
Benchmarks the same kernel with different Tile Sizes to find the optimal configuration.
"""

import cupy as cp
import cuda.tile as ct
import time

# A kernel with some arithmetic intensity
# Calculates: out = in * in + in - in / 2 ...
@ct.kernel
def math_kernel(data, out, tile_size: ct.Constant[int]):
    pid = ct.bid(0)
    
    # Load
    r = ct.load(data, index=(pid,), shape=(tile_size,))
    
    # Burn some cycles with math operations
    # (Simulating complex logic)
    res = r * r
    res = res + r
    res = res * 0.5
    res = res * res
    
    # Store
    ct.store(out, index=(pid,), tile=res)

def benchmark_tile_size(tile_size, N, n_warmup=10, n_iter=100):
    # Prepare Data
    data = cp.random.uniform(0, 1, N).astype(cp.float32)
    out = cp.zeros_like(data)
    
    grid = (ct.cdiv(N, tile_size), 1, 1)
    
    # Warmup
    for _ in range(n_warmup):
        ct.launch(cp.cuda.get_current_stream(),
                  grid, math_kernel, (data, out, tile_size))
    
    # Synchronize before timing
    cp.cuda.Device().synchronize()
    
    # Timing
    start = cp.cuda.Event()
    end = cp.cuda.Event()
    
    start.record()
    for _ in range(n_iter):
        ct.launch(cp.cuda.get_current_stream(),
                  grid, math_kernel, (data, out, tile_size))
    end.record()
    end.synchronize()
    
    total_time_ms = cp.cuda.get_elapsed_time(start, end)
    avg_time_ms = total_time_ms / n_iter
    
    return avg_time_ms

def main():
    print("== Tile Size Autotuning Benchmark ==")
    
    N = 1024 * 1024 * 32 # 32M elements (~128MB)
    print(f"Vector Size: {N:,} elements")
    
    candidate_sizes = [32, 64, 128, 256, 512, 1024]
    
    best_time = float('inf')
    best_size = -1
    
    results = []
    
    for size in candidate_sizes:
        print(f"Benchmarking Tile Size: {size}...", end="", flush=True)
        try:
            t_ms = benchmark_tile_size(size, N)
            print(f" {t_ms:.4f} ms")
            results.append((size, t_ms))
            
            if t_ms < best_time:
                best_time = t_ms
                best_size = size
        except Exception as e:
            print(f" Failed! ({e})")
            results.append((size, float('inf')))

    print("\n== Results ==")
    print(f"{'Tile Size':<15} | {'Time (ms)':<15} | {'Speedup':<15}")
    print("-" * 50)
    
    base_time = results[0][1]
    for size, t in results:
        speedup = base_time / t if t > 0 else 0
        print(f"{size:<15} | {t:<15.4f} | {speedup:<15.2f}x")

    print(f"\nBest Configuration: Tile Size = {best_size}")

if __name__ == "__main__":
    main()
