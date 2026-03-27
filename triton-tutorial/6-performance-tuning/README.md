# Performance Tuning

This tutorial explores how parameters like **Block Size** affect kernel performance in Triton.

## The Trade-off

Choosing the "right" block size balances hardware constraints:

1. **Parallelism (Occupancy)**
   - Small blocks: more programs, better occupancy, more scheduling overhead.
   - Large blocks: fewer programs, risk of underutilization.

2. **Register Pressure**
   - Larger blocks use more registers and can reduce occupancy.

3. **Memory Coalescing**
   - Larger blocks often improve memory transaction efficiency.

## Autotuning

Since the best size depends on GPU, Triton users often benchmark or use `triton.autotune`.

### In this Example

We benchmark `math_kernel` with sizes `[32, 64, ... 1024]`:
- Small sizes may be slow due to overhead.
- Very large sizes may be slow due to register pressure or low occupancy.

Run:

```bash
python autotune_benchmark.py
```
