# Vector Addition with Triton

This example demonstrates the "Hello World" of GPU programming using Triton: adding two vectors together.

## Code Walkthrough

The code performs an element-wise addition of two arrays, `a` and `b`, storing the result in `c`.

### 1. Imports

```python
import torch
import triton
import triton.language as tl
```

- **`torch`**: Used for GPU memory allocation and verification.
- **`triton`**: Kernel launcher and runtime.
- **`triton.language` (tl)**: Triton's DSL for writing GPU kernels.

### 2. The Kernel

```python
@triton.jit
def vector_add_kernel(X, Y, Z, N, BLOCK: tl.constexpr):
```

- **`@triton.jit`**: Tells Triton to compile this function into a GPU kernel.
- **Arguments**:
    - `X`, `Y`, `Z`: Pointers to input/output arrays.
    - `N`: Total number of elements.
    - `BLOCK`: A compile-time constant controlling how many elements each program handles.

```python
pid = tl.program_id(0)
offs = pid * BLOCK + tl.arange(0, BLOCK)
mask = offs < N
```

- **`tl.program_id(0)`**: Unique program index in the 1D launch grid.
- **`tl.arange(0, BLOCK)`**: Vector of offsets inside this program.
- **`mask`**: Prevents out-of-bounds accesses at the tail.

```python
x = tl.load(X + offs, mask=mask, other=0.0)
y = tl.load(Y + offs, mask=mask, other=0.0)
tl.store(Z + offs, x + y, mask=mask)
```

- **`tl.load`** / **`tl.store`**: Read and write global memory.
- Masking ensures safety when `N` is not a multiple of `BLOCK`.

### 3. The Host Code

```python
vector_size = 2**12
block_size = 256
grid = (triton.cdiv(vector_size, block_size),)
```

- **Grid**: We launch enough programs to cover the vector.
- **`triton.cdiv`**: Ceiling division.

### 4. Verification

We compare GPU results with a CPU reference using `torch.testing.assert_close`.

## How to Run

```bash
python vectoradd.py
```
