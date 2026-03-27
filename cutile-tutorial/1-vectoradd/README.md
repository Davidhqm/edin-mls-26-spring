# Vector Addition with cuTile

This example demonstrates the "Hello World" of GPU programming using `cuTile`: adding two vectors together.

## Code Walkthrough

The code performs an element-wise addition of two arrays, `a` and `b`, storing the result in `c`.

### 1. Imports

```python
import cupy as cp
import numpy as np
import cuda.tile as ct
```

- **`cupy`**: An open-source matrix library accelerated with NVIDIA CUDA. We use it to allocate memory on the GPU.
- **`numpy`**: The fundamental package for scientific computing in Python. We use it here for verification on the CPU.
- **`cuda.tile`**: The `cuTile` library, imported as `ct`. This is the core library we are learning.

### 2. The Kernel

The kernel is the function that runs on the GPU.

```python
@ct.kernel
def vector_add(a, b, c, tile_size: ct.Constant[int]):
```

- **`@ct.kernel`**: This decorator tells `cuTile` that this function is a GPU kernel.
- **Arguments**:
    - `a`, `b`, `c`: Input/Output pointers (arrays).
    - `tile_size`: A constant integer indicating how many elements each thread block (or "tile") processes. Using `ct.Constant` allows the compiler to optimize loop unrolling and register usage.

```python
    # Get the 1D pid
    pid = ct.bid(0)
```

- **`ct.bid(0)`**: Gets the "Block ID" in the 0-th dimension. In `cuTile`, we often think of this as the "Program ID" or "Tile ID". If we launch a grid of 100 tiles, this will range from 0 to 99. Each ID is unique to a chunk of work.

```python
    # Load input tiles
    a_tile = ct.load(a, index=(pid,), shape=(tile_size,))
    b_tile = ct.load(b, index=(pid,), shape=(tile_size,))
```

- **`ct.load`**: Moves data from Global Memory (slow, large) to Tile Memory (fast registers/shared memory).
- **`index=(pid,)`**: Tells `cuTile` *where* in the global array to start reading. Here, we interpret the global array as a list of tiles. So `pid` 0 reads the 0th tile, `pid` 1 reads the 1st tile, etc.
- **`shape=(tile_size,)`**: Specifies how much data to load.

```python
    # Perform elementwise addition
    result = a_tile + b_tile
```

- This line looks like standard Python addition, but it happens in parallel on the GPU registers. `a_tile` and `b_tile` are special `cuTile` objects representing the loaded data.

```python
    # Store result
    ct.store(c, index=(pid, ), tile=result)
```

- **`ct.store`**: Moves the computed `result` from Tile Memory back to Global Memory (the `c` array).
- **`tile=result`**: The data to write.

### 3. The Host Code (Main Function)

This code runs on the CPU and orchestrates the GPU work.

```python
    vector_size = 2**12  # 4096 elements
    tile_size = 2**4     # 16 elements per tile
```

- We define the total problem size and the chunk size.

```python
    grid = (ct.cdiv(vector_size, tile_size), 1, 1)
```

- **Grid Calculation**: We need enough tiles to cover the entire vector.
- **`ct.cdiv`**: Ceiling division. $4096 / 16 = 256$. So we launch 256 tiles.

```python
    a = cp.random.uniform(-1, 1, vector_size)
    b = cp.random.uniform(-1, 1, vector_size)
    c = cp.zeros_like(a)
```

- **Data Preparation**: We create random arrays directly on the GPU using CuPy. `c` is allocated to hold the result.

```python
    # Launch kernel
    ct.launch(cp.cuda.get_current_stream(),
              grid,
              vector_add,
              (a, b, c, tile_size))
```

- **`ct.launch`**: Executes the kernel.
    - **Stream**: Uses the current CuPy stream to ensure synchronization.
    - **Grid**: The dimensions of our launch (256, 1, 1).
    - **Kernel**: The function `vector_add`.
    - **Args**: The arguments passed to `vector_add`.

### 4. Verification

```python
    # Copy to host only to compare
    a_np = cp.asnumpy(a)
    b_np = cp.asnumpy(b)
    c_np = cp.asnumpy(c)

    # Verify results
    expected = a_np + b_np
    np.testing.assert_array_almost_equal(c_np, expected)
```

- We copy the data back to the CPU (`cp.asnumpy`) to check correctness using NumPy. This is crucial for debugging but skipped in production for performance.

## How to Run

```bash
python vectoradd.py
```
