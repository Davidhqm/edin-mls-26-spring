# cuTile Tutorial

NVIDIA cuTile Python GPU programming tutorials.

## Environment Setup

### 1. Install Environment

```bash
# Run from project root
bash utils/setup-cutile.sh
```

This creates a conda environment named `mls` with:
- CUDA Toolkit (nvidia::cuda)
- CuPy, cuda-python, cuda-tile
- HuggingFace (transformers, datasets)
- PyTorch, Streamlit, etc.

### 2. Activate Environment

```bash
conda activate mls
source utils/setup-cutile.sh  # Run from project root
```

If `conda` isn't on PATH in a new shell (`<conda>` is the conda install prefix, e.g. `~/miniconda3` or `/opt/conda`):

```bash
source <conda>/bin/activate
conda activate mls
```

## Running Tutorials

### Blackwell GPU (RTX 50 series, B100, B200)

Run directly:

```bash
python cutile-tutorial/1-vectoradd/vectoradd.py
```

### Non-Blackwell GPU (RTX 40/30 series, A100, H100, etc.)

**Recommended**: Run setup-cutile.sh then continue (supports both cutile and hw1-asr):

```bash
# From project root
source utils/setup-cutile.sh

# Run cuTile tutorials
python cutile-tutorial/1-vectoradd/vectoradd.py
python cutile-tutorial/7-attention/attention.py

# Run hw1-asr
python hw1-asr/benchmark_student.py glm_asr_scratch
```

`setup-cutile.sh` automatically:
- Sets CUDA environment variables (CUDA_PATH, CUDA_HOME, LD_LIBRARY_PATH, CUPY_CUDA_PATH)
- Sets CuPy compilation include paths (CFLAGS, CXXFLAGS)
- Creates/activates the conda environment

## Tutorial Directories

| Directory | Content |
|-----------|---------|
| 0-environment | Environment check |
| 1-vectoradd | Vector addition (Hello World) |
| 2-execution-model | Execution model (1D/2D grid) |
| 3-data-model | Data types (FP16/FP32) |
| 4-transpose | Matrix transpose |
| 5-secret-notes | Advanced notes |
| 6-performance-tuning | Performance tuning |
| 7-attention | Attention mechanism |

## Supported GPUs

| GPU | Compute Capability | Support Method |
|-----|-------------------|----------------|
| RTX 5090/5080 | 12.x | Native support |
| B100/B200/GB200 | 10.x | Native support |
| RTX 4090/4080 | 8.9 | Not supported |
| RTX 3090/3080 | 8.6 | Not supported |
| A100/H100 | 8.0/9.0 | Not supported |
