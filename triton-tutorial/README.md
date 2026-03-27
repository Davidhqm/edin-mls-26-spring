# Triton Tutorial

Triton GPU programming tutorials corresponding to the cuTile examples.

## Environment Setup

### 1. Install Environment

If you are using the project environment:

```bash
# Run from project root
bash utils/setup-triton.sh
```

This installs `torch`, `numpy`, and `triton` into the shared `mls` conda environment.

Or install Triton + PyTorch directly:

```bash
pip install torch numpy triton
```

### 2. Activate Environment

```bash
conda activate mls
```

If `conda` isn't on PATH in a new shell (`<conda>` is the conda install prefix, e.g. `~/miniconda3` or `/opt/conda`):

```bash
source <conda>/bin/activate
conda activate mls
```

## Running Tutorials

```bash
python triton-tutorial/1-vectoradd/vectoradd.py
```

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

Triton relies on the underlying PyTorch backend. A recent NVIDIA GPU (sm70+ recommended) is typically required for CUDA builds.
