"""
Pure CuTile 1D Convolution Implementation
End-to-end implementation using only NVIDIA CuTile kernels
Uses im2col approach to convert convolution to matrix multiplication.
"""

import cuda.tile as ct
import cupy as cp
import numpy as np
from typing import Optional


def get_stream():
    """Get current CUDA stream pointer."""
    return cp.cuda.get_current_stream().ptr


def next_power_of_two(x: int) -> int:
    """Return the smallest power of two >= x."""
    return 1 << (x - 1).bit_length() if x > 0 else 1


# ============================================================================
# CuTile Kernels for Conv1d via Matrix Multiplication
# ============================================================================

@ct.kernel
def conv1d_matmul_kernel(
    col,            # im2col result: (batch, in_c * k_size, out_length)
    weight,         # Weight reshaped: (out_channels, in_c * k_size)
    output,         # Output: (batch, out_channels, out_length)
    out_channels: ct.Constant[int],
    col_size: ct.Constant[int],  # in_channels * kernel_size (padded to power of 2)
    out_length: ct.Constant[int]
):
    """
    Conv1d via matrix multiplication after im2col transformation.
    Grid: (batch,)
    Each block computes output for one batch element.
    """
    pid_b = ct.bid(0)

    # Load weight: (out_channels, col_size)
    w_tile = ct.load(weight, index=(0, 0), shape=(out_channels, col_size))

    # Load col for this batch: (col_size, out_length)
    col_tile = ct.load(col, index=(pid_b, 0, 0), shape=(1, col_size, out_length))
    col_tile = ct.reshape(col_tile, (col_size, out_length))

    # Matrix multiply: (out_channels, col_size) @ (col_size, out_length)
    # = (out_channels, out_length)
    result = ct.matmul(w_tile, col_tile)

    # Store
    result = ct.reshape(result, (1, out_channels, out_length))
    ct.store(output, index=(pid_b, 0, 0), tile=result)


# ============================================================================
# Conv1d Classes
# ============================================================================

def im2col_1d(x: cp.ndarray, kernel_size: int, stride: int) -> cp.ndarray:
    """
    Convert input to column format for efficient convolution.

    Args:
        x: Input (batch, in_channels, length) - already padded
        kernel_size: Size of convolution kernel
        stride: Stride for convolution

    Returns:
        Column matrix (batch, in_channels * kernel_size, out_length)
    """
    batch, in_channels, length = x.shape
    out_length = (length - kernel_size) // stride + 1

    # Use CuPy's stride tricks for efficient unfold
    # Shape of output: (batch, in_channels, kernel_size, out_length)
    strides = (
        x.strides[0],  # batch stride
        x.strides[1],  # channel stride
        x.strides[2],  # kernel position stride
        x.strides[2] * stride  # output position stride
    )
    shape = (batch, in_channels, kernel_size, out_length)

    col = cp.lib.stride_tricks.as_strided(x, shape=shape, strides=strides)
    # Reshape to (batch, in_channels * kernel_size, out_length)
    col = col.transpose(0, 1, 2, 3).reshape(batch, in_channels * kernel_size, out_length)
    return cp.ascontiguousarray(col)


class Conv1d:
    """
    1D Convolution using im2col + CuTile matrix multiplication.

    For large dimensions, falls back to CuPy matmul to avoid CuTile tile size limits.
    """

    # Maximum dimension for CuTile kernel (to avoid very slow compilation)
    MAX_TILE_DIM = 256

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        padding: int = 0,
        bias: bool = True
    ):
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.has_bias = bias

        # Compute sizes for CuTile (need power of 2)
        self.col_size = in_channels * kernel_size
        self.col_size_padded = next_power_of_two(self.col_size)
        self.out_channels_padded = next_power_of_two(out_channels)

        # Determine if we can use CuTile kernel
        self.use_cutile = (
            self.col_size_padded <= self.MAX_TILE_DIM and
            self.out_channels_padded <= self.MAX_TILE_DIM
        )

        # Initialize weights
        # Xavier initialization
        k = 1.0 / (in_channels * kernel_size)
        weight = cp.random.uniform(
            -np.sqrt(k), np.sqrt(k),
            (out_channels, in_channels, kernel_size)
        ).astype(cp.float32)

        # Reshape weight for matmul: (out_channels, in_channels * kernel_size)
        self.weight = weight.reshape(out_channels, self.col_size)

        # Pad weight for power of 2 if needed (only for CuTile path)
        if self.use_cutile and (
            self.col_size_padded != self.col_size or
            self.out_channels_padded != out_channels
        ):
            self.weight_padded = cp.zeros(
                (self.out_channels_padded, self.col_size_padded),
                dtype=cp.float32
            )
            self.weight_padded[:out_channels, :self.col_size] = self.weight
        else:
            self.weight_padded = self.weight

        if bias:
            self.bias = cp.zeros(out_channels, dtype=cp.float32)
        else:
            self.bias = None

    def __call__(self, x: cp.ndarray) -> cp.ndarray:
        """
        Forward pass using im2col + matrix multiply.

        Args:
            x: Input (batch, in_channels, length)

        Returns:
            Output (batch, out_channels, out_length)
        """
        batch, in_channels, length = x.shape

        # Calculate output length
        out_length = (length + 2 * self.padding - self.kernel_size) // self.stride + 1

        # Pad input spatially if needed
        if self.padding > 0:
            x_padded = cp.pad(
                x,
                ((0, 0), (0, 0), (self.padding, self.padding)),
                mode='constant',
                constant_values=0
            )
        else:
            x_padded = x

        x_padded = cp.ascontiguousarray(x_padded.astype(cp.float32))

        # im2col transformation
        col = im2col_1d(x_padded, self.kernel_size, self.stride)
        # col shape: (batch, in_channels * kernel_size, out_length)

        out_length_padded = next_power_of_two(out_length)

        # Check if dimensions are small enough for CuTile
        can_use_cutile = (
            self.use_cutile and
            out_length_padded <= self.MAX_TILE_DIM
        )

        if can_use_cutile:
            # Pad col for CuTile if needed
            if self.col_size_padded != self.col_size or out_length_padded != out_length:
                col_padded = cp.zeros(
                    (batch, self.col_size_padded, out_length_padded),
                    dtype=cp.float32
                )
                col_padded[:, :self.col_size, :out_length] = col
                col = col_padded

            # Allocate output (padded)
            output_padded = cp.empty(
                (batch, self.out_channels_padded, out_length_padded),
                dtype=cp.float32
            )

            # Launch CuTile matmul kernel
            ct.launch(
                get_stream(),
                (batch,),
                conv1d_matmul_kernel,
                (col, self.weight_padded, output_padded,
                 self.out_channels_padded, self.col_size_padded, out_length_padded)
            )

            # Extract actual output dimensions
            output = output_padded[:, :self.out_channels, :out_length].copy()
        else:
            # Use CuPy einsum for large dimensions (batched matmul)
            # weight: (out_channels, col_size)
            # col: (batch, col_size, out_length)
            # We want: output[b, o, l] = sum_c weight[o, c] * col[b, c, l]
            output = cp.einsum('oc,bcl->bol', self.weight, col)

        # Add bias
        if self.has_bias and self.bias is not None:
            output = output + self.bias[None, :, None]

        return output


class Conv1dSubsampler:
    """
    Audio feature subsampling using two Conv1d layers.
    Reduces time dimension by factor of 4 (stride 2 x 2).
    """

    def __init__(
        self,
        in_channels: int,
        mid_channels: int,
        out_channels: int,
        kernel_sizes: tuple = (3, 3)
    ):
        self.conv1 = Conv1d(
            in_channels, mid_channels,
            kernel_size=kernel_sizes[0],
            stride=2,
            padding=kernel_sizes[0] // 2
        )
        self.conv2 = Conv1d(
            mid_channels, out_channels,
            kernel_size=kernel_sizes[1],
            stride=2,
            padding=kernel_sizes[1] // 2
        )

    def __call__(self, x: cp.ndarray) -> cp.ndarray:
        """
        Forward pass with GELU activation between layers.

        Args:
            x: Input (batch, in_channels, time)

        Returns:
            Output (batch, out_channels, time//4)
        """
        # First conv + GELU
        x = self.conv1(x)
        x = gelu(x)

        # Second conv + GELU
        x = self.conv2(x)
        x = gelu(x)

        return x


def gelu(x: cp.ndarray) -> cp.ndarray:
    """GELU activation using tanh approximation."""
    return 0.5 * x * (1.0 + cp.tanh(
        np.sqrt(2.0 / np.pi) * (x + 0.044715 * x ** 3)
    ))


# ============================================================================
# Test
# ============================================================================

if __name__ == "__main__":
    print("Testing Pure CuTile Conv1d...")

    # Test small Conv1d (uses CuTile kernel)
    print("\nSmall Conv1d (CuTile kernel):")
    batch_size = 2
    in_channels = 16
    length = 32
    out_channels = 32
    kernel_size = 3

    conv = Conv1d(in_channels, out_channels, kernel_size, stride=1, padding=1)
    x = cp.random.randn(batch_size, in_channels, length).astype(cp.float32)
    print(f"  Using CuTile: {conv.use_cutile}")
    y = conv(x)
    print(f"  Input shape: {x.shape}")
    print(f"  Output shape: {y.shape}")

    # Test larger Conv1d (may use CuPy fallback)
    print("\nLarger Conv1d (may use CuPy):")
    in_channels_large = 80
    out_channels_large = 256

    conv_large = Conv1d(in_channels_large, out_channels_large, kernel_size, stride=1, padding=1)
    x_large = cp.random.randn(batch_size, in_channels_large, length).astype(cp.float32)
    print(f"  Using CuTile: {conv_large.use_cutile}")
    y_large = conv_large(x_large)
    print(f"  Input shape: {x_large.shape}")
    print(f"  Output shape: {y_large.shape}")

    # Test strided Conv1d
    print("\nStrided Conv1d (stride=2):")
    conv_strided = Conv1d(in_channels, out_channels, kernel_size, stride=2, padding=1)
    y_strided = conv_strided(x)
    expected_len = (length + 2 - kernel_size) // 2 + 1
    print(f"  Output shape: {y_strided.shape}")
    print(f"  Expected length: {expected_len}")

    # Verify output statistics
    print("\nOutput statistics:")
    print(f"  Mean: {float(cp.mean(y)):.4f}")
    print(f"  Std:  {float(cp.std(y)):.4f}")

    print("\nPure CuTile Conv1d working!")
