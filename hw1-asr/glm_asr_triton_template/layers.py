"""
Triton Neural Network Layers — H200-optimised v2
Key optimisations:
1. Fused QKV projection (3 GEMMs → 1 GEMM) — biggest win
2. FP16 throughout — 2-4x GEMM speedup via Tensor Cores
3. Fused SwiGLU kernel
4. Fused Linear+GELU for encoder
5. Safe Hopper tile sizes (avoids Ampere MMA assert)
6. RMSNorm/LayerNorm always Triton with power-of-two block padding
"""

import math
from typing import Optional, Tuple

import numpy as np
import torch
import triton
import triton.language as tl


# ============================================================================
# Shared Utilities
# ============================================================================

def get_stream():
    if torch.cuda.is_available():
        return torch.cuda.current_stream().cuda_stream
    return None

def next_power_of_two(x: int) -> int:
    return 1 << (x - 1).bit_length() if x > 0 else 1

def pad_to_multiple(size: int, multiple: int) -> int:
    return ((size + multiple - 1) // multiple) * multiple

def _is_power_of_two(x: int) -> bool:
    return x > 0 and (x & (x - 1)) == 0


# ============================================================================
# Triton Kernels
# ============================================================================

@triton.jit
def rmsnorm_kernel(
    x_ptr, w_ptr, y_ptr,
    stride_x, stride_y,
    hidden_size, eps,
    BLOCK_SIZE: tl.constexpr,
):
    pid  = tl.program_id(0)
    offs = tl.arange(0, BLOCK_SIZE)
    mask = offs < hidden_size
    x    = tl.load(x_ptr + pid * stride_x + offs, mask=mask, other=0.0).to(tl.float32)
    var  = tl.sum(x * x, axis=0) / hidden_size
    xn   = x * tl.rsqrt(var + eps)
    w    = tl.load(w_ptr + offs, mask=mask, other=1.0).to(tl.float32)
    tl.store(y_ptr + pid * stride_y + offs, xn * w, mask=mask)


@triton.jit
def layernorm_kernel(
    x_ptr, w_ptr, b_ptr, y_ptr,
    stride_x, stride_y,
    hidden_size, eps,
    BLOCK_SIZE: tl.constexpr,
):
    pid  = tl.program_id(0)
    offs = tl.arange(0, BLOCK_SIZE)
    mask = offs < hidden_size
    x    = tl.load(x_ptr + pid * stride_x + offs, mask=mask, other=0.0).to(tl.float32)
    mean = tl.sum(x, axis=0) / hidden_size
    xc   = x - mean
    var  = tl.sum(xc * xc, axis=0) / hidden_size
    xn   = xc * tl.rsqrt(var + eps)
    w    = tl.load(w_ptr + offs, mask=mask, other=1.0).to(tl.float32)
    b    = tl.load(b_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    tl.store(y_ptr + pid * stride_y + offs, xn * w + b, mask=mask)


@triton.jit
def gelu_kernel(x_ptr, y_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid  = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < n_elements
    x    = tl.load(x_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    c    = 0.7978845608028654
    y    = x * 0.5 * (1.0 + tl.extra.cuda.libdevice.tanh(c * (x + 0.044715 * x * x * x)))
    tl.store(y_ptr + offs, y, mask=mask)


@triton.jit
def silu_kernel(x_ptr, y_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid  = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < n_elements
    x    = tl.load(x_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    tl.store(y_ptr + offs, x / (1.0 + tl.exp(-x)), mask=mask)


@triton.jit
def linear_kernel_fp16(
    a_ptr, b_ptr, c_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    """
    FP16 matmul using Tensor Cores: C = A @ B
    Hopper-safe tile sizes: BLOCK_N <= BLOCK_M, BLOCK_K <= BLOCK_N
    Configs tried:
      A: 64/32/32  — safe baseline
      B: 128/64/32 — best balance for H200  ← chosen
      C: 64/64/32  — alternative
    """
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BLOCK_K)):
        k_off = k * BLOCK_K + offs_k
        a = tl.load(
            a_ptr + offs_m[:, None] * stride_am + k_off[None, :] * stride_ak,
            mask=(offs_m[:, None] < M) & (k_off[None, :] < K), other=0.0,
        ).to(tl.float16)
        b = tl.load(
            b_ptr + k_off[:, None] * stride_bk + offs_n[None, :] * stride_bn,
            mask=(k_off[:, None] < K) & (offs_n[None, :] < N), other=0.0,
        ).to(tl.float16)
        acc += tl.dot(a, b, allow_tf32=True).to(tl.float32)
    tl.store(
        c_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn,
        acc,
        mask=(offs_m[:, None] < M) & (offs_n[None, :] < N),
    )


@triton.jit
def linear_gelu_kernel(
    a_ptr, b_ptr, c_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    """Fused Linear + GELU in FP16."""
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BLOCK_K)):
        k_off = k * BLOCK_K + offs_k
        a = tl.load(
            a_ptr + offs_m[:, None] * stride_am + k_off[None, :] * stride_ak,
            mask=(offs_m[:, None] < M) & (k_off[None, :] < K), other=0.0,
        ).to(tl.float16)
        b = tl.load(
            b_ptr + k_off[:, None] * stride_bk + offs_n[None, :] * stride_bn,
            mask=(k_off[:, None] < K) & (offs_n[None, :] < N), other=0.0,
        ).to(tl.float16)
        acc += tl.dot(a, b, allow_tf32=True).to(tl.float32)
    c = 0.7978845608028654
    acc = acc * 0.5 * (1.0 + tl.extra.cuda.libdevice.tanh(c * (acc + 0.044715 * acc * acc * acc)))
    tl.store(
        c_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn,
        acc,
        mask=(offs_m[:, None] < M) & (offs_n[None, :] < N),
    )


@triton.jit
def swiglu_fused_kernel(
    a_ptr, gate_ptr, up_ptr, c_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_gk, stride_gn,
    stride_uk, stride_un,
    stride_cm, stride_cn,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    """Fused SwiGLU in FP16: out = SiLU(x @ gate) * (x @ up)."""
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)
    gate_acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    up_acc   = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BLOCK_K)):
        k_off = k * BLOCK_K + offs_k
        a = tl.load(
            a_ptr + offs_m[:, None] * stride_am + k_off[None, :] * stride_ak,
            mask=(offs_m[:, None] < M) & (k_off[None, :] < K), other=0.0,
        ).to(tl.float16)
        gw = tl.load(
            gate_ptr + k_off[:, None] * stride_gk + offs_n[None, :] * stride_gn,
            mask=(k_off[:, None] < K) & (offs_n[None, :] < N), other=0.0,
        ).to(tl.float16)
        uw = tl.load(
            up_ptr + k_off[:, None] * stride_uk + offs_n[None, :] * stride_un,
            mask=(k_off[:, None] < K) & (offs_n[None, :] < N), other=0.0,
        ).to(tl.float16)
        gate_acc += tl.dot(a, gw, allow_tf32=True).to(tl.float32)
        up_acc   += tl.dot(a, uw, allow_tf32=True).to(tl.float32)
    out = (gate_acc / (1.0 + tl.exp(-gate_acc))) * up_acc
    tl.store(
        c_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn,
        out,
        mask=(offs_m[:, None] < M) & (offs_n[None, :] < N),
    )


@triton.jit
def softmax_kernel(x_ptr, y_ptr, stride_x, stride_y, n_cols, BLOCK_SIZE: tl.constexpr):
    row  = tl.program_id(0)
    offs = tl.arange(0, BLOCK_SIZE)
    mask = offs < n_cols
    x    = tl.load(x_ptr + row * stride_x + offs, mask=mask, other=float("-inf"))
    x    = x - tl.max(x, axis=0)
    ex   = tl.exp(x)
    tl.store(y_ptr + row * stride_y + offs, ex / tl.sum(ex, axis=0), mask=mask)


@triton.jit
def embedding_kernel(
    indices_ptr, weight_ptr, output_ptr,
    embedding_dim, stride_w0, stride_w1, stride_out0,
    BLOCK_SIZE: tl.constexpr,
):
    pid0 = tl.program_id(0)
    pid1 = tl.program_id(1)
    idx  = tl.load(indices_ptr + pid0)
    offs = pid1 * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < embedding_dim
    w    = tl.load(weight_ptr + idx * stride_w0 + offs * stride_w1, mask=mask, other=0.0)
    tl.store(output_ptr + pid0 * stride_out0 + offs, w, mask=mask)


# ============================================================================
# Layer Classes
# ============================================================================

class RMSNorm:
    def __init__(self, hidden_size: int, eps: float = 1e-6):
        self.hidden_size = hidden_size
        self.eps = eps
        self.weight = torch.ones(hidden_size, dtype=torch.float32)

    def _move_weight(self, device):
        if self.weight.device != device:
            self.weight = self.weight.to(device)

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        if not x.is_cuda:
            xf  = x.float()
            var = (xf * xf).mean(dim=-1, keepdim=True)
            xn  = xf * torch.rsqrt(var + self.eps)
            self._move_weight(x.device)
            return (self.weight * xn).to(x.dtype)
        original_shape = x.shape
        B   = int(np.prod(x.shape[:-1]))
        xf  = x.reshape(B, self.hidden_size).contiguous().float()
        out = torch.empty_like(xf)
        self._move_weight(x.device)
        block = next_power_of_two(self.hidden_size)
        rmsnorm_kernel[(B,)](
            xf, self.weight, out,
            xf.stride(0), out.stride(0),
            self.hidden_size, self.eps,
            BLOCK_SIZE=block,
        )
        return out.reshape(original_shape)


class LayerNorm:
    def __init__(self, hidden_size: int, eps: float = 1e-5):
        self.hidden_size = hidden_size
        self.eps = eps
        self.weight = torch.ones(hidden_size, dtype=torch.float32)
        self.bias   = torch.zeros(hidden_size, dtype=torch.float32)

    def _move_params(self, device):
        if self.weight.device != device:
            self.weight = self.weight.to(device)
        if self.bias.device != device:
            self.bias = self.bias.to(device)

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        if not x.is_cuda:
            xf   = x.float()
            mean = xf.mean(dim=-1, keepdim=True)
            var  = xf.var(dim=-1, keepdim=True, unbiased=False)
            xn   = (xf - mean) * torch.rsqrt(var + self.eps)
            self._move_params(x.device)
            return (self.weight * xn + self.bias).to(x.dtype)
        original_shape = x.shape
        B   = int(np.prod(x.shape[:-1]))
        xf  = x.reshape(B, self.hidden_size).contiguous().float()
        out = torch.empty_like(xf)
        self._move_params(x.device)
        block = next_power_of_two(self.hidden_size)
        layernorm_kernel[(B,)](
            xf, self.weight, self.bias, out,
            xf.stride(0), out.stride(0),
            self.hidden_size, self.eps,
            BLOCK_SIZE=block,
        )
        return out.reshape(original_shape)


# ============================================================================
# Activation Functions
# ============================================================================

def gelu(x: torch.Tensor) -> torch.Tensor:
    if not x.is_cuda:
        return torch.nn.functional.gelu(x)
    original_shape = x.shape
    total = x.numel()
    block = 256
    xf  = x.reshape(-1).contiguous().float()
    out = torch.empty_like(xf)
    gelu_kernel[(triton.cdiv(total, block),)](xf, out, total, BLOCK_SIZE=block)
    return out.reshape(original_shape).to(x.dtype)


def silu(x: torch.Tensor) -> torch.Tensor:
    if not x.is_cuda:
        return torch.nn.functional.silu(x)
    original_shape = x.shape
    total = x.numel()
    block = 256
    xf  = x.reshape(-1).contiguous().float()
    out = torch.empty_like(xf)
    silu_kernel[(triton.cdiv(total, block),)](xf, out, total, BLOCK_SIZE=block)
    return out.reshape(original_shape).to(x.dtype)


def get_activation(name: str):
    acts = {"gelu": gelu, "silu": silu}
    if name not in acts:
        raise ValueError(f"Unknown activation: {name}")
    return acts[name]


def softmax(x: torch.Tensor, axis: int = -1) -> torch.Tensor:
    if axis not in (-1, len(x.shape) - 1):
        x = torch.movedim(x, axis, -1)
    original_shape = x.shape
    B   = int(np.prod(x.shape[:-1]))
    seq = x.shape[-1]
    xf  = x.reshape(B, seq).float().contiguous()
    out = torch.empty_like(xf)
    if x.is_cuda:
        block = next_power_of_two(seq)
        softmax_kernel[(B,)](xf, out, xf.stride(0), out.stride(0), seq, BLOCK_SIZE=block)
        result = out.reshape(original_shape)
    else:
        result = torch.softmax(x, dim=-1)
    if axis not in (-1, len(original_shape) - 1):
        result = torch.movedim(result, -1, axis)
    return result


# ============================================================================
# Linear Layer (FP16 Triton backend)
# ============================================================================

class Linear:
    # Hopper-safe tile sizes: BLOCK_N <= BLOCK_M, BLOCK_K <= BLOCK_N
    TILE_M = 128
    TILE_N = 128
    TILE_K = 32

    BACKEND = "auto"   # "torch" | "triton" | "auto"

    def __init__(self, in_features: int, out_features: int, bias: bool = True):
        self.in_features  = in_features
        self.out_features = out_features
        self.has_bias     = bias
        self.weight       = torch.zeros((out_features, in_features), dtype=torch.float32)
        self.bias_param   = torch.zeros(out_features) if bias else None
        self._wt_padded   = None
        self._K_padded    = None
        self._N_padded    = None
        self._weight_device = None

    def _ensure_weight_prepared(self, device):
        if self._wt_padded is not None and self._weight_device == device:
            return
        K = self.in_features
        N = self.out_features
        self._K_padded = pad_to_multiple(K, self.TILE_K)
        self._N_padded = pad_to_multiple(N, self.TILE_N)
        wt = self.weight.t().contiguous()
        if self._K_padded > K or self._N_padded > N:
            wp = torch.zeros((self._K_padded, self._N_padded), dtype=torch.float32, device=device)
            wp[:K, :N] = wt
            self._wt_padded = wp
        else:
            self._wt_padded = wt
        self._weight_device = device

    def _move_params(self, device):
        if self.weight.device != device:
            self.weight = self.weight.to(device)
            self._wt_padded = None
        if self.has_bias and self.bias_param is not None and self.bias_param.device != device:
            self.bias_param = self.bias_param.to(device)

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        M = int(np.prod(x.shape[:-1]))
        if Linear.BACKEND == "triton" or (Linear.BACKEND == "auto" and M >= self.TILE_M and x.is_cuda):
            return self._forward_triton(x)
        return self._forward_torch(x)

    def _forward_torch(self, x: torch.Tensor) -> torch.Tensor:
        self._move_params(x.device)
        original_shape = x.shape
        M   = int(np.prod(original_shape[:-1]))
        x2d = x.reshape(M, self.in_features).float()
        out = x2d @ self.weight.t()
        if self.has_bias and self.bias_param is not None:
            out = out + self.bias_param
        return out.reshape(*original_shape[:-1], self.out_features)

    def _forward_triton(self, x: torch.Tensor) -> torch.Tensor:
        self._move_params(x.device)
        self._ensure_weight_prepared(x.device)
        original_shape = x.shape
        M = int(np.prod(original_shape[:-1]))
        K = self.in_features
        N = self.out_features
        x2d   = x.reshape(M, K).float().contiguous()
        M_pad = pad_to_multiple(M, self.TILE_M)

        if M_pad > M or self._K_padded > K:
            xp = torch.zeros((M_pad, self._K_padded), dtype=torch.float32, device=x.device)
            xp[:M, :K] = x2d
        else:
            xp = x2d

        out = torch.zeros((M_pad, self._N_padded), dtype=torch.float32, device=x.device)
        grid = (triton.cdiv(M_pad, self.TILE_M), triton.cdiv(self._N_padded, self.TILE_N))
        linear_kernel_fp16[grid](
            xp, self._wt_padded, out,
            M_pad, self._N_padded, self._K_padded,
            xp.stride(0), xp.stride(1),
            self._wt_padded.stride(0), self._wt_padded.stride(1),
            out.stride(0), out.stride(1),
            BLOCK_M=self.TILE_M, BLOCK_N=self.TILE_N, BLOCK_K=self.TILE_K,
        )
        out = out[:M, :N]
        if self.has_bias and self.bias_param is not None:
            out = out + self.bias_param
        return out.reshape(*original_shape[:-1], N)


# ============================================================================
# Fused QKV Linear
# Replaces 3 separate Q, K, V projections with a single GEMM
# Saves 2 full memory passes — biggest single win
# ============================================================================

class FusedQKVLinear:
    """
    Fused QKV projection: single GEMM instead of 3.
    x @ [Wq | Wk | Wv]  →  split into q, k, v

    Usage: replace
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)
    with:
        q, k, v = self.qkv_proj(x)
    """

    TILE_M = 128
    TILE_N = 128
    TILE_K = 32

    def __init__(
        self,
        in_features: int,
        q_out: int,
        k_out: int,
        v_out: int,
        bias: bool = False,
    ):
        self.in_features = in_features
        self.q_out = q_out
        self.k_out = k_out
        self.v_out = v_out
        self.total_out = q_out + k_out + v_out
        self.has_bias = bias

        # Single combined weight [total_out, in_features]
        self.weight = torch.zeros((self.total_out, in_features), dtype=torch.float32)
        self.bias_param = torch.zeros(self.total_out) if bias else None

        self._wt_padded = None
        self._weight_device = None

    def _move_params(self, device):
        if self.weight.device != device:
            self.weight = self.weight.to(device)
            self._wt_padded = None
        if self.has_bias and self.bias_param is not None and self.bias_param.device != device:
            self.bias_param = self.bias_param.to(device)

    def _ensure_prepared(self, device):
        if self._wt_padded is not None and self._weight_device == device:
            return
        K = self.in_features
        N = self.total_out
        K_p = pad_to_multiple(K, self.TILE_K)
        N_p = pad_to_multiple(N, self.TILE_N)
        wt = self.weight.t().contiguous()
        if K_p > K or N_p > N:
            wp = torch.zeros((K_p, N_p), dtype=torch.float32, device=device)
            wp[:K, :N] = wt
            self._wt_padded = wp
        else:
            self._wt_padded = wt
        self._K_padded = K_p
        self._N_padded = N_p
        self._weight_device = device

    def __call__(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        self._move_params(x.device)
        original_shape = x.shape
        M = int(np.prod(original_shape[:-1]))
        K = self.in_features
        N = self.total_out

        # Use torch for small batches or CPU
        if not x.is_cuda or M < self.TILE_M:
            x2d = x.reshape(M, K).float()
            out = x2d @ self.weight.t()
            if self.has_bias and self.bias_param is not None:
                out = out + self.bias_param
            out = out.reshape(*original_shape[:-1], N)
            q = out[..., :self.q_out]
            k = out[..., self.q_out:self.q_out + self.k_out]
            v = out[..., self.q_out + self.k_out:]
            return q, k, v

        self._ensure_prepared(x.device)
        x2d   = x.reshape(M, K).float().contiguous()
        M_pad = pad_to_multiple(M, self.TILE_M)

        if M_pad > M or self._K_padded > K:
            xp = torch.zeros((M_pad, self._K_padded), dtype=torch.float32, device=x.device)
            xp[:M, :K] = x2d
        else:
            xp = x2d

        out_padded = torch.zeros((M_pad, self._N_padded), dtype=torch.float32, device=x.device)
        grid = (triton.cdiv(M_pad, self.TILE_M), triton.cdiv(self._N_padded, self.TILE_N))
        linear_kernel_fp16[grid](
            xp, self._wt_padded, out_padded,
            M_pad, self._N_padded, self._K_padded,
            xp.stride(0), xp.stride(1),
            self._wt_padded.stride(0), self._wt_padded.stride(1),
            out_padded.stride(0), out_padded.stride(1),
            BLOCK_M=self.TILE_M, BLOCK_N=self.TILE_N, BLOCK_K=self.TILE_K,
        )
        out = out_padded[:M, :N].reshape(*original_shape[:-1], N)
        if self.has_bias and self.bias_param is not None:
            out = out + self.bias_param

        q = out[..., :self.q_out].contiguous()
        k = out[..., self.q_out:self.q_out + self.k_out].contiguous()
        v = out[..., self.q_out + self.k_out:].contiguous()
        return q, k, v

    def load_separate_weights(self, wq, wk, wv, bq=None, bk=None, bv=None):
        """Load from separate Q, K, V weight tensors (for compatibility with weight loader)."""
        with torch.no_grad():
            self.weight[:self.q_out] = wq
            self.weight[self.q_out:self.q_out + self.k_out] = wk
            self.weight[self.q_out + self.k_out:] = wv
            if self.has_bias and bq is not None:
                self.bias_param[:self.q_out] = bq
                self.bias_param[self.q_out:self.q_out + self.k_out] = bk
                self.bias_param[self.q_out + self.k_out:] = bv
        self._wt_padded = None  # invalidate cache


# ============================================================================
# Embedding
# ============================================================================

class Embedding:
    def __init__(self, num_embeddings: int, embedding_dim: int):
        self.num_embeddings = num_embeddings
        self.embedding_dim  = embedding_dim
        self.weight = torch.zeros((num_embeddings, embedding_dim), dtype=torch.float32)

    def __call__(self, input_ids: torch.Tensor) -> torch.Tensor:
        original_shape = input_ids.shape
        B = int(np.prod(original_shape))
        if self.weight.device != input_ids.device:
            self.weight = self.weight.to(input_ids.device)
        if not input_ids.is_cuda:
            flat = input_ids.reshape(-1).long()
            return self.weight.index_select(0, flat).reshape(*original_shape, self.embedding_dim)
        flat = input_ids.reshape(-1).to(torch.int32).contiguous()
        out  = torch.empty((B, self.embedding_dim), dtype=torch.float32, device=flat.device)
        block = 256
        grid  = (B, triton.cdiv(self.embedding_dim, block))
        embedding_kernel[grid](
            flat, self.weight, out,
            self.embedding_dim,
            self.weight.stride(0), self.weight.stride(1), out.stride(0),
            BLOCK_SIZE=block,
        )
        return out.reshape(*original_shape, self.embedding_dim)


# ============================================================================
# MLP (SwiGLU — decoder)
# ============================================================================

class MLP:
    FUSED = True
    TILE_M, TILE_N, TILE_K = 32, 64, 32   # Hopper-safe

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        activation: str = "silu",
        bias: bool = False,
        use_gating: bool = True,
    ):
        self.use_gating        = use_gating
        self.act_fn            = get_activation(activation)
        self.hidden_size       = hidden_size
        self.intermediate_size = intermediate_size
        self.bias_enabled      = bias

        if use_gating:
            self.gate_proj = Linear(hidden_size, intermediate_size, bias=bias)
            self.up_proj   = Linear(hidden_size, intermediate_size, bias=bias)
        else:
            self.up_proj   = Linear(hidden_size, intermediate_size, bias=bias)
        self.down_proj = Linear(intermediate_size, hidden_size, bias=bias)

        self._gate_wt      = None
        self._up_wt        = None
        self._fused_device = None

    def _prepare_fused(self, device):
        if self._gate_wt is not None and self._fused_device == device:
            return
        self._gate_wt      = self.gate_proj.weight.t().contiguous().to(device)
        self._up_wt        = self.up_proj.weight.t().contiguous().to(device)
        self._fused_device = device

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        if self.use_gating and MLP.FUSED and x.is_cuda:
            return self._forward_fused(x)
        return self._forward_standard(x)

    def _forward_standard(self, x):
        if self.use_gating:
            return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
        return self.down_proj(self.act_fn(self.up_proj(x)))

    def _forward_fused(self, x: torch.Tensor) -> torch.Tensor:
        if self.gate_proj.weight.device != x.device:
            self.gate_proj.weight = self.gate_proj.weight.to(x.device)
            self._gate_wt = None
        if self.up_proj.weight.device != x.device:
            self.up_proj.weight = self.up_proj.weight.to(x.device)
            self._up_wt = None
        self._prepare_fused(x.device)

        orig = x.shape
        x2d  = x.reshape(-1, self.hidden_size).float().contiguous()
        M, K, N = x2d.shape[0], self.hidden_size, self.intermediate_size
        M_p = pad_to_multiple(M, self.TILE_M)
        K_p = pad_to_multiple(K, self.TILE_K)
        N_p = pad_to_multiple(N, self.TILE_N)

        xp = x2d
        if M_p > M or K_p > K:
            xp = torch.zeros((M_p, K_p), dtype=torch.float32, device=x.device)
            xp[:M, :K] = x2d

        gp, up = self._gate_wt, self._up_wt
        if K_p > K or N_p > N:
            gp = torch.zeros((K_p, N_p), dtype=torch.float32, device=x.device)
            gp[:K, :N] = self._gate_wt
            up = torch.zeros((K_p, N_p), dtype=torch.float32, device=x.device)
            up[:K, :N] = self._up_wt

        inter = torch.zeros((M_p, N_p), dtype=torch.float32, device=x.device)
        grid  = (triton.cdiv(M_p, self.TILE_M), triton.cdiv(N_p, self.TILE_N))
        swiglu_fused_kernel[grid](
            xp, gp, up, inter,
            M_p, N_p, K_p,
            xp.stride(0), xp.stride(1),
            gp.stride(0), gp.stride(1),
            up.stride(0), up.stride(1),
            inter.stride(0), inter.stride(1),
            BLOCK_M=self.TILE_M, BLOCK_N=self.TILE_N, BLOCK_K=self.TILE_K,
        )
        if M_p > M or N_p > N:
            inter = inter[:M, :N]
        return self.down_proj(inter.reshape(*orig[:-1], self.intermediate_size))


# ============================================================================
# EncoderMLP (BERT/Whisper style, fused Linear+GELU)
# ============================================================================

class EncoderMLP:
    FUSED = True
    TILE_M, TILE_N, TILE_K = 32, 64, 32

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        activation: str = "gelu",
        bias: bool = True,
    ):
        self.fc1 = Linear(hidden_size, intermediate_size, bias=bias)
        self.fc2 = Linear(intermediate_size, hidden_size, bias=bias)
        self.act_fn            = get_activation(activation)
        self.hidden_size       = hidden_size
        self.intermediate_size = intermediate_size
        self.bias_enabled      = bias
        self.activation        = activation
        self._fc1_wt           = None
        self._fused_device     = None

    def _prepare_fused(self, device):
        if self._fc1_wt is not None and self._fused_device == device:
            return
        self._fc1_wt       = self.fc1.weight.t().contiguous().to(device)
        self._fused_device = device

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        if EncoderMLP.FUSED and self.activation == "gelu" and x.is_cuda:
            return self._forward_fused(x)
        return self.fc2(self.act_fn(self.fc1(x)))

    def _forward_fused(self, x: torch.Tensor) -> torch.Tensor:
        if self.fc1.weight.device != x.device:
            self.fc1.weight = self.fc1.weight.to(x.device)
            self._fc1_wt = None
        self._prepare_fused(x.device)

        orig = x.shape
        x2d  = x.reshape(-1, self.hidden_size).float().contiguous()
        M, K, N = x2d.shape[0], self.hidden_size, self.intermediate_size
        M_p = pad_to_multiple(M, self.TILE_M)
        K_p = pad_to_multiple(K, self.TILE_K)
        N_p = pad_to_multiple(N, self.TILE_N)

        xp = x2d
        if M_p > M or K_p > K:
            xp = torch.zeros((M_p, K_p), dtype=torch.float32, device=x.device)
            xp[:M, :K] = x2d

        wp = self._fc1_wt
        if K_p > K or N_p > N:
            wp = torch.zeros((K_p, N_p), dtype=torch.float32, device=x.device)
            wp[:K, :N] = self._fc1_wt

        inter = torch.zeros((M_p, N_p), dtype=torch.float32, device=x.device)
        grid  = (triton.cdiv(M_p, self.TILE_M), triton.cdiv(N_p, self.TILE_N))
        linear_gelu_kernel[grid](
            xp, wp, inter,
            M_p, N_p, K_p,
            xp.stride(0), xp.stride(1),
            wp.stride(0), wp.stride(1),
            inter.stride(0), inter.stride(1),
            BLOCK_M=self.TILE_M, BLOCK_N=self.TILE_N, BLOCK_K=self.TILE_K,
        )
        if M_p > M or N_p > N:
            inter = inter[:M, :N]

        if self.bias_enabled and self.fc1.bias_param is not None:
            if self.fc1.bias_param.device != x.device:
                self.fc1.bias_param = self.fc1.bias_param.to(x.device)
            inter = inter + self.fc1.bias_param

        return self.fc2(inter.reshape(*orig[:-1], self.intermediate_size))


# ============================================================================
# Smoke Test
# ============================================================================

if __name__ == "__main__":
    print("Testing H200-optimised layers v2...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    x = torch.randn(2, 16, 256, device=device)

    print("RMSNorm:", RMSNorm(256)(x).shape)
    print("LayerNorm:", LayerNorm(256)(x).shape)
    print("GELU:", gelu(x).shape)
    print("SiLU:", silu(x).shape)

    lin = Linear(256, 512)
    print("Linear:", lin(x).shape)

    print("\nFused QKV:")
    qkv = FusedQKVLinear(256, 128, 64, 64)
    q, k, v = qkv(x)
    print(f"  Q:{q.shape} K:{k.shape} V:{v.shape}")

    print("\nMLP (SwiGLU):", MLP(256, 512)(x).shape)
    print("EncoderMLP:", EncoderMLP(256, 512)(x).shape)
    print("\nAll OK!")