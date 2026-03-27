#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Quick Triton sanity check:
- Verifies imports and CUDA availability
- Prints GPU info
- Compiles & runs a minimal Triton kernel (vector add)
- Validates results against torch

Install (typical):
  pip install triton torch
"""

from __future__ import annotations

import sys
import traceback


class Colors:
    RESET = "\033[0m"
    BOLD = "\033[1m"
    GREEN = "\033[32m"
    YELLOW = "\033[33m"
    RED = "\033[31m"
    CYAN = "\033[36m"

    @staticmethod
    def ok(text: str) -> str:
        return f"{Colors.BOLD}{Colors.GREEN}[OK]{Colors.RESET} {text}"

    @staticmethod
    def info(text: str) -> str:
        return f"{Colors.BOLD}{Colors.CYAN}[INFO]{Colors.RESET} {text}"

    @staticmethod
    def warn(text: str) -> str:
        return f"{Colors.BOLD}{Colors.YELLOW}[WARN]{Colors.RESET} {text}"

    @staticmethod
    def fail(text: str) -> str:
        return f"{Colors.BOLD}{Colors.RED}[FAIL]{Colors.RESET} {text}"

    @staticmethod
    def passed(text: str) -> str:
        return f"{Colors.BOLD}{Colors.GREEN}[PASS]{Colors.RESET} {text}"


def _try_imports():
    try:
        import torch  # noqa: F401
    except Exception as e:
        raise RuntimeError("Failed to import torch. Install: pip install torch") from e

    try:
        import triton  # noqa: F401
        import triton.language as tl  # noqa: F401
    except Exception as e:
        raise RuntimeError("Failed to import triton. Install: pip install triton") from e


def _gpu_checks():
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available. Check your GPU driver and PyTorch build.")

    dev = torch.cuda.current_device()
    name = torch.cuda.get_device_name(dev)
    cap = torch.cuda.get_device_capability(dev)
    return dev, name, cap


def _triton_vector_add_selftest():
    import torch
    import triton
    import triton.language as tl

    @triton.jit
    def vector_add_kernel(X, Y, Z, N, BLOCK: tl.constexpr):
        pid = tl.program_id(0)
        offs = pid * BLOCK + tl.arange(0, BLOCK)
        mask = offs < N
        x = tl.load(X + offs, mask=mask, other=0.0)
        y = tl.load(Y + offs, mask=mask, other=0.0)
        tl.store(Z + offs, x + y, mask=mask)

    N = 2**12
    BLOCK = 256

    x = torch.randn(N, device="cuda", dtype=torch.float32)
    y = torch.randn(N, device="cuda", dtype=torch.float32)
    z = torch.empty_like(x)

    grid = (triton.cdiv(N, BLOCK),)
    vector_add_kernel[grid](x, y, z, N, BLOCK=BLOCK)

    torch.testing.assert_close(z.cpu(), (x + y).cpu(), rtol=1e-5, atol=1e-6)


def main():
    print(f"{Colors.BOLD}== Triton quick check =={Colors.RESET}")

    # 1) Imports
    try:
        _try_imports()
        print(Colors.ok("imports: torch / triton"))
    except Exception as e:
        print(Colors.fail(f"imports: {e}"))
        return 2

    # 2) GPU checks
    try:
        dev, name, cap = _gpu_checks()
        print(Colors.info(f"GPU {dev}: {name}"))
        print(Colors.info(f"Compute Capability: {cap[0]}.{cap[1]}"))
        if cap[0] < 7:
            print(Colors.warn("This GPU is older than Volta (sm70). Triton may be unsupported or slow."))
    except Exception as e:
        print(Colors.fail(f"GPU checks: {e}"))
        return 3

    # 3) Run a minimal Triton kernel
    try:
        _triton_vector_add_selftest()
        print(Colors.passed("Triton kernel ran successfully and results matched torch"))
        return 0
    except Exception as e:
        print(Colors.fail("Triton kernel self-test failed."))
        print(f"Reason: {e}")
        print("--- traceback ---")
        traceback.print_exc()
        return 4


if __name__ == "__main__":
    sys.exit(main())
