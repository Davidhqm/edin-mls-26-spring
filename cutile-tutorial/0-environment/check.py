#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Quick cuTile (cuda-tile) sanity check:
- Verifies prerequisites (GPU CC, driver/runtime versions)
- Compiles & runs a minimal cuTile kernel (vector add)
- Validates results against NumPy

Install (per NVIDIA quickstart):
  pip install cuda-tile
  pip install cupy-cuda13x
  pip install numpy
Refs:
  https://docs.nvidia.com/cuda/cutile-python/quickstart.html
"""

from __future__ import annotations

import sys
import traceback
import warnings


# Minimum requirements for cuTile Python
MIN_CUDA_VERSION = 13000  # CUDA 13.0
MIN_DRIVER_VERSION = 570  # Driver 570.x


# ANSI color codes
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


def _format_cuda_version(ver: int) -> str:
    """Convert numeric CUDA version (e.g., 13010) to human-readable (e.g., 13.1.0)."""
    major = ver // 1000
    minor = (ver % 1000) // 10
    patch = ver % 10
    return f"{major}.{minor}.{patch}"


def _format_driver_version(ver_str: str) -> tuple[str, int]:
    """Parse driver version string and return (formatted_str, major_int)."""
    try:
        major = int(ver_str.split(".")[0])
        return ver_str, major
    except (ValueError, IndexError):
        return ver_str, 0


def _try_imports():
    try:
        import cupy as cp  # noqa: F401
    except Exception as e:
        raise RuntimeError(
            "Failed to import cupy. Install: pip install cupy-cuda13x"
        ) from e

    try:
        import cuda.tile as ct  # noqa: F401
    except Exception as e:
        raise RuntimeError(
            "Failed to import cuda.tile (cuda-tile). Install: pip install cuda-tile"
        ) from e

    try:
        import numpy as np  # noqa: F401
    except Exception as e:
        raise RuntimeError(
            "Failed to import numpy. Install: pip install numpy"
        ) from e


def _version_checks():
    import cupy as cp

    # Driver/runtime version numbers are typically like 13010 for 13.1, 12040 for 12.4, etc.
    drv = cp.cuda.runtime.driverGetVersion()
    rtm = cp.cuda.runtime.runtimeGetVersion()

    # Try to also read NVML driver string (optional, nicer "580.xx")
    nvml_str = None
    try:
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message=".*pynvml.*deprecated.*")
            import pynvml  # type: ignore

        pynvml.nvmlInit()
        ver = pynvml.nvmlSystemGetDriverVersion()
        nvml_str = ver.decode("utf-8", errors="ignore") if isinstance(ver, bytes) else str(ver)
        pynvml.nvmlShutdown()
    except Exception:
        pass

    return drv, rtm, nvml_str


def _gpu_checks():
    import cupy as cp

    n = cp.cuda.runtime.getDeviceCount()
    if n <= 0:
        raise RuntimeError("No CUDA device found.")

    dev_id = cp.cuda.runtime.getDevice()
    prop = cp.cuda.runtime.getDeviceProperties(dev_id)

    name = prop.get("name", b"").decode("utf-8", errors="ignore") if isinstance(prop.get("name"), (bytes, bytearray)) else str(prop.get("name"))
    major = int(prop.get("major", -1))
    minor = int(prop.get("minor", -1))

    return dev_id, name, major, minor


def _cutile_vector_add_selftest():
    import cupy as cp
    import numpy as np
    import cuda.tile as ct

    # cuTile evaluates type hints in module globals; ensure ct is visible there.
    globals()["ct"] = ct

    @ct.kernel
    def vector_add(a, b, c, tile_size: ct.Constant[int]):
        pid = ct.bid(0)  # 1D block-id / tile-processor id

        a_tile = ct.load(a, index=(pid,), shape=(tile_size,))
        b_tile = ct.load(b, index=(pid,), shape=(tile_size,))
        result = a_tile + b_tile
        ct.store(c, index=(pid,), tile=result)

    vector_size = 2**12
    tile_size = 2**4
    grid = (ct.cdiv(vector_size, tile_size), 1, 1)

    a = cp.random.uniform(-1, 1, vector_size).astype(cp.float32)
    b = cp.random.uniform(-1, 1, vector_size).astype(cp.float32)
    c = cp.zeros_like(a)

    # Launch on current CuPy stream
    ct.launch(
        cp.cuda.get_current_stream(),
        grid,
        vector_add,
        (a, b, c, tile_size),
    )

    # Validate
    a_np = cp.asnumpy(a)
    b_np = cp.asnumpy(b)
    c_np = cp.asnumpy(c)
    np.testing.assert_allclose(c_np, a_np + b_np, rtol=1e-5, atol=1e-6)


def main():
    print(f"{Colors.BOLD}== cuTile Python quick check =={Colors.RESET}")

    # 1) Imports
    try:
        _try_imports()
        print(Colors.ok("imports: cupy / cuda.tile / numpy"))
    except Exception as e:
        print(Colors.fail(f"imports: {e}"))
        return 2

    # 2) Version checks
    try:
        drv, rtm, nvml_str = _version_checks()

        # Format and check CUDA runtime version
        rtm_str = _format_cuda_version(rtm)
        rtm_ok = rtm >= MIN_CUDA_VERSION
        rtm_compat = "compatible" if rtm_ok else f"requires >= {_format_cuda_version(MIN_CUDA_VERSION)}"
        msg = f"CUDA runtime: {rtm_str} ({rtm_compat})"
        print(Colors.ok(msg) if rtm_ok else Colors.warn(msg))

        # Format and check CUDA driver version (from runtime API)
        drv_str = _format_cuda_version(drv)
        drv_ok = drv >= MIN_CUDA_VERSION
        drv_compat = "compatible" if drv_ok else f"requires >= {_format_cuda_version(MIN_CUDA_VERSION)}"
        msg = f"CUDA driver API: {drv_str} ({drv_compat})"
        print(Colors.ok(msg) if drv_ok else Colors.warn(msg))

        # Format and check NVIDIA driver version (from NVML)
        if nvml_str:
            _, nvml_major = _format_driver_version(nvml_str)
            nvml_ok = nvml_major >= MIN_DRIVER_VERSION
            nvml_compat = "compatible" if nvml_ok else f"requires >= {MIN_DRIVER_VERSION}.x"
            msg = f"NVIDIA driver: {nvml_str} ({nvml_compat})"
            print(Colors.ok(msg) if nvml_ok else Colors.warn(msg))
    except Exception as e:
        print(Colors.warn(f"version checks failed: {e}"))

    # 3) GPU checks
    try:
        dev_id, name, major, minor = _gpu_checks()
        print(Colors.info(f"GPU {dev_id}: {name}"))
        print(Colors.info(f"Compute Capability: {major}.{minor}"))

        # cuTile Python quickstart says CC must be 10.x or 12.x
        if major not in (10, 12):
            print(Colors.fail("This GPU CC is not 10.x/12.x; cuTile Python is expected to be unsupported here."))
            return 3
        print(Colors.ok("GPU compute capability meets cuTile Python requirement (10.x/12.x)."))
    except Exception as e:
        print(Colors.fail(f"GPU checks: {e}"))
        return 4

    # 4) Run a minimal cuTile kernel
    try:
        _cutile_vector_add_selftest()
        print(Colors.passed("cuTile kernel ran successfully and results matched NumPy"))
        return 0
    except Exception as e:
        print(Colors.fail("cuTile kernel self-test failed."))
        print(f"Reason: {e}")
        print("--- traceback ---")
        traceback.print_exc()
        return 5


if __name__ == "__main__":
    sys.exit(main())
