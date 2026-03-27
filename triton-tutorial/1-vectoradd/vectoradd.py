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


def test():
    # Create input data
    vector_size = 2**12
    block_size = 256

    a = torch.randn(vector_size, device="cuda", dtype=torch.float32)
    b = torch.randn(vector_size, device="cuda", dtype=torch.float32)
    c = torch.empty_like(a)

    grid = (triton.cdiv(vector_size, block_size),)

    # Launch kernel
    vector_add_kernel[grid](a, b, c, vector_size, BLOCK=block_size)

    # Verify results
    torch.testing.assert_close(c.cpu(), (a + b).cpu(), rtol=1e-5, atol=1e-6)
    print("[PASS] vector_add_example passed!")


if __name__ == "__main__":
    test()
