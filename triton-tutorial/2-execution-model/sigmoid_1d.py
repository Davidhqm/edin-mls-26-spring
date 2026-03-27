import torch
import triton
import triton.language as tl


@triton.jit
def sigmoid_kernel(X, Y, N, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < N
    x = tl.load(X + offs, mask=mask, other=0.0)
    y = 1.0 / (1.0 + tl.exp(-x))
    tl.store(Y + offs, y, mask=mask)


def test_sigmoid_1d():
    N = 1024
    BLOCK = 256

    print(f"Vector Size: {N}")
    print(f"Block Size: {BLOCK}")
    print(f"Number of Programs: {triton.cdiv(N, BLOCK)}")

    x = torch.linspace(-5, 5, N, device="cuda", dtype=torch.float32)
    y = torch.empty_like(x)

    grid = (triton.cdiv(N, BLOCK),)
    sigmoid_kernel[grid](x, y, N, BLOCK=BLOCK)

    # Verify against torch reference
    expected = torch.sigmoid(x).cpu()
    out = y.cpu()

    print("\nVerifying output...")
    print(f"Input range: [{x.min().item():.2f}, {x.max().item():.2f}]")
    print(f"Output range: [{out.min().item():.4f}, {out.max().item():.4f}]")
    print("\nSample values:")
    print(f"  sigmoid(-5.0) = {out[0].item():.6f} (expected: {expected[0].item():.6f})")
    print(f"  sigmoid(0.0)  = {out[N//2].item():.6f} (expected: {expected[N//2].item():.6f})")
    print(f"  sigmoid(5.0)  = {out[-1].item():.6f} (expected: {expected[-1].item():.6f})")

    torch.testing.assert_close(out, expected, rtol=1e-5, atol=1e-5)
    print("\n[PASS] 1D Sigmoid Test Passed!")


if __name__ == "__main__":
    test_sigmoid_1d()
