import torch
import triton
import triton.language as tl


@triton.jit
def mixed_precision_scale(X, Y, N, scale, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < N

    x = tl.load(X + offs, mask=mask, other=0.0)
    x_fp32 = x.to(tl.float32)

    y_fp32 = x_fp32 * scale
    y_fp16 = y_fp32.to(tl.float16)

    tl.store(Y + offs, y_fp16, mask=mask)


def test_data_model():
    N = 1024
    BLOCK = 256

    data_in = torch.empty((N,), device="cuda", dtype=torch.float16).uniform_(-10, 10)
    data_out = torch.empty_like(data_in)

    factor = 2.5

    grid = (triton.cdiv(N, BLOCK),)

    print(f"Input Type: {data_in.dtype}")
    print(f"Output Type: {data_out.dtype}")
    print(f"Scale Factor: {factor} (float)")

    mixed_precision_scale[grid](data_in, data_out, N, factor, BLOCK=BLOCK)

    h_in = data_in.cpu().float()
    h_out = data_out.cpu().float()
    expected = h_in * factor

    torch.testing.assert_close(h_out, expected, rtol=1e-3, atol=1e-3)
    print("[PASS] Mixed Precision Test Passed!")


if __name__ == "__main__":
    test_data_model()
