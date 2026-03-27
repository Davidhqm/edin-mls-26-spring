import cupy as cp
import numpy as np
import cuda.tile as ct

@ct.kernel
def mixed_precision_scale(input_ptr, output_ptr, scale_factor, tile_size: ct.Constant[int]):
    # Get Block ID
    pid = ct.bid(0)

    # 1. Load FP16 Data
    # 'input_ptr' is a float16 array.
    # ct.load will load this into a Tile.
    input_tile = ct.load(input_ptr, index=(pid,), shape=(tile_size,))

    # 2. Compute in FP32
    # The scale_factor is passed as a standard float (FP32).
    # Operations between the FP16 tile and FP32 scalar will promote the tile to FP32.
    # This is standard Python/NumPy behavior preserved in cuTile.
    result_fp32 = input_tile * scale_factor

    # 3. Convert back to FP16 and Store
    # 'output_ptr' is float16.
    # cuTile requires explicit casting - use ct.astype() to convert FP32 result to FP16.
    result_tile = ct.astype(result_fp32, ct.float16)
    ct.store(output_ptr, index=(pid,), tile=result_tile)

def test_data_model():
    N = 1024
    TILE_SIZE = 32
    
    # Setup data in Half Precision (float16)
    data_in = cp.random.uniform(-10, 10, N).astype(cp.float16)
    data_out = cp.zeros_like(data_in)
    
    # Scale factor
    factor = 2.5
    
    grid = (ct.cdiv(N, TILE_SIZE), 1, 1)
    
    print(f"Input Type: {data_in.dtype}")
    print(f"Output Type: {data_out.dtype}")
    print(f"Scale Factor: {factor} (float)")

    # Launch
    ct.launch(cp.cuda.get_current_stream(),
              grid,
              mixed_precision_scale,
              (data_in, data_out, factor, TILE_SIZE))

    # Verify
    # We verify in FP32 on CPU to check precision
    h_in = cp.asnumpy(data_in).astype(np.float32)
    h_out = cp.asnumpy(data_out).astype(np.float32)
    expected = h_in * factor
    
    # Comparison
    # Note: FP16 has lower precision, so we allow a larger epsilon
    np.testing.assert_allclose(h_out, expected, rtol=1e-3, atol=1e-3)
    
    print("✓ Mixed Precision Test Passed!")

if __name__ == "__main__":
    test_data_model()
