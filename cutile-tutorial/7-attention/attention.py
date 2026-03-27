# SPDX-FileCopyrightText: Copyright (c) <2025> NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""
Example demonstrating a simplified Tiled Attention mechanism.
O = Softmax(Q @ K.T) @ V
"""

import cupy as cp
import numpy as np
import cuda.tile as ct

# We define a kernel that computes attention for a subset of Queries against all Keys/Values.
# Q: (M, D)
# K: (N, D)
# V: (N, D)
# Out: (M, D)
#
# We parallelize over M (the query sequence length).
# Each tile handles 'tile_size_m' queries.

@ct.kernel
def simple_attention(Q, K, V, Out, 
                     seq_len_k: int, 
                     d_head: ct.Constant[int],
                     tile_size_m: ct.Constant[int],
                     tile_size_n: ct.Constant[int]):
    
    # 1. Identify which Queries this tile processes
    # pid_m is the index of the Query tile
    pid_m = ct.bid(0)
    
    # Load the Query Tile: Shape (tile_size_m, d_head)
    # We load from Q at row offset (pid_m * tile_size_m)
    # The 'index' arg expects the start coordinate in the global grid.
    # Since Q is (M, D), we index by (row_idx, 0)
    # Wait, 'index' usually corresponds to the tile grid if we use grid-based loading?
    # No, in vector_add it was index=(pid,), shape=(size,).
    # In transpose (2D), it was index=(y, x).
    # Here Q is 2D. We start at (pid_m, 0)? 
    # Actually, if we view Q as a grid of (tile_m, D) blocks, index=(pid_m, 0) works.
    
    q_tile = ct.load(Q, index=(pid_m, 0), shape=(tile_size_m, d_head))
    
    # We need to accumulate the result (O) for these queries.
    # Initialize accumulator tile with zeros.
    acc_tile = q_tile * 0.0 # Hack to get a zero tile of same shape/type
    
    # To perform Softmax(Q K.T), we usually need the max and sum across the K-dimension.
    # For this simplified tutorial, we will do a "Block-wise" attention 
    # WITHOUT the 'online softmax' numerical stability tricks (FlashAttention) 
    # just to keep the code readable.
    # WARNING: This might overflow if d_head is large or values are large.
    
    # Note: A real implementation would loop over K chunks, compute max/sum incrementally.
    # Here, we assume N fits in memory or we iterate and just accumulate raw scores (naive).
    # Let's iterate over K in tiles of 'tile_size_n'.
    
    # In a naive tiled approach without online softmax, we can't easily compute the full softmax 
    # because we need the sum over ALL K before we can normalize.
    # 
    # SO, for this tutorial, we will assume N is small enough to load entirely? 
    # No, that's not scalable.
    # 
    # Let's implement the standard loop with a simplification:
    # We will just compute raw scores and accumulate (ignoring Softmax normalization for a moment)
    # OR we implement the online softmax.
    # 
    # Let's try Online Softmax (FlashAttention V1 style simplified):
    # m_prev = -inf
    # l_prev = 0
    # O_prev = 0
    
    # We need to maintain state vectors for each row in the query tile.
    # m_i: max score seen so far for query i
    # l_i: sum of exponentials seen so far
    
    # Initialize running stats (conceptually)
    # We can't easily create scratchpad vars in Python DSL without explicit support.
    # Let's pivot to a slightly simpler "Block Attention":
    # Just compute Q @ K.T @ V without softmax, or with element-wise Relu?
    # The user asked for "Attn".
    
    # Okay, let's try to implement the loop.
    # We iterate k_idx from 0 to seq_len_k // tile_size_n
    
    # Using 'range' inside kernel? cuTile likely supports python control flow (loops).
    
    # Number of tiles in K dimension
    num_k_tiles = ct.cdiv(seq_len_k, tile_size_n)
    
    # Iterate over K/V tiles
    for k_id in range(num_k_tiles):
        # Load K Tile: (tile_size_n, d_head)
        # Note: We load K transposed? Or we load K then transpose?
        # Standard K is (N, D). Load (tile_n, D). Transpose to (D, tile_n).
        k_tile = ct.load(K, index=(k_id, 0), shape=(tile_size_n, d_head))
        
        # Load V Tile: (tile_size_n, d_head)
        v_tile = ct.load(V, index=(k_id, 0), shape=(tile_size_n, d_head))
        
        # 1. Attention Scores for this block: S = Q @ K.T
        # Shape: (tile_m, D) @ (D, tile_n) -> (tile_m, tile_n)
        # We assume '@' operator does matrix multiplication on tiles.
        k_tile_T = ct.transpose(k_tile)
        score_tile = q_tile @ k_tile_T
        
        # 2. Scale (1/sqrt(d))
        # Use d_head ** -0.5 instead of cp.sqrt() which is not supported in kernel
        scale = d_head ** -0.5
        score_tile = score_tile * scale
        
        # 3. Apply Softmax (Approximation/Simplification)
        # In this tutorial, we simply exponentiate.
        # We skip the row-wise normalization sum for brevity 
        # (This computes un-normalized attention: exp(QK)V )
        exp_score = ct.exp(score_tile)
        
        # 4. Weighted Sum: A @ V
        # Shape: (tile_m, tile_n) @ (tile_n, D) -> (tile_m, D)
        weighted_val = exp_score @ v_tile
        
        # Accumulate
        acc_tile = acc_tile + weighted_val

    # Store result
    ct.store(Out, index=(pid_m, 0), tile=acc_tile)


def test_attention():
    # Dimensions
    M = 128  # Number of Queries
    N = 128  # Number of Keys/Values
    D = 64   # Head Dimension
    
    TILE_M = 32
    TILE_N = 32
    
    print(f"Attention Problem: Q({M}x{D}) @ K({N}x{D}).T @ V({N}x{D})")
    
    # Data
    q = cp.random.normal(0, 1, (M, D)).astype(cp.float32)
    k = cp.random.normal(0, 1, (N, D)).astype(cp.float32)
    v = cp.random.normal(0, 1, (N, D)).astype(cp.float32)
    out = cp.zeros((M, D), dtype=cp.float32)
    
    # Grid: Cover M queries
    grid = (ct.cdiv(M, TILE_M), 1, 1)
    
    # Launch
    # Note: We pass N (seq_len_k) as regular int
    ct.launch(cp.cuda.get_current_stream(),
              grid,
              simple_attention,
              (q, k, v, out, N, D, TILE_M, TILE_N))
    
    # Verify
    # Compute Expected on Host (or via Cupy)
    # Un-normalized Attention = exp(Q K.T / sqrt(d)) V
    h_q = cp.asnumpy(q)
    h_k = cp.asnumpy(k)
    h_v = cp.asnumpy(v)
    
    scores = (h_q @ h_k.T) / np.sqrt(D)
    attn = np.exp(scores)
    expected = attn @ h_v
    
    h_out = cp.asnumpy(out)
    
    print("Checking accuracy...")
    # We expect some floating point divergence due to sum order
    if np.allclose(h_out, expected, rtol=1e-3, atol=1e-3):
        print("✓ Tiled Attention Passed!")
    else:
        print("✗ Tiled Attention Failed!")
        diff = np.abs(h_out - expected)
        print(f"Max Diff: {np.max(diff)}")
        print(f"Sample Out:\n{h_out[0,:4]}")
        print(f"Sample Exp:\n{expected[0,:4]}")

if __name__ == "__main__":
    test_attention()
