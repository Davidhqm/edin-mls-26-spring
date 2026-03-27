# Attention Mechanism

This tutorial brings everything together to implement a simplified version of the **Attention Mechanism** using Triton.

## The Math

Standard Attention is defined as:

$$ \text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V $$

In this simplified tutorial, we implement:

$$ \text{Out} = \exp\left(\frac{QK^T}{\sqrt{d_k}}\right)V $$

(We skip the normalization step to keep the kernel readable.)

## Key Concepts Demonstrated

1. **Complex Data Flow**: Loading three different matrices (Q, K, V).
2. **Tiled Matrix Multiplication**:
   - `scores = Q @ K.T`
   - `weighted = scores @ V`
3. **Kernel Loops**: Iterating over the Key/Value sequence length (`N`) in chunks (`BLOCK_N`).
4. **Math Functions**: Using `tl.exp` on tiles.

## Execution Flow

1. **Grid**: We launch one program per Query block (`BLOCK_M`).
2. **Loop**: Each program iterates over `K/V` blocks to accumulate the output.
3. **Compute**:
   - Load a block of K and V.
   - Compute similarity between our Q-block and the current K-block.
   - Apply `exp`.
   - Multiply by V-block.
   - Accumulate into the result.
4. **Store**: Write the accumulated result back to global memory.

## Notes

- `HEAD_DIM` and `SEQ_LEN_K` are passed as **compile-time constants** (Triton `tl.constexpr`) so the loop is static.
- This kernel is a teaching example, not a production FlashAttention implementation.
