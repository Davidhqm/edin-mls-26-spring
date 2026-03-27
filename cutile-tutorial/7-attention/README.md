# Attention Mechanism

This tutorial brings everything together to implement a simplified version of the **Attention Mechanism** used in Transformers (like LLMs).

## The Math

Standard Attention is defined as:

$$ \text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V $$

In our simplified implementation (to avoid the complexity of online-softmax normalization logic in a tutorial), we implement:

$$ \text{Out} = \sum \exp\left(\frac{QK^T}{\sqrt{d_k}}\right)V $$

## Key Concepts Demonstrated

1.  **Complex Data Flow**: Loading three different matrices (Q, K, V).
2.  **Tiled Matrix Multiplication**:
    -   `score = q_tile @ k_tile.T`
    -   `weighted = score @ v_tile`
3.  **Kernel Loops**: Iterating over the Key/Value sequence length (`N`) in chunks (`TILE_N`) while holding a Query chunk (`TILE_M`) in registers. This is the foundation of **FlashAttention**.
4.  **Math Functions**: Using `ct.exp` (or `cp.exp` if overloaded) on tiles.

## Execution Flow

1.  **Grid**: We calculate how many tiles we need to cover the Query sequence (`M`). Each block handles `TILE_M` queries.
2.  **Loop**: Inside the kernel, we loop over the Key/Value sequence (`N`) in steps of `TILE_N`.
3.  **Compute**:
    -   Load a block of K and V.
    -   Compute similarity between our Q-block and the current K-block.
    -   Apply non-linearity (Exp).
    -   Multiply by V-block.
    -   Accumulate into the result.
4.  **Store**: Finally, write the accumulated result for the Q-block back to global memory.

This "Tiling" strategy allows us to compute Attention for very long sequences without creating the massive $(M \times N)$ attention matrix in global memory, saving huge amounts of memory.
