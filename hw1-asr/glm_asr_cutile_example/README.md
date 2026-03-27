# Example - Initial CuPy Implementation

## Performance: baseline

## Key Characteristics:
1. Pure CuPy tensor operations for compute
2. Basic CuTile linear kernel with 16x16 tiles
3. Standard scaled_dot_product_attention using einsum

## Implementation Details:
- Uses CuPy's `einsum` for attention: `scores = cp.einsum('bnqd,bnkd->bnqk', q, k)`
- High memory bandwidth from large intermediate tensors
- Lack of kernel fusion optimizations
- No tensor core utilization

## Bottlenecks:
- Materializing full attention matrix (O(n^2) memory)
- Non-optimized attention computation
- Small tile sizes causing excessive kernel launch overhead

## Key Files:
- attention.py: Standard scaled dot-product attention (einsum fallback)
- layers.py: Basic CuTile linear kernel
