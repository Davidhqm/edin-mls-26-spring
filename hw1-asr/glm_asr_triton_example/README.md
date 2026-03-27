# Example - Initial Triton Implementation

## Performance: TBD (baseline)

## Key Characteristics:
1. Pure Torch tensor operations for compute
2. Triton kernels for linear, norm, activation, and attention (small sizes)
3. Standard scaled_dot_product_attention using einsum fallback

## Implementation Details:
- Uses Torch `einsum` fallback for large attention: `scores = torch.einsum('bnqd,bnkd->bnqk', q, k)`
- Triton kernels for small attention dimensions (<=256)
- No aggressive kernel fusion enabled in baseline

## Bottlenecks:
- Materializing full attention matrix (O(n^2) memory)
- Non-optimized attention computation for large seq_len
- Kernel launch overhead for small tensors

## Key Files:
- attention.py: Triton attention kernels with Torch fallback
- layers.py: Triton kernels for core layers
