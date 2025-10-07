# SM120 Separate Tensor Implementation Plan

## Goal
Implement proper grouped GEMM support for SM120/121 using separate scale tensors per group (matching CUTLASS example architecture).

## Current vs Target Architecture

### Current (Concatenated - NOT WORKING)
```python
# Single concatenated tensors
a_scale: (k//128, total_m) for MN-major OR (total_m, k//128) for K-major
b_scale: (num_groups, k//128 or n//128, ...)

# Pass single tensor, kernel tries to slice
```

### Target (Separate - CUTLASS Pattern)
```python
# Independent tensors per group
a_scales_list = [a_scale_group_0, a_scale_group_1, ...]
b_scales_list = [b_scale_group_0, b_scale_group_1, ...]

# Each group gets its own tensor with independent memory
```

## Implementation Approach

### Python Side (`flashinfer/gemm.py`)

For SM120 multi-group, restructure scales before calling kernel:

```python
if is_sm120a_supported(a.device) and num_groups > 1:
    # Create separate scale tensors (clone to get independent memory)
    a_scales_separate = []
    b_scales_separate = []

    for i in range(num_groups):
        m_start, m_end = m_indptr[i].item(), m_indptr[i+1].item()

        if scale_major_mode == "K":
            a_scale_i = a_scale[m_start:m_end, :].clone()
        else:  # MN
            sf_start = m_start // scale_granularity_mnk[0]
            sf_end = (m_end + scale_granularity_mnk[0] - 1) // scale_granularity_mnk[0]
            a_scale_i = a_scale[:, sf_start:sf_end].clone()

        b_scale_i = b_scale[i].clone()

        a_scales_separate.append(a_scale_i)
        b_scales_separate.append(b_scale_i)

    # Now call kernel with restructured tensors
    # Need to pass list/concat differently
```

### C++ Side (`group_gemm_fp8_groupwise_sm120.cuh`)

Modify argument preparation to handle per-group scales:
- Each group's scale pointer points to independent memory
- Layout uses actual group's M (not max_m)
- Matches CUTLASS example pattern exactly

## Implementation Steps

1. ✅ Revert to baseline (done)
2. Modify C++ argument kernel to expect separate tensors
3. Modify Python to restructure scales
4. Test with quick_test.sh
5. Run full suite

Let's proceed!
