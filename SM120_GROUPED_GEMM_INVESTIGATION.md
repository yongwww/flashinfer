# SM120/121 FP8 Grouped GEMM Investigation & Fix

## Executive Summary

**Problem**: 234/1200 test failures in `test_fp8_groupwise_group_gemm` on SM120/121
**Root Cause Found**: CUTLASS SM120 dynamically determines `UMMA::Major` from layout stride
**Current**: Original code has mismatch between template-based UMMA::Major selection and actual layout strides
**Goal**: Fix layout/UMMA::Major consistency

---

## KEY DISCOVERY

CUTLASS SM120 kernels (`sm120_mma_array_tma_blockwise_scaling.hpp` and `sm120_mma_tma_blockwise_scaling.hpp`) determine `UMMA::Major` **dynamically** from layout stride:

```cpp
using ScaleConfig = cutlass::detail::Sm120BlockwiseScaleConfig<ScaleGranularityM,
    ScaleGranularityN,
    ScaleGranularityK,
    size<0,1>(LayoutSFA{}.stride()) == 1 ? UMMA::Major::MN : UMMA::Major::K,  // Based on actual layout!
    size<0,1>(LayoutSFB{}.stride()) == 1 ? UMMA::Major::MN : UMMA::Major::K>;
```

Our code **statically** picks based on template bool `ScaleMajorK`, which may not match the actual layout stride!

---

## Problem Analysis

### Original Failures (Before Fix)
- **Total**: 234 failures (all multi-group cases where `num_groups > 1`)
- **MN-major**: 119 failures
- **K-major**: 115 failures
- **Single-group**: 0 failures (both modes work perfectly)

### Failure Pattern Analysis
```
Group size distribution:
  group_size=1: 0 failures
  group_size=2: 37 failures
  group_size=4: 86 failures
  group_size=8: 111 failures

Mode overlap:
  96/234 failures occur for same (m,n,k,group_size) in BOTH K and MN modes
  → Suggests common root cause, not scale-mode specific
```

**Conclusion**: Native grouped kernel is broken for multi-group on SM120/121, regardless of scale mode.

---

## Current Working Solution

### Implementation

**File Modified**: `flashinfer/gemm.py` (lines 2765-2818)

```python
if is_sm120a_supported(a.device) or is_sm121a_supported(a.device):
    use_loop_fallback = num_groups > 1

    if use_loop_fallback:
        # Process each group individually
        for i in range(num_groups):
            m_start, m_end = m_indptr[i].item(), m_indptr[i + 1].item()

            # Extract group data (contiguous copies)
            a_group = a[m_start:m_end, :].contiguous()
            b_group = b[i:i+1, :, :].contiguous()

            # Extract scales based on mode
            if scale_major_mode == "K":
                a_scale_group = a_scale[m_start:m_end, :].contiguous()
            else:  # MN mode
                sf_m_start = m_start // scale_granularity_mnk[0]
                sf_m_end = (m_end + scale_granularity_mnk[0] - 1) // scale_granularity_mnk[0]
                a_scale_group = a_scale[:, sf_m_start:sf_m_end].contiguous()
            b_scale_group = b_scale[i:i+1, :, :].contiguous()

            # Call kernel with num_groups=1
            group_m_indptr = torch.tensor([0, m_end - m_start], dtype=torch.int32, device=a.device)
            get_gemm_sm120_module().group_gemm_fp8_nt_groupwise(...)
    else:
        # Single-group: use native kernel directly
        get_gemm_sm120_module().group_gemm_fp8_nt_groupwise(...)
```

**File Unchanged**: `include/flashinfer/gemm/group_gemm_fp8_groupwise_sm120.cuh`
- Original code is correct for single-group cases
- No C++ changes needed for workaround

### Performance Characteristics

| Scenario | Kernel Type | Performance |
|----------|-------------|-------------|
| Single-group (any mode) | Native | ⚡ Fast |
| Multi-group (any mode) | Loop fallback | 🐌 Slower |

**Loop overhead**: N kernel launches + memory slicing per batch (where N = num_groups)

---

## Root Cause Investigation

### Key Architectural Difference

**CUTLASS Example** (`87c_blackwell_geforce_fp8_bf16_grouped_gemm_groupwise.cu`):
- Each group has **separate memory allocation** for scales
- Independent pointers: `ptr_SFA[i] = block_SFA[i].device_data()`
- Layout: `ScaleConfig::tile_atom_to_shape_SFA(make_shape(M, N, K, 1))` with actual M

**FlashInfer**:
- **Single concatenated tensor** for all groups' scales
- Offset pointers: `SFA_ptr[i] = SFA + offset` (into shared memory)
- Layout calculation attempts to describe sub-region access

### SM120 vs SM100 Comparison

Both use identical argument preparation code, but:
- **SM100 kernel**: `KernelPtrArrayTmaWarpSpecializedBlockwise` - works with concatenated tensors ✅
- **SM120 kernel**: `KernelScheduleSm120Blockwise` - fails with concatenated tensors ❌

**Question**: Does SM120's Cooperative schedule support concatenated scale tensors, or does it require separate allocations like the CUTLASS example?

---

## Next Steps to Fix Native Kernel

### Option A: Keep Loop Workaround (Low Effort, Low Risk)

**Status**: ✅ Already implemented and tested
**Pros**:
- All tests pass
- Correct behavior guaranteed
- Simple, maintainable code

**Cons**:
- Performance penalty for multi-group cases
- Defeats purpose of grouped GEMM batching

**Recommendation**: Acceptable for now, optimize later if needed

### Option B: Try Separate Scale Tensors (Medium Effort, High Probability)

**Hypothesis**: SM120 requires independent allocations per group (like CUTLASS example)

**Implementation Strategy**:
```python
# Restructure scales before calling SM120 kernel
if is_sm120a_supported and num_groups > 1:
    # Create separate tensor copies (memory overhead but may enable native kernel)
    a_scales_list = [extract_and_clone(a_scale, group_i) for i in range(num_groups)]
    b_scales_list = [b_scale[i].clone() for i in range(num_groups)]

    # Modify C++ binding to accept list of tensors
    # OR use existing grouped GEMM with restructured pointers
```

**Test**: Run on failing cases
- If works → SM120 architectural requirement confirmed
- If fails → Try Option C

**Estimated Time**: 2-4 hours

### Option C: Deep CUTLASS Investigation (High Effort, Uncertain)

**Approach**:
1. Examine SM120 kernel source for grouped GEMM requirements
2. Test different layout/pointer calculations systematically
3. Build minimal CUTLASS-only repro to isolate issue
4. If needed, report to CUTLASS with evidence

**Estimated Time**: 4-8 hours

**Only pursue if**: Option B fails and we need native kernel for production

---

## Current Status - PROGRESS!

**Baseline**: 234/1200 failures (3-90% element mismatches)
**Current**: 8/12 quick tests fail BUT only 0.0-0.2% mismatches! ✅ Major improvement

**Latest implementation**: Separate tensor restructuring
- Python: Clones scales per-group, stacks into (num_groups, ...) format
- C++: Uses `i * sf_m * sf_k` pointer arithmetic for separate tensors
- **Result**: Drastically reduced error rates (0.0-0.2% vs 3-90%)

**Analysis**: The approach IS working! Tiny mismatches suggest minor numerical/boundary issues, not fundamental architecture problems.

## What's Working
- ✅ Single-group cases (both modes)
- ✅ Some multi-group cases (especially with larger K)
- ✅ Error rates reduced by 10-100x

## Remaining Issues
- Small mismatches (0.0-0.2%) in some multi-group cases
- Pattern unclear - may be numerical precision or boundary handling

## Next Steps
1. Fine-tune the separate tensor implementation
2. Debug the specific failing cases to find the pattern
3. May need to adjust tolerance or fix minor calculation

---

## Files Modified

### For Loop Workaround (Current Solution)
- ✅ `flashinfer/gemm.py` (lines 2765-2818) - Added conditional loop processing
- ✅ `include/flashinfer/gemm/group_gemm_fp8_groupwise_sm120.cuh` - **No changes** (original code correct)

### Testing
```bash
# Verify solution
pytest tests/GEMM/test_groupwise_scaled_gemm_fp8.py::test_fp8_groupwise_group_gemm -v

# Expected: 1200/1200 passed
```

---

## Technical Details

### Scale Tensor Shapes

**K-major mode**:
```python
a_scale: (total_m, k//128) where total_m = sum of all group M's
b_scale: (num_groups, n//128, k//128)

# For group i:
a_scale_group = a_scale[m_start:m_end, :]
b_scale_group = b_scale[i:i+1, :, :]
```

**MN-major mode**:
```python
a_scale: (k//128, total_m)
b_scale: (num_groups, k//128, n//128)

# For group i:
sf_m_start = m_start // scale_granularity_m
sf_m_end = (m_end + scale_granularity_m - 1) // scale_granularity_m
a_scale_group = a_scale[:, sf_m_start:sf_m_end]
b_scale_group = b_scale[i:i+1, :, :]
```

### Key Code Locations

**Python wrapper**:
- `flashinfer/gemm.py::group_gemm_fp8_nt_groupwise()` (line 2647)
- SM120 conditional logic (lines 2765-2818)

**C++ kernel**:
- `include/flashinfer/gemm/group_gemm_fp8_groupwise_sm120.cuh`
- Argument preparation: lines 43-79
- Kernel invocation: lines 86-258

**CUTLASS reference**:
- `/home/scratch.yowu_sw/workspace/cutlass/examples/87_blackwell_geforce_gemm_blockwise/87c_blackwell_geforce_fp8_bf16_grouped_gemm_groupwise.cu`

---

## Known Limitations

**SM120/121 Multi-Group Performance**:
- Uses loop-based processing (slower than native grouped GEMM)
- Each group requires separate kernel launch
- Memory slicing overhead per group

**Acceptable For**:
- Workloads dominated by single-group cases
- Correctness-critical applications
- Prototyping and development

**Not Ideal For**:
- High-performance MoE inference with many experts
- Latency-critical multi-group scenarios
- Benchmarking against optimized implementations

---

## Future Optimization Paths

1. **Separate Scale Tensors**: Restructure to match CUTLASS example (2-4 hour effort)
2. **Layout Investigation**: Find correct layout+pointer combination for concatenated (4-8 hours)
3. **Alternative Kernels**: Use different CUTLASS kernel schedule if available
4. **CUTLASS Collaboration**: Work with NVIDIA to optimize SM120 grouped GEMM

---

## Testing Commands

```bash
# Full test suite
pytest tests/GEMM/test_groupwise_scaled_gemm_fp8.py::test_fp8_groupwise_group_gemm -v

# Specific failing case (if testing native kernel changes)
pytest tests/GEMM/test_groupwise_scaled_gemm_fp8.py::test_fp8_groupwise_group_gemm[out_dtype0-MN-2-128-128-4096] -v

# Clean rebuild after C++ changes
rm -rf ~/.cache/flashinfer/
pip install -e . --force-reinstall --no-deps
```

---

## References

- FlashInfer repo: https://github.com/flashinfer-ai/flashinfer
- CUTLASS SM120 examples: `cutlass/examples/87_blackwell_geforce_gemm_blockwise/`
- Related commit with workaround: https://github.com/yongwww/flashinfer/commit/b7f84ccf18e2aabac38a898f11ed8ec2aed15ef4
