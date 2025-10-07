# SM120 Grouped GEMM - Current State

## Summary

**Baseline**: 234/1200 test failures (original SM120 code)
**Current**: 213/1200 failures (21 fewer with separate tensor approach)
**Progress**: ~9% improvement, but still not working completely

## What We've Tried

1. ❌ Using `m` instead of `max_m` for MN-major → Made things worse
2. ❌ 3-parameter ScaleConfig → 611 failures (made things much worse)
3. ❌ Dynamic stride-based UMMA::Major selection → 607 failures
4. ✅ Separate tensor restructuring → **213 failures** (best so far, 9% improvement)

## Current Implementation

### Python (`flashinfer/gemm.py`, lines 2765-2828)
- Detects multi-group on SM120
- Extracts each group's scales from concatenated tensor
- Clones to create independent memory
- Stacks into (num_groups, ...) format
- Passes restructured tensors to C++

### C++ (`group_gemm_fp8_groupwise_sm120.cuh`, lines 70-83)
- Uses `i * sf_m * sf_k` for K-major pointer calculation
- Uses `i * sf_k * sf_m` for MN-major pointer calculation
- Layout uses actual group's `m` (not `max_m`)

## Why It's Still Not Working

The separate tensor approach IS helping (21 fewer failures), but:
1. Still have ~200 failing cases
2. Error rates 0.1-3.9% (better than original 3-90%)
3. Pattern unclear - no obvious dimension or mode correlation

## Possible Remaining Issues

1. **Single-group path incompatibility**: C++ now uses new pointer arithmetic for ALL cases, but Python only restructures for multi-group
2. **Incorrect stacking dimension**: Maybe torch.stack isn't creating the right memory layout
3. **Missing ScaleMajorK handling**: The separate tensors may need different organization for K vs MN modes
4. **Fundamental CUTLASS limitation**: SM120 might truly require the CUTLASS example's exact architecture (completely separate allocations per group, not stacked)

## Files Modified

- `flashinfer/gemm.py` - Lines 2765-2828 (separate tensor restructuring)
- `include/flashinfer/gemm/group_gemm_fp8_groupwise_sm120.cuh` - Lines 59, 73-83 (new pointer arithmetic)

## Next Steps to Consider

1. **Debug single specific failing case** to understand exact issue
2. **Compare memory layout** of our stacked tensors vs what CUTLASS expects
3. **Try completely independent allocations** (not stacked) per group
4. **Consult CUTLASS team** or create minimal repro to verify if our approach is supported
5. **Consider alternative**: Use TRT-LLM backend or other CUTLASS kernel schedule if available

## Test Commands

```bash
# Quick test (10 cases, ~30-50 sec)
bash quick_test.sh

# Full test (~2 min)
pytest tests/GEMM/test_groupwise_scaled_gemm_fp8.py::test_fp8_groupwise_group_gemm --tb=no -q

# Single failing case for debugging
pytest -xvs tests/GEMM/test_groupwise_scaled_gemm_fp8.py::test_fp8_groupwise_group_gemm[out_dtype0-MN-2-128-4096-4]
```

## Recommendation

Given the time invested and complexity, recommend:
1. Document current state (this file)
2. Create detailed bug report for deeper investigation
3. Consider reaching out to CUTLASS team with our findings
4. May need to accept that SM120's grouped GEMM has architectural constraints we can't easily work around
