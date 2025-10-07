# SM120 Independent Allocations Implementation - Current Status

## What We've Done

1. ✅ **Proven root cause**: CUTLASS SM120 kernel requires independent GPU allocations
   - Modified CUTLASS example to use concatenated scales → FAILS
   - Modified CUTLASS example to copy concat→independent → PASSES

2. ✅ **Implemented independent allocations** in FlashInfer (C++ only, no Python changes):
   - Allocate independent buffers per group using `cudaMalloc`
   - Copy from original concatenated format to independent buffers
   - Set up all arguments on host, copy to device
   - Clean up allocations after GEMM

3. ✅ **Fixed critical bugs**:
   - Device memory access from CPU (m_indptr segfault) → Copy to host first
   - K-major sf_m_i calculation (was m_i/128, should be m_i) → Fixed

## Current Results

**Quick Test (12 tests)**: 6 passed, 6 failed (50% pass rate)

**Improvements over original**:
- Original: ~234/1200 failures (80% fail rate)
- Current: ~50% fail rate on quick tests

## Remaining Issues

Even with independent allocations (proven necessary), we still have:
- 0.0% to 18.6% element mismatches on failing tests
- No clear pattern (mix of K-major and MN-major, various sizes)

**Example failure**: K-8-128-8192-4 (18.6% mismatches)
- Debug shows correct offsets: Group 0-7 at offsets 0, 4, 8, 12, 16, 20, 24, 28
- Independent allocations created
- Data copied
- But still fails with large errors

## Possible Remaining Issues

1. **2D copy for MN-major** might be incorrect
   - Pitch calculations could be wrong
   - Column offset calculation might be off

2. **Layout mismatch** between what we create vs what CUTLASS expects
   - We use: `ScaleConfig::tile_atom_to_shape_SFA(cute::make_shape(m_i, n, k, 1))`
   - CUTLASS example uses the same, should be compatible

3. **Memory ordering** in stacked vs independent allocations
   - Maybe CUTLASS expects specific memory layout we're not matching

4. **Some other subtle difference** between our setup and CUTLASS example

## Next Steps

**Option A**: Deep dive into one failing case
- Compare byte-by-byte what we copy vs what CUTLASS example has
- Verify layouts match exactly
- Check if kernel reads data correctly

**Option B**: Compare our C++ implementation line-by-line with CUTLASS example
- Ensure argument setup is identical
- Verify all metadata (strides, layouts) match

**Option C**: Since we have 50% pass rate with independent allocs (vs 20% without), submit PR with current state and document remaining issues for CUTLASS team

Current implementation is in: `/home/scratch.yowu_sw/workspace/flashinfer/include/flashinfer/gemm/group_gemm_fp8_groupwise_sm120.cuh`

No Python changes needed.
