# SM120/121 FP8 Grouped GEMM - Final Investigation Summary

## Bottom Line

**Status**: 212/234 failures remaining (22 tests fixed, 9% improvement)
**Root Cause**: DATA-DEPENDENT numerical issue (not architectural)
**Evidence**: All-ones inputs PASS perfectly, random inputs FAIL

## Critical Discoveries

### What Works ✅
1. **CUTLASS kernel is fine** - example passes all tests including our failing dimensions
2. **Loop workaround** - all 1200 tests pass (proves kernel + data both work)
3. **Single-group** - 0 failures (our integration is correct for this case)
4. **All-ones inputs** - ALL test cases pass perfectly
5. **Isolated tests** - Some "failing" tests pass when run alone

### What Fails ❌
1. **Multi-group with random inputs** - 212/234 original failures remain
2. **Error pattern**: 0.05-28% element mismatches, max_diff 4-7
3. **Both K and MN modes** affected (not mode-specific)
4. **Various dimensions** affected (no clear pattern)

## The Paradox

**Test case: m=4096, n=4096, k=128, gs=2, MN-major**
- With all-ones: ✓ PASS (max_diff=0.0000)
- With random (seed=0): ✗ FAIL (max_diff=5.9, 0.05% mismatches)
- Architecture identical for both!

**Conclusion**: Issue is in DATA PROCESSING, not setup/pointers/allocations!

## What We've Tried (10+ approaches)

1. ❌ Line 74 fix only (`max_m`→`m`) - minimal help
2. ❌ 3-parameter ScaleConfig - made worse
3. ❌ Dynamic UMMA::Major selection - made worse
4. ✅ Separate tensor restructuring - helped (22 tests fixed)
5. ✅ `.contiguous().clone()` order - correct approach
6. ❌ Independent GPU allocations experiment - segfault
7. ✅ Verified data integrity - scales preserved correctly
8. ✅ Verified memory layout - torch.stack creates correct offsets

## Current Best Implementation

**Files Modified**:
- `flashinfer/gemm.py` (lines 2765-2807): Restructures scales with torch.stack
- `group_gemm_fp8_groupwise_sm120.cuh` (lines 59, 73-83): Uses offset-based pointers

**Result**: 212 failures (from 234) - 9% improvement

## Root Cause Hypothesis

**The SM120 kernel processes data differently than SM100**, causing numerical issues with:
- Concatenated scale tensors (our approach)
- Certain value ranges from quantization
- Accumulated errors across groups

**Why loop works**: Each group processed independently with clean state.

## Recommendations

### Option 1: Accept Loop Workaround (Temporary)
- **Pros**: All tests pass, proven correct
- **Cons**: Performance penalty, you said not acceptable
- **Use case**: Temporary until proper fix found

### Option 2: Accept Current State (212 failures)
- **Pros**: 9% improvement, no performance penalty
- **Cons**: 17.7% of tests still failing
- **Use case**: If failures are in non-critical dimensions

### Option 3: Deep Investigation with CUTLASS Team
- **Action**: File detailed bug report with all findings
- **Evidence**: CUTLASS example works, our approach partially works
- **Ask**: Why does random data cause issues with concatenated scales?

### Option 4: Complete Separate Allocation Refactor
- **Effort**: 2-3 days of careful implementation
- **Risk**: May not fix if issue is kernel-internal
- **Benefit**: Matches CUTLASS example exactly

## Files for Handoff

**Documentation**:
- `FINAL_INVESTIGATION_SUMMARY.md` (this file)
- `SM120_GROUPED_GEMM_INVESTIGATION.md` (technical details)
- `current_failures.txt` (212 unique failing tests)
- `tests/failed_tests.txt` (234 original failures)

**Test Scripts**:
- `find_failing_ones_case.py` - All-ones tests (all pass!)
- `reproduce_pytest_failure.py` - Exact pytest reproduction
- `test_scale_corruption.py` - Verifies data integrity
- `quick_test.sh` - Fast 12-case validation

**Key Finding**: Issue is DATA-DEPENDENT, not architectural!
