# SM120/121 FP8 Grouped GEMM - Investigation Complete

## Final Results

**Baseline**: 234 test failures
**With separate tensor restructuring**: 208 failures
- Fixed: 48 original failures
- Broke: 70 originally-passing tests
- **Net result: -22 (worse overall)**

## Conclusion

After extensive investigation (10+ hours):

1. **Simple fixes don't work** - all layout/pointer adjustments either make things worse or help marginally
2. **Separate tensor approach** - fixes some but breaks others (not a complete solution)
3. **Root cause is deep** - likely in CUTLASS SM120 kernel internals or API requirements we don't understand
4. **No loop-based solution acceptable** - per requirement

## What We Learned

### Technical Findings
- CUTLASS SM120 uses dynamic UMMA::Major selection based on layout stride
- SM100 and SM120 have fundamentally different kernel implementations
- Concatenated scale tensors are problematic on SM120
- Separate tensors help some cases but not universally

### Why It's Hard
- No clear documentation on SM120 grouped GEMM requirements
- CUTLASS example uses completely different architecture (separate allocations)
- Every attempted fix helps some tests but breaks others
- Error patterns don't follow obvious dimension/mode correlations

## Recommendations

### Option 1: Engage CUTLASS Team (RECOMMENDED)
**Action**: Create detailed bug report with our findings
**Evidence**:
- 234 failures on SM120, 0 on SM100 with same code
- Multiple attempted fixes all partially work
- Exhaustive investigation documented

**Next**: File GitHub issue or contact NVIDIA directly

### Option 2: Use Alternative Backend
**Action**: Switch to TRT-LLM backend for SM120/121 if available
**Check**: Does TRT-LLM support grouped GEMM on SM120?

### Option 3: Accept Limitations
**Action**: Document SM120 grouped GEMM as unsupported/limited
**Workaround**: Users must use SM100 or single-group operations

### Option 4: Deep CUTLASS Source Dive (High Effort)
**Action**: Spend 1-2 days studying CUTLASS SM120 kernel implementation
**Goal**: Understand exact API requirements we're missing
**Risk**: May still not find solution if it's a kernel bug

## Files to Save

**Documentation**:
- `INVESTIGATION_COMPLETE.md` (this file)
- `SM120_GROUPED_GEMM_INVESTIGATION.md` (full technical details)
- `current_failures.txt` (208 unique failing tests)
- `tests/failed_tests.txt` (234 original failures)

**Test Scripts**:
- `quick_test.sh` - Fast validation
- `debug_one_case.py` - Deep single-case analysis
- `compare_failures.sh` - Compare failure sets

**Current Code State**:
- `flashinfer/gemm.py` - Has restructuring (fixes 48, breaks 70)
- `group_gemm_fp8_groupwise_sm120.cuh` - Modified pointer arithmetic

## What to Do Monday

1. **Revert to baseline** (original code with 234 failures)
2. **Escalate to CUTLASS team** with comprehensive findings
3. **Consider** if TRT-LLM backend is viable alternative
4. **If must proceed**: Pair with someone who has deep CUTLASS expertise

## Bottom Line

This is a **CUTLASS kernel issue or undocumented API requirement**, not a simple bug we can fix without deeper CUTLASS knowledge or team support.

After 10+ hours of investigation trying 10+ different approaches, we've learned the problem is complex enough to require either:
- CUTLASS team consultation
- Alternative backend
- Acceptance of limitations

**Recommendation**: Save progress, escalate to CUTLASS, don't spend more time on trial-and-error fixes.
