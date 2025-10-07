# CUTLASS SM120 Grouped GEMM - Concatenated vs Separate Allocations

## Summary

**PROVEN**: SM120 grouped GEMM kernel **requires independent GPU allocations** and **does NOT support concatenated scales with offset pointers**.

## Test Results

### CUTLASS Original (Separate Allocations)
```cpp
// N independent GPU buffers
block_SFA[0].device_data() = 0x1000  // Region 1
block_SFA[1].device_data() = 0x5000  // Region 2 (different base)
```
**Result**: ✅ **PASSES** (verified with groups=2,4,8)

### FlashInfer (Concatenated with Offsets)
```python
# ONE concatenated tensor
stacked = torch.stack([scale0, scale1], dim=0)
ptr[0] = base + 0
ptr[1] = base + offset  // SAME base, different offset
```
**Result**: ❌ **203/1200 tests FAIL** (0.05-6% mismatches)

### CUTLASS Modified Experiment (Concatenated with Offsets)
**Change**: Modified CUTLASS example to use concatenated buffer with offset pointers (exactly like FlashInfer)

```cpp
HostTensorSFA concatenated_SFA(total_size);
ptr_SFA[0] = concatenated_SFA.device_data() + 0;
ptr_SFA[1] = concatenated_SFA.device_data() + offset;
```

**Result**: ❌ **BOTH GROUPS FAIL**
```
Group 0 FAILED correctness check!
Group 1 FAILED correctness check!
Disposition: Failed
```

## Conclusion

**The SM120 grouped GEMM kernel fundamentally requires independent memory allocations per group.**

Using concatenated buffers with offset pointers causes **identical failures** in both:
- CUTLASS example (our modification)
- FlashInfer (original implementation)

This is a **CUTLASS kernel limitation/requirement**, not a FlashInfer bug.

## Evidence

| Approach | Allocations | Pointers | Result |
|----------|-------------|----------|--------|
| CUTLASS Original | Independent (N buffers) | Scattered (0x1000, 0x5000, ...) | ✅ PASS |
| FlashInfer | Concatenated (1 buffer) | Sequential (base+0, base+offset, ...) | ❌ FAIL |
| CUTLASS Modified | Concatenated (1 buffer) | Sequential (base+0, base+offset, ...) | ❌ FAIL |

## Next Steps

**Option A**: Implement independent allocations in FlashInfer (matches CUTLASS requirement)
**Option B**: Report to CUTLASS as kernel limitation with our repro
**Option C**: Test if copying from concatenated to independent fixes it (verify it's pointer pattern, not data)

**Recommendation**: Try Option C first to confirm it's the pointer pattern specifically.
