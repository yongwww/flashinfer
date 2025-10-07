# SM120 Grouped GEMM - Start Here Tomorrow

## Current Status (End of Day)

**Baseline**: 234 test failures (original SM120 code)
**Current Code**: 208 unique failures (with separate tensor restructuring)
- Fixed: 48 original failures
- Broke: 70 originally-passing tests
- Net: -22 (slightly worse overall, but some progress on originally-failing cases)

## What's Implemented Now

### Python (`flashinfer/gemm.py`, lines 2765-2807)
```python
# Always restructures scales for SM120
# Extracts per-group, clones with .contiguous().clone()
# Stacks into (num_groups, ...) format
```

### C++ (`group_gemm_fp8_groupwise_sm120.cuh`, lines 73-83)
```cpp
// K-major: SFA_ptr[i] = SFA + i * sf_m * sf_k
// MN-major: SFA_ptr[i] = SFA + i * sf_k * sf_m
// Layout: make_shape(m, n, k, 1) for both modes
```

## Key Discovery from Tonight

**CUTLASS SM120 kernels dynamically determine UMMA::Major** from layout stride:
```cpp
size<0,1>(LayoutSFA{}.stride()) == 1 ? UMMA::Major::MN : UMMA::Major::K
```

Our static template-based selection may not align with this.

## Tomorrow's Debugging Strategy

### Step 1: Verify Current State
```bash
cd /workspace/flashinfer
git status  # Check what's modified
python debug_one_case.py  # Verify test case passes standalone
```

### Step 2: Analyze What We Fixed vs Broke
```bash
# Compare sets carefully
comm -23 <(grep "\[out_dtype" tests/failed_tests.txt | sed 's/.*\[/[/' | sed 's/\].*/]/' | sort -u) \
         <(grep "test_fp8_groupwise_group_gemm\[" current_failures.txt | sed 's/.*\[/[/' | sed 's/\].*/]/' | sort -u) > fixed_tests.txt

comm -13 <(grep "\[out_dtype" tests/failed_tests.txt | sed 's/.*\[/[/' | sed 's/\].*/]/' | sort -u) \
         <(grep "test_fp8_groupwise_group_gemm\[" current_failures.txt | sed 's/.*\[/[/' | sed 's/\].*/]/' | sort -u) > broken_tests.txt

# Analyze patterns
echo "Fixed tests:"
cat fixed_tests.txt | head -20

echo "Broken tests:"
cat broken_tests.txt | head -20
```

### Step 3: Find Pattern in Fixed vs Broken
- Do fixed tests share common dimensions?
- Do broken tests have specific characteristics?
- Is there a mode/group_size pattern?

### Step 4: Hypothesis to Test
**Theory**: Maybe we need conditional restructuring:
- If certain condition → use restructured scales
- Otherwise → use original concatenated scales

### Step 5: Alternative Approach
Try reverting to baseline and applying ONLY the line 74 fix from your working commit:
```cpp
// In SM120 cuh file, line 74
- layout_SFA[i] = ScaleConfig::tile_atom_to_shape_SFA(cute::make_shape(max_m, n, k, 1));
+ layout_SFA[i] = ScaleConfig::tile_atom_to_shape_SFA(cute::make_shape(m, n, k, 1));
```

## Files to Keep

**Critical Documentation**:
- `START_HERE_TOMORROW.md` (this file)
- `INVESTIGATION_COMPLETE.md` (comprehensive summary)
- `SM120_GROUPED_GEMM_INVESTIGATION.md` (technical details)
- `tests/failed_tests.txt` (234 original failures)
- `current_failures.txt` (208 current failures)

**Test Scripts**:
- `quick_test.sh`
- `debug_one_case.py`
- `compare_failures.sh`

**Can Delete** (temporary):
- `debug_stack.py`
- `debug_layout.py`
- `test_k128.py`
- All other SEPARATE_TENSOR_*, CURRENT_STATE.md, etc.

## Quick Commands for Tomorrow

```bash
# Check state
git status
git diff include/flashinfer/gemm/group_gemm_fp8_groupwise_sm120.cuh
git diff flashinfer/gemm.py

# Test current
python debug_one_case.py

# Revert if needed
git checkout flashinfer/gemm.py include/flashinfer/gemm/group_gemm_fp8_groupwise_sm120.cuh

# Rebuild
rm -rf ~/.cache/flashinfer/ && pip install -e . --force-reinstall --no-deps

# Test
bash quick_test.sh
```

## Key Insight for Tomorrow

The fact that we fixed 48 but broke 70 suggests our approach is **partially correct** but **incomplete**.

Maybe we need:
1. **Hybrid approach**: Restructure only for specific cases
2. **Different pointer arithmetic**: Not `i * sf_m * sf_k` but something else
3. **Consult your working commit more carefully**: What exactly did the line 74 fix do?

## Reference

Your working commit (loop-based workaround):
https://github.com/yongwww/flashinfer/commit/b7f84ccf18e2aabac38a898f11ed8ec2aed15ef4

Key change was line 74: `max_m` → `m` in MN-major branch.

Have a good night! 🌙
