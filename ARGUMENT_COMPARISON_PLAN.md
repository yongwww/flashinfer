# FlashInfer Arguments → CUTLASS Example Comparison Plan

## Objective
Test if CUTLASS example produces correct results when given FlashInfer's exact arguments for a failing case.

## Hypothesis
If CUTLASS example fails with FlashInfer's arguments → proves our argument setup is correct and issue is in CUTLASS kernel behavior
If CUTLASS example passes with FlashInfer's arguments → proves our argument setup has bugs

## Step-by-Step Plan

### **Step 1: Choose Failing Test Case**
**Selected**: `K-2-128-4096-8192`
- Parameters: m=8192, n=4096, k=128, group_size=2, scale_major_mode='K'
- Currently fails with 0.0% element mismatches but large absolute errors
- Simple case (only 2 groups)
- K-major (simpler layout than MN-major)

### **Step 2: Dump FlashInfer Arguments**
Location: `include/flashinfer/gemm/group_gemm_fp8_groupwise_sm120.cuh`

Before calling `gemm.run()` (around line 240), add code to dump:

**Metadata** (text file):
```
num_groups
For each group i:
  - problem_sizes[i].m(), problem_sizes[i].n(), problem_sizes[i].k()
  - stride_A[i], stride_B[i], stride_D[i]
  - layout_SFA[i], layout_SFB[i]
  - A_ptr[i] offset from base
  - B_ptr[i] offset from base
  - D_ptr[i] offset from base
  - SFA_ptr[i] offset from base
  - SFB_ptr[i] offset from base
```

**Tensor Data** (binary files):
- A.bin (input data)
- B.bin (input data)
- SFA.bin (scales)
- SFB.bin (scales)

### **Step 3: Modify CUTLASS Example**
Add new command-line option: `--load-flashinfer-args`

```cpp
if (load_flashinfer_args) {
    // Read metadata file
    // Allocate tensors with correct sizes
    // Load binary data
    // Set up arguments exactly as FlashInfer did
    // Run kernel
    // Compare output
}
```

### **Step 4: Run Comparison**
```bash
# Capture FlashInfer args
pytest test_case... # Dumps to /tmp/flashinfer_args/

# Run CUTLASS with those args
cd /workspace/cutlass/build
./87c_... --load-flashinfer-args=/tmp/flashinfer_args/

# Analyze:
# - Exit code (pass/fail)
# - Output differences
# - Error patterns
```

### **Step 5: Analysis**

**Scenario A**: CUTLASS PASSES with FlashInfer's args
- Conclusion: FlashInfer's argument setup is buggy
- Action: Compare dumped args vs what CUTLASS generates for same problem
- Find: Which field differs (strides? layouts? pointers?)

**Scenario B**: CUTLASS FAILS with same errors as FlashInfer
- Conclusion: Arguments are correct, CUTLASS kernel has issue with this config
- Action: Report to CUTLASS with minimal repro
- Evidence: Their own example fails with these specific arguments

**Scenario C**: CUTLASS FAILS with different errors
- Conclusion: Data format/corruption issue
- Action: Verify data is correctly transferred

## Implementation Files

1. `/workspace/flashinfer/include/flashinfer/gemm/group_gemm_fp8_groupwise_sm120.cuh`
   - Add argument dumping code

2. `/workspace/cutlass/examples/.../87c_blackwell_geforce_fp8_bf16_grouped_gemm_groupwise.cu`
   - Add `--load-flashinfer-args` option
   - Load and use dumped arguments

3. Helper scripts:
   - `dump_args.py` - Trigger specific test to generate dump
   - `compare_outputs.py` - Compare FlashInfer vs CUTLASS outputs

## Expected Timeline
- Step 2 (dump): 30 min
- Step 3 (modify CUTLASS): 45 min
- Step 4 (run): 10 min
- Step 5 (analyze): 30 min
**Total**: ~2 hours

## Success Criteria
Definitive answer to: "Is the issue in FlashInfer's argument setup or CUTLASS kernel behavior?"
