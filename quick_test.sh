#!/bin/bash
# Quick validation of SM120 fix with representative failing cases
cd /workspace/flashinfer

echo "Quick SM120 Fix Validation - 10 Representative Cases"
echo "================================================================"

pytest -v tests/GEMM/test_groupwise_scaled_gemm_fp8.py::test_fp8_groupwise_group_gemm \
  -k "out_dtype0-MN-2-128-128-4096 or \
      out_dtype0-MN-4-256-4096-8192 or \
      out_dtype0-MN-8-512-8192-8192 or \
      out_dtype0-K-2-128-4096-8192 or \
      out_dtype0-K-4-256-8192-8192 or \
      out_dtype0-K-8-512-4096-8192 or \
      out_dtype0-MN-2-128-4096-4 or \
      out_dtype0-K-8-128-8192-4 or \
      out_dtype0-MN-1-128-128-4096 or \
      out_dtype0-K-1-512-4096-8192"

echo ""
echo "================================================================"
echo "Summary: Check above for PASSED/FAILED count"
echo "If all 10 pass, run full suite:"
echo "  pytest tests/GEMM/test_groupwise_scaled_gemm_fp8.py::test_fp8_groupwise_group_gemm -v | tail -20"
echo "================================================================"
