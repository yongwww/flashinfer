#!/bin/bash
cd /workspace/flashinfer

echo "Running full test to capture current failures..."
pytest tests/GEMM/test_groupwise_scaled_gemm_fp8.py::test_fp8_groupwise_group_gemm --tb=no -v 2>&1 | grep "FAILED" | sed 's/FAILED //' | sed 's/ -.*//' > current_failures.txt

echo "Comparing with original failures..."
echo ""
echo "Original failures: $(wc -l < tests/failed_tests.txt) lines"
echo "Current failures: $(wc -l < current_failures.txt) tests"
echo ""

# Find tests that were failing but now pass
echo "Tests FIXED (were failing, now passing):"
comm -23 <(grep "test_fp8_groupwise_group_gemm" tests/failed_tests.txt | sed 's/.*:://' | sed 's/ -.*//' | sort) <(sort current_failures.txt) | head -20

echo ""
echo "Tests BROKEN (were passing, now failing):"
comm -13 <(grep "test_fp8_groupwise_group_gemm" tests/failed_tests.txt | sed 's/.*:://' | sed 's/ -.*//' | sort) <(sort current_failures.txt) | head -20

echo ""
echo "Run 'diff tests/failed_tests.txt current_failures.txt' for detailed comparison"
