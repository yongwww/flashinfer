#!/usr/bin/env python3
"""Reproduce exact pytest failure - compare ones vs random inputs"""

import torch
import math
from flashinfer.gemm import group_gemm_fp8_nt_groupwise
from flashinfer.testing.utils import quantize_fp8, dequantize_fp8
from einops import einsum


def test_case(input_type, m, n, k, group_size, scale_major_mode):
    """Test with specified input type"""
    tile_size = 128

    # Create inputs based on type
    if input_type == "ones":
        a_val = torch.ones((group_size * m, k), dtype=torch.float, device="cuda")
        b_val = torch.ones((group_size, n, k), dtype=torch.float, device="cuda")
    elif input_type == "random":
        torch.random.manual_seed(0)  # EXACT pytest seed
        a_val = torch.randn((group_size * m, k), dtype=torch.float, device="cuda")
        b_val = torch.randn(
            (group_size, n, k), dtype=torch.float, device="cuda"
        ) / math.sqrt(k)
    else:
        raise ValueError(f"Unknown input_type: {input_type}")

    # Quantize
    if scale_major_mode == "K":
        a_scale_shape = (group_size * m, k // tile_size)
        b_scale_shape = (group_size, n // tile_size, k // tile_size)
    else:
        a_scale_shape = (k // tile_size, m * group_size)
        b_scale_shape = (group_size, k // tile_size, n // tile_size)

    a_tile_shape = (1, tile_size)
    b_tile_shape = (1, tile_size, tile_size)

    a_fp8, a_scale = quantize_fp8(a_val, a_scale_shape, a_tile_shape, scale_major_mode)
    b_fp8, b_scale = quantize_fp8(b_val, b_scale_shape, b_tile_shape, scale_major_mode)

    # Compute
    a_dequant = dequantize_fp8(a_fp8, a_scale, scale_major_mode)
    b_dequant = dequantize_fp8(b_fp8, b_scale, scale_major_mode)
    m_indptr = torch.arange(0, group_size + 1, dtype=torch.int32, device="cuda") * m

    out = group_gemm_fp8_nt_groupwise(
        a_fp8,
        b_fp8,
        a_scale,
        b_scale,
        m_indptr,
        scale_major_mode=scale_major_mode,
        out_dtype=torch.bfloat16,
    )

    ref = (
        einsum(
            a_dequant.view((group_size, m, k)),
            b_dequant,
            "b m k, b n k -> b m n",
        )
        .view((group_size * m, n))
        .to(torch.bfloat16)
    )

    # Check difference
    diff = (out - ref).abs()
    max_diff = diff.max().item()
    mismatches = (diff > 0.01).sum().item()
    pct = 100.0 * mismatches / diff.numel()

    status = "✓ PASS" if max_diff < 0.01 else "✗ FAIL"
    print(
        f"  {status}: max_diff={max_diff:.4f}, mismatches={mismatches}/{diff.numel()} ({pct:.2f}%)"
    )

    if max_diff > 0.01:
        # Find first mismatch
        bad_idx = (diff > 0.01).nonzero()[0]
        i, j = bad_idx[0].item(), bad_idx[1].item()
        print(
            f"    First error at [{i},{j}]: out={out[i, j]:.4f}, ref={ref[i, j]:.4f}, diff={diff[i, j]:.4f}"
        )

    return max_diff < 0.01


if __name__ == "__main__":
    # Failing case from quick_test
    m, n, k, group_size = 4096, 4096, 128, 2
    scale_major_mode = "MN"

    print(f"Testing: m={m}, n={n}, k={k}, groups={group_size}, mode={scale_major_mode}")
    print("=" * 80)

    print("\nTest 1: ALL-ONES INPUTS")
    pass_ones = test_case("ones", m, n, k, group_size, scale_major_mode)

    print("\nTest 2: RANDOM INPUTS (seed=0 - exact pytest setup)")
    pass_random = test_case("random", m, n, k, group_size, scale_major_mode)

    print("\n" + "=" * 80)
    if pass_ones and not pass_random:
        print("CONCLUSION: All-ones PASS, random FAIL → DATA-DEPENDENT BUG CONFIRMED!")
        print(
            "This proves our architecture is correct but has numerical issues with certain data."
        )
    elif not pass_ones and not pass_random:
        print("Both fail → fundamental bug")
    elif pass_ones and pass_random:
        print("Both pass → this case doesn't trigger the bug")
    print("=" * 80)
