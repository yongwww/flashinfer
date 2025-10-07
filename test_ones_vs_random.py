#!/usr/bin/env python3
"""Compare ones vs random inputs for failing case"""

import torch
import math
from flashinfer.gemm import group_gemm_fp8_nt_groupwise
from flashinfer.testing.utils import quantize_fp8, dequantize_fp8
from einops import einsum


def test_case(input_type, m, n, k, group_size, scale_major_mode):
    """
    Test with specified input type

    Args:
        input_type: "ones" or "random"
        m, n, k: problem dimensions
        group_size: number of groups
        scale_major_mode: "K" or "MN"

    Returns:
        bool: True if passes, False if fails
    """
    tile_size = 128

    # Create inputs
    if input_type == "ones":
        a_val = torch.ones((group_size * m, k), dtype=torch.float32, device="cuda")
        b_val = torch.ones((group_size, n, k), dtype=torch.float32, device="cuda")
    elif input_type == "random":
        torch.random.manual_seed(0)
        a_val = torch.randn((group_size * m, k), dtype=torch.float, device="cuda")
        b_val = torch.randn(
            (group_size, n, k), dtype=torch.float, device="cuda"
        ) / math.sqrt(k)
    else:
        raise ValueError(f"input_type must be 'ones' or 'random', got {input_type}")

    # Quantize
    if scale_major_mode == "K":
        a_scale_shape = (group_size * m, k // tile_size)
        b_scale_shape = (group_size, n // tile_size, k // tile_size)
    else:
        a_scale_shape = (k // tile_size, m * group_size)
        b_scale_shape = (group_size, k // tile_size, n // tile_size)

    a_fp8, a_scale = quantize_fp8(
        a_val, a_scale_shape, (1, tile_size), scale_major_mode
    )
    b_fp8, b_scale = quantize_fp8(
        b_val, b_scale_shape, (1, tile_size, tile_size), scale_major_mode
    )

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
        einsum(a_dequant.view((group_size, m, k)), b_dequant, "b m k, b n k -> b m n")
        .view((group_size * m, n))
        .to(torch.bfloat16)
    )

    # Check
    diff = (out - ref).abs()
    max_diff = diff.max().item()
    mismatches = (diff > 0.01).sum().item()
    pct = 100.0 * mismatches / diff.numel()

    passed = max_diff < 0.01
    status = "✓ PASS" if passed else "✗ FAIL"

    print(f"  {status}: max_diff={max_diff:.4f}, mismatches={mismatches} ({pct:.2f}%)")

    if not passed:
        bad_idx = (diff > 0.01).nonzero()[0]
        i, j = bad_idx[0].item(), bad_idx[1].item()
        print(f"    First error [{i},{j}]: out={out[i, j]:.4f}, ref={ref[i, j]:.4f}")

    return passed


if __name__ == "__main__":
    # Known failing case from quick_test
    m, n, k, gs, mode = 4096, 4096, 128, 2, "MN"

    print(f"Test Case: m={m}, n={n}, k={k}, groups={gs}, mode={mode}")
    print("=" * 80)

    print("\nOnes:")
    pass_ones = test_case("ones", m, n, k, gs, mode)

    print("\nRandom (seed=0):")
    pass_random = test_case("random", m, n, k, gs, mode)

    print("\n" + "=" * 80)
    if pass_ones and pass_random:
        print("BOTH PASS - This case doesn't trigger bug")
    elif pass_ones and not pass_random:
        print("ONES PASS, RANDOM FAIL - Data-dependent bug!")
    elif not pass_ones:
        print("ONES FAIL - Fundamental bug!")
    print("=" * 80)
