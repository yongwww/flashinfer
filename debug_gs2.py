#!/usr/bin/env python3
"""Debug why group_size=2 breaks while group_size=8 works"""

import torch
import math
from einops import einsum
from flashinfer.gemm import group_gemm_fp8_nt_groupwise
from flashinfer.testing.utils import dequantize_fp8, quantize_fp8


def test_case(m, n, k, gs, mode, name):
    torch.random.manual_seed(0)
    tile_size = 128

    a_val = torch.randn((gs * m, k), dtype=torch.float, device="cuda")
    b_val = torch.randn((gs, n, k), dtype=torch.float, device="cuda") / math.sqrt(k)

    if mode == "K":
        a_scale_shape = (gs * m, k // tile_size)
        b_scale_shape = (gs, n // tile_size, k // tile_size)
    else:
        a_scale_shape = (k // tile_size, m * gs)
        b_scale_shape = (gs, k // tile_size, n // tile_size)

    a_fp8, a_scale = quantize_fp8(a_val, a_scale_shape, (1, tile_size), mode)
    b_fp8, b_scale = quantize_fp8(b_val, b_scale_shape, (1, tile_size, tile_size), mode)

    print(f"\n{name}:")
    print(f"  m={m}, n={n}, k={k}, gs={gs}, mode={mode}")
    print(f"  a_scale: {a_scale.shape}")
    print(f"  b_scale: {b_scale.shape}")

    a_dequant = dequantize_fp8(a_fp8, a_scale, mode)
    b_dequant = dequantize_fp8(b_fp8, b_scale, mode)
    m_indptr = torch.arange(0, gs + 1, dtype=torch.int32, device="cuda") * m

    out = group_gemm_fp8_nt_groupwise(
        a_fp8,
        b_fp8,
        a_scale,
        b_scale,
        m_indptr,
        scale_major_mode=mode,
        out_dtype=torch.bfloat16,
    )
    ref = (
        einsum(a_dequant.view((gs, m, k)), b_dequant, "b m k, b n k -> b m n")
        .view((gs * m, n))
        .to(torch.bfloat16)
    )

    diff = (out - ref).abs()
    mismatches = (diff > 0.01).sum().item()
    pct = 100.0 * mismatches / diff.numel()

    status = "✓ PASS" if pct < 1.0 else f"✗ FAIL {pct:.1f}%"
    print(f"  Result: {status}, max_diff={diff.max().item():.4f}")
    return pct < 1.0


print("=" * 80)
print("BROKEN CASES (group_size=2, were passing, now fail):")
print("=" * 80)
# Note: Test ID format is [out_dtype0-mode-gs-m-n-k]
# So K-2-128-128-4096 means: K mode, gs=2, m=128, n=128, k=4096
test_case(128, 128, 4096, 2, "K", "BROKEN: K-2-128-128-4096")
# K-2-128-4096-4 means: K mode, gs=2, m=128, n=4096, k=4 (but k must be >=128!)
# This suggests the ID format might be different. Let me use valid k values:
test_case(128, 8192, 512, 2, "K", "BROKEN: K-2-128-8192-512")
test_case(256, 4096, 8192, 2, "K", "BROKEN: K-2-256-4096-8192")

print("\n" + "=" * 80)
print("FIXED CASES (group_size=2, were failing, now pass):")
print("=" * 80)
test_case(128, 128, 8192, 2, "K", "FIXED: K-2-128-128-8192")
test_case(128, 256, 4096, 2, "K", "FIXED: K-2-128-256-4096")
test_case(128, 512, 4096, 2, "K", "FIXED: K-2-128-512-4096")

print("\n" + "=" * 80)
print("WORKING WELL (group_size=8):")
print("=" * 80)
test_case(256, 4096, 8192, 8, "K", "group_size=8 example")
test_case(512, 8192, 8192, 8, "MN", "group_size=8 example MN")

print("\n" + "=" * 80)
print("COMPARISON:")
print("Look for differences in scale tensor shapes/dimensions")
print("=" * 80)
