#!/usr/bin/env python3
"""Debug the exact failing cases from quick_test.sh"""

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
    max_diff = diff.max().item()
    mismatches = (diff > 0.01).sum().item()
    pct = 100.0 * mismatches / diff.numel()

    # Use same criteria as torch.testing.assert_close
    passes = (diff <= 0.01).all().item() or max_diff < 0.01
    status = "✓ PASS" if passes else f"✗ FAIL {pct:.1f}%"
    print(f"  Result: {status}, max_diff={max_diff:.4f}, mismatches={mismatches}")
    return passes


print("EXACT FAILING CASES FROM QUICK_TEST:")
print("=" * 80)

# Test ID: MN-2-128-128-4096 = mode=MN, gs=2, m=128, n=128, k=4096
# But error shows: m = 4096, n = 128, k = 128
# So format must be: mode-gs-SOMETHING-SOMETHING-SOMETHING
# Let me check by testing what the error message says:

test_case(
    4096, 128, 128, 2, "MN", "MN-2-128-128-4096 (from error: m=4096, n=128, k=128)"
)
test_case(128, 4096, 128, 2, "MN", "Alternative: m=128, n=4096, k=128")

test_case(
    4096, 4096, 128, 2, "MN", "MN-2-128-4096-4096 (from error: m=4096, n=4096, k=128)"
)

test_case(
    8192, 4096, 128, 2, "K", "K-2-128-4096-8192 (from error: m=8192, n=4096, k=128)"
)

test_case(4, 8192, 128, 8, "K", "K-8-128-8192-4 (from error: m=4, n=8192, k=128)")

print("\n" + "=" * 80)
print("COMPARISON: Check which parameterization matches the error messages")
print("=" * 80)
