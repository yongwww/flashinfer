#!/usr/bin/env python3
"""Debug: Understand what's happening with scale major modes"""

import torch
from flashinfer.testing.utils import quantize_fp8

# Test case parameters
m, n, k, group_size = 128, 128, 4096, 2
tile_size = 128

# Create dummy data
a_val = torch.randn((group_size * m, k), dtype=torch.float, device="cuda")
b_val = torch.randn((group_size, n, k), dtype=torch.float, device="cuda")

print("=" * 80)
print("K-MAJOR MODE")
print("=" * 80)
scale_major_mode = "K"
a_scale_shape = (group_size * m, k // tile_size)
b_scale_shape = (group_size, n // tile_size, k // tile_size)
print(f"a_scale_shape: {a_scale_shape}")
print(f"b_scale_shape: {b_scale_shape}")

a_fp8, a_scale = quantize_fp8(a_val, a_scale_shape, (1, tile_size), scale_major_mode)
b_fp8, b_scale = quantize_fp8(
    b_val, b_scale_shape, (1, tile_size, tile_size), scale_major_mode
)

print(f"a_scale actual: {a_scale.shape}, stride: {a_scale.stride()}")
print(f"b_scale actual: {b_scale.shape}, stride: {b_scale.stride()}")
print(f"a_scale is_contiguous: {a_scale.is_contiguous()}")
print(
    f"a_scale layout: {'Row-major' if a_scale.stride()[0] > a_scale.stride()[1] else 'Col-major'}"
)

print("\n" + "=" * 80)
print("MN-MAJOR MODE")
print("=" * 80)
scale_major_mode = "MN"
a_scale_shape = (k // tile_size, m * group_size)
b_scale_shape = (group_size, k // tile_size, n // tile_size)
print(f"a_scale_shape: {a_scale_shape}")
print(f"b_scale_shape: {b_scale_shape}")

a_fp8, a_scale = quantize_fp8(a_val, a_scale_shape, (1, tile_size), scale_major_mode)
b_fp8, b_scale = quantize_fp8(
    b_val, b_scale_shape, (1, tile_size, tile_size), scale_major_mode
)

print(f"a_scale actual: {a_scale.shape}, stride: {a_scale.stride()}")
print(f"b_scale actual: {b_scale.shape}, stride: {b_scale.stride()}")
print(f"a_scale is_contiguous: {a_scale.is_contiguous()}")
print(
    f"a_scale layout: {'Row-major' if a_scale.stride()[0] > a_scale.stride()[1] else 'Col-major'}"
)

print("\n" + "=" * 80)
print("INTERPRETATION")
print("=" * 80)
print("For CUTLASS stride check: size<0,1>(LayoutSFA{}.stride()) == 1")
print("This checks if the stride in dimension (0,1) equals 1")
print("  - If stride==1: Column-major in that dimension → UMMA::Major::MN")
print("  - If stride!=1: Row-major in that dimension → UMMA::Major::K")
