#!/usr/bin/env python3
"""Save exact failing tensors for CUTLASS comparison"""

import torch
import math
from flashinfer.testing.utils import quantize_fp8
import struct

torch.random.manual_seed(0)
m, n, k, gs = 4096, 4096, 128, 2
mode = "MN"

# Generate exact inputs
a_val = torch.randn((gs * m, k), dtype=torch.float, device="cuda")
b_val = torch.randn((gs, n, k), dtype=torch.float, device="cuda") / math.sqrt(k)

# Quantize
a_scale_shape = (k // 128, m * gs)
b_scale_shape = (gs, k // 128, n // 128)

a_fp8, a_scale = quantize_fp8(a_val, a_scale_shape, (1, 128), mode)
b_fp8, b_scale = quantize_fp8(b_val, b_scale_shape, (1, 128, 128), mode)

print(f"Saving tensors for m={m}, n={n}, k={k}, groups={gs}, mode={mode}")
print(f"  a_fp8: {a_fp8.shape}")
print(f"  b_fp8: {b_fp8.shape}")
print(f"  a_scale: {a_scale.shape}")
print(f"  b_scale: {b_scale.shape}")


# Save as binary files that C++ can read
def save_tensor_binary(tensor, filename):
    """Save tensor as binary file"""
    t = tensor.cpu().contiguous()
    with open(filename, "wb") as f:
        # Write shape
        f.write(struct.pack("i", len(t.shape)))
        for dim in t.shape:
            f.write(struct.pack("i", dim))
        # Write data
        f.write(t.numpy().tobytes())


save_tensor_binary(a_fp8, "a_fp8.bin")
save_tensor_binary(b_fp8, "b_fp8.bin")
save_tensor_binary(a_scale, "a_scale.bin")
save_tensor_binary(b_scale, "b_scale.bin")

print(f"Parameters: m={m}, n={n}, k={k}, groups={gs}")
print(f"Mode: {mode}")
