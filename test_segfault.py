#!/usr/bin/env python3
"""Standalone test to debug segfault"""

import torch
import sys

print("Step 1: Importing flashinfer...", flush=True)
sys.path.insert(0, "/workspace/flashinfer")
from flashinfer.gemm import group_gemm_fp8_nt_groupwise

print("Step 2: Creating test data...", flush=True)
device = torch.device("cuda:0")
num_groups = 1
m = 128
n = 128
k = 4096

# Simple test case - MN-major, group_size=1
m_indptr = torch.tensor([0, 128], dtype=torch.int32, device=device)
a = torch.randn(128, 4096, dtype=torch.float16, device=device).to(torch.float8_e4m3fn)
b = torch.randn(1, 128, 4096, dtype=torch.float16, device=device).to(
    torch.float8_e4m3fn
)

# Scales
a_scale = torch.randn(32, 1, dtype=torch.float32, device=device)  # k//128 = 32
b_scale = torch.randn(1, 1, 32, dtype=torch.float32, device=device)

print("Step 3: Calling group_gemm_fp8_nt_groupwise...", flush=True)
try:
    out = group_gemm_fp8_nt_groupwise(
        a,
        b,
        a_scale,
        b_scale,
        m_indptr,
        scale_major_mode="MN",
        scale_granularity_mnk=(128, 128, 128),
    )
    print("Step 4: SUCCESS!", flush=True)
    print(f"Output shape: {out.shape}", flush=True)
except Exception as e:
    print(f"Step 4: EXCEPTION: {e}", flush=True)
    import traceback

    traceback.print_exc()
