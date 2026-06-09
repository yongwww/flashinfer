"""Verify the C++/CUDA Trace Apply loader: build a minimal TVM-FFI CUDA kernel
from a Solution and run it. Requires GPU + the flashinfer build env."""

import torch
from flashinfer.trace.solution import Solution
from flashinfer.trace_apply import loaders

CU = r"""
#include "tvm_ffi_utils.h"
using tvm::ffi::TensorView;

__global__ void _add_one_kernel(const float* in, float* out, long long n) {
  long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) out[i] = in[i] + 1.0f;
}

void add_one(TensorView input, TensorView output) {
  long long n = 1;
  for (int d = 0; d < input.ndim(); ++d) n *= input.size(d);
  ffi::CUDADeviceGuard guard(input.device().device_id);
  cudaStream_t stream = get_stream(input.device());
  int threads = 256;
  int blocks = (int)((n + threads - 1) / threads);
  _add_one_kernel<<<blocks, threads, 0, stream>>>(
      static_cast<const float*>(input.data_ptr()),
      static_cast<float*>(output.data_ptr()), n);
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(add_one, add_one);
"""


def main():
    sol = Solution.from_dict(
        {
            "name": "cuda_add_one",
            "definition": "dummy",
            "author": "tester",
            "spec": {
                "language": "cuda",
                "target_hardware": ["NVIDIA B200"],
                "entry_point": "add_one.cu::add_one",
            },
            "sources": [{"path": "add_one.cu", "content": CU}],
        }
    )
    print("building CUDA solution via flashinfer.jit ...")
    fn = loaders.load(sol)
    print("loaded entry:", fn)
    x = torch.arange(16, dtype=torch.float32, device="cuda")
    out = torch.empty_like(x)
    fn(x, out)  # positional TVM-FFI call (inputs..., outputs...)
    torch.cuda.synchronize()
    ok = torch.allclose(out, x + 1.0)
    print("out[:5]=", out[:5].tolist(), "correct=", ok)
    assert ok, "C++/CUDA loader produced wrong result"
    print("C++/CUDA LOADER OK")


if __name__ == "__main__":
    main()
