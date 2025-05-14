import flashinfer
from flashinfer.gemm import group_gemm_fp8_nt_groupwise

import torch

def test_trigger():
    m, n, k = 128, 128, 128
    group_size = 1
    tile_size = 128

    x_a = torch.rand((m, k), device='cuda').to(torch.float8_e4m3fn)
    x_b = torch.rand((group_size, n, k), device='cuda').to(torch.float8_e4m3fn)
    a_scale = torch.rand((k // tile_size, m), device='cuda', dtype=torch.float32)
    b_scale = torch.rand((group_size, k // tile_size, n // tile_size), device='cuda', dtype=torch.float32)
    m_indptr = torch.tensor([0, m], dtype=torch.int32, device='cuda')
    out_dtype = torch.bfloat16
    out = torch.empty(x_a.shape[0], n, dtype=out_dtype, device=x_a.device)

    # Ensure all JIT/init is done before capture
    m_indptr_end = int(m_indptr[-1].item())
    workspace_buffer = torch.empty(128 * 1024 * 1024, dtype=torch.uint8, device=x_a.device)
    group_gemm_fp8_nt_groupwise(x_a, x_b, a_scale, b_scale, m_indptr, m_indptr_end, out=out, workspace_buffer=workspace_buffer)
    torch.cuda.synchronize()

    # Use a non-default stream
    stream = torch.cuda.Stream()
    graph = torch.cuda.CUDAGraph()

    # Prepare static input/output for capture
    static_out = out.clone()

    # Start capturing on the custom stream
    torch.cuda.synchronize()
    with torch.cuda.stream(stream):
        stream.wait_stream(torch.cuda.current_stream())
        graph.capture_begin()
        group_gemm_fp8_nt_groupwise(x_a, x_b, a_scale, b_scale, m_indptr, m_indptr_end, out=static_out, workspace_buffer=workspace_buffer)
        graph.capture_end()
        torch.cuda.current_stream().wait_stream(stream)

    # Replay on default stream
    graph.replay()
    torch.cuda.synchronize()
    print("Graph replayed successfully.")

test_trigger()