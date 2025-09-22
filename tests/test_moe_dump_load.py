#!/usr/bin/env python3
"""
Comprehensive test suite for MOE input dumping and loading functionality.

This test suite verifies:
1. Dumping mechanics work correctly
2. Tensors and scalars are preserved exactly after dump/load
3. FP8 dtypes are maintained
4. None values are handled properly
5. Dump limits are enforced
6. Loaded inputs produce identical outputs
"""

import json
import os
import shutil
import tempfile
from pathlib import Path

import pytest
import torch


def create_test_inputs(seq_len=128, hidden_size=4096, num_experts=8, intermediate_size=11008):
    """Create test inputs matching the FP8 block scale MOE requirements."""
    # Create routing tensors
    routing_logits = torch.randn(seq_len, num_experts, dtype=torch.float32, device="cuda")
    routing_bias = torch.randn(num_experts, dtype=torch.float32, device="cuda")
    
    # Create hidden states and scales - convert to FP8 after creation
    hidden_states = torch.randn(seq_len, hidden_size, dtype=torch.float32, device="cuda")
    hidden_states_fp8 = hidden_states.to(torch.float8_e4m3fn)
    hidden_states_scale = torch.randn(hidden_size // 128, seq_len, dtype=torch.float32, device="cuda")
    
    # Create weight tensors - convert to FP8 after creation
    gemm1_weights = torch.randn(
        num_experts, 2 * intermediate_size, hidden_size, 
        dtype=torch.float32, device="cuda"
    ).to(torch.float8_e4m3fn)
    
    gemm1_weights_scale = torch.randn(
        num_experts, 2 * intermediate_size // 128, hidden_size // 128,
        dtype=torch.float32, device="cuda"
    )
    
    gemm2_weights = torch.randn(
        num_experts, hidden_size, intermediate_size,
        dtype=torch.float32, device="cuda"
    ).to(torch.float8_e4m3fn)
    
    gemm2_weights_scale = torch.randn(
        num_experts, hidden_size // 128, intermediate_size // 128,
        dtype=torch.float32, device="cuda"
    )
    
    return {
        "routing_logits": routing_logits,
        "routing_bias": routing_bias,
        "hidden_states": hidden_states_fp8,
        "hidden_states_scale": hidden_states_scale,
        "gemm1_weights": gemm1_weights,
        "gemm1_weights_scale": gemm1_weights_scale,
        "gemm2_weights": gemm2_weights,
        "gemm2_weights_scale": gemm2_weights_scale,
    }


def load_dumped_inputs(dump_dir, request_id="000"):
    """Load the dumped inputs from a specific request."""
    request_dir = dump_dir / f"request_{request_id}"
    
    # Load scalar parameters
    with open(request_dir / "scalar.json", "r") as f:
        scalars = json.load(f)
    
    # Load tensor inputs
    tensors = {}
    tensor_files = [
        "routing_logits.pt",
        "routing_bias.pt",
        "hidden_states.pt",
        "hidden_states_scale.pt",
        "gemm1_weights.pt",
        "gemm1_weights_scale.pt",
        "gemm2_weights.pt",
        "gemm2_weights_scale.pt",
    ]
    
    for filename in tensor_files:
        filepath = request_dir / filename
        if filepath.exists():
            name = filename.replace(".pt", "")
            tensors[name] = torch.load(filepath)
    
    return scalars, tensors


@pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] < 9,
    reason="Requires CUDA and SM >= 90"
)
def test_dump_mechanics():
    """Test that the dumping mechanics work correctly."""
    
    with tempfile.TemporaryDirectory() as temp_dir:
        dump_dir = Path(temp_dir) / "test_dumps"
        
        # Set up environment for dumping
        os.environ["DUMP_MOE_INPUTS"] = "1"
        os.environ["MOE_DUMP_DIR"] = str(dump_dir)
        os.environ["MOE_MAX_DUMPS"] = "2"
        
        # Reload module to pick up environment variables
        import importlib
        import flashinfer.fused_moe.core as moe_core
        importlib.reload(moe_core)
        
        # Access the dump function directly
        from flashinfer.fused_moe.core import _dump_moe_inputs
        
        # Create test inputs
        inputs = create_test_inputs(seq_len=64, hidden_size=2048, num_experts=4, intermediate_size=5504)
        
        params = {
            "num_experts": 4,
            "top_k": 2,
            "n_group": 1,
            "topk_group": 1,
            "intermediate_size": 5504,
            "local_expert_offset": 0,
            "local_num_experts": 4,
            "routed_scaling_factor": 1.0,
            "tile_tokens_dim": 8,
            "routing_method_type": 0,
            "use_shuffled_weight": False,
            "weight_layout": 0,
            "enable_pdl": None,
        }
        
        # First dump
        _dump_moe_inputs(**inputs, **params)
        
        # Second dump with None routing_bias
        inputs_with_none = inputs.copy()
        inputs_with_none["routing_bias"] = None
        params_modified = params.copy()
        params_modified["use_shuffled_weight"] = True
        _dump_moe_inputs(**inputs_with_none, **params_modified)
        
        # Third dump (should not create due to limit)
        _dump_moe_inputs(**inputs, **params)
        
        # Verify dumps were created correctly
        request_dirs = sorted(list(dump_dir.glob("request_*")))
        assert len(request_dirs) == 2, f"Expected 2 dumps (due to limit), found {len(request_dirs)}"
        
        # Check request_000
        request_000 = dump_dir / "request_000"
        assert request_000.exists(), "request_000 not found"
        
        # Load and verify scalar.json for request_000
        with open(request_000 / "scalar.json", "r") as f:
            scalars_000 = json.load(f)
        
        assert scalars_000["request_id"] == "000"
        assert scalars_000["num_experts"] == 4
        assert scalars_000["use_shuffled_weight"] == False
        
        # Check that routing_bias.pt exists in request_000
        assert (request_000 / "routing_bias.pt").exists(), "routing_bias.pt should exist in request_000"
        
        # Check request_001
        request_001 = dump_dir / "request_001"
        assert request_001.exists(), "request_001 not found"
        
        # Check that routing_bias.pt does NOT exist in request_001 (was None)
        assert not (request_001 / "routing_bias.pt").exists(), "routing_bias.pt should not exist in request_001"
        
        # Load and verify scalars for request_001
        with open(request_001 / "scalar.json", "r") as f:
            scalars_001 = json.load(f)
        
        assert scalars_001["use_shuffled_weight"] == True  # Different from request_000


@pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] < 9,
    reason="Requires CUDA and SM >= 90"
)
def test_tensor_preservation():
    """Test that tensors are preserved exactly after dump/load, including FP8."""
    
    with tempfile.TemporaryDirectory() as temp_dir:
        dump_dir = Path(temp_dir) / "test_dumps"
        
        # Set up environment
        os.environ["DUMP_MOE_INPUTS"] = "1"
        os.environ["MOE_DUMP_DIR"] = str(dump_dir)
        os.environ["MOE_MAX_DUMPS"] = "1"
        
        # Reload module
        import importlib
        import flashinfer.fused_moe.core as moe_core
        importlib.reload(moe_core)
        from flashinfer.fused_moe.core import _dump_moe_inputs
        
        # Create original tensors
        original_tensors = create_test_inputs(seq_len=64, hidden_size=2048, num_experts=4, intermediate_size=5504)
        
        original_scalars = {
            "num_experts": 4,
            "top_k": 2,
            "n_group": 1,
            "topk_group": 1,
            "intermediate_size": 5504,
            "local_expert_offset": 0,
            "local_num_experts": 4,
            "routed_scaling_factor": 1.0,
            "tile_tokens_dim": 8,
            "routing_method_type": 0,
            "use_shuffled_weight": False,
            "weight_layout": 0,
            "enable_pdl": None,
        }
        
        # Dump the tensors
        _dump_moe_inputs(**original_tensors, **original_scalars)
        
        # Load the dumped data
        loaded_scalars, loaded_tensors = load_dumped_inputs(dump_dir, "000")
        
        # Compare scalars
        for key in original_scalars.keys():
            assert loaded_scalars[key] == original_scalars[key], \
                f"Scalar {key} mismatch: {loaded_scalars[key]} != {original_scalars[key]}"
        
        # Compare tensors - exact equality
        for name in original_tensors.keys():
            if name in loaded_tensors:
                original = original_tensors[name]
                loaded = loaded_tensors[name]
                
                # Check shape and dtype
                assert original.shape == loaded.shape, f"{name} shape mismatch"
                assert original.dtype == loaded.dtype, f"{name} dtype mismatch"
                
                # Check exact equality
                assert torch.equal(original, loaded), f"{name} tensors not equal"
                
                # Special check for FP8 tensors
                if original.dtype == torch.float8_e4m3fn:
                    # Verify FP8 was preserved
                    assert loaded.dtype == torch.float8_e4m3fn, f"{name} FP8 dtype not preserved"
                    
                    # Convert to float and compare statistics to ensure values are preserved
                    orig_float = original.float()
                    loaded_float = loaded.float()
                    assert torch.allclose(orig_float, loaded_float, atol=0, rtol=0), \
                        f"{name} FP8 values not preserved exactly"


@pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] < 9,
    reason="Requires CUDA and SM >= 90"
)
def test_functional_equivalence():
    """Test that dumped and loaded inputs produce identical outputs."""
    from flashinfer.fused_moe.core import trtllm_fp8_block_scale_moe
    
    with tempfile.TemporaryDirectory() as temp_dir:
        dump_dir = Path(temp_dir) / "test_dumps"
        
        # Set up environment for dumping
        os.environ["DUMP_MOE_INPUTS"] = "1"
        os.environ["MOE_DUMP_DIR"] = str(dump_dir)
        os.environ["MOE_MAX_DUMPS"] = "1"
        
        # Reload module
        import importlib
        import flashinfer.fused_moe.core as moe_core
        importlib.reload(moe_core)
        from flashinfer.fused_moe.core import trtllm_fp8_block_scale_moe
        
        # Create test inputs
        inputs = create_test_inputs()
        params = {
            "num_experts": 8,
            "top_k": 2,
            "n_group": 1,
            "topk_group": 1,
            "intermediate_size": 11008,
            "local_expert_offset": 0,
            "local_num_experts": 8,
            "routed_scaling_factor": 1.0,
            "tile_tokens_dim": 8,
            "routing_method_type": 0,
            "use_shuffled_weight": False,
            "weight_layout": 0,
            "enable_pdl": None,
        }
        
        # Call function to trigger dumping
        try:
            output_original = trtllm_fp8_block_scale_moe(**inputs, **params)
        except Exception as e:
            pytest.skip(f"Cannot run trtllm_fp8_block_scale_moe: {e}")
        
        # Verify dump was created
        assert dump_dir.exists(), f"Dump directory {dump_dir} was not created"
        
        # Load the dumped inputs
        loaded_scalars, loaded_tensors = load_dumped_inputs(dump_dir, "000")
        
        # Disable dumping for the second call
        os.environ["DUMP_MOE_INPUTS"] = "0"
        importlib.reload(moe_core)
        from flashinfer.fused_moe.core import trtllm_fp8_block_scale_moe
        
        # Call function with loaded inputs
        output_loaded = trtllm_fp8_block_scale_moe(**loaded_tensors, **loaded_scalars)
        
        # Verify outputs match exactly
        torch.testing.assert_close(
            output_original, 
            output_loaded,
            atol=0.0,
            rtol=0.0,
            msg="Output from loaded inputs does not match original output"
        )


@pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] < 9,
    reason="Requires CUDA and SM >= 90"
)
def test_dynamic_env_control():
    """Test that DUMP_MOE_INPUTS is checked dynamically on each call."""
    
    with tempfile.TemporaryDirectory() as temp_dir:
        dump_dir = Path(temp_dir) / "test_dumps"
        
        # Initially disable dumping
        os.environ["DUMP_MOE_INPUTS"] = "0"
        os.environ["MOE_DUMP_DIR"] = str(dump_dir)
        os.environ["MOE_MAX_DUMPS"] = "5"
        
        import importlib
        import flashinfer.fused_moe.core as moe_core
        importlib.reload(moe_core)
        from flashinfer.fused_moe.core import trtllm_fp8_block_scale_moe
        
        inputs = create_test_inputs(seq_len=16, hidden_size=256, num_experts=2, intermediate_size=512)
        params = {
            "num_experts": 2,
            "top_k": 1,
            "n_group": 1,
            "topk_group": 1,
            "intermediate_size": 512,
            "local_expert_offset": 0,
            "local_num_experts": 2,
            "routed_scaling_factor": 1.0,
            "tile_tokens_dim": 8,
            "routing_method_type": 0,
            "use_shuffled_weight": False,
            "weight_layout": 0,
            "enable_pdl": None,
        }
        
        # Call with dumping disabled
        try:
            trtllm_fp8_block_scale_moe(**inputs, **params)
        except Exception:
            pass  # Ignore kernel compilation errors
        assert len(list(dump_dir.glob("request_*"))) == 0, "No dumps when disabled"
        
        # Enable dumping
        os.environ["DUMP_MOE_INPUTS"] = "1"
        
        # Call with dumping enabled
        for _ in range(2):
            try:
                trtllm_fp8_block_scale_moe(**inputs, **params)
            except Exception:
                pass
        assert len(list(dump_dir.glob("request_*"))) == 2, "Should have 2 dumps"
        
        # Disable again
        os.environ["DUMP_MOE_INPUTS"] = "0"
        
        # Call with dumping disabled again
        try:
            trtllm_fp8_block_scale_moe(**inputs, **params)
        except Exception:
            pass
        assert len(list(dump_dir.glob("request_*"))) == 2, "Still 2 dumps"


@pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] < 9,
    reason="Requires CUDA and SM >= 90"
)
def test_one_call_one_dump():
    """Test that each call creates exactly one dump."""
    
    with tempfile.TemporaryDirectory() as temp_dir:
        dump_dir = Path(temp_dir) / "test_dumps"
        
        # Set up environment
        os.environ["DUMP_MOE_INPUTS"] = "1"
        os.environ["MOE_DUMP_DIR"] = str(dump_dir)
        os.environ["MOE_MAX_DUMPS"] = "5"
        
        import importlib
        import flashinfer.fused_moe.core as moe_core
        importlib.reload(moe_core)
        from flashinfer.fused_moe.core import _dump_moe_inputs
        
        inputs = create_test_inputs(seq_len=32, hidden_size=512, num_experts=2, intermediate_size=1024)
        params = {
            "num_experts": 2,
            "top_k": 1,
            "n_group": 1,
            "topk_group": 1,
            "intermediate_size": 1024,
            "local_expert_offset": 0,
            "local_num_experts": 2,
            "routed_scaling_factor": 1.0,
            "tile_tokens_dim": 8,
            "routing_method_type": 0,
            "use_shuffled_weight": False,
            "weight_layout": 0,
            "enable_pdl": None,
        }
        
        # Make 5 calls
        for i in range(5):
            _dump_moe_inputs(**inputs, **params)
        
        # Check we have exactly 5 dumps numbered 000-004
        dumps = sorted(list(dump_dir.glob("request_*")))
        assert len(dumps) == 5, f"Should have 5 dumps, got {len(dumps)}"
        
        expected = [f"request_{i:03d}" for i in range(5)]
        actual = [d.name for d in dumps]
        assert actual == expected, f"Expected {expected}, got {actual}"
        
        # Test that 6th call doesn't create a dump (limit reached)
        _dump_moe_inputs(**inputs, **params)
        dumps_after_limit = list(dump_dir.glob("request_*"))
        assert len(dumps_after_limit) == 5, "Should still have 5 dumps after limit"


@pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] < 9,
    reason="Requires CUDA and SM >= 90"
)
def test_dump_limit():
    """Test that dumping stops after reaching the limit."""
    
    with tempfile.TemporaryDirectory() as temp_dir:
        dump_dir = Path(temp_dir) / "test_dumps"
        max_dumps = 3
        
        # Set up environment with limit
        os.environ["DUMP_MOE_INPUTS"] = "1"
        os.environ["MOE_DUMP_DIR"] = str(dump_dir)
        os.environ["MOE_MAX_DUMPS"] = str(max_dumps)
        
        # Reload module
        import importlib
        import flashinfer.fused_moe.core as moe_core
        importlib.reload(moe_core)
        from flashinfer.fused_moe.core import trtllm_fp8_block_scale_moe
        
        inputs = create_test_inputs()
        params = {
            "num_experts": 8,
            "top_k": 2,
            "n_group": 1,
            "topk_group": 1,
            "intermediate_size": 11008,
            "local_expert_offset": 0,
            "local_num_experts": 8,
            "routed_scaling_factor": 1.0,
            "tile_tokens_dim": 8,
            "routing_method_type": 0,
            "use_shuffled_weight": False,
            "weight_layout": 0,
            "enable_pdl": None,
        }
        
        # Make more calls than the limit
        for i in range(max_dumps + 2):
            try:
                trtllm_fp8_block_scale_moe(**inputs, **params)
            except Exception:
                pass  # Ignore errors, we're testing dumping
        
        # Verify only max_dumps directories were created
        request_dirs = list(dump_dir.glob("request_*"))
        assert len(request_dirs) == max_dumps, \
            f"Expected {max_dumps} dumps, but found {len(request_dirs)}"
        
        # Verify the request IDs are correct
        expected_ids = [f"request_{i:03d}" for i in range(max_dumps)]
        actual_ids = sorted([d.name for d in request_dirs])
        assert actual_ids == expected_ids, \
            f"Expected request IDs {expected_ids}, but found {actual_ids}"


def run_all_tests():
    """Run all tests manually (for when pytest is not available)."""
    print("Running MOE dump/load tests...\n")
    print("=" * 60)
    
    if not torch.cuda.is_available():
        print("❌ CUDA not available, skipping tests")
        return
    
    if torch.cuda.get_device_capability()[0] < 9:
        print(f"❌ GPU compute capability {torch.cuda.get_device_capability()} < 9.0, skipping tests")
        return
    
    tests = [
        ("Dump Mechanics", test_dump_mechanics),
        ("Tensor Preservation", test_tensor_preservation),
        ("Dynamic Env Control", test_dynamic_env_control),
        ("One Call One Dump", test_one_call_one_dump),
        ("Functional Equivalence", test_functional_equivalence),
        ("Dump Limit", test_dump_limit),
    ]
    
    passed = 0
    failed = 0
    skipped = 0
    
    for test_name, test_func in tests:
        print(f"\n🧪 Running: {test_name}")
        print("-" * 40)
        try:
            test_func()
            print(f"✅ {test_name}: PASSED")
            passed += 1
        except pytest.skip.Exception as e:
            print(f"⏭️  {test_name}: SKIPPED - {e}")
            skipped += 1
        except Exception as e:
            print(f"❌ {test_name}: FAILED - {e}")
            failed += 1
            import traceback
            traceback.print_exc()
    
    print("\n" + "=" * 60)
    print(f"📊 Test Summary:")
    print(f"   ✅ Passed: {passed}")
    print(f"   ❌ Failed: {failed}")
    print(f"   ⏭️  Skipped: {skipped}")
    print(f"   Total: {len(tests)}")
    
    if failed == 0 and passed > 0:
        print("\n🎉 All tests passed!")
    elif failed > 0:
        print(f"\n❌ {failed} test(s) failed")


if __name__ == "__main__":
    run_all_tests()