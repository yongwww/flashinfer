# MOE Input Dumping Feature

This feature allows dumping inputs to the `trtllm_fp8_block_scale_moe` function during inference for debugging and analysis.

## Quick Start

### Enable Dumping During Inference
```bash
# Basic usage - dumps to ./dumps/
export DUMP_MOE_INPUTS=1
python your_inference_script.py

# Custom settings
export DUMP_MOE_INPUTS=1
export MOE_DUMP_DIR=/path/to/dumps  # Custom directory
export MOE_MAX_DUMPS=50             # Limit to 50 requests (default: 100)
python your_inference_script.py
```

### Skip Warmup Phase
```python
import os

# During warmup - disable dumping
os.environ["DUMP_MOE_INPUTS"] = "0"

# Run warmup iterations
for i in range(warmup_iterations):
    model.forward(...)

# After warmup - enable dumping
os.environ["DUMP_MOE_INPUTS"] = "1"

# Run actual inference (will dump)
for request in requests:
    model.forward(...)
```

### Directory Structure
```
dumps/
├── request_000/
│   ├── scalar.json            # All scalar parameters
│   ├── routing_logits.pt      # Tensor: routing logits
│   ├── routing_bias.pt        # Tensor: routing bias (if not None)
│   ├── hidden_states.pt       # Tensor: FP8 hidden states
│   ├── hidden_states_scale.pt # Tensor: hidden states scale
│   ├── gemm1_weights.pt       # Tensor: first layer weights (FP8)
│   ├── gemm1_weights_scale.pt # Tensor: first layer scales
│   ├── gemm2_weights.pt       # Tensor: second layer weights (FP8)
│   └── gemm2_weights_scale.pt # Tensor: second layer scales
├── request_001/
│   └── ...
└── request_099/  # Max 100 requests by default
```

## Utility Scripts

### 1. Load and Replay Dumps (`load_moe_dumps.py`)
```bash
# List all dumps
python load_moe_dumps.py --list

# View specific dump details
python load_moe_dumps.py --request 0 --print-scalars --print-shapes

# Replay a dump (call function with saved inputs)
python load_moe_dumps.py --request 0 --replay
```

### 2. Manage Dumps (`manage_moe_dumps.py`)
```bash
# Show statistics
python manage_moe_dumps.py --stats

# Show disk usage breakdown
python manage_moe_dumps.py --disk-usage

# Clean dumps older than 7 days
python manage_moe_dumps.py --clean-older-than 7

# Clean all dumps
python manage_moe_dumps.py --clean
```

## Testing

**`tests/test_moe_dump_load.py`** - Comprehensive test suite that verifies:
- Dumping mechanics work correctly
- Tensors and scalars are preserved exactly after dump/load
- FP8 dtypes are maintained
- None values are handled properly
- Dump limits are enforced
- Loaded inputs produce identical outputs (when kernel available)

Run tests:
```bash
# Run all tests
python tests/test_moe_dump_load.py

# Or use pytest
pytest tests/test_moe_dump_load.py -v
```

## Implementation Details

### Code Changes
The implementation adds:
1. Import statements for dumping utilities
2. Global variables to track dumping state (`_dump_counter`, `_dump_lock`, etc.)
3. `_dump_moe_inputs()` helper function
4. Call to dump function in `trtllm_fp8_block_scale_moe()` when enabled

### Features
- **One call = One dump**: Each call to `trtllm_fp8_block_scale_moe` creates exactly one dump
- **Sequential numbering**: Dumps are numbered sequentially (request_000, request_001, ...)
- **Dynamic control**: Environment variables are checked on each call - can toggle dumping at runtime
- **Skip warmup**: Easily disable dumps during warmup, enable after warmup completes
- **Thread-safe**: Uses locks for concurrent access
- **Limited dumps**: Stops at configurable limit (default 100 calls = 100 dumps)
- **FP8 preservation**: Maintains original tensor dtypes
- **Zero overhead**: No impact when disabled
- **None handling**: Correctly handles optional parameters
- **Session behavior**: New sessions (program restarts) will overwrite from request_000

### scalar.json Format
```json
{
  "request_id": "000",
  "timestamp": "2024-01-01T12:00:00.000000",
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
  "use_shuffled_weight": false,
  "weight_layout": 0,
  "enable_pdl": null
}
```

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `DUMP_MOE_INPUTS` | `"0"` | Set to `"1"` to enable dumping |
| `MOE_DUMP_DIR` | `"./dumps"` | Directory to save dumps |
| `MOE_MAX_DUMPS` | `"100"` | Maximum number of requests to dump |

## Troubleshooting

1. **No dumps created**: Ensure `DUMP_MOE_INPUTS=1` is set
2. **Dumps stop at limit**: Check `MOE_MAX_DUMPS` setting
3. **Disk space**: Use `manage_moe_dumps.py --disk-usage` to check usage
4. **Old dumps**: Use `--clean-older-than` to remove old dumps
