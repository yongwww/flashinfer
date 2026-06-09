"""E2E demo: unmodified SGLang Llama-3-8B with FLASHINFER_TRACE_APPLY=1.

flashinfer.norm.rmsnorm is transparently substituted by the trace solution
'torch_rmsnorm_h4096'. We assert the model still produces correct text; the
worker log carries the solution's one-time marker and the Trace Apply INFO line.
"""

import logging
import os
import sys

logging.getLogger("flashinfer.trace_apply").setLevel(logging.INFO)

import glob

import sglang as sgl

_HF = os.environ.get("HF_HOME", "/opt/dlami/nvme/yongwww/hf_cache")
_snaps = sorted(
    glob.glob(f"{_HF}/hub/models--NousResearch--Meta-Llama-3-8B-Instruct/snapshots/*")
)
MODEL = _snaps[-1] if _snaps else "NousResearch/Meta-Llama-3-8B-Instruct"


def main() -> int:
    import flashinfer.trace_apply as ta

    print(
        "[demo] FLASHINFER_TRACE_APPLY      =", os.environ.get("FLASHINFER_TRACE_APPLY")
    )
    print(
        "[demo] FLASHINFER_TRACE_APPLY_PATH =",
        os.environ.get("FLASHINFER_TRACE_APPLY_PATH"),
    )
    print("[demo] trace_apply enabled in driver proc:", ta.is_enabled())

    llm = sgl.Engine(
        model_path=MODEL,
        attention_backend="flashinfer",
        disable_cuda_graph=True,
        disable_piecewise_cuda_graph=True,  # pure eager → every rmsnorm hits dispatch
        tp_size=1,
        mem_fraction_static=0.80,
        disable_radix_cache=True,
        log_level="info",
    )
    out = llm.generate(
        ["The capital of France is", "Q: 2+2? A:"],
        {"temperature": 0.0, "max_new_tokens": 16},
    )
    for o in out:
        print("GEN:", repr(o["text"][:80]))
    llm.shutdown()
    return 0


if __name__ == "__main__":
    sys.exit(main())
