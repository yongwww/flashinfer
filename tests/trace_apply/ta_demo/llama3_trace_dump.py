"""Run a short Llama-3-8B generation under SGLang with the flashinfer attention
backend and FLASHINFER_TRACE_DUMP=1, to capture which flashinfer APIs (and
shapes) the model actually exercises. Prints the captured definition files.
"""

import glob
import os
import sys

import sglang as sgl

MODEL = "/opt/dlami/nvme/yongwww/hf_cache/hub/models--NousResearch--Meta-Llama-3-8B-Instruct/snapshots/53346005fb0ef11d3b6a83b12c895cca40156b6c"
DUMP_DIR = os.environ["FLASHINFER_TRACE_DUMP_DIR"]


def main() -> int:
    llm = sgl.Engine(
        model_path=MODEL,
        attention_backend="flashinfer",
        disable_cuda_graph=True,  # eager → flashinfer ops run outside capture
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

    files = sorted(
        os.path.basename(p) for p in glob.glob(os.path.join(DUMP_DIR, "*.json"))
    )
    print(f"\n=== {len(files)} flashinfer definitions captured in {DUMP_DIR} ===")
    for f in files:
        print("  ", f)
    return 0


if __name__ == "__main__":
    sys.exit(main())
