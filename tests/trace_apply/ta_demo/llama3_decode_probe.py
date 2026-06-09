import glob
import logging
import os

logging.getLogger("flashinfer.trace_apply").setLevel(logging.INFO)
_HF = os.environ.get("HF_HOME", "/opt/dlami/nvme/yongwww/hf_cache")
_snaps = sorted(
    glob.glob(f"{_HF}/hub/models--NousResearch--Meta-Llama-3-8B-Instruct/snapshots/*")
)
MODEL = _snaps[-1] if _snaps else "NousResearch/Meta-Llama-3-8B-Instruct"


def main():
    import sglang as sgl

    llm = sgl.Engine(
        model_path=MODEL,
        attention_backend="flashinfer",
        disable_cuda_graph=True,
        disable_piecewise_cuda_graph=True,
        tp_size=1,
        mem_fraction_static=0.80,
        disable_radix_cache=True,
        log_level="info",
    )
    out = llm.generate(
        ["The capital of France is"], {"temperature": 0.0, "max_new_tokens": 8}
    )
    for o in out:
        print("GEN:", repr(o["text"][:60]))
    llm.shutdown()


if __name__ == "__main__":
    main()
