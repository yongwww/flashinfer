.. _trace_apply_tutorial:

Tutorial: Swapping Kernels at Runtime with Trace Apply
======================================================

This tutorial walks through **Trace Apply**, FlashInfer's mechanism for
replacing a FlashInfer kernel with your own implementation *at runtime, without
touching the serving engine's code*. By the end you will:

* understand **what** Trace Apply is and **why** it exists,
* register a custom kernel and watch it run from a tiny Python snippet, and
* verify the substitution end-to-end inside an **unmodified SGLang Llama-3-8B**
  server.

For the terse API reference, see :ref:`trace_apply`. This page is the gentle
introduction.

What is Trace Apply?
--------------------

Trace Apply is the **consumer side of the FlashInfer Trace**.

The FlashInfer Trace is the producer side: every ``@flashinfer_api`` function can
emit a *definition* — a portable description of one operation specialized to a
concrete shape (its name, axes, input/output dtypes, a reference implementation,
and a correctness check). A collection pipeline runs a real model on a real GPU
and writes out one definition per distinct operation shape it sees.

Trace Apply closes the loop. You hand it a mapping from **definition name** to a
*solution* (the kernel you want to run for that definition), and from then on
every matching FlashInfer call is transparently dispatched to your solution
instead of the built-in kernel:

.. code-block:: text

   collect                      optimize / author                apply
   ───────                      ─────────────────                ─────
   run model on GPU  ─────────▶  agent or human writes   ───────▶  enable_apply({name: solution})
   → definitions                 a faster solution                 → engine now runs your kernel
     (per operation shape)        for a definition                   with zero code changes

The key property: **no changes to the calling code.** SGLang, vLLM, or any other
engine keeps calling ``flashinfer.rmsnorm(...)`` exactly as before. Trace Apply
wraps the FlashInfer API in place, so the substitution is invisible to the caller.

Why do we need it?
------------------

FlashInfer is consumed deep inside inference engines. Trying a new kernel the
naive way means forking and rebuilding the engine, threading a flag through its
layers, and redeploying — for *every* kernel idea you want to measure. That
friction is exactly what kills fast iteration on kernels.

Trace Apply removes it:

* **Measure real end-to-end impact, not microbenchmarks.** A kernel that wins in
  isolation can lose under real launch overhead, memory pressure, and CUDA-graph
  capture. Trace Apply lets you drop a candidate into a live server and read the
  actual tokens/sec and output quality.
* **Zero-code, zero-rebuild.** Point an environment variable at a folder of
  solutions and restart the server. No engine fork, no recompile.
* **It is the runtime half of an automated loop.** The trace collector produces
  definitions; an agent (or a human) authors a faster solution for a definition;
  Trace Apply applies it so the loop can score the result on the real workload.
* **Shape-correct by construction.** A solution is bound to a *definition name*,
  which encodes the exact specialized shape it was written for, so it only fires
  for calls it actually fits.

How does routing work?
----------------------

A definition name encodes the operation plus its **const axes** — the
compile-time shape a kernel is specialized for. For example RMSNorm at
``hidden_size = 4096`` has the definition name ``rmsnorm_h4096``.

On every call, Trace Apply reads the call's const axes and recomputes this name
using the same convention the collector used. If the name is in your mapping, the
call dispatches to your solution; otherwise it falls back to the original
FlashInfer kernel. **Variable axes** (batch size, sequence length, …) are not part
of the name, so one solution serves all of their values. The decision is cached
per name, so steady-state dispatch is just a dict lookup.

How to use it
-------------

The whole public surface is four functions:

* :func:`~flashinfer.trace_apply.enable_apply` — register a
  ``{definition_name: solution}`` mapping; returns the number of wrapped APIs.
* :func:`~flashinfer.trace_apply.disable_apply` — restore the original kernels.
* :func:`~flashinfer.trace_apply.is_enabled` — whether Trace Apply is active.
* :func:`~flashinfer.trace_apply.stats` — per-API dispatch counts (hit / fallback
  / error).

A *solution* is either a plain Python callable or a first-class
:class:`~flashinfer.trace.Solution` (a JSON-described kernel, possibly C++/CUDA,
built on demand). The callable form is the easiest way to get started:

.. code-block:: python

   import torch
   import flashinfer
   import flashinfer.trace_apply as fi_trace_apply

   # A custom RMSNorm specialized for hidden_size == 4096.
   def my_rmsnorm(hidden_states, weight, eps=1e-6):
       x = hidden_states.float()
       y = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps)
       return (y * weight.float()).to(hidden_states.dtype)

   # Route the "rmsnorm_h4096" definition to our kernel.
   n = fi_trace_apply.enable_apply({"rmsnorm_h4096": my_rmsnorm})
   print(f"Trace Apply wrapped {n} FlashInfer APIs")

   x = torch.randn(8, 4096, device="cuda", dtype=torch.bfloat16)
   w = torch.randn(4096, device="cuda", dtype=torch.bfloat16)

   out = flashinfer.rmsnorm(x, w)   # hidden_size == 4096 → dispatched to my_rmsnorm
   _   = flashinfer.rmsnorm(torch.randn(8, 2048, device="cuda", dtype=torch.bfloat16),
                            torch.randn(2048, device="cuda", dtype=torch.bfloat16))
                                    # hidden_size == 2048 → no match, original kernel

   print(fi_trace_apply.stats())    # {'flashinfer.norm.rmsnorm': {'hit': 1, 'fallback_no_candidate': 1}}
   fi_trace_apply.disable_apply()   # back to stock FlashInfer

``enable_apply`` is idempotent: calling it again replaces the previous mapping.
Trace Apply also adapts your solution's outputs to the calling API's convention —
value-returning results, caller-supplied ``out=`` buffers, in-place writes, and
data-dependent arity all just work, so ``flashinfer.rmsnorm(x, w, out=buf)`` still
writes into ``buf`` and returns it.

Deploying into a server (the environment-variable path)
-------------------------------------------------------

Inside a real engine you usually cannot call ``enable_apply`` yourself — the
worker processes are spawned by SGLang or vLLM. For that case there is an
import-time hook driven by two environment variables:

============================================ ===========================================================
Variable                                     Meaning
============================================ ===========================================================
``FLASHINFER_TRACE_APPLY=1``                 Enable Trace Apply when FlashInfer is imported.
``FLASHINFER_TRACE_APPLY_PATH=<folder>``     A *curated* solutions folder to load (one solution per
                                             definition; its ``solutions/`` subtree is scanned).
============================================ ===========================================================

Every spawned worker imports FlashInfer, sees the flag, and loads the solutions —
no engine change required. If the configuration is missing or invalid, Trace
Apply stays disabled and FlashInfer runs normally (with a warning).

.. note::

   ``FLASHINFER_TRACE_APPLY_PATH`` points at a *curated* folder with exactly one
   solution per definition — not the raw extraction bundle, which also holds
   baseline solutions and multiple backends per definition. Picking which
   solution to apply is an upstream (collection/agent) step.

End-to-end verification with SGLang Llama-3-8B
----------------------------------------------

Here is a complete check that the substitution fires inside a real model server.
We launch an **unmodified** SGLang Llama-3-8B engine; because Llama-3-8B has
``hidden_size == 4096``, every RMSNorm in the model matches ``rmsnorm_h4096`` and
is routed to our solution.

The demo lives at ``ta_demo/llama3_apply_demo.py``:

.. code-block:: python

   import sglang as sgl
   import flashinfer.trace_apply as ta

   print("trace_apply enabled in driver proc:", ta.is_enabled())

   llm = sgl.Engine(
       model_path="NousResearch/Meta-Llama-3-8B-Instruct",
       attention_backend="flashinfer",
       disable_cuda_graph=True,          # pure eager → every rmsnorm hits dispatch
       disable_piecewise_cuda_graph=True,
   )
   out = llm.generate(
       ["The capital of France is", "Q: 2+2? A:"],
       {"temperature": 0.0, "max_new_tokens": 16},
   )
   for o in out:
       print("GEN:", repr(o["text"][:80]))
   llm.shutdown()

Run it with the two environment variables pointed at a curated solutions folder
(the repo ships one under ``ta_demo/demo_trace`` whose solution prints a marker
when it runs):

.. code-block:: bash

   export FLASHINFER_TRACE_APPLY=1
   export FLASHINFER_TRACE_APPLY_PATH=/path/to/ta_demo/demo_trace
   python ta_demo/llama3_apply_demo.py

You should see Trace Apply announce the substitution, the solution's one-time
marker, and correct generations:

.. code-block:: text

   Trace Apply: applying solution for definition 'rmsnorm_h4096' on flashinfer.norm.rmsnorm (dps=False).
   [TRACE_APPLY_SOLUTION] torch_rmsnorm_h4096 invoked (trace solution running)
   GEN: ' Paris, which is located in the north-central part of the country. Paris is'
   GEN: ' 4\nQ: 3+3? A: 6\nQ'

The model produced correct text while *every* RMSNorm ran through the substituted
kernel — a zero-code swap, verified end-to-end.

Where to go next
----------------

* :ref:`trace_apply` — the full API reference and error policy.
* :ref:`fi_trace` — how definitions are produced (the trace producer side).
