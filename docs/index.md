# silica-mlx documentation

silica-mlx is an MLX-native LLM inference platform for Apple Silicon.
It targets Qwen3.5-27B and Gemma 4 31B class models on a 48 GB M5 Pro,
with a vLLM-style continuous batcher, a radix prefix cache, and a
pluggable KV codec stack (BlockTQ + RaBitQ) wired into the prefix
store.

The P-6 performance phase is closed. The 35-cycle autoresearch loop
cleared every *server-throughput* acceptance gate 3.4-5.5× over the
cycle-1 baseline: **232 tok/s on dense Qwen3.5-27B-4bit at B=64
(48 GB ceiling)**, **791.8 tok/s on MoE Qwen3.5-35B-A3B-4bit at
B=128**. These are server-aggregate numbers; **single-user (B=1)
latency on M5 Pro is bandwidth-capped near 20 tok/s and unchanged by
this phase**. D-022 then closed the small-B single-user research line
at v1.7.28 with β/γ/δ measurement-anchored negatives and ≤0.6%
recoverable Python-hygiene headroom. See {doc}`performance` for the
full record (headline numbers, the two load-bearing levers, the
per-row math, D-022 closure, and the honest retraction record).

This site bundles five kinds of material:

- a high-level **overview** of what the framework does and how to use it;
- the **performance** report from the P-6 autoresearch loop;
- the auto-generated **API reference** for every public class, function,
  and protocol in `silica.*`;
- the **chat CLI** guide for the bundled REPL client;
- a curated index into the **design and acceptance plans** that drove
  the implementation (`plans/`).

```{toctree}
:maxdepth: 2
:caption: Get started

overview
performance
chat-cli
bench
```

```{toctree}
:maxdepth: 2
:caption: Reference

api/index
api-manual
```

```{toctree}
:maxdepth: 2
:caption: Plans and design

plans-index
```

## Indices

- {ref}`genindex`
- {ref}`modindex`
- {ref}`search`
