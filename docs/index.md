# silica-mlx documentation

silica-mlx is an MLX-native LLM inference platform for Apple Silicon.
It targets Qwen3.5-27B and Gemma 4 31B class models on a 48 GB M5 Pro,
with a vLLM-style continuous batcher, a radix prefix cache, and a
pluggable KV codec stack (BlockTQ + RaBitQ) wired into the prefix
store.

The 35-cycle P-6 autoresearch loop closed in May 2026 with every
*server-throughput* acceptance gate cleared 3.4-5.5× over the cycle-1
baseline: **232 tok/s on dense Qwen3.5-27B-4bit at B=64 (48 GB ceiling)**,
**791.8 tok/s on MoE Qwen3.5-35B-A3B-4bit at B=128**. These are
server-aggregate numbers; **single-user (B=1) latency on M5 Pro is
bandwidth-capped near 20 tok/s and unchanged by this phase**. Closing
per-step time at small batch is the **D-022** research line, in
progress at v1.7.24. See {doc}`performance` for the full record
(headline numbers, the two load-bearing levers, the per-row math,
and the honest closures + cycle-27 retraction).

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
