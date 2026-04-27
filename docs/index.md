# silica-mlx documentation

silica-mlx is an MLX-native LLM inference platform for Apple Silicon.
It targets Qwen3.5-27B and Gemma 4 31B class models on a 48 GB M5 Pro,
with a vLLM-style continuous batcher, a radix prefix cache, and a
pluggable KV codec stack (BlockTQ + RaBitQ) wired into the prefix
store.

This site bundles four kinds of material:

- a high-level **overview** of what the framework does and how to use it;
- the auto-generated **API reference** for every public class, function,
  and protocol in `silica.*`;
- the **chat CLI** guide for the bundled REPL client;
- a curated index into the **design and acceptance plans** that drove
  the implementation (`plans/`).

```{toctree}
:maxdepth: 2
:caption: Get started

overview
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
