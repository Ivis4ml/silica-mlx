# Overview

silica-mlx is an inference framework written **MLX-native** — every
hot-path tensor on the inference loop is `mx.array`, with no PyTorch
runtime dependency. The library sits at three layers:

1. a vLLM-style core (continuous batcher, paged KV cache, radix prefix
   cache, request lifecycle, sampling);
2. a pluggable KV codec stack (BlockTurboQuant, RaBitQ-1, ExtRaBitQ),
   integrated via the prefix-block store under D-003 — no
   compressed-domain attention in v0.1;
3. a chat REPL plus an OpenAI-compatible HTTP server that exercises the
   framework's signature features (cross-turn KV reuse, codec savings,
   admission headroom).

## Quickstart

The bundled `silica` CLI exposes both the chat REPL and the HTTP
server. Bare invocation drops into the chat client:

```bash
# install with optional extras
pip install -e '.[chat,serve]'

# chat against a local model
silica --model Qwen/Qwen3-0.6B --kv-codec block_tq_b64_b4

# OpenAI-compatible server (P-7+)
silica run --model Qwen/Qwen3-0.6B
```

For the full guide on the chat client, see {doc}`chat-cli`.

## Module layout

The `silica` namespace splits into one subpackage per concern:

| Subpackage | Responsibility |
| --- | --- |
| `silica.core` | Request FSM, sampling params, profiling hooks |
| `silica.engine` | Top-level Engine driving generate / generate_batch |
| `silica.scheduler` | ContinuousBatcher, MemoryBudgeter, prefix admission |
| `silica.kvcache` | PagedKVCache, RadixPrefixCache, KVCodec protocol |
| `silica.models` | ModelAdapter protocol + Qwen3, Qwen3.5, Gemma4 adapters |
| `silica.weights` | WeightProvider protocol + resident impls |
| `silica.vq` | BlockTQ / RaBitQ codec implementations |
| `silica.speculative` | Speculative decoding (P-6+) |
| `silica.bench` | Benchmark runner over Engine |
| `silica.chat` | Chat session + CLI front-end |
| `silica.server` | CLI entry point + HTTP server |
| `silica.llm` | High-level Python API (P-8+) |

The auto-generated reference under {doc}`api/index` walks every public
symbol; the hand-curated **{doc}`api-manual`** classifies each symbol
as public / internal / protocol / stub and explains why downstream
units depend on it.

## Where the design lives

Every architectural decision and every acceptance gate is recorded in
`plans/` — `PLAN.md` is the single source of truth, accompanied by per-phase
opening / prep / survey / acceptance documents and the raw measurement
artifacts under `plans/P*_INVESTIGATION/` and `plans/P*_ACCEPTANCE_SWEEP/`.
The {doc}`plans-index` page in this site offers a curated entry into
that material.

## Hardware target

Primary inference target is the M5 Pro 48 GB. The codec stack and the
admission budgeter are tuned around that envelope: a Qwen3.5-27B-class
model in fp16 dominates the budget by itself; under BlockTQ the prefix
store compresses 3.8x, freeing headroom that the budgeter then turns
into additional admitted requests (P-5 acceptance evidence in
`plans/P5_ACCEPTANCE_SWEEP/admission_headroom.md`).
