# Silica-MLX

MLX-native LLM inference platform for Apple Silicon. vLLM-style core (paged KV,
continuous batching, prefix cache, memory budget) plus a mini-sglang outer layer
planned for Phase 8. Target: run dense 27B-31B class models on a 48 GB M5 Pro.

> **Status: P-2 complete, P-3 in progress.** Today the package runs
> single-request and batched generation end-to-end on both plain Qwen3
> (0.6B / 4B / 7B / ...) and hybrid DeltaNet Qwen3.5-0.8B, with
> shared-prefix caching and budget-aware preemption on the plain-Qwen3
> path. Qwen3.5-27B loads cleanly via `mlx_lm.load` and runs
> single-request; its batched path uses the same hybrid scheduler as
> 0.8B and is ready for benchmarking. Gemma4-31B, MoE variants
> (Qwen3.5-35B-A3B / gemma-4-26B-A4B), VQ KV compression, weight
> streaming, and speculative decoding are stubbed at the interface
> level but not yet implemented. See [`docs/PLAN.md`](docs/PLAN.md) for
> the full roadmap and [`docs/API.md`](docs/API.md) for the per-module
> function reference.

---

## Install

Requires Python 3.12+ and an Apple Silicon Mac. Managed via [uv](https://github.com/astral-sh/uv).

```bash
uv pip install -e .
# or, for the planned HTTP-serve extras (not yet wired):
uv pip install -e '.[serve]'
```

Runtime dependencies: `mlx>=0.22`, `mlx-lm>=0.18`, `numpy>=1.26`, `pydantic>=2`.
Dev deps (`pytest`, `ruff`, `mypy`) live in the `dev` group.

---

## Quickstart

### 1. CLI — single prompt

```bash
python -m silica run \
    --model Qwen/Qwen3-0.6B \
    --prompt "The capital of France is" \
    --max-tokens 64 \
    --temperature 0.0
```

Prints `prompt + generation` to stdout and a one-line metrics row
(`ttft=… prefill=…tok/s decode=…tok/s resident=…MB`) to stderr. The console
script `silica run …` is equivalent once the package is installed.

### 2. Python API — single request

```python
from silica import Engine
from silica.core.sampling import SamplingParams
from silica.models.factory import adapter_for_repo

adapter, kv = adapter_for_repo("Qwen/Qwen3-0.6B")
engine = Engine(adapter, kv)

tokenizer = adapter.tokenizer()
params = SamplingParams(
    temperature=0.7,
    top_p=0.9,
    max_tokens=128,
    stop_token_ids=tuple(tokenizer.eos_token_ids or ()),
)

token_ids = list(engine.generate("Write a haiku about silicon.", params))
print(tokenizer.decode(token_ids))
print(engine.metrics.snapshot())
```

`Engine.generate` is an iterator — consume it however you want (streaming
to stdout, collecting into a list, piping into a detokenizer). The iterator
auto-releases KV on exhaustion or exception.

### 3. Python API — continuous batching (8 concurrent)

```python
from silica import Engine
from silica.core.sampling import SamplingParams
from silica.kvcache.prefix import RadixPrefixCache
from silica.kvcache.store import SyntheticPrefixBlockStore
from silica.models.factory import adapter_for_repo

adapter, kv = adapter_for_repo("Qwen/Qwen3-0.6B")
engine = Engine(adapter, kv)

# Optional: prefix cache persists across generate_batch calls.
block_size = 16
pc = RadixPrefixCache(
    block_size=block_size,
    store=SyntheticPrefixBlockStore(block_size=block_size),
)

prompts = [
    "The capital of France is",
    "The capital of Germany is",
    "The capital of Japan is",
    # ... up to len(prompts) prompts; max_batch_size bounds active rows
]
params = SamplingParams(temperature=0.0, max_tokens=64)

outputs: dict[int, list[int]] = {}
for event in engine.generate_batch(
    prompts, params, max_batch_size=8, prefix_cache=pc
):
    if event.kind == "token":
        outputs.setdefault(event.req_index, []).append(event.token_id)
    elif event.kind == "done":
        print(f"[{event.req_index}] done: {event.finish_reason}")
    elif event.kind == "aborted":
        print(f"[{event.req_index}] aborted: {event.finish_reason}")
```

`generate_batch` yields [`BatchEvent`](docs/API.md#batchevent) values of three
kinds: `token` (one per emitted token), `done` (normal terminal), and
`aborted` (budget exhaustion or error). The `req_index` matches the position
in the `prompts` list. For budget-aware scheduling, construct a
[`MemoryBudgeter`](docs/API.md#memorybudgeter) and install it on a
`ContinuousBatcher` directly (see the API doc for the lower-level path).

---

## What works today

| Capability | Status | Where |
| --- | --- | --- |
| Single-request generation (greedy / temperature / top-k / top-p / repetition penalty) | ✅ | `silica.engine.Engine.generate` |
| Token streaming as iterator | ✅ | same |
| Per-request metrics (TTFT / throughput / resident MB) | ✅ | `Engine.metrics.snapshot()` |
| 8-way continuous batching with left-padded prefill | ✅ | `silica.scheduler.ContinuousBatcher` |
| Mid-run admission when slots free | ✅ | `Engine.generate_batch(max_batch_size=...)` |
| Shared-prefix caching (block-granular radix trie) | ✅ | `silica.kvcache.prefix.RadixPrefixCache` |
| Memory-budget admission: admit → evict → preempt → reject ladder | ✅ | `silica.scheduler.MemoryBudgeter` |
| Preempt + replay (save composite prompt, re-enter queue) | ✅ | `ContinuousBatcher._apply_preempt` |
| Plain Qwen3 family adapter (0.6B / 4B / 7B / 14B / 32B) | ✅ | `silica.models.qwen3.Qwen3Adapter` |
| Qwen3.5 hybrid family adapter (0.8B) — single-request + batched (greedy parity pinned) | ✅ | `silica.models.qwen3_5.Qwen3_5Adapter`; parity vs single-request in `tests/test_p3_hybrid_batched_parity.py` |
| CLI: `python -m silica run` | ✅ | `silica.server.cli` |
| Qwen3.5-27B load via `mlx_lm.load` verified (single-request) | ✅ | `scripts/probe_qwen3_5_27b_load.py` (batched validation pending bench) |
| DeltaNet recurrent state + `state_delta` plumbing (single-request + batched path) | ✅ | D-015 + P-3-C0..C3d; `Qwen3_5Adapter.make_batch_cache` interleaves `ArraysCache` / `BatchKVCache` per layer |
| Gemma4-31B dense adapter — single-request + batched miss-only path (sliding + full attention hybrid) | ✅ | `silica.models.gemma4.Gemma4Adapter`; single-request smoke in `tests/test_p3_gemma4_single_request_smoke.py`; batched smoke in `tests/test_p3_gemma4_batched_smoke.py` (dual-gated); **batched parity** pending P-3-D3.1; **prefix-cache + SLIDING** rejected at construction (P-3-D3 local follow-up) |
| MoE adapters (Qwen3.5-35B-A3B / gemma-4-26B-A4B) | ⏳ | P-3 |
| Preempt/replay with recurrent state snapshot | ⏳ | P-3-C5 |
| VQ KV compression (BlockTQ / RaBitQ) | Stub | P-5 (`IdentityCodec` today) |
| Weight streaming | Stub | P-6 (`ResidentWeightProvider` today) |
| Speculative decoding (DraftTarget / EAGLE / Medusa) | Stub | P-7 (`NoopDraftEngine` today) |
| OpenAI-compatible HTTP server + session layer | ⏳ | P-8 |
| Unified benchmark harness | ⏳ | P-4 |

Legend: ✅ shipped · Stub = wired to the main loop as the fp16 baseline,
real implementation in the named phase · ⏳ = not started.

---

## Running the tests

```bash
uv run pytest tests           # full suite (~510 tests)
uv run ruff check .
uv run mypy silica
```

The test suite includes the PLAN §7 P-2 acceptance triad:

- `tests/test_p2_batched_parity.py::test_generate_batch_real_8_concurrent` — 8 concurrent prompts run stably.
- `tests/test_p2_batched_parity.py::test_prefix_hit_reduces_forward_tokens` — shared-prefix hit verifiable from `ContinuousBatcher.forward_prompt_tokens`.
- `tests/test_p2_batched_parity.py::test_budget_overflow_aborts_cleanly` — budget overflow queues (preempt branch) or aborts (reject branch) without crashing.

Real-model tests download Qwen3-0.6B via mlx-lm on first run.

---

## Layout

```text
silica/
    __init__.py           # re-exports Engine
    __main__.py           # python -m silica → CLI
    core/                 # events, logger, profiler, request FSM, sampler, sampling params
    kvcache/              # I-2 KVManager, I-3 KVCodec, paged bookkeeping, radix prefix cache, prefix block store
    scheduler/            # ContinuousBatcher, MemoryBudgeter, seed_kv helper
    models/               # I-1 ModelAdapter + family adapters (qwen3, qwen3_5) + factory
    mlx/                  # thin wrappers over mlx-lm's forward signature
    weights/              # I-4 WeightProvider + ResidentWeightProvider (dense, full residency)
    speculative/          # I-5 DraftEngine + NoopDraftEngine
    engine/               # top-level Engine (generate + generate_batch)
    server/               # CLI (HTTP server is P-8)
    bench/  llm/  vq/     # empty — reserved for P-4/P-8/P-5
```

The five frozen interfaces — `ModelAdapter` (I-1), `KVManager` (I-2),
`KVCodec` (I-3), `WeightProvider` (I-4), `DraftEngine` (I-5) — are defined
in their respective subpackages as `typing.Protocol`s and are the integration
points for native capabilities (Principle 9 in PLAN.md). Each phase from P-3
onwards replaces a stub with a real implementation behind the same interface.

---

## Documentation

- [`docs/PLAN.md`](docs/PLAN.md) — single source of truth for phases,
  decisions log, open questions.
- [`docs/API.md`](docs/API.md) — per-module function and class reference
  (the wiki).
- [`docs/P2_OPENING.md`](docs/P2_OPENING.md) — Phase 2 architectural opening
  document (why `ContinuousBatcher` looks the way it does).
- [`docs/P2_UNIT_16D_PREP.md`](docs/P2_UNIT_16D_PREP.md) — preemption +
  replay prep doc, including the B-1 through B-9 invariants enforced by
  the batcher's tests.

---

## License

Apache-2.0.
