# Silica-MLX

MLX-native LLM inference platform for Apple Silicon. vLLM-style core
(paged KV, continuous batching, prefix cache, memory-budget admission)
plus a planned mini-sglang outer layer for Phase 8. Target: run dense
27B–31B class models on a 48 GB M5 Pro.

## Status

| Phase | Scope | State |
| --- | --- | --- |
| P-0 | Core skeleton, frozen interfaces, sampler | ✅ complete |
| P-1 | Single-request `Engine.generate` | ✅ complete |
| P-2 | Continuous batching, radix prefix cache, memory-budget admission, preempt+replay | ✅ complete |
| P-3 | Family adapters — Qwen3 dense, Qwen3.5 hybrid DeltaNet, Gemma4-31B dense, Qwen3.5-MoE, Gemma4-MoE | ✅ mostly (`C5` preempt/replay with recurrent-state snapshot pending; `E4` batched MoE pending) |
| P-4 | Unified bench harness — runner, oracles, 15 registered scenarios, JSONL + Markdown reports, vqbench subprocess PPL | ✅ complete |
| P-5 | VQ KV compression (BlockTQ / RaBitQ) | Stub (`IdentityCodec` today) |
| P-6 | Weight streaming | Stub (`ResidentWeightProvider` today) |
| P-7 | Speculative decoding (DraftTarget / EAGLE / Medusa) | Stub (`NoopDraftEngine` today) |
| P-8 | OpenAI-compatible HTTP server + session layer | ⏳ planned |

Legend: ✅ shipped · Stub = wired as the fp16 baseline behind the
frozen interface · ⏳ = not started.

Single-source-of-truth for the roadmap and decisions log:
[`docs/PLAN.md`](docs/PLAN.md).

---

## Install

Requires Python 3.12+ and an Apple Silicon Mac. Managed via
[uv](https://github.com/astral-sh/uv).

```bash
uv pip install -e .
# optional extras for the planned HTTP serve path (P-8, not yet wired):
uv pip install -e '.[serve]'
```

Runtime deps: `mlx>=0.22`, `mlx-lm>=0.18`, `numpy>=1.26`, `pydantic>=2`.
Dev deps (`pytest`, `ruff`, `mypy`) live in the `dev` group.

---

## Quickstart

Five entry points: `silica run` (single-shot CLI), `scripts/chat.py`
(REPL chatbot), the Python API (single request + continuous batching),
and `scripts/bench.py` (benchmark harness).

### 1. CLI — single prompt

```bash
python -m silica run \
    --model Qwen/Qwen3-0.6B \
    --prompt "The capital of France is" \
    --max-tokens 64 \
    --temperature 0.0
```

Prints `prompt + generation` to stdout and a one-line metrics row
(`ttft=… prefill=…tok/s decode=…tok/s resident=…MB`) to stderr.
The console script `silica run …` is equivalent once the package is
installed.

### 2. CLI — chat REPL

```bash
python scripts/chat.py --model Qwen/Qwen3-0.6B \
    --system "You are a concise assistant." \
    --temperature 0.7 --top-p 0.9 --max-tokens 256
```

Opens a multi-turn REPL that streams each reply token-by-token. After
every turn stderr prints a single-line metrics record:

```
[ttft=25.1ms prefill=596.9tok/s decode=151.4tok/s
 resident_kv=29.4MB peak=1261.5MB logical_kv=29.4MB
 prompt=15 out=64 wall=0.44s finish=max_tokens]
```

REPL commands:

| Command | Effect |
| --- | --- |
| `/reset` | drop all messages except the initial `--system` prompt |
| `/stats` | print cumulative session metrics (avg TTFT, avg decode tok/s, total tokens, wall time) |
| `/exit` · EOF · Ctrl-C | quit |

Flags:

| Flag | Purpose |
| --- | --- |
| `--model` | HuggingFace repo id (e.g. `Qwen/Qwen3-0.6B`, `Qwen/Qwen3.5-0.8B`) |
| `--system` | system prompt; omit for an empty system |
| `--temperature` · `--top-p` · `--top-k` · `--max-tokens` | sampling |
| `--no-stream` | print the full reply only after generation completes |

### 3. Python API — single request

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

`Engine.generate` is an iterator — consume it however you want
(streaming to stdout, collecting into a list, piping into a
detokenizer). The iterator auto-releases KV on exhaustion or
exception.

### 4. Python API — continuous batching (8 concurrent)

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
    # ... up to N prompts; max_batch_size bounds active rows
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

`generate_batch` yields `BatchEvent` values of three kinds: `token`
(one per emitted token), `done` (normal terminal), and `aborted`
(budget exhaustion or error). `req_index` matches the position in
the `prompts` list. For budget-aware scheduling, construct a
[`MemoryBudgeter`](docs/API.md#memorybudgeter) and install it on a
`ContinuousBatcher` directly (see the API doc for the lower-level
path).

### 5. Python API — multi-turn chat session

```python
from silica import Engine
from silica.chat import ChatSession
from silica.models.factory import adapter_for_repo

adapter, kv = adapter_for_repo("Qwen/Qwen3-0.6B")
engine = Engine(adapter, kv)
session = ChatSession(adapter, engine, system_prompt="Be concise.")

import sys
def stream_to_stdout(delta: str) -> None:
    sys.stdout.write(delta); sys.stdout.flush()

m = session.chat("Explain TTFT in one sentence.", stream_to=stream_to_stdout)
print()  # newline after stream
print(f"TTFT={m.ttft_ms:.1f}ms  decode={m.decode_tok_s:.1f}tok/s  "
      f"peak={m.peak_memory_mb:.1f}MB  finish={m.finish_reason}")

m = session.chat("Now explain decode tok/s.", stream_to=stream_to_stdout)
```

`ChatSession` wraps `Engine.generate` with OpenAI-style message
history, the tokenizer's `apply_chat_template` (Qwen-style
`<|im_start|>` fallback when unavailable), and per-turn
[`TurnMetrics`](silica/chat/session.py) — TTFT, prefill tok/s,
decode tok/s, resident KV, peak memory, logical KV bytes, wall
seconds, and finish reason. `session.reset()` drops every message
except the system prompt.

### 6. Benchmark harness

Run the full built-in catalog (15 scenarios; dual-gated rows skip
when their env var is not `=1`):

```bash
python -m scripts.bench --all \
    --out bench-results.jsonl \
    --report-md bench-results.md
```

Run a single scenario:

```bash
python -m scripts.bench --scenario qwen3-0.6b-bgt1-parity
```

List the catalog:

```bash
python -m scripts.bench --list
```

Typical on-device output for `--all` without any dual-gate env vars
set (cache-only rows run, dual-gated rows skip):

```
| id | status | reason | ttft_ms | decode_tok_s | resident_mb | peak_mb | wall_s | tokens |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| qwen3-0.6b-smoke | ok |  | 11.6 | 151.8 | 29.4 | 1222.8 | 0.43 | 4 |
| qwen3-0.6b-b1-parity | ok |  | 16.3 | 151.1 | 29.4 | 1252.0 | 0.56 | 4 |
| qwen3-0.6b-bgt1-parity | ok |  |  |  |  | 1279.3 | 0.53 | 16 |
| qwen3-0.6b-short-in-long-out | ok |  | 13.3 | 167.8 | 29.4 | 1222.9 | 0.90 | 64 |
| qwen3-0.6b-long-in-short-out | ok |  | 46.1 | 165.1 | 58.7 | 1815.2 | 0.45 | 4 |
| ... |
| gemma4-31b-smoke | skipped | env_var_not_set:SILICA_REAL_GEMMA4_31B | | ... |
| qwen3.5-moe-smoke | skipped | env_var_not_set:SILICA_REAL_QWEN3_5_MOE | | ... |
```

Opt into the heavy dual-gated rows by exporting the corresponding
env var to `1`:

| Env var | Unlocks |
| --- | --- |
| `SILICA_REAL_GEMMA4_31B` | `gemma4-31b-smoke`, `gemma4-31b-b1-parity`, `gemma4-31b-bgt1-parity` |
| `SILICA_REAL_QWEN3_5_27B` | `qwen3.5-27b-smoke` |
| `SILICA_REAL_QWEN3_5_MOE` | `qwen3.5-moe-smoke` |
| `SILICA_REAL_GEMMA4_MOE` | `gemma4-moe-smoke` |

### 7. vqbench_baseline (P-5 PPL reference column)

For the numeric cross-check PLAN §P-5 requires, silica-mlx wraps
the existing `vqbench/` scripts in a subprocess and parses the PPL
headline row:

```bash
python scripts/vqbench_baseline.py \
    --python-executable /path/to/vqbench/venv/bin/python \
    --out vqbench-baseline.jsonl
```

`--python-executable` is mandatory for a real run — the
silica-mlx venv does **not** depend on `torch` / `transformers` /
`datasets` (D-009 hot-path constraint), so the vqbench script must
run under its own Python. Output is a `VqbenchBaselineResult` with
`model`, `method`, `bits`, `ppl_fp16`, `ppl_quant`, `delta_ppl`,
`delta_pct` (emitted as one JSONL row when `--out` is set).

---

## What works today

| Capability | Status | Where |
| --- | --- | --- |
| Single-request generation (greedy / temperature / top-k / top-p / repetition penalty) | ✅ | `silica.engine.Engine.generate` |
| Token streaming as iterator | ✅ | same |
| Per-request metrics (TTFT / throughput / resident MB / logical KV bytes) | ✅ | `Engine.metrics.snapshot()` |
| Multi-turn chat session + apply_chat_template + per-turn metrics | ✅ | `silica.chat.ChatSession` + `scripts/chat.py` |
| 8-way continuous batching with left-padded prefill | ✅ | `silica.scheduler.ContinuousBatcher` |
| Mid-run admission when slots free | ✅ | `Engine.generate_batch(max_batch_size=...)` |
| Shared-prefix caching (block-granular radix trie) | ✅ | `silica.kvcache.prefix.RadixPrefixCache` |
| Memory-budget admission ladder (admit → evict → preempt → reject) | ✅ | `silica.scheduler.MemoryBudgeter` |
| Preempt + replay (save composite prompt, re-enter queue) | ✅ | `ContinuousBatcher._apply_preempt` |
| Plain Qwen3 family adapter (0.6B / 4B / 7B / 14B / 32B) | ✅ | `silica.models.qwen3.Qwen3Adapter` |
| Qwen3.5 hybrid DeltaNet family (0.8B, 27B) — single-request + batched | ✅ | `silica.models.qwen3_5.Qwen3_5Adapter`; B>1 greedy parity on 0.8B in `tests/test_p3_hybrid_batched_parity.py` |
| Gemma4-31B dense — single-request + batched miss-only path; B=1 parity + B>1 direct mlx-lm reference | ✅ (dual-gated) | `silica.models.gemma4.Gemma4Adapter`; strict B>1 batched-vs-single greedy parity not claimed |
| Qwen3.5-35B-A3B MoE / Gemma4-26B-A4B MoE — single-request only | ✅ (dual-gated) | `silica.models.qwen3_5_moe`, `silica.models.gemma4_moe`; option-(c) dispatch-observation seam; batched MoE pending P-3-E4 |
| Per-kind KV budget for heterogeneous attention (Gemma4 sliding+full) | ✅ | `KVLayout.bytes_per_token_total` |
| DeltaNet recurrent state + `state_delta` plumbing (single + batched) | ✅ | `Qwen3_5Adapter.make_batch_cache` interleaves `ArraysCache` / `BatchKVCache` per layer |
| Unified bench runner (SMOKE / B=1 parity / B>1 direct-reference / teacher-forced-argmax oracles) | ✅ | `silica.bench.BenchRunner` + `scripts/bench.py`; 15 registered scenarios (see below) |
| Bench: JSONL + Markdown report output (paste-into-PR) | ✅ | `BenchRunner(out_path=...)` + `scripts/bench.py --report-md` + `silica.bench.render_markdown_report` |
| Bench: per-row first-token offset for B>1 SMOKE (TTFT-under-concurrency signal for Q-010) | ✅ | `metadata.rows[].first_token_ms_offset` in the JSONL row |
| Bench: vqbench subprocess PPL reference column | ✅ | `silica.bench.vqbench_baseline` + `scripts/vqbench_baseline.py` |
| CLI: `python -m silica run` (single-shot) + `scripts/chat.py` (REPL) | ✅ | `silica.server.cli`, `scripts/chat.py` |
| Preempt/replay with recurrent state snapshot | ⏳ | P-3-C5 |
| Batched MoE | ⏳ | P-3-E4 |
| VQ KV compression (BlockTQ / RaBitQ) | Stub | P-5 (`IdentityCodec` today) |
| Weight streaming | Stub | P-6 (`ResidentWeightProvider` today) |
| Speculative decoding (DraftTarget / EAGLE / Medusa) | Stub | P-7 (`NoopDraftEngine` today) |
| OpenAI-compatible HTTP server + session layer | ⏳ | P-8 |

---

## Bench catalog (15 scenarios)

Run `python -m scripts.bench --list` for the current roster.

### Cache-only (run on any dev box that has pulled Qwen/Qwen3-0.6B)

| id | oracle | shape |
| --- | --- | --- |
| `qwen3-0.6b-smoke` | SMOKE | 1 prompt, max_tokens=4 |
| `qwen3-0.6b-b1-parity` | B1_PARITY_VS_SINGLE | 1 prompt, B=1 |
| `qwen3-0.6b-bgt1-parity` | BGT1_DIRECT_BATCHED_REFERENCE | 2 prompts, B=2, different tokenized lengths |
| `qwen3-0.6b-short-in-long-out` | SMOKE | 1 prompt ("Hello"), max_tokens=64 |
| `qwen3-0.6b-long-in-short-out` | SMOKE | 301-token prompt, max_tokens=4 |
| `qwen3-0.6b-concurrent-shared-prefix` | SMOKE | 4 prompts w/ shared "The capital of" prefix, prefix_cache=True |
| `qwen3-0.6b-ttft-under-concurrency` | SMOKE | 1 long + 3 short prompts, B=4 (Q-010 signal) |
| `qwen3-0.6b-teacher-forced-argmax` | TEACHER_FORCED_ARGMAX | silica vs direct mlx-lm positional argmax, ≥0.98 agreement |
| `qwen3.5-0.8b-b1-parity` | B1_PARITY_VS_SINGLE | Qwen3.5 hybrid DeltaNet, 1 prompt |

### Dual-gated (cache + `SILICA_REAL_<family>=1`)

| id | gate env var | shape |
| --- | --- | --- |
| `qwen3.5-27b-smoke` | `SILICA_REAL_QWEN3_5_27B` | SMOKE, ~16 GB checkpoint |
| `qwen3.5-moe-smoke` | `SILICA_REAL_QWEN3_5_MOE` | MoE SMOKE, ~20 GB checkpoint, ~30 GB peak |
| `gemma4-31b-smoke` | `SILICA_REAL_GEMMA4_31B` | SMOKE, ~18 GB checkpoint |
| `gemma4-31b-b1-parity` | `SILICA_REAL_GEMMA4_31B` | B=1 parity on dense 31B |
| `gemma4-31b-bgt1-parity` | `SILICA_REAL_GEMMA4_31B` | B=2 parity vs direct mlx-lm |
| `gemma4-moe-smoke` | `SILICA_REAL_GEMMA4_MOE` | MoE SMOKE, ~16 GB checkpoint |

---

## Running the tests

```bash
uv run pytest tests           # full suite (currently 778 tests, ~20 s)
uv run ruff check .
uv run mypy silica
```

The suite includes the PLAN §7 P-2 acceptance triad:

- `tests/test_p2_batched_parity.py::test_generate_batch_real_8_concurrent` — 8 concurrent prompts run stably.
- `tests/test_p2_batched_parity.py::test_prefix_hit_reduces_forward_tokens` — shared-prefix hit verifiable from `ContinuousBatcher.forward_prompt_tokens`.
- `tests/test_p2_batched_parity.py::test_budget_overflow_aborts_cleanly` — budget overflow queues (preempt branch) or aborts (reject branch) cleanly.

Real-model tests download Qwen3-0.6B via mlx-lm on first run
(~1.2 GB). Dual-gated real-model smokes (Gemma4-31B, Qwen3.5-27B,
both MoE families) are skipped unless the corresponding
`SILICA_REAL_*=1` env var is set.

---

## Layout

```text
silica/
    __init__.py           # re-exports Engine
    __main__.py           # python -m silica → CLI
    core/                 # events, logger, profiler, request FSM, sampler, sampling params
    kvcache/              # I-2 KVManager, I-3 KVCodec, paged bookkeeping, radix prefix cache, prefix block store
    scheduler/            # ContinuousBatcher, MemoryBudgeter, seed_kv helper
    models/               # I-1 ModelAdapter + family adapters (qwen3, qwen3_5, qwen3_5_moe, gemma4, gemma4_moe) + factory
    mlx/                  # thin wrappers over mlx-lm's forward signature
    weights/              # I-4 WeightProvider + ResidentWeightProvider (dense, full residency)
    speculative/          # I-5 DraftEngine + NoopDraftEngine
    engine/               # top-level Engine (generate + generate_batch)
    chat/                 # ChatSession + TurnMetrics (multi-turn chat over Engine)
    bench/                # P-4 harness (scenario schema, runner, oracles, built-in catalog, report, vqbench_baseline)
    server/               # CLI (HTTP server is P-8)
    llm/  vq/             # empty — reserved for P-8/P-5
```

The five frozen interfaces — `ModelAdapter` (I-1), `KVManager`
(I-2), `KVCodec` (I-3), `WeightProvider` (I-4), `DraftEngine`
(I-5) — are defined in their respective subpackages as
`typing.Protocol`s and are the integration points for native
capabilities (Principle 9 in PLAN.md). Each phase from P-3 onwards
replaces a stub with a real implementation behind the same
interface.

---

## Documentation

Core:

- [`docs/PLAN.md`](docs/PLAN.md) — single source of truth for
  phases, deliverables, acceptance criteria, decisions log, open
  questions, and empirical findings per sub-commit.
- [`docs/API.md`](docs/API.md) — per-module function and class
  reference (the wiki).

Phase-specific context:

- [`docs/P2_OPENING.md`](docs/P2_OPENING.md) — why
  `ContinuousBatcher` looks the way it does.
- [`docs/P2_UNIT_16D_PREP.md`](docs/P2_UNIT_16D_PREP.md) —
  preemption + replay prep, including the B-1 through B-9
  invariants enforced by the batcher's tests.
- [`docs/P3_DELTANET_SURVEY.md`](docs/P3_DELTANET_SURVEY.md) —
  Qwen3.5 hybrid DeltaNet architecture survey.
- [`docs/P3_GEMMA4_SURVEY.md`](docs/P3_GEMMA4_SURVEY.md) —
  Gemma4-31B source-code survey (sliding + full hybrid).
- [`docs/P3_BATCH_ROTATING_KV_SURVEY.md`](docs/P3_BATCH_ROTATING_KV_SURVEY.md)
  — mlx-lm's `BatchRotatingKVCache` audit for the batched sliding
  path.
- [`docs/P3_MOE_SURVEY.md`](docs/P3_MOE_SURVEY.md) — Qwen3.5-MoE
  and Gemma4-MoE architecture differences + D-011 "per-expert at
  dispatch, fused at fetch" decision.

Gate docs (for historical context):

- [`docs/P1_ACCEPTANCE.md`](docs/P1_ACCEPTANCE.md),
  [`docs/P2_GATE_0.md`](docs/P2_GATE_0.md),
  [`docs/P2_GATE_0_5.md`](docs/P2_GATE_0_5.md),
  [`docs/P2_UNIT_16C_PREP.md`](docs/P2_UNIT_16C_PREP.md),
  [`docs/P2_UNIT_16C_2_PREP.md`](docs/P2_UNIT_16C_2_PREP.md).

---

## Roadmap

Immediate next units (see `docs/PLAN.md` for the full plan):

- **P-3-C5** — preempt/replay with recurrent-state snapshot on
  hybrid DeltaNet. Closes the last P-3 bullet.
- **P-3-E4** — batched MoE: lift the `has_moe=True` capability
  gate on `ContinuousBatcher` once the SwitchGLU + per-row active
  expert routing round-trips cleanly.
- **P-5** — BlockTQ / RaBitQ `KVCodec` implementations (MLX-native
  rewrite of the vqbench numeric reference; cross-check via
  `scripts/vqbench_baseline.py`).
- **P-6** — weight streaming (per-expert residency for MoE + layer
  streaming for dense, driven by the `WeightProvider` interface
  that already exists).
- **P-7** — speculative decoding behind the `DraftEngine`
  interface (DraftTarget, EAGLE, Medusa).
- **P-8** — OpenAI-compatible HTTP server + session layer (wraps
  `ChatSession` with routing, auth, streaming SSE / WebSocket).

---

## License

Apache-2.0.
