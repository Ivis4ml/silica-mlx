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
| P-4 | Unified bench harness — runner, oracles, 15 registered scenarios, JSONL + Markdown reports, vqbench subprocess PPL | ✅ complete; P-4 exit surfaced Q-010 chunked-prefill trigger → P-4.5 bridge planned |
| P-4.5 | P-4 exit bridge — chunked-prefill minimal + VectorCodec runtime integration spike | ✅ complete (v1.6.9) |
| P-5 | VQ KV compression (BlockTQ / RaBitQ) | ✅ complete (v1.7.4 — P-5-A / B / C / D sub-units + P-5 Acceptance (1)–(4) all closed: codec-swap neutrality by inspection, `--all-kv-codecs` one-command report, `n_block > n_fp16` admission-headroom gate, vqbench-aligned mean-over-seeds PPL cross-check. P-5-F pre-RoPE production routing closed at v1.7.6 via the (3b) projection-output capture path; production (4-b) anchor row measures ΔPPL +0.012 inside D.2a envelope. (b-static) Qwen3.5-4B vs vqbench REPORT.md baseline closed at v1.7.7. v1.7.8 closes the v1.7.6 follow-up trail: slice-regime + pre_norm hybrid Qwen3.5-0.8B E2E discriminator, opt-in per-head Haar rotation default OFF. `PagedPrefixBlockStore` codec injection intentionally deferred under D-003) |
| P-6 | Weight streaming | Stub (`ResidentWeightProvider` today) |
| P-7 | Speculative decoding (DraftTarget / EAGLE / Medusa) | Stub (`NoopDraftEngine` today) |
| P-8 | OpenAI-compatible HTTP server + session layer | ⏳ planned (leaning T1 tail, after P-5) |

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
    kvcache/              # I-2 KVManager, I-3 VectorCodec, paged bookkeeping, radix prefix cache, prefix block store
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
(I-2), `VectorCodec[P]` (I-3, side-level since P-5-A.0.4),
`WeightProvider` (I-4), `DraftEngine` (I-5) — are defined in their
respective subpackages as `typing.Protocol`s and are the integration
points for native capabilities (Principle 9 in PLAN.md). Each phase
from P-3 onwards replaces a stub with a real implementation behind
the same interface.

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

Immediate next units (see `docs/PLAN.md` for the full plan). The P-4
exit surfaced two signals that reshuffle the near-term order:
(1) Q-010 chunked-prefill promotion triggered — cohort-level prefill
drags short-row TTFT behind the long-row `T_max`; (2) the
`IdentityCodec` stub lives outside the real forward hot path, so P-5
needs a runtime-integration spike before any BlockTQ coding starts.
P-4.5 bridges both.

- **P-4.5** — P-4 exit bridge (three sub-units, see `docs/PLAN.md`
  §7 P-4.5):
  - **P-4.5-A** — PLAN / README decision sync (this revision).
  - **P-4.5-B** — chunked prefill minimal: three-option opening
    doc (in-cohort chunking / cohort splitting / admission
    ordering) → scheduler change under `silica/scheduler/batcher.py`
    → TTFT-under-concurrency ratio `< 3×` and greedy-output
    bit-identity regression lock.
  - **P-4.5-C** — VectorCodec runtime integration spike (complete,
    v1.6.9): `docs/P4_5_C_KVCODEC_OPENING.md` enumerated the three
    integration points (active `BatchKVCache` / detached prefix store
    / cache wrapper) and landed Option (B); `IdentityCodec` is wired
    through `SyntheticPrefixBlockStore.register_detached` /
    `fetch_detached` end-to-end on the Qwen3-0.6B path.
- **P-5** — VQ KV compression platform. All implementation
  sub-units landed (v1.7.0 through v1.7.4, between 2026-04-22 and
  2026-04-24); **§7 P-5 Acceptance items (1) / (2) / (3) / (4)
  all closed at v1.7.4**. (1) codec-swap neutrality by inspection
  (zero concrete-codec dispatch across scheduler / model adapters;
  evidence in `docs/P5_ACCEPTANCE_SWEEP/codec_swap_neutrality.md`).
  (2) `scripts/bench.py --all --all-kv-codecs --seeds 42,43,44`
  produces a coherent 924-row report covering 28 scenarios × 11
  codecs × 3 seeds; all 564 failed rows classified into three
  expected compatibility classes (`codec_override_invalid`,
  K-only `rabitq_b1`, vqbench-aligned symmetric-codec guard);
  evidence in `docs/P5_ACCEPTANCE_SWEEP/all_kv_codecs.{jsonl,md}`.
  (3) `qwen3-0.6b-admission-headroom-prefix-heavy` passes
  `n_block > n_fp16` (`7 > 4`, `admit_ratio ≈ 1.75`,
  `residency_ratio ≈ 0.266`, ≈ 1 / 3.76 per vqbench §3.1);
  evidence in `docs/P5_ACCEPTANCE_SWEEP/admission_headroom.{jsonl,md}`.
  (4) vqbench cross-check closed at v1.7.3 via the D.2a
  vqbench-aligned oracle (mean-over-seeds gate on the
  `qwen3-0.6b-wikitext-ppl-block-tq-b64-b4-vqbench-aligned` row:
  `|mean_gap| ≤ 2·SEM_diff` AND `|mean_gap| < 1.0` PPL).
  **Post-P-5 follow-up closed at P-5-F (v1.7.6):** the production
  `prefix_store_post_rope` prefix-cache arm at the same codec
  config previously paid a ~5–10 PPL ΔPPL quality cost (post-RoPE
  noise injection through RoPE-coupled attention). P-5-F lands a
  pre-norm K/V store via the (3b) projection-output capture path
  (`silica/models/pre_norm_capture.py` Protocol + per-family
  proxy on `attn.k_proj`); production wikitext PPL rows now
  default to `codec_quality_path="prefix_store_pre_norm"` (F.3
  default flip). The (4-b) anchor row measures ΔPPL +0.012 on
  the production path post-F.3 (was +20.83 PPL pre-F.3 on the
  legacy post-RoPE store), inside D.2a's `+0.51 ± 0.35 PPL`
  envelope. Three legacy comparison arms (`prefix_store_post_rope`,
  `prefix_store_pre_rope`, `vqbench_aligned`) retained as bench-
  only opt-ins per `docs/P5_F_OPENING.md` §6.9 reading order.
  **(b-static) closed at v1.7.7:** the originally v1.5.1-named
  Qwen3.5-4B PPL gate vs `vqbench/REPORT.md` static baseline
  closed on the production hot path via the same (3b) capture
  path (mean ΔPPL = +0.0016 across 3 seeds, inside vqbench's
  reported `+0.000% ± 0.000%` lossless-at-measurement-precision
  envelope; ~16x SEM headroom on the (4-b)-style aggregated
  gate). Evidence:
  `docs/P5_ACCEPTANCE_SWEEP/qwen35_4b_b_static_close.md`.
  **v1.7.8 closes the remaining v1.7.6 follow-up trail:** Item 1
  — slice-regime + `pre_norm=True` end-to-end on hybrid
  Qwen3.5-0.8B (`tests/test_p5_f_pre_norm_e2e_hybrid.py`,
  IdentityCodec discriminator on slice-regime helpers); Item 3 —
  opt-in per-head Haar rotation across BlockTQ / RaBitQ1Bit /
  ExtRaBitQ (`per_head_rotation: bool = False`, default OFF
  preserves the closed (4-b) baseline; seed convention `seed *
  1000 + head_idx` matches vqbench). The single
  intentionally-deferred Deliverable remains
  `PagedPrefixBlockStore` codec injection (`NotImplementedError`
  stub per D-003 no-compressed-domain-attention scope; lands when
  the paged-attention kernel track advances).
  - **P-5-A** — Codec scaffolding + BlockTQ hot path + memory
    accounting + decode-speed gate. Side-level `VectorCodec[P]`
    Protocol + `CodedPayload` hierarchy + MLX-native bit-packing +
    Lloyd-Max / Haar calibration quarantine + K/V split store (A.0,
    v1.7.0); `BlockTurboQuantMSE` on the `mx.array` hot path + codec
    registry + vqbench algorithmic parity (A.1);
    `MemoryBudgeter` three-mode residency accounting (A.2);
    decode-speed acceptance gate on
    `qwen3-0.6b-prefix-hit-decode-{fp16,block-tq-b64-b4}` (A.3).
  - **P-5-B** — RaBitQ family. `RaBitQ1Bit` K-only codec + registry +
    vqbench parity (B.1); `ExtRaBitQ` at bits ∈ {2, 3, 4} + registry +
    vqbench parity (B.2); `ext_rabitq_b4` decode-speed gate on the
    `qwen3-0.6b-prefix-hit-decode-ext-rabitq-b4` row (B.3).
  - **P-5-C** — Bench harness extensions. Teacher-forced streaming
    PPL oracle + `forward_batched_full` runner entry point (C.1);
    codec-backed PPL oracle + WikiText loader + four
    `qwen3-0.6b-wikitext-ppl-*` rows (C.2); `STORAGE` +
    `ADMISSION_HEADROOM` `OracleKind` + compression / prefix-heavy
    rows (C.3); `--seeds` fan-out + per-scenario mean ± std
    aggregation (C.4); `--kv-codec` / `--all-kv-codecs` CLI
    (C.5); `--vqbench-xcheck` ΔPPL divergence gate + per-arm
    `vqbench_gap` column (C.6).
  - **P-5-D** — vqbench-aligned PPL path + (4-b) gate close.
    Bench-runner seed propagation into codec Haar rotations (D.1,
    commit `2b3868d`); pre-RoPE projection-patch oracle
    `teacher_forced_chunked_nll_vqbench_aligned` +
    `qwen3-0.6b-wikitext-ppl-block-tq-b64-b4-vqbench-aligned`
    scenario + 3-seed verification data (D.2a, commit `ed57be1`);
    (4-b) gate reinterpretation as mean-over-seeds + PLAN / OPENING
    / README sync + top-level (4) checkbox flip (D.3, v1.7.3).
- **P-8** — OpenAI-compatible HTTP server + session layer (wraps
  `ChatSession` with routing, auth, streaming SSE / WebSocket).
  Leaning T1 tail per Q-002 progress; sequenced after P-5 so the
  HTTP product face ships with VQ compression live.
- **P-3-C5** — preempt/replay with recurrent-state snapshot on
  hybrid DeltaNet. Deferred through P-4.5 and P-5; revisited at P-7
  speculative-rollback entry since the commit / rollback semantics
  co-design there.
- **P-3-E4** — batched MoE: lift the `has_moe=True` capability
  gate on `ContinuousBatcher` once the SwitchGLU + per-row active
  expert routing round-trips cleanly. Re-scoped as pre-P-6 work
  given per-expert routing shares primitives with streaming.
- **P-6** — weight streaming (per-expert residency for MoE + layer
  streaming for dense, driven by the `WeightProvider` interface
  that already exists).
- **P-7** — speculative decoding behind the `DraftEngine`
  interface (DraftTarget, EAGLE, Medusa).

---

## License

Apache-2.0.
