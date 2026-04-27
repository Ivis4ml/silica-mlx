# silica-mlx

**Continuous-batching LLM serving on Apple Silicon — vLLM-core
architecture, MLX-native.**

The continuous batching, memory-budget admission, preempt/replay, and
radix prefix cache that production-grade serving frameworks rely on,
ported to MLX's unified-memory model. Native KV codec compression
(BlockTQ / RaBitQ) shipped. Multi-family adapters with batched-output
parity validated against a direct mlx-lm batched reference: Qwen3
dense, Qwen3.5 hybrid DeltaNet, Gemma4-31B dense, Qwen3.5-35B-A3B MoE,
gemma-4-26B-A4B MoE.

Target hardware: M5 Pro 48 GB. Runs Qwen3 (0.6B / 4B / 7B / 14B /
32B), Qwen3.5 hybrid (0.8B / 4B / 27B), Gemma4-31B dense,
Qwen3.5-35B-A3B MoE, gemma-4-26B-A4B MoE.

> **Status:** the scheduler core (continuous batching, prefix cache,
> memory budget), multi-family adapters, and KV codec compression are
> shipped. OpenAI HTTP server, speculative decoding, and weight
> streaming for MoE residency remain stubs behind frozen interfaces,
> scheduled next.

> **Documentation.** Build the full Sphinx site with
> `pip install -e '.[docs]' && make -C docs html`, then open
> `docs/_build/html/index.html`. The site bundles the
> [overview](docs/overview.md), the [chat-CLI guide](docs/chat-cli.md),
> the [benchmark harness guide](docs/bench.md), the
> [auto-generated per-module API](docs/api/index.md), the hand-curated
> [API reference](docs/API.md), and a
> [curated index into `plans/`](docs/plans-index.md).

---

## Why silica-mlx

| Capability | mlx-lm | vLLM | SGLang | silica-mlx |
| --- | --- | --- | --- | --- |
| Backend | MLX | CUDA | CUDA | MLX |
| Continuous batching | ✗ | ✅ | ✅ | ✅ |
| Radix prefix cache | ✗ | limited (block-level) | ✅ | ✅ |
| Memory-budget admission ladder | ✗ | ✅ | ✅ | ✅ |
| Preempt + replay | ✗ | ✅ | ✅ | ✅ |
| Native KV codec compression | ✗ | limited (FP8 / INT8) | limited | ✅ (BlockTQ + RaBitQ family) |
| Hybrid DeltaNet (Qwen3.5) — batched | limited (single-request) | ✗ | ✗ | ✅ |
| MoE batched dispatch | limited (single-request) | ✅ | ✅ | ✅ |
| OpenAI-compatible HTTP server | ✗ | ✅ | ✅ | planned |
| Speculative decoding | ✗ | ✅ | ✅ | planned |
| Per-expert MoE residency | ✗ | limited | ✗ | planned |

**Niche.** silica-mlx is an MLX-native serving framework that combines
the vLLM scheduler core (continuous batching + memory budget +
preempt/replay) with a radix prefix cache and a KV codec compression
layer on a single integrated runtime. mlx-lm is single-request and
solves a different problem; vLLM and SGLang are CUDA-first and don't
run on Apple Silicon. silica-mlx is not feature-complete relative to
either yet — see "What's planned" below for the gap.

---

## Highlights — what's shipped

- **vLLM-core scheduler.** Continuous batching with a memory-budget
  admission ladder (admit → evict → preempt → reject) and
  preempt+replay (save composite prompt, re-enter queue).
  Single-request and batched generation share one code path.
- **Radix prefix cache.** Block-granular radix-trie reuse with a
  per-codec store seam. Block-aligned prefix hits seed the per-row
  KV cache; miss-prefill chunks insert blocks back into the tree on
  row termination.
- **Native KV codec compression.** `BlockTurboQuantMSE` (B=64 4-bit)
  matches vqbench/REPORT.md's lossless-at-measurement-precision
  baseline on Qwen3.5-4B WikiText-2 (mean ΔPPL = +0.0016 PPL across
  three seeds, statistically indistinguishable from vqbench's
  reported `+0.000% ± 0.000%`). `RaBitQ1Bit` and `ExtRaBitQ`
  (2 / 3 / 4-bit) ship alongside.
- **Multi-family adapters.** Qwen3 dense (0.6B–32B), Qwen3.5 hybrid
  DeltaNet (0.8B / 4B / 27B), Gemma4-31B dense, Qwen3.5-MoE (35B-A3B,
  256 experts × top-8), Gemma4-MoE (26B-A4B, 128 experts × top-8).
  Each family has batched output parity validated against a direct
  mlx-lm batched reference using the same per-layer cache types and
  left-padding convention.
- **Hybrid DeltaNet on the batched path.** Recurrent-state snapshot
  and restore unlock `RadixPrefixCache + Qwen3.5 hybrid` cooperation
  end-to-end on real Qwen3.5-0.8B.
- **Bench harness.** 15+ registered scenarios across five oracle
  types (smoke, B=1 parity, B>1 direct-reference, teacher-forced
  argmax, WikiText-2 perplexity) producing JSONL + Markdown reports
  with an optional vqbench subprocess cross-check column. One
  command runs the entire catalogue.
- **CLI + Python API.** `silica run` single-shot, `scripts/chat.py`
  streaming REPL with `/reset` and `/stats`, plus
  `Engine.generate` / `Engine.generate_batch` for direct embedding.

---

## What's planned

The engine main loop already carries stub implementations behind
frozen interfaces; the planned phases progressively replace those
stubs without changing call sites.

- **Weight streaming for MoE residency.** Dense layer prefetch plus
  per-expert residency for MoE checkpoints. Target: Qwen3.5-35B-A3B's
  active-3B / total-35B residency footprint under tighter memory
  budgets without OOM, with decode throughput close to the resident
  baseline. Today, everything stays resident.
- **Speculative decoding.** Draft-target speculation lands on the
  frozen `DraftEngine` interface that already lives in the engine
  main loop. EAGLE / Medusa style schemes are out of scope for v0.1.
- **OpenAI-compatible HTTP server.** `silica-server` binary with
  OpenAI-compatible chat / completions endpoints, plus a session
  layer for per-conversation prefix caching across HTTP requests.
  This is the SGLang-style outer layer of the framework.

Paged-attention codec integration is deferred until MLX exposes a
variable-length SDPA kernel.

---

## Status — phase board

| Phase | Scope | State |
| --- | --- | --- |
| P-0 | Core skeleton, frozen interfaces, sampler | ✅ complete |
| P-1 | Single-request `Engine.generate` | ✅ complete |
| P-2 | Continuous batching, radix prefix cache, memory-budget admission, preempt+replay | ✅ complete |
| P-3 | Family adapters — Qwen3 dense, Qwen3.5 hybrid DeltaNet, Gemma4-31B dense, Qwen3.5-MoE, Gemma4-MoE | ✅ complete on the supported batched surface (C5 slice-prefill α-MVP; E4 batched MoE smoke + scheduler-glue parity at v1.7.9) |
| P-4 | Unified bench harness — runner, oracles, 15 scenarios, JSONL + Markdown reports, vqbench subprocess PPL | ✅ complete |
| P-4.5 | P-4 exit bridge — chunked-prefill minimal + VectorCodec runtime integration spike | ✅ complete (v1.6.9) |
| P-5 | VQ KV compression (BlockTQ / RaBitQ) | ✅ complete (v1.7.4 — Acceptance (1)–(4) closed; P-5-F production routing closed at v1.7.6; (b-static) Qwen3.5-4B baseline closed at v1.7.7; per-head opt-in + measurements at v1.7.8 / v1.7.10 / v1.7.11) |
| P-6 | Weight streaming (dense + per-expert MoE residency) | Stub (`ResidentWeightProvider` today) |
| P-7 | Speculative decoding (DraftTarget / EAGLE / Medusa) | Stub (`NoopDraftEngine` today) |
| P-8 | OpenAI-compatible HTTP server + session layer | ⏳ planned (T1 tail, after P-5) |

Legend: ✅ shipped · Stub = wired as the baseline implementation
behind the frozen interface, swappable in P-6 / P-7 · ⏳ = not started.

Single-source-of-truth for the roadmap and decisions log:
[`plans/PLAN.md`](plans/PLAN.md).

---

## Install

Requires Python 3.12+ and an Apple Silicon Mac. Managed via
[uv](https://github.com/astral-sh/uv).

```bash
uv pip install -e .
# chat REPL extras (prompt_toolkit + pygments)
uv pip install -e '.[chat]'
# optional extras for the planned HTTP serve path (P-8, not yet wired)
uv pip install -e '.[serve]'
# documentation tooling (Sphinx + MyST + Furo theme)
uv pip install -e '.[docs]'
```

Runtime deps: `mlx>=0.22`, `mlx-lm>=0.18`, `numpy>=1.26`, `pydantic>=2`.
Dev deps (`pytest`, `ruff`, `mypy`) live in the `dev` group.

Build the docs site (after the `[docs]` extras):

```bash
make -C docs html
open docs/_build/html/index.html
```

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

```text
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

### 4. Python API — continuous batching

```python
from silica import Engine
from silica.kvcache.prefix import RadixPrefixCache
from silica.kvcache.store import SyntheticPrefixBlockStore
from silica.models.factory import adapter_for_repo

adapter, kv = adapter_for_repo("Qwen/Qwen3-0.6B")
engine = Engine(adapter, kv)

block_size = 16
pc = RadixPrefixCache(
    block_size=block_size,
    store=SyntheticPrefixBlockStore(block_size=block_size),
)

for event in engine.generate_batch(prompts, params, max_batch_size=8, prefix_cache=pc):
    ...   # event.kind ∈ {"token", "done", "aborted"}
```

`generate_batch` yields `BatchEvent` values; `req_index` matches the
position in the `prompts` list. The full surface (`MemoryBudgeter`,
`ContinuousBatcher`, the codec wiring on `SyntheticPrefixBlockStore`)
is documented under
[`silica.engine`](docs/api/silica.engine.md) and
[`silica.scheduler`](docs/api/silica.scheduler.md) in the docs site.

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

```bash
python -m scripts.bench --all --out bench-results.jsonl --report-md bench-results.md
python -m scripts.bench --scenario qwen3-0.6b-bgt1-parity
python -m scripts.bench --list
```

Full guide — scenario catalog, dual-gate env vars, `--all-kv-codecs`
sweep, `--vqbench-xcheck` cross-check column — lives in
[`docs/bench.md`](docs/bench.md).

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

The repository ships fifteen `silica.*` subpackages — `core`,
`engine`, `scheduler`, `kvcache`, `models`, `weights`, `vq`,
`speculative`, `bench`, `chat`, `server`, `mlx`, plus the `llm`
stub. The five frozen interfaces — `ModelAdapter` (I-1), `KVManager`
(I-2), `VectorCodec[P]` (I-3, side-level since P-5-A.0.4),
`WeightProvider` (I-4), `DraftEngine` (I-5) — are `typing.Protocol`s
defined in their respective subpackages and are the integration
points for native capabilities (Principle 9 in PLAN.md).

For the per-subpackage walk-through with auto-generated symbol
listings, see [`docs/api/index.md`](docs/api/index.md) (or
`docs/_build/html/api/index.html` after `make -C docs html`).

---

## Documentation

- [`docs/_build/html/index.html`](docs/index.md) — the full Sphinx
  site (`make -C docs html` to build). Bundles the overview, chat-CLI
  guide, benchmark guide, auto-generated API, manual API, and a
  curated index into `plans/`.
- [`plans/PLAN.md`](plans/PLAN.md) — single source of truth for
  phases, deliverables, acceptance criteria, decisions log, open
  questions, and empirical findings per sub-commit.
- [`docs/API.md`](docs/API.md) — hand-curated per-module API
  reference (also rendered as the *Manual API* page on the docs
  site).
- [`docs/plans-index.md`](docs/plans-index.md) — curated entry into
  `plans/` (per-phase opening / prep / survey / acceptance docs and
  measurement artifacts).

---

## Roadmap

Phases P-0 through P-5 are closed (see the *Status* table). The
detailed sub-unit decomposition, decisions log, and acceptance
evidence live in [`plans/PLAN.md`](plans/PLAN.md) — what follows is
the structural picture only.

- **P-3-E4** *(closed at v1.7.9)* — batched MoE smoke + scheduler-
  glue parity vs direct mlx-lm batched reference. Per-expert
  streaming is P-6 scope.
- **P-3-C5** *(closed at C5.5 α-MVP)* — preempt/replay with
  recurrent-state snapshot on hybrid DeltaNet, slice-prefill regime.
- **P-5** *(closed at v1.7.4)* — KV codec stack (BlockTQ + RaBitQ).
  Acceptance items (1)–(4) all closed; production-path pre-norm
  routing closed at P-5-F (v1.7.6); Qwen3.5-4B (b-static) PPL
  baseline closed at v1.7.7. Per-head Haar rotation lands as opt-in
  (default OFF) at v1.7.8 with empirical re-measurements at v1.7.10
  / v1.7.11. The single intentionally-deferred deliverable is
  `PagedPrefixBlockStore` codec injection — waiting on the paged-
  attention kernel track per D-003.
- **P-6** *(planned)* — weight streaming. Per-expert residency for
  MoE checkpoints + layer streaming for dense models, driven by the
  `WeightProvider` interface already in place.
- **P-7** *(planned)* — speculative decoding behind the
  `DraftEngine` interface (DraftTarget / EAGLE / Medusa).
- **P-8** *(planned)* — OpenAI-compatible HTTP server + session
  layer wrapping `ChatSession` with routing, auth, streaming SSE /
  WebSocket. Leaning T1 tail per Q-002, sequenced so the HTTP
  product face ships with VQ compression live.

---

## License

Apache-2.0.
