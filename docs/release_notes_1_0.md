# silica-mlx 1.0 release notes (announce draft)

| Field        | Value                                                                                |
| ------------ | ------------------------------------------------------------------------------------ |
| Status       | announce draft — internal v1.7.33; package version is `0.0.1` in `pyproject.toml`; semver bump and PyPI publish are deferred decisions and have not been cut yet. Install is from source (editable) until then — see "Install" below. |
| Internal version | v1.7.33 (per `plans/PLAN.md` §13)                                                |
| Draft date   | 2026-05-07                                                                           |
| Target hardware | Apple Silicon, M5 Pro 48 GB (developed and measured on this machine)              |
| Pinned stack | `mlx==0.31.1`, `mlx-lm==0.31.2`, `mlx-metal==0.31.1` (see `pyproject.toml` lines 24-28 and `plans/P6_AUTORESEARCH_FINAL_REPORT.md` for the cycle-11 0.31.2 determinism break that motivated the pin) |
| Scope        | local single-user serving on Apple Silicon — see "What's in 1.0" below              |

## TL;DR

silica-mlx 1.0 is an MLX-native LLM serving framework: a vLLM-style
continuous batcher, a radix prefix cache, a pluggable KV codec
(BlockTQ + RaBitQ) at the prefix-block store, multi-family adapters
with batched parity against `mlx-lm`, and an OpenAI-compatible HTTP
server that the official `openai` Python client can drive. P-6
performance research and P-8 OpenAI HTTP server are both closed; M-9
milestone is cleared. The 1.0 surface is positioned as a *local
single-user serving framework*, not a multi-user serving cluster —
see "Out of scope" below.

## What's in 1.0

Phases shipped (all per `plans/PLAN.md` §7):

- **P-0..P-2** — frozen `typing.Protocol` seams; `Engine.generate`;
  vLLM-core scheduler (continuous batching, memory-budget admission
  ladder, preempt + replay, radix prefix cache).
- **P-3** — five model families with batched-output parity validated
  against `mlx-lm` references: Qwen3 dense (0.6B – 32B), Qwen3.5
  hybrid DeltaNet (0.8B / 4B / 27B), Gemma 4 31B dense, Qwen3.5
  35B-A3B MoE, Gemma 4 26B-A4B MoE.
- **P-4 / P-4.5** — unified bench harness (15+ scenarios across five
  oracle types); chunked-prefill minimal + VectorCodec runtime spike.
- **P-5** — VQ KV compression (BlockTQ B=64 4-bit ties the vqbench
  baseline at measurement precision; RaBitQ-1 / ExtRaBitQ 2/3/4-bit;
  per-head Haar rotation as opt-in).
- **P-6** — performance phase closed at v1.7.28; 35-cycle autoresearch
  loop; the two load-bearing levers (cycle-10 batched-aggregate
  axis-shift × cycle-12 bf16 DeltaNet recurrent state) carried the
  aggregate result; D-022 closed the small-B single-user research
  line with β/γ/δ measurement-anchored negatives.
- **P-7 (foundation)** — `DraftTargetEngine` + three rollback paths
  (target-side KV via `PagedKVCache.rollback` / `SimpleKVCache`
  per-layer trim, recurrent state via `Qwen3_5Adapter.snapshot_pre_draft_state`,
  draft-side via `DraftTargetEngine.commit`); cycle-1 byte-equal
  greedy parity on cached Qwen3-0.6B / Qwen3.5-0.8B; spec-metrics
  schema; `--speculative draft_target` bench switch.
- **P-8** — OpenAI-compatible HTTP server + session layer (this
  release): `silica serve`; chat / completions / models endpoints;
  SSE streaming; `X-Silica-Session-ID` cross-request prefix reuse;
  bearer auth + token-bucket rate limit; `--trust-proxy-headers`
  opt-in for reverse-proxy deployments; OpenAI-shaped error envelope;
  `silica.llm.LLM` Python facade.

## Performance — what is and isn't measured

Server-aggregate decode throughput on M5 Pro 48 GB:

- Dense **Qwen3.5-27B-4bit** at **B=64**: **232 tok/s** (48 GB
  hardware ceiling; cycle 28 of the autoresearch loop).
- MoE **Qwen3.5-35B-A3B-4bit** at **B=128**: **791.8 tok/s** (peak
  47.96 GB resident; cycle 35).

These are *server-aggregate* numbers — they describe how many tokens
per second the engine emits across all concurrent rows, not how fast
any one user sees a stream.

Per-row decode at the same batch sizes is the inverse:
**10.5 tok/s at B=4**, **3.92 tok/s at B=52**. **B=1 single-user
latency sits near ~20 tok/s, bandwidth-capped on M5 Pro 48 GB and
unchanged by P-6**. D-022 closed the small-B single-user research
line at v1.7.28 with β narrow-scope, γ tiny-gain, and δ
real-compute-overhead measurement-anchored negatives (≤0.6%
recoverable Python-hygiene headroom). The full performance record
lives at `docs/performance.md` and `plans/P6_AUTORESEARCH_FINAL_REPORT.md`.

## M-9 acceptance attestation (P-8 disposition)

P-8 closes against three M-9 acceptance rows. The attestation framing
is honest about what is load-bearing vs. supportive:

- **M-9.1** chat-completions stream: cleared by R-c + R-d unit tests
  + manual openai-SDK round-trips on Qwen3.5-0.8B and Qwen3.5-27B-4bit.
- **M-9.2** cross-request prefix reuse: **load-bearing attestation is
  the deterministic R-f unit test**
  `tests/test_server_session_routing.py::test_three_turn_shared_prefix_demo_logs_prefix_hits_after_turn_one`
  (pins `prefix_hit_tokens > 0` on turns 2 + 3 through a near-real
  engine). Manual smoke is supportive only — `prompt_tokens` grows
  monotonically across shared-`X-Silica-Session-ID` turns. The
  route's `prefix_hit_tokens` INFO log line is not in the captured
  v1.7.33 smoke logs because the capture predates the v1.7.34 (h)
  follow-up #3 fix that wired `silica.core.logger.setup_logging`
  into `silica.server.cli._serve()`; future smoke runs surface the
  INFO line, but the deterministic R-f test remains the canonical
  load-bearing attestation regardless.
- **M-9.3** locally behaves like a small serving engine: the
  server-side test suite (eleven `tests/test_server_*.py` files,
  198 tests collected) is clean. Manual openai-SDK surface enumerated
  per model — Qwen3.5-0.8B exercises `/healthz`, `/v1/models`,
  `/v1/chat/completions` (streaming + non-streaming),
  `/v1/completions`, and a 3-turn `X-Silica-Session-ID` session;
  Qwen3.5-27B-4bit exercises the same surface minus `/v1/models`
  (single-model registry, pinned by R-e unit tests on every commit)
  with a 2-turn session.

Smoke factbundle: `plans/P8_R_H_SMOKE/qwen3_5_0_8b.log` and
`plans/P8_R_H_SMOKE/qwen3_5_27b_4bit.log`. Disposition narrative:
`plans/PLAN.md` §13 v1.7.33 + `plans/P8_OPENING.md` §9.

## Out of scope in 1.0

The following items remain post-announce. None block 1.0; each has a
named follow-on if a real workload demands lifting it.

- **Multi-customer scheduler routing** — Options B/C in
  `plans/P8_OPENING.md` §6.1.1. v0.1 is design-locked to Option A
  (single-user local server; concurrent requests serialise on the
  engine).
- **Cross-session shared system-prompt prefix reuse** — same-session
  reuse only in v0.1 per the G-2 design lock. Each persisted
  `ChatSession` owns its own `RadixPrefixCache`.
- **SLIDING-attention adapters + persistent prefix cache** —
  Gemma 4 31B today. Naming `session_id` against a sliding-bearing
  model returns 501; drop the header for the fresh-per-call path.
- **Structured-output execution** — `response_format=json_schema`
  parses but is rejected as 501. The reservation slot is in place
  for a future grammar engine.
- **`tools` / `tool_choice` / `logprobs` / `top_logprobs` /
  `logit_bias` / `presence_penalty` / `frequency_penalty` / `n>1`** —
  all return 501 by design (matches the v0.1 single-user / local-
  developer framing).
- **Multi-process uvicorn** — `--workers > 1` and `--reload` are
  rejected at startup; lifespan slot is process-local.
- **Persistent rate-limit / auth state** — in-memory token-bucket
  and in-memory `AuthState`; restart resets both. Redis-backed
  shared state is post-announce.
- **Speculative-decoding production payoff** — P-7 ships the
  foundation + bench switch; the ≥1.2× decode-throughput payoff
  was settled with a measurement-anchored negative on this hardware
  / model stack at cycle 23 of the autoresearch loop.
- **Weight streaming for MoE residency** — stays a stub behind the
  frozen `WeightProvider` interface.

## Install

PyPI publish has not been cut for 1.0 yet (see Status in the header
table above). Install from source in editable mode:

```bash
git clone https://github.com/Ivis4ml/silica-mlx
cd silica-mlx
uv pip install -e .
uv pip install -e '.[serve]'   # OpenAI-compatible HTTP server extras
uv pip install -e '.[chat]'    # REPL chat extras
```

Requires Python 3.12+ and Apple Silicon. Core dependencies are
declared in `pyproject.toml`; the MLX stack is pinned at exact
versions `mlx==0.31.1` / `mlx-lm==0.31.2` /
`mlx-metal==0.31.1` (Darwin only) for 1.0. mlx 0.32+ is post-1.0:
cycle-11 of the P-6 autoresearch loop found that mlx-metal 0.31.2
broke the `tests/test_p2_preload_parity` argmax determinism gate
— see `plans/P6_AUTORESEARCH_FINAL_REPORT.md` cycle-11 closure
section and `plans/P6_AUTORESEARCH_NOTES.md` cycle 24.

## Minimal usage

Boot the server:

```bash
silica serve --model Qwen/Qwen3.5-0.8B --api-key sk-local-dev
```

Drive it with the openai Python client:

```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8000/v1", api_key="sk-local-dev")
resp = client.chat.completions.create(
    model="Qwen/Qwen3.5-0.8B",
    messages=[{"role": "user", "content": "Capital of France?"}],
)
print(resp.choices[0].message.content)
```

Cross-request prefix reuse via session id:

```python
client.chat.completions.create(
    model="Qwen/Qwen3.5-0.8B",
    messages=[...],
    extra_headers={"X-Silica-Session-ID": "session-42"},
)
```

See `docs/openai_server.md` for the full surface (auth, rate limit,
`--trust-proxy-headers`, error envelope, structured-output reservation
slot, observability, limitations).

## References

- `plans/PLAN.md` §7 (phase board with Status fields and acceptance
  bullets); §13 v1.7.33 (P-8 disposition narrative).
- `plans/P8_OPENING.md` §6.2 (R-a..R-h sub-unit acceptance rows);
  §6.3 (M-9 verdict matrix); §9 (disposition with commit ladder
  and factbundle pointers).
- `plans/P8_R_H_SMOKE/{qwen3_5_0_8b.log, qwen3_5_27b_4bit.log}`
  (the R-h smoke factbundle).
- `docs/openai_server.md` (user-facing HTTP server surface).
- `docs/performance.md` and `plans/P6_AUTORESEARCH_FINAL_REPORT.md`
  (the P-6 measurement record).

## What's *not* in this document

- A new feature pitch. silica-mlx 1.0 ships what `plans/PLAN.md`
  says it ships; nothing more.
- Multi-user benchmark numbers. The B=64 / B=128 figures above are
  *engine-aggregate* throughput, not user-facing latency under
  concurrent load. P-8 v0.1 routing is single-customer per request
  (one active decode turn at a time; concurrent requests serialise
  on the engine).
- A roadmap for what's next. Post-1.0 work attaches to the
  follow-on items listed in "Out of scope" above; ordering depends
  on real-workload signal that does not yet exist.
