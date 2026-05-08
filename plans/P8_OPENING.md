# P-8 — OpenAI HTTP server opening doc

| Field        | Value                                                              |
| ------------ | ------------------------------------------------------------------ |
| Phase        | P-8 (Mini-SGLang Layer, T2 per §8.1)                               |
| Milestone    | M-9 (Platform usable — OpenAI API + session usable)                |
| Status       | done (v1.7.33 disposition; sub-units (a)–(h) all landed; M-9 cleared) |
| Created      | 2026-05-06                                                         |
| Closed       | 2026-05-07 (v1.7.33 — see PLAN.md §13)                             |
| Origin       | PLAN.md §7 P-8 contract; M-9 acceptance is the announce blocker    |
| Phase order  | Runs after D-022/D-023 (both closed); precedes silica-mlx 1.0      |
| Pinned stack | mlx 0.31.1 (`project_mlx_031_2_blocked.md` still active)           |

---

## TL;DR

P-8 wraps the existing engine + chat-session stack in an OpenAI-compatible
HTTP server so the `openai` Python client can reach silica-mlx as if it were
a hosted LLM endpoint, and so cross-request prefix reuse becomes
demonstrable rather than only inferable from internal tests. The work splits
into eight sub-units (a)–(h).

**Routing posture (G-1 / G-2 decided at §6.1.2):** P-8 v0.1 is positioned
as a *local single-user OpenAI-compatible server* — Option A endpoint
routing, one active decode turn at a time, per-`ChatSession`
`RadixPrefixCache`. Multi-customer scheduler routing (Options B/C) is a
post-announce follow-on. The canonical session selector is the
`X-Silica-Session-ID` header (or `extra_body.extension.session_id`);
the OpenAI `user` field is not used as session id.

The acceptance gate is **the `openai` Python client streams a chat
completion end-to-end**; everything else is a sub-unit hard-block on the
ladder up to that wall. No new top-level dependencies are needed —
`pyproject.toml [serve]` already declares FastAPI + Uvicorn + the OpenAI
client.

---

## §1 Why P-8 now

P-8 is the M-9 gate, and M-9 is the silica-mlx 1.0 announce blocker. The
project state at v1.7.31 is:

- D-022 P-6 small-B line CLOSED (`project_p6_small_b_direction.md`).
- D-023 Gemma 4 MTP PASS-PREPROJECTION at v1.7.30
  (`project_d023_mtp_gemma4_next.md`).
- D-024 native ladder parked as post-announce TODO at v1.7.31.
- P-6 phase advanced to `done`.
- Engine + ChatSession + RadixPrefixCache + ContinuousBatcher are all in
  place; the chat-CLI demonstrates the single-customer path; tests cover
  the multi-customer `generate_batch` path.

What is missing for "usable Mac inference platform" framing: the HTTP
server. PLAN.md §7 P-8 Notes calls this out directly — "this Phase is what
upgrades Silica-MLX from 'an engine library' to 'a usable Mac inference
platform'". M-9's acceptance row reads "OpenAI API + session usable".

User intent at v1.7.31 handoff: complete P-8 before silica-mlx 1.0
announce, no parking. Recorded in `project_p8_opening_next.md`.

---

## §2 Scope

### §2.1 In scope (PLAN.md §7 P-8 deliverables, verbatim)

- `silica.server.openai_api` — FastAPI, `/v1/chat/completions`,
  `/v1/completions`.
- `silica.server.session.SessionManager` — session management,
  cross-request prefix reuse.
- An interface slot for structured generation / grammar (unimplemented).
- `silica.llm.LLM` — Python-friendly high-level interface.

### §2.2 In scope (P-8 acceptance, verbatim)

- The `openai` Python client can stream responses from the silica server.
- Cross-request prefix reuse is verifiable (send N shared-prefix requests
  in one session, check prefix cache hit rate).
- Locally behaves like a small serving engine.

### §2.3 Out of scope (deferred to v0.2 or post-announce)

- Distributed serving, multi-GPU sharding, multi-host clustering.
- Tool calling / function calling / structured output execution (only the
  *interface slot* is in scope; the actual grammar engine is not).
- Token-level usage billing, multi-tenant quota enforcement beyond a
  rate-limit floor.
- Speculative-decoding integration in the server endpoint surface (the
  Engine already supports it under-the-hood; the HTTP layer should not
  expose spec knobs in v0.1).
- Per-expert MoE residency demonstration via the API surface (P-6
  streaming work was deferred to v0.2 per D-018).
- Native MTP integration (D-024 — parked as post-announce TODO).
- TLS termination — assume reverse-proxy in front for production.
- Authentication beyond a single shared bearer token.
- Persistent session storage (SQLite / Redis) — sessions are in-memory
  for v0.1; restart loses state. SessionManager interface should not
  preclude a persistent backend in v0.2.

### §2.4 Deferred to a follow-on sub-unit (post-(h), pre-1.0)

- `silica chat` web UI bundling — the chat-CLI is `prompt_toolkit` and
  ships under `[chat]` extra; an HTTP-side equivalent (a tiny static SPA)
  is a "demo-ergonomics" follow-on, not a P-8 acceptance gate.
- OpenAI API compatibility surface beyond `/v1/chat/completions`,
  `/v1/completions`, `/v1/models` — embeddings, moderations, files,
  fine-tuning, batch are deferred.

---

## §3 Entry-point inventory (verified at v1.7.31, 2026-05-06)

### §3.1 Empty packages — what P-8 fills

| Path                            | Lines | Notes                                  |
| ------------------------------- | ----- | -------------------------------------- |
| `silica/server/__init__.py`     | 0     | empty                                  |
| `silica/server/cli.py`          | 178   | one-shot `python -m silica run` CLI    |
| `silica/llm/__init__.py`        | 0     | empty                                  |

`silica/server/cli.py` is a `python -m silica run --model ... --prompt
...` dispatcher. It is **not** the HTTP entry point and should not be
extended into one — the HTTP server gets its own module
(`silica.server.openai_api`) and its own console-script entry.

### §3.2 Already declared dependencies

`pyproject.toml [project.optional-dependencies]` (lines 41-44):

```toml
serve = [
    "fastapi>=0.115",
    "uvicorn>=0.30",
    "openai>=1.0",
]
```

The `openai>=1.0` dep is for end-to-end tests against the silica server
itself, matching the M-9 acceptance "the `openai` Python client can
stream responses". No additional top-level dependencies are needed for
P-8's first slice; if `python-multipart`, `sse-starlette`, or similar is
pulled in for sub-unit (d), it should land alongside the sub-unit, not
pre-emptively.

### §3.3 API surfaces P-8 will wrap

Both core surfaces are stream-shaped. The HTTP layer's job is to bridge
them to OpenAI's SSE wire format.

| Surface | Path | Shape | Use site |
| --- | --- | --- | --- |
| `Engine.generate(prompt, sampling_params)` | `silica/engine/__init__.py:143` | `Iterator[int]` | single-request hot path |
| `Engine.generate_batch(prompts, params, *, max_batch_size, prefix_cache, length_spread_threshold)` | `silica/engine/__init__.py:579` | `Iterator[BatchEvent]` | multi-row scheduling |
| `BatchEvent.{token,done,aborted}` | `silica/core/events.py:48,52,56` | dataclass classmethod factories | `req_index` is into prompts list, not a request ID |
| `ChatSession.chat(user_text, *, sampling_params=None, stream_to=None)` | `silica/chat/session.py:574` | returns `TurnMetrics` | high-level conversation API; `stream_to` is a token callback |
| `ChatSession.continue_last(...)` | `silica/chat/session.py:814` | returns `TurnMetrics` | finish_reason=length truncation recovery |
| `ContinuousBatcher.{add_request,has_work,step}` | `silica/scheduler/batcher.py:394,477,489` | `step()` returns `list[BatchEvent]` | inside `Engine.generate_batch` |
| `RadixPrefixCache.{peek,lookup,insert,stats}` | `silica/kvcache/prefix.py:188,210,305,499` | `stats()` returns `PrefixCacheStats` | acceptance #2 leans on `stats()` |
| `TurnMetrics` dataclass | `silica/chat/session.py:105` | `reply, prompt_tokens, output_tokens, finish_reason, raw_reply, ttft_ms, prefill_tok_s, decode_tok_s, resident_mb, peak_memory_mb, logical_kv_bytes, wall_s, prefix_hit_blocks, prefix_hit_tokens, prefix_store_resident_bytes, prefix_store_logical_bytes` | maps cleanly to OpenAI `usage` + extension fields |

### §3.4 Things NOT to assume

- **No existing FastAPI app, router, or app factory in `silica/server/`** —
  do not grep expecting to find one.
- **`silica/server/cli.py` is not the HTTP entry point** — it is a
  one-shot generation CLI. The HTTP server needs its own
  `python -m silica.server.openai_api` (or `silica serve`) entry.
- **chat-CLI is not the API server** — `silica/chat/` is `prompt_toolkit`-
  based under `[chat]` extra; P-8 reuses `ChatSession` as a building
  block, not the chat-CLI shell.
- **`silica.llm.LLM` is not yet defined** — the module file exists
  empty; sub-unit (g) creates the class.
- **mlx 0.31.1 pin is still in effect** — see
  `project_mlx_031_2_blocked.md`. P-8 must not introduce upgrade
  pressure on the mlx stack.

---

## §4 Architecture sketch

### §4.1 Process model

One Uvicorn worker process owns one `Engine` instance. The Engine owns
the model/adapter path and constructs the per-generation KV/scheduler
machinery; under the v0.1 G-2 decision, prefix-cache ownership lives
with `SessionManager` / `ChatSession`, not with the Engine. Each
persisted `ChatSession` carries the `RadixPrefixCache` passed into the
batched generation path for that turn. The Engine is single-threaded
MLX; the FastAPI request handlers run on Uvicorn's asyncio loop and
submit work to the engine via:

- **Single-customer path:** `ChatSession.chat(stream_to=...)` for the
  most common request shape (one chat conversation per HTTP request,
  no concurrent decode rows).
- **Multi-customer path:** `Engine.generate_batch(...)` + a request
  multiplexer when concurrent in-flight requests share the engine
  instance.

The boundary between the two routes is the load-bearing design choice
in P-8 — see §6.1 stop-and-ask gate (G-1).

### §4.2 SessionManager responsibilities

`silica.server.session.SessionManager` owns:

- `session_id → ChatSession` map (in-memory, dict-of-`ChatSession`).
- Lifetime / eviction policy: LRU with a configurable cap (default
  `--max-sessions=64`); idle TTL (default `--session-ttl=30min`).
- Cross-request prefix-reuse story: each persisted `ChatSession`
  carries its own `RadixPrefixCache` (G-2 decided in favour of
  per-session — see §6.1.2). The v0.1 acceptance gate proves
  *same-session* cross-request reuse only; cross-session shared
  system-prompt reuse is post-P-8.
- An exposed metric pair (`prefix_hit_tokens`, `prefix_hit_blocks`)
  surfaced via `/metrics` or in the response trailers so the
  acceptance #2 measurement is automatable.

### §4.3 OpenAI-API request shapes

The OpenAI Chat Completions schema (Pydantic v2) covers:

- `model`, `messages`, `max_tokens`, `temperature`, `top_p`, `stream`,
  `stop`, `seed` (optional). The OpenAI `user` field is **not** the
  canonical session id — its semantics in the OpenAI spec are
  abuse-monitoring identifier, not conversation continuity. P-8 uses an
  explicit `X-Silica-Session-ID` HTTP header as the preferred session
  selector. For OpenAI Python client callers that prefer body-only
  extensions, `extra_body.extension.session_id` is also accepted; the
  server normalises both inputs into the same internal session identity.
  `user` is at most a development fallback and may be left unsupported
  in v0.1.
- silica extensions go under `extra_body` / a single `extension`
  envelope so the standard schema stays clean — proposed extensions:
  `extension.session_id` (same internal session identity as
  `X-Silica-Session-ID`), `extension.thinking_mode`,
  `extension.continue_truncated`.
- Response: standard OpenAI `ChatCompletion` / `ChatCompletionChunk`
  envelope; `usage.prompt_tokens` / `usage.completion_tokens` populated
  from `TurnMetrics`; SSE `finish_reason ∈ {stop, length, content_filter}`.

### §4.4 Streaming wire format

Sub-unit (d) emits Server-Sent Events conformant to OpenAI's published
streaming format:

```text
data: {"id":"...","object":"chat.completion.chunk","choices":[{"delta":{"content":"hi"}}]}\n\n
data: {"id":"...","object":"chat.completion.chunk","choices":[{"delta":{},"finish_reason":"stop"}]}\n\n
data: [DONE]\n\n
```

Content-Type `text/event-stream`; `Cache-Control: no-cache`; the
generator coroutine drains tokens from `ChatSession.chat`'s `stream_to`
callback (or `Engine.generate_batch`'s BatchEvent yield) and serializes
each into a chunk envelope. Backpressure is the asyncio queue between
the engine's stream callback (called from MLX-thread context) and the
SSE generator coroutine — see (d) deliverables.

---

## §5 Sub-unit ladder

Eight sub-units (a)–(h). Each lands behind a sub-unit acceptance gate
(see §6) before the next one starts. Sub-units (a)–(d) are the M-9
critical path; (e)–(h) round out the OpenAI surface and the
production-readiness floor.

### (a) FastAPI scaffold + lifespan

- **Goal:** stand up `silica.server.openai_api:app` as an empty FastAPI
  app with a lifespan manager that owns one `Engine` instance, a
  `/healthz` route, and structured logging via `silica.core.logger`.
- **Deliverables:**
  - `silica/server/openai_api.py` — FastAPI app + lifespan.
  - `silica/server/runtime.py` — `Engine`-owning singleton wrapper.
  - Console script `silica serve` in `pyproject.toml [project.scripts]`,
    invoking `uvicorn silica.server.openai_api:app`.
  - One smoke test asserting `/healthz` returns `{"status":"ok"}` with
    the engine resident.
- **Acceptance row:** `silica serve --model Qwen/Qwen3.5-0.8B` boots,
  `/healthz` returns 200, no MLX leakage on shutdown.

### (b) Request/response schemas

- **Goal:** Pydantic v2 models matching OpenAI Chat Completions and
  Completions strictly enough that the `openai` Python client
  serialises into them without errors.
- **Deliverables:**
  - `silica/server/schemas.py` — `ChatCompletionRequest`,
    `ChatCompletionResponse`, `ChatCompletionChunk`, `Completion*`,
    `Usage`, `ResponseFormat`, the silica `Extension` envelope.
  - Schema-only unit tests: round-trip a captured `openai` Python
    client request payload, assert no validation errors.
- **Acceptance row:** five canonical OpenAI request fixtures
  (greedy / temperature / top-p / stream / non-stream) all parse;
  `extra_body` extensions parse into the `Extension` envelope.

### (c) `/v1/chat/completions` non-streaming

- **Goal:** synchronous chat turn driven by `ChatSession.chat(
  stream_to=None)`. One conversation per HTTP request, no SSE.
- **Deliverables:**
  - `silica/server/routes/chat_completions.py` — non-streaming branch
    only.
  - SessionManager v0 stub: every request gets a fresh `ChatSession`,
    no reuse (defer reuse to (f)).
  - `usage` block populated from `TurnMetrics`.
  - Integration test: `openai.OpenAI(base_url=...).chat.completions
    .create(..., stream=False)` round-trips and returns a non-empty
    reply.
- **Acceptance row:** one openai-client non-streaming round trip
  passes against a real Qwen3.5-0.8B model load; `usage.prompt_tokens`
  matches `TurnMetrics.prompt_tokens` exactly.

### (d) `/v1/chat/completions` streaming SSE

- **Goal:** SSE-formatted streaming responses driven by
  `ChatSession.chat(stream_to=callback)`. The asyncio generator
  forwards each callback-delivered token into a chunk envelope.
- **Deliverables:**
  - Streaming branch in
    `silica/server/routes/chat_completions.py`.
  - Asyncio-queue bridge between MLX-thread `stream_to` callback and
    the SSE coroutine. Backpressure semantics documented in the
    module docstring.
  - SSE chunk format: matches OpenAI's published streaming spec
    (`object:"chat.completion.chunk"`, `delta` with `content` /
    `role` / `finish_reason`, terminal `data: [DONE]`).
  - Integration test: `openai.OpenAI(...).chat.completions.create(...,
    stream=True)` round-trips, asserts every chunk parses and the
    concatenation matches the non-streaming reply.
- **Acceptance row:** one openai-client streaming round trip passes;
  TTFT measured at HTTP layer is within 50 ms of `TurnMetrics.ttft_ms`
  for a localhost call (acceptance is correctness, not latency).

### (e) `/v1/completions` and `/v1/models`

- **Goal:** legacy text-completion endpoint (`/v1/completions`) and
  the model-list endpoint (`/v1/models`).
- **Deliverables:**
  - `silica/server/routes/completions.py` — text-completions branch
    using `Engine.generate` directly (no chat template).
  - `silica/server/routes/models.py` — `/v1/models` returns the
    currently loaded model with metadata from the adapter's
    `capabilities()`.
  - Integration tests for both endpoints via the openai client.
- **Acceptance row:** openai-client text completion + models list
  round-trips pass.

### (f) `SessionManager` + cross-request prefix reuse

- **Goal:** real `SessionManager` with `session_id → ChatSession`
  mapping, eviction policy, and a verifiable prefix-reuse demo.
- **Deliverables:**
  - `silica/server/session.py` — `SessionManager` class with
    `get_or_create`, `evict_idle`, `evict_lru`, `stats()` returning
    aggregate `PrefixCacheStats`.
  - Server-side mapping: incoming request's `X-Silica-Session-ID`
    header (equivalently `extra_body.extension.session_id`) selects the
    `ChatSession`. Absent header → a fresh per-request session is
    used (no cross-request reuse for that call); the OpenAI `user`
    field is **not** consulted as session id.
  - Per-session `RadixPrefixCache`: each persisted `ChatSession` owns
    its own `RadixPrefixCache` (G-2 decision per §6.1.2). The
    `RadixPrefixCache` API stays as it is today — token-block keyed,
    no session/tenant scope — and isolation is enforced by which
    `ChatSession` instance holds the cache reference. Process-global
    cache + session-scoped keys is **out of scope** for v0.1; revisit
    only if cross-session shared-system-prompt reuse becomes a
    post-announce requirement.
  - Integration test: send 3 requests with shared system prompt prefix
    in one `session_id`; assert second + third request's
    `prefix_hit_tokens > 0`.
- **Acceptance row:** N-shared-prefix demo passes in a deterministic
  test (acceptance #2 verbatim).

### (g) `silica.llm.LLM` Python facade

- **Goal:** mlx-lm-style `LLM(model="...").generate("prompt")`
  ergonomic wrapper over `Engine` + optional `ChatSession` so existing
  notebooks can drop silica in with one import.
- **Deliverables:**
  - `silica/llm/__init__.py` exposing `LLM`.
  - `silica/llm/_facade.py` — `LLM` class with `generate(prompt,
    sampling_params=None, stream=False)`, `chat(messages, ...)`,
    `unload()`.
  - Lazy-load semantics: model is loaded on first call, not at
    `LLM(...)` construction (avoid surprising cost in notebook
    imports).
  - Migration smoke test: the chat-CLI's startup path can be rewritten
    to use `silica.llm.LLM` without behaviour drift (this rewrite is
    deferred to a follow-on, but the smoke test asserts feasibility).
- **Acceptance row:** `silica.llm.LLM` round-trips a single greedy
  generation matching `Engine.generate` byte-for-byte under the same
  seed.

### (h) Auth, rate-limit, errors, structured-output slot, tests, docs

- **Goal:** production-readiness floor for v0.1 — bearer-token auth,
  rate-limit ceiling, OpenAI-shaped error responses, structured
  output reservation slot, full test coverage, user-facing docs.
- **Deliverables:**
  - `silica/server/auth.py` — single shared bearer token from
    `SILICA_API_KEY` env; 401 when absent or wrong.
  - Rate limit: token-bucket per `Authorization` header, configurable
    via `--rate-limit-rpm` flag.
  - Error mapper: OpenAI's standard error envelope
    (`{"error":{"message":"...","type":"..."}}`).
  - `response_format` slot: accept it in the request schema, return
    a 501-like notice "structured output not yet implemented" but
    log the requested format for future use.
  - End-to-end test suite covering all routes + auth failure modes +
    abort handling (client disconnect mid-stream).
  - User docs: `docs/openai_server.md` covering startup, env vars,
    extension envelope, prefix reuse demo.
- **Acceptance row:** the server-side test suite (the
  `tests/test_server_*.py` family — eleven files at disposition,
  198 tests collected) passes; manual openai client end-to-end on
  Qwen3.5-0.8B + on Qwen3.5-27B-4bit; M-9 acceptance #1, #2, #3
  all marked.

---

## §6 Acceptance gate matrix

### §6.1 Stop-and-ask gates (decision rows)

| Gate | Where | Decision required | Hard block? |
| --- | --- | --- | --- |
| **G-1** | (c)/(d) before commit | Routing model for `/v1/chat/completions`. See §6.1.1 for the three concrete options and their failure modes under N>1 concurrent streaming clients. | Yes — choosing wrong forces a rewrite |
| **G-2** | (f) before commit | One `RadixPrefixCache` per `ChatSession` vs one process-global cache with session-scoped insert keys. Per-session is isolated; global is more reuse but harder to evict and audit. | Yes — affects prefix-hit accounting |
| **G-3** | (d) before commit | Asyncio-queue backpressure policy when SSE client is slow. Drop / block / disconnect. | No — soft-warn, can iterate |

Each G-* gate must be surfaced to the user before the implementing
sub-unit's PR lands. The convention from D-022 / D-023 applies: hard-
block gates evaluate first; the OPENING doc records the user's
decision verbatim.

### §6.1.1 G-1 routing options (the load-bearing P-8 design choice)

The endpoint contract is "one HTTP request → one chat completion (stream
or non-stream)". The engine offers two streams: `ChatSession.chat
(stream_to=callback)` (single-row, blocking the calling thread for the
duration of the turn) and `Engine.generate_batch(prompts, ...) →
Iterator[BatchEvent]` (multi-row, scheduler-managed). Three concrete
routing options:

**Option A — single-customer per request, no shared engine queue.**
The endpoint handler routes each HTTP request through a `ChatSession`
and calls `session.chat(stream_to=callback)` synchronously on a worker
thread. *One active decode turn at a time*; before sub-unit (f),
requests use a fresh `ChatSession` per call (no cross-request reuse);
after (f), `X-Silica-Session-ID` selects a persisted `ChatSession`
whose `RadixPrefixCache` carries forward across requests in that
session. Simple, mirrors chat-CLI exactly. Breakage under N>1
concurrent streaming clients: the engine is single-threaded MLX; only
one chat turn can decode at a time. Concurrent requests serialise on
a global engine lock (or compete via Python's GIL). The 2nd through
Nth client sees increasing TTFT proportional to in-flight chat turns.
No queueing visibility — the OS scheduler decides who waits.
Acceptable for a "local mac inference platform" with low concurrency;
not acceptable for multi-user deployment.

**Option B — multi-customer scheduler-driven, single endpoint shape.**
Every HTTP request enqueues into the `ContinuousBatcher` via
`Engine.generate_batch`-style admission, demultiplexed by `req_index`.
A central asyncio-queue dispatcher fans `BatchEvent`s back to per-
request SSE generators. Adds a queue-and-dispatch layer (request
admission, BatchEvent fan-out, abort-on-disconnect propagation) but
gives the scheduler real concurrency. Breakage modes: queue dispatcher
becomes a bottleneck if not carefully written; SSE generator must be
robust to "BatchEvent for my req_index" filtering; abort handling is
harder (cancelling a request mid-flight requires the scheduler to honour
abort, which it does, but the wiring is more involved).

**Option C — hybrid: single-customer endpoint shape, multi-customer
engine.** The endpoint handler still calls `ChatSession.chat
(stream_to=callback)`, but `ChatSession` is rewritten to enqueue into
the multi-row scheduler under the hood (concretely, `ChatSession.chat`
becomes a thin facade over an internal `generate_batch([prompt])` call
with per-row state). Each HTTP request is a 1-row "batch"; the scheduler
sees them as concurrent rows and amortises decode across them.
Breakage modes: requires a touch to `ChatSession` that crosses a
package boundary (chat-session is currently single-row by contract); the
internal-batcher path may differ from current behaviour in edge cases
(spec decoding, recurrent state, prefix-cache integration) and would
need a parity test before the rewrite lands. The cleanest long-term
design but the largest scope expansion.

**Decision criteria for G-1:**

- **If announce target is "local single-user inference platform":**
  Option A is sufficient and minimal. Acceptable announce framing:
  "single-process server, concurrent requests serialise on the engine".
- **If announce target is "small multi-user serving engine":** Option B
  or C is required. Option B is the smaller delta from current code;
  Option C is the cleaner design but expands `ChatSession` scope.
- **If P-8 must close in one announce push:** Option A is the only
  choice that fits without scope creep.

### §6.1.2 G-1 / G-2 / G-3 decisions (recorded 2026-05-06)

Recorded verbatim from the OPENING-review turn:

- **G-1 → Option A.** P-8 v0.1 is positioned as a *local single-user
  OpenAI-compatible server*, not a small multi-user scheduler. Options
  B and C would convert P-8 from an HTTP/API layer into a scheduler
  rewrite and expand scope past the announce push. Option A's wording
  is amended to be forward-compatible with sub-unit (f): *one active
  decode turn at a time; before (f) requests use fresh sessions; after
  (f), session_id selects a persistent `ChatSession`*. Multi-customer
  scheduler routing (Options B/C) is a post-announce follow-on, opened
  as a separate phase if needed.
- **G-2 → per-`ChatSession` `RadixPrefixCache`.** The current
  `RadixPrefixCache` is token-block keyed with no session/tenant
  scope; `ChatSession.set_prefix_cache()` already treats reset-via-
  fresh-cache as the anti-leak mechanism. Process-global cache plus
  session-scoped insert keys would touch radix key semantics, store
  accounting, and eviction — too much risk for P-8 v0.1. Each
  persisted `ChatSession` owns its own `RadixPrefixCache`; the v0.1
  acceptance gate proves *same-session cross-request* reuse only.
  Cross-session shared system-prompt reuse is post-P-8.
- **G-3 (soft) default → bounded queue, no token drop, backpressure
  on slow clients, disconnect → cancel/abort.** Streaming SSE that
  drops tokens on a slow consumer breaks the OpenAI client semantics
  (the assembled reply on the consumer side would be silently
  truncated). v0.1 default is: bounded asyncio queue between MLX-
  thread `stream_to` callback and SSE coroutine; if the queue fills,
  the producer (engine callback) blocks, propagating backpressure
  through the chat-session call site and slowing that single request.
  On client disconnect, propagate abort into the engine to cancel the
  request promptly. Adjustable in (h) hardening if a real workload
  requires different behaviour.

These decisions are now binding for sub-units (c)/(d)/(f); subsequent
PR-level adjustments must reference §6.1.2 explicitly.

### §6.2 Sub-unit acceptance rows (hard blocks)

| Row | Sub-unit | Acceptance | Hard block? | Status (v1.7.33) |
| --- | --- | --- | --- | --- |
| R-a | (a) | `silica serve` boots; `/healthz` returns 200 | Yes | **MET** — `405b3d0` (a1) + `d0212ab` (a2)+(a4) + `fb96c3b` /healthz pin + `89729bb` (a3) `silica serve` |
| R-b | (b) | OpenAI request fixtures parse without validation errors | Yes | **MET** — `3039805` Pydantic v2 schemas + `Extension` envelope |
| R-c | (c) | `openai` non-streaming round trip succeeds against Qwen3.5-0.8B | Yes (M-9 acceptance #1 dependency) | **MET** — `0ff3ff0` (c) + R-h smoke `qwen3_5_0_8b.log` (283 ms wall, finish_reason=length) |
| R-d | (d) | `openai` streaming round trip succeeds; chunks concatenate to non-streaming reply | Yes (M-9 acceptance #1 dependency) | **MET** — `5da48a1` (d) SSE + R-h smoke (TTFT 2 ms / total 71 ms / 8 chunks on 0.8B; TTFT 3 ms / 596 ms on 27B-4bit) |
| R-e | (e) | `/v1/completions` + `/v1/models` round trip via openai client | No (M-9 doesn't require beyond chat) | **MET** — `e4a74fb` (e) + R-h smoke shows both endpoints green via openai SDK |
| R-f | (f) | N-shared-prefix demo: 3-turn shared-prefix session shows `prefix_hit_tokens > 0` on turn 2+ | Yes (M-9 acceptance #2) | **MET** — `0f4af00` (f) `SessionManager` + the deterministic test `tests/test_server_session_routing.py::test_three_turn_shared_prefix_demo_logs_prefix_hits_after_turn_one` pins `prefix_hit_tokens > 0` on turns 2 + 3 through a near-real engine. Manual smoke shows `prompt_tokens` growth across turns (0.8B 27→48→73; 27B 24→48), supportive but not load-bearing — the route's `prefix_hit_tokens` INFO line is not in the captured log because the v1.7.33 capture pre-dated the v1.7.34 (h) follow-up #3 fix; `silica serve` did not yet call `setup_logging` at the time of capture. The fix landed at v1.7.34 (see §9.4); future smoke runs will surface the INFO line. |
| R-g | (g) | `silica.llm.LLM` round-trips a greedy generation byte-for-byte vs `Engine.generate` | No (PLAN deliverable, not M-9 gate) | **MET** — `2e33886` (g) facade + 18 tests in `tests/test_llm_facade.py` (`uv run pytest tests/test_llm_facade.py` clean); smoke run on Qwen3.5-0.8B confirms `LLM.generate` + `LLM.chat` both work end-to-end |
| R-h | (h) | server-side test suite passes; manual end-to-end on real Qwen3.5-27B-4bit | Yes (M-9 acceptance #3) | **MET** — `e86a735` (h) + `9eaeaba` follow-up #1 + `776e749` follow-up #2; the eleven `tests/test_server_*.py` files (198 tests) are clean; R-h smoke `qwen3_5_27b_4bit.log` exercises `/healthz`, `/v1/chat/completions` (streaming + non-streaming), `/v1/completions`, and `X-Silica-Session-ID` 2-turn shared-prefix demo on the production-target at ~13.8 tok/s effective decode (the 0.8B sanity log additionally drives `/v1/models` + a 3-turn session). |

### §6.3 M-9 overall acceptance (terminal verdict)

The phase closes only when all three M-9 acceptance rows pass:

- **M-9.1** — openai Python client streams chat completions from silica
  server (cleared by R-c + R-d). **CLEARED** (v1.7.33) — round-trips
  green on Qwen3.5-0.8B (non-streaming 283 ms / streaming TTFT 2 ms,
  total 71 ms / 8 chunks) and on Qwen3.5-27B-4bit (non-streaming
  1737 ms at ~13.8 tok/s effective decode / streaming TTFT 3 ms,
  total 596 ms / 8 chunks). Factbundle: `plans/P8_R_H_SMOKE/`.
- **M-9.2** — cross-request prefix reuse verifiable via shared-prefix
  test (cleared by R-f). **CLEARED** (v1.7.33). **Load-bearing
  attestation is the R-f deterministic unit test**
  `tests/test_server_session_routing.py::test_three_turn_shared_prefix_demo_logs_prefix_hits_after_turn_one`
  (pins `prefix_hit_tokens > 0` on turns 2 + 3 of a 3-turn shared-
  prefix session through a near-real engine). The manual smoke
  (Qwen3.5-0.8B 3-turn `prompt_tokens` 27 → 48 → 73; Qwen3.5-27B-4bit
  2-turn 24 → 48) is supportive only — `prompt_tokens` growth is
  consistent with cache reuse but does not directly attest a hit.
  The route's `prefix_hit_tokens` INFO line is not captured in the
  v1.7.33 smoke logs because the capture predates the v1.7.34 (h)
  follow-up #3 fix; at the time of capture `silica serve` did not
  yet call `silica.core.logger.setup_logging`, the silica.* logger
  had no handler attached at runtime, and `--log-level info`
  configured only uvicorn's loggers. The wiring landed at v1.7.34
  (see §9.4) — future smoke runs will surface the INFO line. The
  deterministic R-f test remains the canonical M-9.2 attestation
  regardless.
- **M-9.3** — locally behaves like a small serving engine (cleared by
  R-h end-to-end on real model). **CLEARED** (v1.7.33). The
  server-side test suite (eleven `tests/test_server_*.py` files,
  198 tests collected) is clean. Manual openai-SDK surface
  enumerated per model:
  - Qwen3.5-0.8B (sanity): `/healthz` 200, `/v1/models` round-trip,
    `/v1/chat/completions` non-streaming + streaming,
    `/v1/completions`, `X-Silica-Session-ID` 3-turn shared-prefix
    demo.
  - Qwen3.5-27B-4bit (production-target): `/healthz` 200,
    `/v1/chat/completions` non-streaming + streaming,
    `/v1/completions`, `X-Silica-Session-ID` 2-turn shared-prefix
    demo. `/v1/models` was not driven on this run; it is
    functionally identical to the 0.8B path (single-model registry,
    no model-specific code) and is pinned by the R-e unit tests
    on every commit.

**P-8 phase status flipped to `done` at v1.7.33** — see PLAN.md §13
v1.7.33 entry for the disposition narrative + factbundle + commit
ladder + scope-creep audit.

### §6.4 Diagnostic rows (not hard blocks)

These are useful to record but do not gate disposition:

- TTFT delta vs chat-CLI on the same prompt (HTTP overhead measurement).
- Concurrent-request stability: 8 simultaneous streaming requests for
  60 seconds without scheduler crash.
- Memory ceiling: peak resident memory under the production-target
  load (Qwen3.5-27B-4bit, B=4 concurrent sessions).
- Prefix-reuse hit rate at session-end for a representative
  multi-turn workload.

---

## §7 Sources

- `plans/PLAN.md` §7 P-8 (the contract); §8.2 M-9 row (the
  milestone); §13 v1.7.30 + v1.7.31 changelog (closure context).
- `plans/MTP_GEMMA4_PRE_PROJECTION.md` — D-023 opening template
  (gate matrix, hard-block ordering, stop-and-ask).
- `plans/CHAT_CLI_OPENING.md` — closest engineering-side opening
  template (sub-unit ladder, in-scope/out-of-scope tiers).
- `plans/P5_OPENING.md`, `plans/P6_OPENING.md` — phase-opening
  layout precedents.
- `silica/engine/__init__.py:143-700` — `Engine.generate` /
  `Engine.generate_batch`.
- `silica/chat/session.py:574-900` — `ChatSession.chat` /
  `continue_last` / `TurnMetrics`.
- `silica/core/events.py` — `BatchEvent` definitions.
- `silica/scheduler/batcher.py:223+` — `ContinuousBatcher`.
- `silica/kvcache/prefix.py:142+` — `RadixPrefixCache`.
- `pyproject.toml [project.optional-dependencies]` — `[serve]`,
  `[chat]` extras.

## §8 Cross-references

- `docs/plans-index.md` — P-8 section added at v1.7.32; keep it in
  sync as sub-units land.
- `~/.claude/projects/-Users-xinyu-Desktop-silica-mlx/memory/project_p8_opening_next.md`
  — entry-point memory authored 2026-05-06; mirrored facts above.
- `~/.claude/projects/-Users-xinyu-Desktop-silica-mlx/memory/project_mlx_031_2_blocked.md`
  — mlx 0.31.1 pin; do not bump during P-8.
- `~/.claude/projects/-Users-xinyu-Desktop-silica-mlx/memory/feedback_incremental_plan_execution.md`
  — pause per sub-unit.
- `~/.claude/projects/-Users-xinyu-Desktop-silica-mlx/memory/feedback_commit_approval.md`
  — confirm before each commit.

---

## §9 Disposition (2026-05-07, v1.7.33)

P-8 closes with all eight sub-units (a)–(h) landed and the M-9
milestone cleared. This section is the per-acceptance-row attestation;
the narrative form (commit ladder, factbundle pointers, scope-creep
audit) lives in `plans/PLAN.md` §13 v1.7.33.

### §9.1 Commit ladder

Thirteen commits closed the (a)–(h) ladder:

| # | Commit | Sub-unit | Subject |
| - | ------ | -------- | ------- |
| 1 | `405b3d0` | (a1) | `Runtime` wrapper + `engine_lock` + `to_thread` contract |
| 2 | `d0212ab` | (a2)+(a4) | FastAPI app + lifespan + `/healthz` strict + smoke |
| 3 | `fb96c3b` | post-(a2) | `/healthz` cleanup-branch pin |
| 4 | `89729bb` | (a3) | `silica serve` subcommand + single-process invariant |
| 5 | `3039805` | (b) | OpenAI-compatible Pydantic v2 schemas + `Extension` envelope |
| 6 | `0ff3ff0` | (c) | `/v1/chat/completions` non-streaming |
| 7 | `5da48a1` | (d) | `/v1/chat/completions` SSE streaming + worker-orphan G-1 fix |
| 8 | `e4a74fb` | (e) | `/v1/models` + `/v1/completions` + R-e SDK round trips |
| 9 | `0f4af00` | (f) | `SessionManager` + cross-request prefix reuse |
| 10 | `2e33886` | (g) | `silica.llm.LLM` Python facade |
| 11 | `e86a735` | (h) | auth + rate-limit + OpenAI error envelope + structured-output 501 slot + docs |
| 12 | `9eaeaba` | (h) follow-up #1 | auth-state-aware rate-limit bucket key + buffered-chat docstring honesty + RequestValidationError input redaction |
| 13 | `776e749` | (h) follow-up #2 | `--trust-proxy-headers` opt-in gating XFF / X-Real-IP trust |

### §9.2 R-h smoke factbundle

Two real-model smoke runs captured at `plans/P8_R_H_SMOKE/`:

- `qwen3_5_0_8b.log` — sanity model (`Qwen/Qwen3.5-0.8B`).
  Endpoints driven: `/healthz`, `/v1/models`, openai SDK
  non-streaming chat (283 ms wall), openai SDK streaming chat
  (TTFT 2 ms / total 71 ms / 8 chunks), legacy `/v1/completions`,
  3-turn `X-Silica-Session-ID` shared-prefix demo
  (`prompt_tokens` 27 → 48 → 73). Surface green.
- `qwen3_5_27b_4bit.log` — production-target
  (`mlx-community/Qwen3.5-27B-4bit`). Endpoints driven:
  `/healthz`, openai SDK non-streaming chat (1737 ms wall,
  ~13.8 tok/s effective decode), openai SDK streaming chat
  (TTFT 3 ms / total 596 ms / 8 chunks), `/v1/completions`,
  2-turn `X-Silica-Session-ID` shared-prefix demo
  (`prompt_tokens` 24 → 48). `/v1/models` was not driven in
  this run; the endpoint is single-model and behaves identically
  to the 0.8B path, and the R-e unit tests pin its shape on every
  commit.

**M-9.2 caveat carried in this factbundle.** The route's
`prefix_hit_tokens` INFO line is **not** present in either log
because the v1.7.33 capture predates the v1.7.34 (h) follow-up
#3 fix. At the time of capture `silica serve` did not yet call
`silica.core.logger.setup_logging`, the silica.* logger
namespace had no handler attached at runtime, and `--log-level
info` configured only uvicorn's loggers. The wiring landed at
v1.7.34 (see §9.4 closure note); future smoke runs will surface
the INFO line. The deterministic R-f unit test
(`tests/test_server_session_routing.py::test_three_turn_shared_prefix_demo_logs_prefix_hits_after_turn_one`)
remains the load-bearing M-9.2 attestation, and the
`prompt_tokens` growth in these archived smoke logs is
supportive evidence consistent with cache reuse.

### §9.3 M-9 verdict (terminal)

| Row | Status | Evidence |
| --- | ------ | -------- |
| **M-9.1** chat-completions stream | **CLEARED** | R-c (`0ff3ff0`) + R-d (`5da48a1`) + R-h smoke openai-SDK chat round trips (streaming + non-streaming) on Qwen3.5-0.8B and Qwen3.5-27B-4bit |
| **M-9.2** cross-request prefix reuse | **CLEARED** | Load-bearing: R-f deterministic test `tests/test_server_session_routing.py::test_three_turn_shared_prefix_demo_logs_prefix_hits_after_turn_one` pins `prefix_hit_tokens > 0` on turns 2 + 3. Supportive: R-h smoke shows `prompt_tokens` growing monotonically across shared-`X-Silica-Session-ID` turns on both real models |
| **M-9.3** locally behaves like a small serving engine | **CLEARED** | R-h (`e86a735` + `9eaeaba` + `776e749`) + the eleven `tests/test_server_*.py` files (198 tests collected) clean + per-model enumerated openai SDK surface in §9.2 |

**P-8 disposition: DONE.** PLAN.md §7 P-8 Status flips to `done` at
v1.7.33; all 4 deliverable checkboxes ticked; all 3 acceptance
checkboxes ticked. With M-9 cleared, the platform-side acceptance
gating for silica-mlx 1.0 announce is green; remaining work is
announce push, which is out of P-8 scope.

### §9.4 Out-of-scope items reaffirmed at disposition

The following items remain post-announce / post-P-8 and are
**not** dispositioned in this commit:

- Multi-customer scheduler routing (Options B/C in §6.1.1) — Option A
  single-user shape was the v0.1 commitment; B/C is opened as a
  separate phase if a real workload demands it.
- Cross-session shared system-prompt prefix reuse — same-session
  reuse only in v0.1 per G-2.
- SLIDING-attention adapter (Gemma 4 31B today) + persistent
  `RadixPrefixCache` — incompatible by `ContinuousBatcher` admission
  rule; route returns 501 with actionable message; drop the header
  to use the fresh-per-call path.
- Structured-output execution (`response_format=json_schema`) — the
  reserved 501 slot is in place; grammar engine is post-announce.
- `tools` / `tool_choice` / `logprobs` / `top_logprobs` /
  `logit_bias` / `presence_penalty` / `frequency_penalty` / `n>1`
   — all return 501 by design.
- Admin endpoints / CLI overrides for session tunables
  (max_sessions=64, TTL=30 min, block_size=4) — fixed in v0.1.
- Multi-process / multi-worker uvicorn — `--workers > 1` and
  `--reload` are explicitly rejected at startup; lifespan slot is
  process-local.
- Persistent rate-limit / auth state — in-memory token-bucket and
  in-memory `AuthState`; restart resets both. Redis-backed shared
  state is post-announce.
- ~~Wiring `silica.core.logger.setup_logging` into the `silica serve`
  CLI so `--log-level` surfaces silica.* INFO logs (including the
  route's `prefix_hit_tokens` line) to stderr~~ — **closed at
  v1.7.34 as (h) follow-up #3**. The fix landed as a single
  `setup_logging(level=...)` call in `silica.server.cli._serve()`
  before `uvicorn.run`, with uvicorn's `trace` log level mapped
  to Python `DEBUG` (no Python `logging` analogue exists for
  `TRACE`). Two new pins in `tests/test_server_cli.py` assert
  that `setup_logging` is called *before* `uvicorn.run` (so the
  lifespan's first INFO line is not silently dropped) and that
  the `trace → DEBUG` mapping holds. The v1.7.33 R-h smoke
  factbundle is a frozen point-in-time record from before the
  fix and stays unchanged; future smoke runs will surface the
  route's `prefix_hit_tokens` INFO line through the wired
  silica.* handler. See `plans/PLAN.md` §13 v1.7.34 changelog
  entry for the full closure narrative.
