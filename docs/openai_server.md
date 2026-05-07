# OpenAI-compatible HTTP server

`silica serve` runs an OpenAI-compatible HTTP server fronting one
loaded model. The standard `openai` Python client (and any
HTTP-equivalent SDK) can drive it as if it were a hosted endpoint.
This page covers boot, configuration, routes, the silica
extension envelope, and the v0.1 limitations.

## Boot

```bash
pip install "silica-mlx[serve]"
silica serve --model Qwen/Qwen3.5-0.8B
```

The server binds `127.0.0.1:8000` by default. Override host / port
with `--host` / `--port`. `silica serve --help` lists every flag.

The server is **single-process** in v0.1: `--workers` must remain
`1` and `--reload` is unsupported. The FastAPI lifespan reads a
module-level configuration slot that does not propagate to
`uvicorn`'s reload / multi-worker subprocesses; both modes are
rejected at startup with an explicit message rather than silently
failing in the worker. Multi-customer scheduler routing is a
post-announce follow-on.

A `/healthz` endpoint returns `200 {"status": "ok"}` once the
runtime has built; `503 {"error": ...}` while the lifespan is
still loading or after shutdown. Liveness probes can hit it
without authentication.

## Routes

| Path | Method | Status |
| --- | --- | --- |
| `/v1/chat/completions` | POST | non-streaming + SSE streaming |
| `/v1/completions` | POST | non-streaming text completion |
| `/v1/models` | GET | lists the single loaded model |
| `/healthz` | GET | strict liveness probe |

Each accepts the standard OpenAI request schema. SSE streaming
follows OpenAI's published wire format (`data: {chunk}\n\n` frames
terminated by `data: [DONE]\n\n`).

### Example: openai Python client

```python
from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:8000/v1",
    api_key="not-used",  # see "Authentication" below
)

response = client.chat.completions.create(
    model="Qwen/Qwen3.5-0.8B",
    messages=[
        {"role": "system", "content": "You are concise."},
        {"role": "user", "content": "Capital of France?"},
    ],
)
print(response.choices[0].message.content)
```

Streaming:

```python
stream = client.chat.completions.create(
    model="Qwen/Qwen3.5-0.8B",
    messages=[{"role": "user", "content": "Tell a joke."}],
    stream=True,
)
for chunk in stream:
    delta = chunk.choices[0].delta.content
    if delta:
        print(delta, end="", flush=True)
```

## Authentication

A single shared bearer token. Two ways to set it:

```bash
# CLI flag (takes precedence):
silica serve --model Qwen/Qwen3.5-0.8B --api-key sk-local-dev

# or env var:
SILICA_API_KEY=sk-local-dev silica serve --model Qwen/Qwen3.5-0.8B
```

When configured, every request to `/v1/...` must carry
`Authorization: Bearer sk-local-dev`. `/healthz` is exempt so
load-balancer probes do not need credentials.

Failed auth returns the OpenAI 401 envelope:

```json
{"error": {"message": "missing Authorization header", "type": "invalid_request_error", "code": "invalid_api_key"}}
```

When neither flag nor env var is set, auth is **disabled** —
every request is accepted. This is the development default and
matches v0.1's local-single-user framing.

## Rate limiting

Per-key token-bucket. Configure via `--rate-limit-rpm`:

```bash
silica serve --model Qwen/Qwen3.5-0.8B --rate-limit-rpm 60
```

The bucket capacity equals the configured RPM, so a fully-refilled
bucket allows up to one minute's burst. The key is the
`Authorization` header value (when auth is enabled) or the client
IP (`X-Forwarded-For` first hop, else `request.client.host`).

Rate limit runs **before** auth — a flood of unauthenticated
requests still decrements the offending IP's bucket, instead of
letting an attacker spam cheap 401s.

429 envelope:

```json
{"error": {"message": "rate limit exceeded (60 requests per minute)", "type": "rate_limit_error", "code": "rate_limit_exceeded"}}
```

The response carries a `Retry-After` header (seconds) so SDK
exponential-backoff implementations can honour it.

Disabled by default. `--rate-limit-rpm 0` (or omitting the flag)
keeps the limiter off.

## Persistent sessions: `X-Silica-Session-ID`

Cross-request prefix-cache reuse — the load-bearing M-9 acceptance
deliverable. A session id selects a persistent `ChatSession` whose
prefix cache survives across requests; the second turn's prompt
hits cached blocks for the system + earlier-history prefix.

Set the id via either:

- HTTP header: `X-Silica-Session-ID: abc-123` (preferred).
- Body extension: `extra_body={"extension": {"session_id": "abc-123"}}`
  (for SDKs that cannot set custom headers).

Header takes precedence when both are set. The OpenAI `user` field
is intentionally **not** consulted as a session id — its spec
semantics are abuse-monitoring identifier, not conversation
continuity.

```python
client.chat.completions.create(
    model="Qwen/Qwen3.5-0.8B",
    messages=[
        {"role": "system", "content": "You are a coding assistant."},
        {"role": "user", "content": "Write a quicksort in Rust."},
    ],
    extra_headers={"X-Silica-Session-ID": "session-42"},
)

# Subsequent turn — same system prompt + history prefix, same
# session id → prefix cache lights up.
client.chat.completions.create(
    model="Qwen/Qwen3.5-0.8B",
    messages=[
        {"role": "system", "content": "You are a coding assistant."},
        {"role": "user", "content": "Write a quicksort in Rust."},
        {"role": "assistant", "content": "<previous response>"},
        {"role": "user", "content": "Now in Python."},
    ],
    extra_headers={"X-Silica-Session-ID": "session-42"},
)
```

The OpenAI client typically sends the full conversation each
request; the server replaces the session's message log per call so
client-side history stays authoritative. The prefix cache is the
state that survives.

### Limits

- LRU cap (`max_sessions=64`) and idle TTL (`30 min`) are fixed in
  v0.1; admin endpoints / CLI overrides are post-announce.
- Sessions are in-memory only — restart loses every persistent
  session.
- One session per id, not per user. Different ids on the same
  process get isolated `ChatSession`s; cross-session shared
  system-prompt reuse is **not** in scope.
- Adapters whose attention pattern includes
  `AttentionKind.SLIDING` (Gemma4 31B today) **cannot** host
  persistent sessions in v0.1: the `ContinuousBatcher` rejects the
  combination of `RadixPrefixCache` + sliding-window admission. A
  request naming `session_id` against a sliding-bearing model
  returns 501 with an actionable message; drop the header to use
  the fresh-per-call path.

## Extension envelope

silica-specific request fields ride under `extra_body.extension`:

| Field | v0.1 status | Effect |
| --- | --- | --- |
| `session_id` | **honoured** | Session selector (header form preferred) |
| `thinking_mode` | 501 | Reserved for Qwen3 / Qwen3.5 reasoning toggle |
| `continue_truncated` | 501 | Reserved for `finish_reason=length` continuation |

Extension fields outside this set raise `422` at schema parse time
(`extra="forbid"` on the envelope) — typo-protection for
silica-owned wire shape.

## Error envelope

Every error response uses OpenAI's wire shape:

```json
{
  "error": {
    "message": "<human-readable>",
    "type": "<taxonomy>",
    "code": "<optional code>"
  }
}
```

`type` mapping by status:

| HTTP | `error.type` |
| --- | --- |
| 400 | `invalid_request_error` |
| 401 | `invalid_request_error` |
| 404 | `invalid_request_error` |
| 422 | `invalid_request_error` |
| 429 | `rate_limit_error` |
| 500 | `server_error` |
| 501 | `invalid_request_error` |
| 503 | `server_error` |

This matches what the openai Python SDK switches on, so its
exception hierarchy lifts cleanly.

## Structured output

`response_format.type` of `text` is the default; `json_object` /
`json_schema` parse but are rejected as 501 in v0.1. The slot is
reserved so a future grammar engine can land here without a
schema change. Requested schemas are logged at INFO so the
implementer can analyse what callers actually need.

## Observability

- Request-shape log at INFO: model, stream, history turns,
  user-text length, max_tokens, session_id.
- Reply log at INFO: finish_reason, prompt/output tokens,
  session_id, prefix_hit_tokens.
- Auth and rate-limit denials log at INFO with a redacted key
  prefix.
- Engine internals (TTFT, decode tok/s, KV residency) populate
  `TurnMetrics`; surfacing them on the wire is post-announce
  ((h) follow-up).

## Limitations summary

- **Single process, single model** — no multi-model routing, no
  --workers > 1.
- **In-memory sessions** — restart loses state.
- **No tool / function calling** — `tools`, `tool_choice`,
  `logprobs`, `top_logprobs`, `logit_bias`, `presence_penalty`,
  `frequency_penalty`, `n>1` all return 501.
- **No structured-output execution** — `response_format` slot is
  reserved but not implemented.
- **No multimodal input** — non-string `content` returns 501.
- **Rate-limit and auth are in-memory** — no shared backend, no
  persistent counters.

These deliberately match the v0.1 single-user / local-developer
framing in `plans/P8_OPENING.md` §6.1.2. Each constraint has a
named follow-on phase if a real workload demands lifting it.
