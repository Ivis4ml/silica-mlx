# Silica-MLX API Reference

Per-module reference for every public class, function, and method in the
`silica` package. Organized by subpackage. For the project overview and
roadmap see [`PLAN.md`](../plans/PLAN.md); for quickstart see [`../README.md`](../README.md).

Symbols are marked:
- **public** — re-exported from the subpackage's `__init__` or the top-level `silica`.
- **internal** — load-bearing but not re-exported; listed because tests and
  downstream units rely on them.
- **protocol** — a `typing.Protocol` (one of I-1 .. I-5 if capitalised; a
  local protocol otherwise).
- **stub** — P-0 conforming implementation that ships the interface
  without real behaviour.

---

## Table of contents

1. [`silica`](#silica-top-level)
2. [`silica.core`](#silicacore)
    - [`silica.core.events`](#silicacoreevents)
    - [`silica.core.logger`](#silicacorelogger)
    - [`silica.core.profiler`](#silicacoreprofiler)
    - [`silica.core.request`](#silicacorerequest)
    - [`silica.core.sampler`](#silicacoresampler)
    - [`silica.core.sampling`](#silicacoresampling)
3. [`silica.kvcache`](#silicakvcache)
    - [`silica.kvcache.manager`](#silicakvcachemanager)
    - [`silica.kvcache.simple`](#silicakvcachesimple)
    - [`silica.kvcache.paged`](#silicakvcachepaged)
    - [`silica.kvcache.codec`](#silicakvcachecodec)
    - [`silica.kvcache.prefix`](#silicakvcacheprefix)
    - [`silica.kvcache.store`](#silicakvcachestore)
4. [`silica.scheduler`](#silicascheduler)
    - [`silica.scheduler.budget`](#silicaschedulerbudget)
    - [`silica.scheduler.batcher`](#silicaschedulerbatcher)
    - [`silica.scheduler.seed_kv`](#silicaschedulerseed_kv)
5. [`silica.models`](#silicamodels)
    - [`silica.models.adapter`](#silicamodelsadapter)
    - [`silica.models.capabilities`](#silicamodelscapabilities)
    - [`silica.models.qwen3`](#silicamodelsqwen3)
    - [`silica.models.qwen3_5`](#silicamodelsqwen3_5)
    - [`silica.models.qwen3_5_moe`](#silicamodelsqwen3_5_moe)
    - [`silica.models.gemma4`](#silicamodelsgemma4)
    - [`silica.models.gemma4_moe`](#silicamodelsgemma4_moe)
    - [`silica.models.recurrent`](#silicamodelsrecurrent)
    - [`silica.models.pre_norm_capture`](#silicamodelspre_norm_capture)
    - [`silica.models.factory`](#silicamodelsfactory)
6. [`silica.weights`](#silicaweights)
    - [`silica.weights.provider`](#silicaweightsprovider)
    - [`silica.weights.resident`](#silicaweightsresident)
7. [`silica.vq`](#silicavq)
    - [`silica.vq.block_tq`](#silicavqblock_tq)
    - [`silica.vq.turboquant`](#silicavqturboquant)
    - [`silica.vq.rabitq`](#silicavqrabitq)
    - [`silica.vq._calibration`](#silicavq_calibration)
8. [`silica.mlx`](#silicamlx)
9. [`silica.speculative`](#silicaspeculative)
10. [`silica.engine`](#silicaengine)
11. [`silica.bench`](#silicabench)
    - [`silica.bench.scenario`](#silicabenchscenario)
    - [`silica.bench.scenarios`](#silicabenchscenarios)
    - [`silica.bench.runner`](#silicabenchrunner)
    - [`silica.bench.codec_registry`](#silicabenchcodec_registry)
    - [`silica.bench.oracles`](#silicabenchoracles)
    - [`silica.bench.ppl_oracle`](#silicabenchppl_oracle)
    - [`silica.bench.report`](#silicabenchreport)
    - [`silica.bench.wikitext`](#silicabenchwikitext)
    - [`silica.bench.vqbench_baseline`](#silicabenchvqbench_baseline)
12. [`silica.chat`](#silicachat)
    - [`silica.chat.session`](#silicachatsession)
    - [`silica.chat.cli`](#silicachatcli)
13. [`silica.server`](#silicaserver)

---

## `silica` top-level

Re-exports the Engine for the one-import user path.

- **`Engine`** — see [`silica.engine.Engine`](#engine).
- **`__version__`** — string, current package version.

```python
from silica import Engine
```

---

## `silica.core`

Data types, logging, profiling. No business logic. `silica.core.*` never
imports from `silica.scheduler`, `silica.kvcache`, or other downstream
packages — this is the acyclic root.

Re-exports (via `silica/core/__init__.py`):

| Name | Origin | Kind |
| --- | --- | --- |
| `BatchEvent`, `EventKind` | events | public |
| `get_logger`, `setup_logging` | logger | public |
| `MetricsRegistry`, `TimingRecord`, `UnifiedMetrics`, `registry`, `time_block` | profiler | public |
| `InvalidTransition`, `Request`, `RequestState`, `RequestStatus` | request | public |
| `LogitProcessor`, `Sampler` | sampler | public/protocol |
| `SamplingParams` | sampling | public |

### `silica.core.events`

#### `BatchEvent` *(public, frozen dataclass)*

One element in the stream produced by `Engine.generate_batch` and
`ContinuousBatcher.step`. Carries a tagged-union shape — `kind` plus optional
`token_id` / `finish_reason`.

Fields:
- **`req_index: int`** — zero-based index into the original `prompts` list.
- **`kind: Literal["token", "done", "aborted"]`** — event type.
- **`token_id: int | None`** — set when `kind == "token"`, else `None`.
- **`finish_reason: str | None`** — set when `kind` is terminal. Conventional
  values: `"stop_token"`, `"max_tokens"`, `"budget-exhausted"`, `"error"`.

Constructors:

- **`BatchEvent.token(req_index: int, token_id: int) -> BatchEvent`**
  Emit when a request samples a new token.
- **`BatchEvent.done(req_index: int, reason: str) -> BatchEvent`**
  Emit once per request on normal termination.
- **`BatchEvent.aborted(req_index: int, reason: str) -> BatchEvent`**
  Emit once per request on error / budget exhaustion.

Terminal events are the last event for a given `req_index`; callers can
safely retire per-request state on `done` / `aborted`.

#### `EventKind`

`Literal["token", "done", "aborted"]` alias. Widening the literal is a
backward-compatible way to introduce new event kinds (e.g. `"preempted"` if
ever surfaced externally).

### `silica.core.logger`

Stdlib `logging` wrapper scoped to the `silica.*` namespace, with a coloured
formatter. Does not hijack the root logger.

#### `setup_logging(level="INFO", use_colors=None, stream=None) -> None`

Install the silica coloured handler. Idempotent — removes any previously
installed silica handler before adding a new one.

- **`level: str | int`** — e.g. `"INFO"`, `logging.DEBUG`.
- **`use_colors: bool | None`** — `None` auto-detects (TTY + no `NO_COLOR`
  env var). `False` disables.
- **`stream: IO[str] | None`** — defaults to `sys.stderr`.

#### `get_logger(name: str) -> logging.Logger`

Return a logger under the silica namespace. Pass `__name__`; external names
are prefixed with `silica.` (`foo.bar` → `silica.foo.bar`).

### `silica.core.profiler`

Per-engine metrics collection. P-0 Strategy: `@contextmanager` + registry
+ a unified schema.

#### `UnifiedMetrics` *(public, dataclass)*

Floor schema mandated by PLAN.md P-0 Acceptance. Every field defaults to
`None` to distinguish "not measured" from "measured zero".

Fields: `ttft_ms`, `prefill_tok_s`, `decode_tok_s`, `resident_mb`,
`logical_kv_bytes`, `extra: dict[str, float | int]`.

#### `TimingRecord` *(public, dataclass)*

One timing block. Fields: `name`, `started_at` (perf_counter), `elapsed_s`,
`tags` (free-form payload, e.g. `{"num_tokens": 128}`).

#### `MetricsRegistry` *(public class)*

Collects `TimingRecord`s and scalar metrics. Unknown metric names fall into
`UnifiedMetrics.extra`.

Methods:
- **`record_timing(record: TimingRecord) -> None`**
- **`set_metric(name: str, value: float | int) -> None`** — stores to typed
  field if `name` is a unified floor field, otherwise to `extra`.
- **`get_metric(name: str) -> float | int | None`** — unified name lookup
  falls back to `extra`.
- **`timings(name: str | None = None) -> list[TimingRecord]`** — filter by
  name if provided; returns a defensive copy.
- **`snapshot() -> UnifiedMetrics`** — detached copy.
- **`reset() -> None`** — clear all state.

#### `registry() -> MetricsRegistry`

Return the process-global registry. Q-005 may move this to per-engine.

#### `time_block(name: str, *, record: bool = True) -> Iterator[TimingRecord]` *(context manager)*

Time a code block with `time.perf_counter`. On exit, `record.elapsed_s` is
populated and (if `record=True`) recorded to the global registry.

```python
with time_block("prefill") as t:
    t.tags["num_tokens"] = 128
    run_forward(...)
# t.elapsed_s is now populated
```

### `silica.core.request`

Request + lifecycle FSM for the engine. The FSM is **pure** — transitions
only mutate `status` + history + terminal reason; they do not release KV
blocks, touch the prefix cache, or zero batch rows. Side effects live with
observers (`ContinuousBatcher`, `PagedKVCache`). This keeps P-7 speculative
rollback tractable.

#### `RequestStatus` *(enum)*

Values: `WAITING`, `PREFILL`, `DECODE`, `PREEMPTED`, `DONE`, `ABORTED`.
Terminal: `DONE`, `ABORTED`. Side state: `PREEMPTED` (reachable from
`PREFILL` / `DECODE` under budget pressure).

Allow-list summary:
- `WAITING → PREFILL | ABORTED`
- `PREFILL → DECODE | PREEMPTED | DONE | ABORTED`
  (`PREFILL → DONE` is legal — first sampled token may be a stop token)
- `DECODE → DONE | PREEMPTED | ABORTED`
- `PREEMPTED → WAITING | ABORTED`
- `DONE`, `ABORTED` — terminal, no outgoing edges.

#### `InvalidTransition(ValueError)` *(exception)*

Raised when `RequestState.transition` is called with a target outside the
allow-list.

#### `Request` *(public, frozen dataclass)*

Immutable input. Fields: `prompt: str`, `sampling_params: SamplingParams`,
`request_id: str` (auto-generated hex16 UUID), `token_ids: tuple[int, ...]`
(populated after tokenization).

#### `RequestState` *(public, mutable dataclass)*

Per-request state owned by the scheduler.

Fields:
- **`request: Request`** — immutable input.
- **`status: RequestStatus`** — current FSM state.
- **`num_computed_tokens: int`** / **`num_output_tokens: int`** — counters.
- **`output_token_ids: list[int]`** — generated tokens so far.
- **`arrival_time: float`** / **`first_token_time: float | None`**.
- **`prefix_hit_tokens: int`** — populated on admission; lets the batcher
  skip that portion of prefill.
- **`state_delta_snapshot: Any | None`** — adapter-owned recurrent-state
  snapshot retained across `PREEMPTED` (D-015).
- **`finish_reason: str | None`** — set on terminal transition.

Properties:
- **`request_id: str`** — delegates to `request.request_id`.
- **`is_terminal: bool`** — true iff status is `DONE` or `ABORTED`.
- **`is_finished: bool`** — alias for `is_terminal` (pre-P-2 API).
- **`history: list[tuple[RequestStatus, str]]`** — chronological transition
  log (defensive copy).

Methods:
- **`transition(to: RequestStatus, *, reason: str) -> RequestStatus`**
  Validate and apply. Returns the previous status. Raises
  `InvalidTransition` if `to` is not in the allow-list. For terminal
  transitions, sets `finish_reason` to `reason`.

### `silica.core.sampler`

Logits → token. Per D-013 the processor chain is fixed:
`temperature → repetition penalty → top-k → top-p → sample`.
Greedy fast path (`temperature == 0`) bypasses the entire chain.

#### `LogitProcessor` *(protocol)*

Shape of one logit-transform stage. Called with
`(logits, token_history, params)`; must be pure (no mutation of `logits`).

```python
def __call__(
    self,
    logits: mx.array,
    token_history: mx.array,
    params: SamplingParams,
) -> mx.array: ...
```

Not one of the I-1..I-5 frozen interfaces — it is a local convenience
type-hint (see D-013).

#### Built-in processors (module-level functions)

- **`apply_temperature(logits, _history, params) -> mx.array`** — divides
  logits by `params.temperature` unless `temperature == 1.0` (identity).
- **`apply_repetition_penalty(logits, token_history, params) -> mx.array`**
  — divides / multiplies logits at already-emitted token positions by
  `params.repetition_penalty` (positive logits divided; negative multiplied).
  Identity when `repetition_penalty == 1.0` or history empty.
- **`apply_top_k(logits, _history, params) -> mx.array`** — masks all
  logits below the k-th largest with `-inf`. Identity when `top_k is None`.
- **`apply_top_p(logits, _history, params) -> mx.array`** — nucleus
  sampling via inverse-permutation: sort descending, cumulative-sum probs,
  mask positions past the first crossing of `p`. Identity when `top_p is
  None`. The highest-prob position is always kept.

#### `Sampler` *(public class)*

Applies the fixed chain and samples.

- **`PROCESSORS: tuple[_ProcessorFn, ...]`** — class-level; override via
  subclassing for v0.2 experiments (e.g. grammar-constrained decoding).
- **`sample(logits, token_history, params, key=None) -> mx.array`**
  Returns a 0-d `mx.array` with the sampled token id. Shortcuts to
  `mx.argmax` when `params.is_greedy`. Accepts `token_history` as a
  `Sequence[int]` or an existing `mx.array`; optionally takes a
  categorical RNG key.

### `silica.core.sampling`

#### `SamplingParams` *(public, frozen pydantic model, `extra="forbid"`)*

Immutable per-request sampling config. Processors skip when parameters are
identity-valued.

Fields:
- **`temperature: float = 1.0`** (ge 0.0). `0.0` activates greedy.
- **`top_k: int | None = None`** (ge 1).
- **`top_p: float | None = None`** (gt 0.0, le 1.0).
- **`repetition_penalty: float = 1.0`** (gt 0.0).
- **`max_tokens: int = 256`** (ge 1). Hard upper bound on generated tokens.
- **`stop: tuple[str, ...] = ()`** — string stop sequences (P-2 not yet
  wired in the engine; planned with incremental decoding).
- **`stop_token_ids: tuple[int, ...] = ()`** — yielded-then-stopped. Populate
  with `tokenizer.eos_token_ids` unless `ignore_eos=True`.
- **`ignore_eos: bool = False`**.
- **`seed: int | None = None`**.

Property:
- **`is_greedy: bool`** — `temperature <= 0.0`.

Pass through `model_copy(update={"max_tokens": ...})` to derive a replay
params object (used by the preempt path — original worst-case bytes are
preserved).

---

## `silica.kvcache`

I-2 `KVManager` Protocol, I-3 `KVCodec` Protocol, paged bookkeeping, radix
prefix cache, and the prefix block store abstractions.

### `silica.kvcache.manager`

#### `KVHandle` *(public, frozen dataclass)*

Binding of `req_id` so adapters reach KV without holding block pointers.
Single field: `req_id: str`. Frozen to prevent re-pointing.

#### `BlockList` *(public, dataclass)*

Ordered block ids for one request. Field: `block_ids: tuple[int, ...]`.
Implements `__len__`.

#### `PrefixHit` *(public, dataclass)*

Result of `KVManager.get_computed_blocks`. Fields:
- `block_ids: tuple[int, ...]` — reusable pinned blocks.
- `num_hit_tokens: int` — covered prompt-token count.

Zero-token / empty-block-ids == cache miss.

#### `MemoryBudget` *(public, dataclass)*

Scheduler-facing snapshot:
- `logical_bytes: int` — fp16-baseline-equivalent.
- `resident_bytes: int` — D-012 canonical physical bytes.
- `headroom_bytes: int` — `target − resident`. Negative = over-budget.

#### `KVManager` *(protocol, I-2)*

Block-allocated KV cache + prefix cache + memory budget. Class attribute
`block_size: int`. Methods:

- **`reserve_for_prefill(req_id, token_ids) -> BlockList`** — allocate blocks
  for the initial prompt. Returns the `BlockList` of allocated block ids.
- **`append_slot(req_id, n) -> BlockList`** — incrementally extend during
  decode. Returns newly-allocated ids (empty if current last block has room).
- **`commit(req_id, n_accepted) -> None`** — P-7 speculative accept
  (bookkeeping no-op in P-2).
- **`rollback(req_id, n_reject) -> None`** — P-7 speculative reject. P-1's
  `SimpleKVCache` trims via mlx-lm; P-2's `PagedKVCache` is a no-op.
- **`free(req_id) -> None`** — release blocks on terminal. Idempotent on
  unknown req_id.
- **`get_computed_blocks(token_ids) -> PrefixHit`** — prefix-cache lookup.
  Returns empty `PrefixHit` in P-2 (prefix cache is on the batcher, not the
  kv manager, under Option B).
- **`available_blocks() -> int`** — free-pool size. Non-paged managers
  report 0.
- **`budget() -> MemoryBudget`** — current snapshot.

#### `NullKVManager` *(public, stub)*

Zero-sized no-op. All operations return empty / zero. Used only for P-0
interface conformance. Constructor: `NullKVManager(block_size=16)`.

### `silica.kvcache.simple`

#### `SimpleKVCache` *(public class)*

Single-request I-2 `KVManager` wrapping mlx-lm's per-layer cache list. P-1
path. **Single-request** — claiming a second `req_id` raises `ValueError`.

Class attribute: `block_size: int = _MLX_KVCACHE_GROWTH_CHUNK` (256 —
mlx-lm's auto-growth chunk, not a true paged block size).

Constructors:
- **`SimpleKVCache(cache_list: list[Any])`** — wrap an mlx-lm cache list.
- **`SimpleKVCache.from_model(model) -> SimpleKVCache`** *(classmethod)* —
  delegates to `mlx_lm.models.cache.make_prompt_cache(model)`, which calls
  `model.make_cache()` if defined (Qwen3.5 returns 18+6 hybrid list) and
  otherwise `[KVCache()] * num_layers`.

Extension beyond the I-2 Protocol:
- **`cache_list(req_id: str) -> list[Any]`** — hand the mlx-lm per-layer
  list to the P-1 adapter's `model(tokens, cache=...)` call. Requires
  `req_id` to be the owner.

I-2 Protocol methods: `reserve_for_prefill`, `append_slot`, `commit`,
`rollback` (delegates to `mlx_cache.trim_prompt_cache` when
`can_trim_prompt_cache`), `free`, `get_computed_blocks` (always empty),
`available_blocks` (always 0), `budget` (aggregates per-cache `nbytes`).

### `silica.kvcache.paged`

#### `RowState` *(public, str enum)*

Physical row lifecycle: `FREE → RESERVED → ACTIVE → FREE`. Rows do not
return to `RESERVED`; a new admission claims a different `FREE` row.

#### `PagedKVCache` *(public class)*

P-2 I-2 `KVManager` — paged bookkeeping over a batched K/V store. Tracks
per batch session:
- **Layer A (physical slot table)**: `slot_table: req_id → int` row index
  fixed for lifetime. Row lifecycle in `row_state: list[RowState]`.
- **Layer B (logical page table)**: `page_table: req_id → list[block_id]`
  + shared `refcount` + `free_blocks` list.

Physical `BatchKVCache` construction happens in `ContinuousBatcher`; this
class is bookkeeping only. `get_computed_blocks` returns an empty
`PrefixHit` — prefix lookup in P-2 happens on the `RadixPrefixCache`
directly, not through kv.

Constructor parameters (all kwargs):
- `num_layers`, `max_batch_size`, `n_kv_heads`, `head_dim`, `num_blocks`
- `block_size=16`, `dtype_bytes=2`

Extension methods (used by `ContinuousBatcher` and `RadixPrefixCache`'s
`PagedPrefixBlockStore` adapter):
- **`slot_of(req_id) -> int`** — row index; raises `KeyError` if
  unreserved.
- **`row_states() -> list[RowState]`** — defensive copy.
- **`mark_active(req_id) -> None`** — `RESERVED → ACTIVE` after prefill.
- **`incref(block_id) -> None`** — add a retention ref. Raises `KeyError`
  if block is not currently held.
- **`decref(block_id) -> None`** — drop a retention ref; returns to
  `free_blocks` when count reaches zero.
- **`num_tokens(req_id) -> int`** — total prompt + decoded tokens.

I-2 Protocol methods: `reserve_for_prefill` (allocates slot + blocks, sets
refcount=1 per block), `append_slot` (extends page table; raises on
insufficient free blocks), `commit` / `rollback` (P-7 no-op),
`free` (releases slot + decrefs blocks; idempotent on unknown req_id),
`get_computed_blocks` (empty), `available_blocks`, `budget`.

Budget approximation: `resident = n_claimed_blocks × bytes_per_block`,
counting claimed not used. `headroom_bytes == 0` since the target cap is
tracked by `MemoryBudgeter`, not here.

### `silica.kvcache.codec`

P-5-A.0.4 rewrite. The interface is now **side-level** — one
`VectorCodec[P]` instance encodes either K or V (not both at once),
and the K and V codecs are held independently on
`SyntheticPrefixBlockStore.k_codec` / `.v_codec`. The legacy block-level
`KVCodec` Protocol and the `CodedBlock` wrapper are removed.

#### Payload hierarchy *(public dataclasses)*

`CodedPayload` is the root. Concrete subclasses add per-codec
`mx.array` fields and call `_verify_resident_bytes` from
`__post_init__` so D-012 honesty (resident_bytes equals the sum of
`.nbytes` across `mx.array` fields) is enforced at construction.

- **`CodedPayload`** — base. Field: `resident_bytes: int`.
- **`RawFp16Payload(CodedPayload)`** — Identity-codec payload.
  Field: `t: mx.array`. Shape matches the input tensor (typically
  `(1, n_kv_heads, kv_block_size, head_dim)` for one side of one
  detached block).
- **`BlockTQPayload(CodedPayload)`** — BlockTurboQuantMSE payload.
  Fields: `packed_indices: mx.array (uint8)`, `scales: mx.array
  (fp16)`. Shapes per `(n_vectors = n_kv_heads × kv_block_size, d =
  head_dim, B = vq_block_size, b = num_bits)`: indices
  `(n_vectors, ceil(d × b / 8))`, scales `(n_vectors, d // B)`.
- **`RaBitQPayload(CodedPayload)`** — RaBitQ-1 1-bit payload (P-5-B
  core shape). Fields: `packed_indices: mx.array (uint8, (n_vectors,
  d/8))`, `norm_o: mx.array (fp16, (n_vectors,))`, `ip_coeff:
  mx.array (fp16, (n_vectors,))`. Centroid is held on the codec
  instance, not the payload.
- **`ExtRaBitQPayload(RaBitQPayload)`** — ExtRaBitQ multi-bit payload
  (P-5-B.2). Adds a per-vector fp16 `scale: mx.array (n_vectors,)`
  used as a dequantization factor. The vqbench reference's `offset`
  field is dropped intentionally — opening §5.4 Amendment.

#### `VectorCodec` *(protocol, I-3)*

Side-level, `@runtime_checkable Protocol[P]` parametrised on a
`CodedPayload` subclass. Class attributes: `block_size: int`,
`dtype: mx.Dtype`. Methods:

- **`encode_tensor(x: mx.array) -> P`** — encode one fp16 tensor
  (one side of one block of one layer) into the codec-specific
  payload.
- **`decode_tensor(payload: P) -> mx.array`** — reconstruct an fp16
  tensor (D-003: output is fp16; no compressed-domain attention in
  v0.1).
- **`logical_bytes(num_tokens: int) -> int`** — fp16-baseline-equivalent
  bytes for `num_tokens` tokens of this side.
- **`resident_bytes(num_blocks: int) -> int`** — physical bytes held
  for `num_blocks` blocks of this side (D-012 canonical).

Pair-level dispatch lives in
[`SyntheticPrefixBlockStore`](#silicakvcachestore), which holds the K
and V codec separately and encodes / decodes independently per side
— resolution of Q-008 per the side-level Protocol shipped at P-5-A.0.

#### `IdentityCodec` *(public, baseline)*

Pass-through `VectorCodec[RawFp16Payload]`. Encoded payload wraps the
input tensor unchanged; `resident == logical`. Stays in tree after P-5
as the fp16 baseline on a `--kv-codec` switch.

Constructor: `IdentityCodec(*, block_size: int, n_kv_heads: int,
head_dim: int, dtype=mx.float16)`.

Implements all four `VectorCodec` methods trivially.

### `silica.kvcache.prefix`

#### `RadixPrefixCache` *(public class)*

Block-granular trie keyed by chunks of `block_size` tokens. Inspired by
mini-sglang. Under P-2 Option B (copy-on-admit), a hit means: on
admission, the batcher copies the source blocks' K/V into the new
request's fresh batch row, skipping that prefix's forward compute. Source
blocks survive their originating request via refcount pinning in the
`PrefixBlockStore`.

Constructor: `RadixPrefixCache(*, block_size: int, store: PrefixBlockStore)`.
Both must agree on `block_size`. Counter: `hits: int`.

Read methods:
- **`peek(tokens) -> PrefixHit`** — side-effect-free walk. No `retain_hit`,
  no LRU touch, no `self.hits` increment. Use during admission planning.
- **`lookup(tokens) -> PrefixHit`** — retained walk. Retains each hit
  block via `store.retain_hit`, advances LRU, increments `self.hits`.
  Must pair with a later `release(block_ids)`.

Write / insert methods:
- **`insert(tokens, block_ids) -> None`** — add caller-allocated blocks
  (Paged-backend path). Idempotent on already-covered prefixes.
  Each new node calls `store.retain_source`.
- **`insert_detached(tokens, detached_blocks) -> tuple[int, ...]`** —
  add tokens with attached K/V slices (Synthetic-backend path, used by
  16c.2's admission). `detached_blocks` indexed `[block_idx][layer_idx]` →
  `(K, V)`. Each new node: `allocate_id` → `retain_source` →
  `register_detached`. Returns ids actually newly inserted.

Release + fetch:
- **`release(block_ids) -> None`** — paired with a prior `lookup`.
  Delegates to `store.release_hit`. Mismatched release raises `KeyError`.
- **`fetch_detached_blocks(block_ids) -> list[Sequence[tuple[mx.array, mx.array]]]`**
  — per-block detached K/V for the given hit block ids, shape
  `[block_idx][layer_idx] → (K, V)`, ready for
  `build_seeded_batch_kv`.

Eviction:
- **`evict_until(n_blocks) -> int`** — evict LRU leaf nodes with zero live
  hits. Returns the actual number of blocks freed (may be fewer than
  requested). Order: `release_detached` (if any) then `release_source`.

Debug:
- **`node_count() -> int`** — total non-root nodes.
- **`live_hits(block_id) -> int`** — delegates to `store.hit_refs`.
- **`block_size: int`** property.

### `silica.kvcache.store`

Backend seam between `RadixPrefixCache` and concrete storage (paged kv or
synthetic). Two refcount dimensions — **source** (radix-node retention,
survives the owning request) and **hit** (live lookup, short-lived). Plus
optional detached K/V for the synthetic path.

#### `PrefixBlockStore` *(protocol)*

Class attribute: `block_size: int`. Methods:

- **`allocate_id() -> int`** — fresh block id.
- **`retain_source(block_id) -> None`** / **`release_source(block_id)`** —
  source-ref pair. Final `release_source` fails if `hit_refs > 0` or
  detached K/V still registered (invariants L-2 ⊆ L-1, L-3 ⊆ L-1).
- **`retain_hit(block_id) -> None`** / **`release_hit(block_id)`** — hit-ref
  pair. `retain_hit` requires existing source ref; `release_hit` with no
  outstanding hit raises `KeyError`.
- **`hit_refs(block_id) -> int`** — live-hit count.
- **`has_detached(block_id) -> bool`**.
- **`register_detached(block_id, per_layer_kv)`** — attach per-layer `(K, V)`
  slices; requires existing source ref; duplicate registration raises.
- **`fetch_detached(block_id) -> Sequence[tuple[mx.array, mx.array]]`** —
  raises `KeyError` if none registered.
- **`release_detached(block_id) -> None`** — must precede paired final
  `release_source` during eviction.

#### `SyntheticPrefixBlockStore` *(public class)*

In-memory id-allocator + K/V dict. Owns its own counter; ids are
monotonic, never reused — simplifies reasoning about insert/evict churn.

Constructor: `SyntheticPrefixBlockStore(*, block_size: int)`.

Implements every Protocol method plus debug helpers: `source_refs`,
`live_block_ids`.

#### `PagedPrefixBlockStore` *(public class)*

Wraps a `PagedKVCache`, preserves its existing refcount + evict behaviour.
Block ids come from the caller (`PagedKVCache.reserve_for_prefill` /
`append_slot`); `allocate_id` raises — use `RadixPrefixCache.insert(tokens,
block_ids)`. Detached K/V is not modelled here (future paged-attention
kernel track owns that model); `register_detached` / `fetch_detached` /
`release_detached` raise `NotImplementedError`; `has_detached` returns
`False` unconditionally.

Constructor: `PagedPrefixBlockStore(kv: PagedKVCache)`.

---

## `silica.scheduler`

P-2 admission + continuous batching policy. Re-exports (via
`silica/scheduler/__init__.py`):

| Name | Origin | Kind |
| --- | --- | --- |
| `ContinuousBatcher` | batcher | public |
| `MemoryBudgeter`, `AdmissionDecision`, `AdmitDecision`, `AdmitAfterEvictDecision`, `AdmitAfterPreemptDecision`, `RejectDecision` | budget | public |

### `silica.scheduler.budget`

Admission + preemption **policy** layer. Pure decision; mutation happens
in `ContinuousBatcher`. Separating decide / apply keeps this unit testable
without wiring up real caches.

Two accountings (PLAN.md §P-2 / P2_OPENING.md):
- **resident_bytes** — D-012 canonical; reference for what the kv backend
  holds. Not read at decision time.
- **reserved_bytes** — admission-time upper bound. Per admitted request:
  `(n_prompt + max_tokens) × bytes_per_token`. Conflating with resident
  causes systematic over-admit.

#### Decision taxonomy *(frozen dataclasses)*

- **`AdmitDecision(reserved_delta: int)`** — fits as-is.
- **`AdmitAfterEvictDecision(n_blocks: int, reserved_delta: int)`** —
  fits after LRU-evicting `n_blocks` unpinned prefix blocks.
- **`AdmitAfterPreemptDecision(preempt_req_id: str, reserved_delta: int)`**
  — fits after preempting the FIFO-newest active request.
- **`RejectDecision(reason: str = "budget-exhausted")`** — no mechanism
  frees enough; caller aborts.

#### `AdmissionDecision` *(type alias)*

Tagged union of the four decision dataclasses above.

#### `MemoryBudgeter` *(public class)*

Budget-aware admission + preemption policy.

Constructor (kwargs):
- `prefix_cache: RadixPrefixCache` — consulted for evictable-block count.
- `weights_bytes: int` — static model weights + activations.
- `bytes_per_token: int` — `2 * num_layers * n_kv_heads * head_dim * dtype_bytes`.
- `block_size: int` — token count per radix block.
- `cap_bytes: int` — target resident cap.

Factory classmethod:
- **`MemoryBudgeter.for_adapter(adapter, *, prefix_cache, weights_bytes, cap_bytes)`**
  — derives `bytes_per_token` from `adapter.kv_layout()` and
  `block_size` from `prefix_cache.block_size`. Use this at callsites that
  would otherwise recompute the arithmetic. When
  `layout.bytes_per_token_total` is set (P-3-D4, used by adapters with
  heterogeneous per-layer KV shapes such as Gemma4), the factory uses
  that value verbatim; otherwise it falls back to
  `2 * num_layers * n_kv_heads * head_dim * dtype.size`.

Read-only properties / methods:
- **`cap_bytes`**, **`weights_bytes`**, **`bytes_per_token`**.
- **`reserved_bytes() -> int`** — current sum of active reservations.
- **`headroom_bytes() -> int`** — `cap − weights − reserved`; can be
  negative.
- **`worst_case_bytes(n_prompt, max_tokens) -> int`**.
- **`active_requests() -> list[str]`** — admitted, non-released req_ids
  in FIFO order (admission order).

Decision + lifecycle:
- **`admit(req_id, n_prompt, max_tokens) -> AdmissionDecision`** — pure.
  Policy steps:
  1. Fits as-is → `AdmitDecision`.
  2. Fits after LRU evicting unpinned prefix blocks →
     `AdmitAfterEvictDecision`.
  3. Fits after preempting FIFO-newest active → `AdmitAfterPreemptDecision`.
  4. None → `RejectDecision`.
- **`apply_admit(req_id, reserved_delta) -> None`** — record reservation
  AFTER caller has applied the decision's action. Raises if `req_id`
  already reserved. Invariant B-8: commit this BEFORE the next `admit()`
  observes `reserved_bytes`.
- **`release(req_id) -> None`** — on `DONE` / `ABORTED` / `PREEMPTED`.
  Idempotent on unknown req_id.

Private:
- **`_count_evictable_prefix_blocks()`** — walks
  `RadixPrefixCache._walk_non_root()` (slight encapsulation break,
  confined to this method).
- **`_pick_preempt_victim()`** — returns `self._admitted[-1]` (FIFO
  newest), or `None` if no active request.

### `silica.scheduler.batcher`

#### `ContinuousBatcher` *(public class)*

P-2 step loop: admission / reclaim / prefill / decode / sample / finalize.
Accepts only adapters whose every layer is `AttentionKind.GLOBAL`
(capability gate — Mamba / DeltaNet / sliding / hybrid families refuse
themselves here).

Constructor:
```python
ContinuousBatcher(
    adapter: ModelAdapter,
    *,
    sampler: Sampler | None = None,
    weight_provider: WeightProvider | None = None,
    max_batch_size: int = 1,
    prefix_cache: RadixPrefixCache | None = None,
    budgeter: MemoryBudgeter | None = None,
)
```

Installs `ResidentWeightProvider` by default. `max_batch_size` bounds
**active physical rows**, not queue length.

Public fields (read-only observability):
- **`forward_prompt_tokens: int`** — effective prompt tokens fed through
  prefill (excludes decode steps and left-pad slots).
- **`prefix_hits: int`** — admissions that used the hit path.
- **`aborts: int`**, **`evictions: int`**, **`preempts: int`** — 16d
  counters.

Public API:

- **`add_request(req_index, prompt_ids, params) -> None`**
  Enqueue. Two paths:
  - Pre-step (cohort not yet prepared): append directly to `self._rows`.
    Raises `RuntimeError` at `max_batch_size`.
  - Mid-run (cohort prepared): append to waiting queue (unbounded).
  Raises `ValueError` on empty `prompt_ids`.

- **`has_active() -> bool`** — literal: any non-terminal row. Used by
  internal phase decisions.
- **`has_work() -> bool`** — any active row, any terminal row pending
  reclaim, or any waiting-queue entry. This is what `Engine.generate_batch`
  loops on.

- **`step() -> list[BatchEvent]`** — advance one scheduler iteration.
  Phase order:
  1. **Reclaim** — drop terminal rows from `self._rows` + `self._batch_cache`
     via `BatchKVCache.filter`. If all rows terminal, resets the batch cache
     to `None` so Phase 2 installs a fresh one. Hooks eager extract-to-
     prefix-cache for each terminating row (16c.2 step 4). Releases
     budgeter reservations (16d-2b).
  2. **Admit** — drain waiting queue. Two sub-phases:
     - **Phase A: decide + apply** — per pending, consult `budgeter.admit`
       (if configured), route to admit / reject / evict / preempt. B-8:
       commit reservation BEFORE the next iteration's admit call.
     - **Phase B: execute** — split accepted into hit/miss; hit rows go
       through `_admit_single_hit_row` (seeded BatchKVCache), miss rows
       batched through `_admit_miss_cohort`.
  3. **Forward** — one batched prefill (if any row PREFILL) or one
     batched decode. Skipped when Phase 2 ran admissions (prefill T vs
     decode T=1 cannot mix).

Internal helpers worth knowing (tests exercise them directly):

- **`_reclaim_terminated()`** — Phase 1. Filters `BatchKVCache` and
  rebuilds `slot_table`. Extracts prefix K/V before filter (Gate-0.75
  probe B). Releases budget reservations before the destructive filter.
- **`_admit_waiting_requests()`** — Phase 2. See Phase A/B above.
- **`_apply_evict(n_blocks)`** — Phase A evict branch. Calls
  `prefix_cache.evict_until(n_blocks)`. Raises `_BudgetEvictUnderrun` if
  fewer blocks freed than asked — the caller converts to an aborted
  admission without touching budgeter state (invariant B-2).
- **`_preempt_active_row(victim_req_id) -> _BatchRow | None`** — steps 1-4
  of preempt: extract prefix K/V (if cache present), state transition
  `PREFILL/DECODE → PREEMPTED → WAITING`, filter victim out of
  BatchKVCache, rebuild slot_table, release budgeter reservation. Returns
  the detached `_BatchRow` or `None` on B-7 race.
- **`_apply_preempt(victim_req_id) -> bool`** — wraps `_preempt_active_row`
  with step 5: builds `composite_prompt = prompt_ids + generated`,
  derives `replay_params = params.model_copy(update={"max_tokens":
  remaining})` (Q-2 algebra preserves worst-case bytes), and
  `appendleft`s an `is_replay=True` pending to the waiting queue. Raises
  `AssertionError` (pre-mutation) if `remaining <= 0`.
- **`_extract_and_insert_prefix(row_idx)`** — slice a terminating row's
  block-aligned prefix K/V out of the batched cache, `mx.eval` to force
  to own memory, and `prefix_cache.insert_detached`. Load-bearing: cache
  holds K/V for `prompt + generated[:-1]` (last emit has no K/V yet), and
  every layer's tensor is `(B, H, T_max, D)` with per-row
  `left_padding[i]` offset.
- **`_find_row_by_req_id(req_id) -> int | None`** — linear scan.
- **`_rebuild_slot_table()`** — called after every reshuffle (`filter`,
  `extend`, admit).
- **`_prepare_cohort()`** — first-step initial cohort seal. Allocates
  BatchKVCache with per-row `left_padding`.
- **`_prefill_phase()` / `_decode_phase()`** — one batched forward over
  `(B, T_max)` or `(B, 1)` tokens respectively.
- **`_sample_and_emit_rows(rows, batched_logits, *, is_prefill)`** — per
  row sample, append to `row.generated`, transition FSM, emit
  `BatchEvent.token` / `done`.
- **`_build_prefill_tokens()` / `_build_decode_tokens()`** — construct
  left-padded `(B, T_max)` or `(B, 1)` input tensors. Terminal rows feed
  `_pad_token_id()` during decode.
- **`_admit_single_hit_row(pending, usable_hit_tokens)`** — per-row hit
  admission. `lookup` → `build_seeded_batch_kv` → suffix prefill forward
  → `_sample_and_emit_rows` → extend main. `finally` releases retained
  hits (mismatched release would strand them).
- **`_admit_miss_cohort(admitted_pending)`** — batched miss admission.
  One prefill forward over K rows, `extend` into main.

Capability gate (static method):
- **`_enforce_capability_gate(adapter)`** — raises `NotImplementedError`
  if any layer's `AttentionKind` ≠ `GLOBAL`.

### `silica.scheduler.seed_kv`

#### `build_seeded_batch_kv(detached_blocks, *, num_layers) -> list[BatchKVCache]` *(public function)*

Construct one `BatchKVCache(B=1)` per transformer layer, pre-populated
from a prefix-cache hit's detached K/V slices. Used by
`ContinuousBatcher._admit_single_hit_row`.

- **`detached_blocks`** — `Sequence[Sequence[tuple[mx.array, mx.array]]]`
  indexed `[block_idx][layer_idx] → (K, V)`. Each K/V shape must be
  `(1, n_kv_heads, block_size, head_dim)`.
- **`num_layers`** — must match every inner sequence's length.

Each returned cache: keys/values concatenated along the sequence axis
(`(1, n_kv_heads, num_blocks * block_size, head_dim)`), `offset` set to
total seq, `left_padding == [0]`.

Loud-fail on shape / dtype mismatches — silent coercion would manifest
as divergent decode one step later.

---

## `silica.models`

### `silica.models.adapter`

#### `Tokenizer` *(protocol)*

Minimal tokenizer surface. Attribute `vocab_size: int`. Methods:
- **`encode(text: str) -> list[int]`**
- **`decode(token_ids: Sequence[int]) -> str`**

P-1 / P-3 adapters carry mlx-lm's concrete tokenizer; the Protocol is
tightened at P-0 exit.

#### `ModelConfig` *(public, dataclass)*

Architecture-level info. Fields: `model_name`, `num_layers`, `hidden_size`,
`vocab_size`, `extra: dict[str, Any]`.

#### `KVLayout` *(public, dataclass)*

Per-layer KV shape. Required fields: `num_layers`, `n_kv_heads`,
`head_dim`, `dtype: mx.Dtype`. Optional field (P-3-D4):
- `bytes_per_token_total: int | None = None` — aggregate per-token KV
  bytes across all layers. Homogeneous-shape adapters (plain Qwen3,
  Qwen3.5 dense) leave this at `None` and `MemoryBudgeter.for_adapter`
  derives the total from the four required fields. Adapters whose
  per-layer KV shapes disagree (Gemma4: sliding 16×256 + full 4×512)
  populate this with an explicit per-kind sum so the budget is not
  computed from the single-shape summary fields.

Consumed by `KVCodec` and the KV allocator (which read the four
required fields) and by `MemoryBudgeter.for_adapter` (which prefers
`bytes_per_token_total` when set).

#### `AttentionKind` *(str enum)*

Per-layer classification (D-015):
- `GLOBAL` — standard causal KV.
- `SLIDING` — sliding-window KV.
- `HYBRID` — per-layer sliding/global mix.
- `RECURRENT` — pure linear / Mamba-like (no KV).
- `HYBRID_DELTANET` — Qwen3.5's interleaved Gated-DeltaNet + Gated-Attn stack.

#### `AttentionPattern` *(public, frozen dataclass)*

Field: `per_layer: tuple[AttentionKind, ...]`, length matches
`KVLayout.num_layers`. The scheduler walks this to decide KV routing.

#### `StateDelta` *(public, frozen dataclass)*

Read-only snapshot returned by `prefill` / `decode_step` (D-015). Opaque
`_payload` (adapter-owned); single public method
**`recurrent_bytes() -> int`** used by the budgeter.

#### `ModelAdapter` *(protocol, I-1)*

Per-model bridge between the engine and the MLX-native model graph.
Attribute `config: ModelConfig`. Methods:

- **`build(weight_provider: WeightProvider) -> Module`** — realise the
  model graph. Called once per batcher session.
- **`kv_layout() -> KVLayout`**.
- **`attention_pattern() -> AttentionPattern`**.
- **`tokenizer() -> Tokenizer`**.
- **`prefill(tokens: mx.array, kv_handle: KVHandle) -> tuple[mx.array, StateDelta]`**
  — single-request prefill (P-1). Returns last-position logits +
  state_delta.
- **`decode_step(token: mx.array, kv_handle: KVHandle) -> tuple[mx.array, StateDelta]`**
  — single-request one-token decode.

Key constraints:
1. `attention_pattern()` is the per-layer routing authority.
2. KV mutation ownership belongs to `KVManager` — the adapter never
   holds block pointers directly.
3. `StateDelta` carries non-KV runtime state only (no KV blocks).

#### `StubModelAdapter` *(public, stub)*

P-0 I-1 conformance stub — zero logits, empty state, no weights. Not a
runnable model.

Constructor (kwargs): `model_name="stub"`, `num_layers=2`, `hidden_size=32`,
`vocab_size=8`, `n_kv_heads=2`, `head_dim=16`.

### `silica.models.capabilities`

P-3 capability gate (D-016). The scheduler reads
`adapter.capabilities()` once at construction time and refuses to admit
adapters whose declared needs the current batcher cannot honour.

#### `ModelCapabilities` *(public, frozen dataclass)*

Typed summary of what an adapter needs from the scheduler. Fields:

- **`attention_kinds: frozenset[AttentionKind]`** — every kind that
  appears in the model's `AttentionPattern`. Used by the batcher to
  pick KV layouts.
- **`has_recurrent_state: bool`** — true for hybrid families (Qwen3.5
  DeltaNet) where `StateDelta._payload` carries non-KV runtime state.
  `ContinuousBatcher` refuses adapters with this flag set in v0.1
  unless paired with `BatchRecurrentStateStore` (P-3-C5).
- **`has_moe: bool`** — true when at least one layer routes through a
  mixture-of-experts MLP (Qwen3.5-MoE, Gemma4-MoE). The continuous
  batcher rejects MoE adapters with a P-3-E4 pointer until the
  batched MoE smoke gate lifts.

#### `capabilities_from_attention_pattern(pattern, *, has_moe=False) -> ModelCapabilities` *(public function)*

Derive `ModelCapabilities` from an `AttentionPattern`. Sets
`has_recurrent_state=True` when `HYBRID_DELTANET` appears in the
pattern; `has_moe` is passed through.

### `silica.models.qwen3`

#### `Qwen3Adapter` *(public class)*

I-1 `ModelAdapter` for the plain Qwen3 family (Qwen3-0.6B / 4B / 7B /
14B / 32B — `model_type == "qwen3"`). All layers pure GQA causal
attention (`AttentionPattern` = all `GLOBAL`).

Constructor: `Qwen3Adapter(model, tokenizer, kv_manager: SimpleKVCache)`.

Factory classmethod:
- **`Qwen3Adapter.from_hf_repo(repo) -> tuple[Qwen3Adapter, SimpleKVCache]`**
  — load via `mlx_lm.utils.load`, build `SimpleKVCache.from_model`,
  construct adapter.

Implements every I-1 method. Reads `args.head_dim` primary (Qwen3-0.6B
uses `head_dim=128`, not the naive `hidden_size/num_heads`); falls back
to computed default if absent. Load-bearing for
`MemoryBudgeter.bytes_per_token` accuracy.

### `silica.models.qwen3_5`

#### `Qwen3_5Adapter` *(public class)*

I-1 `ModelAdapter` for the Qwen3.5 hybrid family (0.8B / 27B / 35B-A3B —
`model_type == "qwen3_5"`). Per-layer `is_linear` flag drives
`HYBRID_DELTANET` vs `GLOBAL`. Handles Qwen3.5's multimodal sanitize
(`vision_tower` / `visual` dropped on load) and `mtp.*` head sanitize
(RMSNorm +1.0 shift). P-1 runs single-request through `SimpleKVCache`.
P-2 `ContinuousBatcher` refuses this adapter at construction time
(DeltaNet batching arrives in P-3 via `BatchRecurrentStateStore`).

Constructor + `from_hf_repo` mirror `Qwen3Adapter`.

### `silica.models.qwen3_5_moe`

#### `Qwen3_5MoeAdapter` *(public class)*

I-1 `ModelAdapter` for the Qwen3.5-MoE family
(Qwen3.5-35B-A3B — `model_type == "qwen3_5_moe"`). Thin wrapper around
the dense `Qwen3_5Adapter` — silica does not reimplement MoE math; the
upstream `mlx_lm` `SwitchGLU` / `gather_mm` path owns expert dispatch.
Silica contributes:

- the `has_moe=True` capability so the scheduler refuses batched
  execution at admission;
- variant guards on `num_experts`, `num_experts_per_tok`,
  `mlp_only_layers`;
- `MoeExtra`-style metadata under `config.extra` (`num_experts`,
  `num_experts_per_tok`, `moe_intermediate_size`,
  `shared_expert_intermediate_size`, `norm_topk_prob_runtime`,
  `attn_output_gate_config`, `attn_output_gate_mlx_lm_honors`);
- an opt-in dispatch-observation seam:
  **`install_dispatch_proxy(observer) -> None`** wraps each MoE
  layer's `switch_mlp` with a forwarding proxy reporting
  `(layer_idx, indices)` before delegating to the real `SwitchGLU`.
  Not installed by default.

Constructor + `from_hf_repo` mirror `Qwen3Adapter`. Per-expert MoE
streaming under D-011 lands in P-6.

### `silica.models.gemma4`

#### `Gemma4Adapter` *(public class)*

I-1 `ModelAdapter` for the Gemma4-31B dense family — sliding +
full-attention layers in a `5:1` ratio
(`AttentionPattern` mixes `SLIDING(window=4096)` and `GLOBAL`). Loaded
via `mlx_lm.utils.load`, with weight-loading sanitize for the Gemma4
multimodal heads. The MoE branch (Gemma4-26B-A4B) is rejected at
construction; see `silica.models.gemma4_moe`.

KV layout reports the per-kind sum via
`KVLayout.bytes_per_token_total` so the budgeter sees the real
heterogeneous footprint (sliding 16×256 + full 4×512 mix).

Constructor + `from_hf_repo` mirror `Qwen3Adapter`. Batched execution
delegates to `mlx_lm`'s `BatchRotatingKVCache` for sliding layers and
`BatchKVCache` for full layers (P-3-D2 close).

### `silica.models.gemma4_moe`

`Gemma4MoeAdapter` — variant of `Gemma4Adapter` for the
Gemma4-26B-A4B MoE branch. Branch-selected by an `enable_moe_block`
flag inside the factory's `_build_gemma4` helper because Gemma4-MoE
shares `model_type == "gemma4"` with the dense family. Like
Qwen3.5-MoE, sets `has_moe=True` and is rejected at the batcher
admission gate until P-3-E4 lifts.

Unique to Gemma4-MoE: the dense MLP branch is **not** replaced — the
MoE-mode forward sums `h = h_dense_mlp + h_experts`. KV layout
inherits from `Gemma4Adapter` (sliding + full mix).

### `silica.models.recurrent`

P-3-C5 recurrent-state snapshot interface.

#### `RecurrentSnapshot` *(public class, opaque)*

Read-only snapshot of one batch row's recurrent state at slice-prefill
boundaries. Adapter-defined payload; the scheduler treats it as
opaque. Single public method **`recurrent_bytes() -> int`** for
memory accounting.

#### `RecurrentStateAdapter` *(public, mixin)*

Marker mixin implemented by adapters that carry recurrent state across
slice boundaries (Qwen3.5 hybrid families). Adds:

- **`snapshot_recurrent(batch_row: int) -> RecurrentSnapshot`** —
  capture state at a slice boundary.
- **`restore_recurrent(batch_row: int, snap: RecurrentSnapshot) -> None`**
  — restore at slice resume.

`ContinuousBatcher` calls these only when
`adapter.capabilities().has_recurrent_state` is true and the slice-
prefill regime is active (C5.5 α-MVP).

### `silica.models.pre_norm_capture`

P-5-F (3b) pre-RoPE production routing — stores keys at the
projection-output stage *before* RMSNorm and RoPE so the prefix store
holds normalisation- and rotation-free K, matching the (b-static)
codec quality target.

#### `install_pre_norm_capture_proxies(model) -> None` *(public function)*

Wrap every attention layer's `k_proj` with a `_PreNormCaptureProxy`
that captures the projection output (pre-norm, pre-RoPE) into a
per-layer slot. Installed once per `build()`; idempotent.

#### `apply_k_norm_then_rope_to_block(k_block, *, layer, position_ids, sin, cos) -> mx.array` *(public function)*

Reconstruct post-RoPE K from a pre-norm K block on demand. Called by
`SyntheticPrefixBlockStore.fetch_detached` when the codec is
configured for the (3b) production-store path. Output matches the
shape contract `build_seeded_batch_kv` expects.

#### `PreNormCaptureAdapter` *(public, mixin)*

Marker mixin for adapters that ship the (3b) prefix-cache store
path. Implemented by `Qwen3_5Adapter`. Adds the per-layer
`PreNormSlot` slot list so the proxies have a write target.

### `silica.models.factory`

#### `adapter_for_repo(repo: str) -> tuple[ModelAdapter, SimpleKVCache]` *(public function)*

Load `repo` via mlx-lm, build matching family adapter.

- Dispatches on `model.model_type`.
- Supported: `"qwen3"` → `Qwen3Adapter`, `"qwen3_5"` →
  `Qwen3_5Adapter`, `"qwen3_5_moe"` → `Qwen3_5MoeAdapter`,
  `"gemma4"` → `Gemma4Adapter` or `Gemma4MoeAdapter` depending on
  `enable_moe_block`.
- Unknown `model_type` raises `NotImplementedError` listing supported
  families and pointing at the registry.

Used by the CLI (`silica/server/cli.py::_run`).

#### `supported_model_types() -> tuple[str, ...]` *(public function)*

Sorted list of registered `model_type` strings.

Internals: `_ADAPTERS: dict[str, _AdapterBuilder]` is the greppable
dispatch table. Adding a family is a two-step change: write
`silica/models/<family>.py`, register the builder in `_ADAPTERS`.

---

## `silica.weights`

### `silica.weights.provider`

#### `LayerWeights` *(public, dataclass)*

Fields: `tensors: dict[str, mx.array]` (free-form layout), `resident_bytes: int`.
Adapters name parameters per their own layout (q_proj / k_proj / gate / up /
down vary across Qwen3.5, Gemma4, MoE).

#### `ExpertWeights` *(public, dataclass)*

Analogous to `LayerWeights` but scoped to one MoE expert. Used by the
per-expert WeightProvider methods (D-011).

#### `WeightProvider` *(protocol, I-4)*

Residency / streaming abstraction.

- **`get_layer(layer_idx) -> LayerWeights`** — synchronous, immediately
  usable.
- **`prefetch(layer_indices) -> None`** — hint; implementations may no-op.
- **`release(layer_idx) -> None`** — signal "done with this layer".
- **`resident_bytes() -> int`** — D-012 canonical.
- **`get_expert(layer_idx, expert_id) -> ExpertWeights`** *(D-011)*.
- **`prefetch_experts(layer_idx, expert_ids) -> None`** *(D-011)*.
- **`release_expert(layer_idx, expert_id) -> None`** *(D-011)*.

Dense providers **must** raise `NotImplementedError` on the per-expert
path — pairing a MoE adapter with a dense provider fails loudly at
wiring time (not silently at runtime).

### `silica.weights.resident`

#### `ResidentWeightProvider` *(public class, dense)*

Dense, fully-resident implementation. P-0 stub today (dict-backed); P-3
extends with real safetensors loading via mlx-lm.

Constructor: `ResidentWeightProvider(layers: dict[int, LayerWeights] | None = None)`.

- `get_layer`, `prefetch` (no-op), `release` (no-op), `resident_bytes`
  implemented.
- `get_expert` / `prefetch_experts` / `release_expert` raise
  `NotImplementedError` with message `"dense provider has no per-expert path"`.

---

## `silica.vq`

P-5 KV codec implementations. The codecs satisfy the
[`VectorCodec`](#silicakvcachecodec) Protocol and are wired into the
prefix-block store via `SyntheticPrefixBlockStore.k_codec` /
`.v_codec`. All hot-path bodies are MLX-native (D-009 — no NumPy /
torch in runtime).

Re-exports (via `silica/vq/__init__.py`):

| Name | Origin | Kind |
| --- | --- | --- |
| `BlockTurboQuantMSE` | block_tq | public class |
| `TurboQuantMSE` | turboquant | public function (factory) |
| `RaBitQ1Bit` | rabitq.rabitq_1bit | public class |
| `ExtRaBitQ` | rabitq.rabitq_ext | public class |

### `silica.vq.block_tq`

#### `BlockTurboQuantMSE` *(public class)*

Block-wise TurboQuant with per-vq-block fp16 scales. Each vector
(one head's slice of one token) is split into `d / B` contiguous spans
of length `B = vq_block_size`; each span is normalised by its own L2
norm (scale) and quantised against a Lloyd-Max codebook of width
`b = num_bits` bits. Decode multiplies the scalar-quantised
codebook value by the saved scale.

Constructor:

```python
BlockTurboQuantMSE(
    *,
    block_size: int,           # number of tokens per detached block
    n_kv_heads: int,
    head_dim: int,
    vq_block_size: int,        # B; head_dim must be a multiple
    num_bits: int,             # b; one of {2, 3, 4}
    seed: int = 42,
    per_head_rotation: bool = False,
    norm_correction: bool = True,
    dtype: mx.Dtype = mx.float16,
)
```

Implements `VectorCodec[BlockTQPayload]`. The `per_head_rotation`
flag (D.2a finding, opt-in at v1.7.8 default-off) draws an
independent Haar rotation per head; default tightens variance
across seeds without changing the mean (P-5 acceptance evidence in
`plans/P5_ACCEPTANCE_SWEEP/`).

### `silica.vq.turboquant`

#### `TurboQuantMSE(*, block_size, n_kv_heads, head_dim, num_bits, seed=42, norm_correction=True, dtype=mx.float16) -> BlockTurboQuantMSE` *(public factory)*

Scalar TurboQuantMSE — `BlockTurboQuantMSE(vq_block_size=head_dim)`.
Kept as a separate symbol for catalogue clarity: when `B == d` there
is exactly one scale per vector, and the codec collapses to the
classical TurboQuant scheme.

### `silica.vq.rabitq`

Subpackage with the RaBitQ codec family — vector-level rotated
sign-bit quantisation.

#### `RaBitQ1Bit` *(public class)*

`VectorCodec[RaBitQPayload]`. 1-bit (sign-bit) quantisation against a
fixed Haar rotation, with per-vector `||x - centroid||` and inner-
product coefficient saved on the payload. P-5-B opening §5.3 pins
the centroid at zero. Symmetric in K / V.

#### `ExtRaBitQ` *(public class)*

`VectorCodec[ExtRaBitQPayload]`. Multi-bit extension of `RaBitQ1Bit`
(`num_bits ∈ {2, 3, 4}`). Adds a per-vector fp16 dequantisation
scale to the payload. Symmetric in K / V.

### `silica.vq._calibration`

Underscore-prefixed helper module that hosts NumPy-side calibration
routines (centroid fitting, Haar-rotation sampling, codebook
fitting). Held under `_calibration` because v0.1 codecs do not
calibrate at runtime — calibration data is loaded from pickled
artifacts produced offline by vqbench-equivalent scripts. D-009 keeps
the runtime hot path MLX-native; calibration is the documented
exception.

Public surface is intentionally thin: helpers are imported by codec
constructors, not re-exported via `silica.vq.__init__`.

---

## `silica.mlx`

### `silica.mlx.runner`

Thin wrappers over mlx-lm's `model(tokens, cache=...)` call signature.

#### `forward_batched(model, tokens: mx.array, cache_list: list) -> mx.array` *(public function)*

One batched forward. `tokens` must be 2-D `(B, T)`; returns per-row
last-position logits `(B, V)`. Used by `ContinuousBatcher` for both
prefill (`T = T_max`) and decode (`T = 1`). Raises `ValueError` on
non-2-D input or zero B/T.

#### `forward(model, tokens: mx.array, cache_list: list) -> mx.array` *(public function)*

Single-request path — wraps `forward_batched` at `B=1`. `tokens` is 1-D
`(T,)`; returns `(V,)`. Used by `Engine.generate` and the family
adapters. Raises `ValueError` on non-1-D / empty input.

---

## `silica.speculative`

### `silica.speculative.engine`

#### `DraftTokens` *(public, dataclass)*

Speculative draft. Fields: `token_ids: tuple[int, ...]`,
`draft_logprobs: tuple[float, ...] | None`. `NoopDraftEngine` emits empty;
P-7 `DraftTargetEngine` fills both (target verification needs draft
logprobs for accept/reject probabilities).

#### `DraftEngine` *(protocol, I-5)*

- **`propose(ctx: RequestState, k: int) -> DraftTokens`** — up to k
  draft tokens; fewer allowed; zero legal (== "no proposal this step").
- **`commit(ctx: RequestState, accepted_len: int) -> None`** — called
  after target verification with the accepted-count.

#### `NoopDraftEngine` *(public, stub)*

Draft disabled. `propose` returns empty; `commit` no-op. Default wired to
the main loop from P-0 so speculative decoding is toggled by swapping
this implementation, not by adding conditional branches.

---

## `silica.engine`

### `silica.engine.Engine` *(public class)*

Top-level generation orchestrator. One instance per `(adapter,
kv_manager)` pair; call `generate` / `generate_batch` any number of times.
`req_id` is auto-assigned.

Constructor:
```python
Engine(
    adapter: ModelAdapter,
    kv_manager: KVManager,
    sampler: Sampler | None = None,
    metrics: MetricsRegistry | None = None,
)
```

Attribute: **`metrics: MetricsRegistry`** — read via `engine.metrics.snapshot()`
once the generator is exhausted. Populated fields:
- `ttft_ms` — wall-clock ms from prefill start to first yielded token
  (prefill forward + first sample).
- `prefill_tok_s` — `len(prompt_ids) / ttft_s`.
- `decode_tok_s` — `n_decode / decode_s` where `decode_s` is generator
  active time (caller-side latency between yields is excluded, since
  `perf_counter` inside a generator only ticks during `__next__`).
- `resident_mb` — `kv_manager.budget().resident_bytes / 1e6` at end.
- `logical_kv_bytes` — `kv_manager.budget().logical_bytes` at end.

Methods:

#### `generate(prompt: str, params: SamplingParams | None = None) -> Iterator[int]`

Yield generated token ids one at a time. Empty prompt yields nothing
(mirrors mlx-lm's non-empty requirement).

Stop policy (P-1):
- `max_tokens` is the hard upper bound on yielded tokens.
- `stop_token_ids` tokens are **yielded-then-stopped** (vLLM convention —
  the caller needs to see the stop token to know why we stopped).
- Caller populates `stop_token_ids` with `tokenizer.eos_token_ids` unless
  `ignore_eos=True`.
- String-sequence `stop` patterns are a P-2 concern (require incremental
  decoding).

Internal flow: `tokenizer.encode` → `kv.reserve_for_prefill` →
`adapter.prefill` + first sample → `try … finally kv.free`. Decode loop:
`adapter.decode_step` + sample per step until `max_tokens` or stop.

#### `generate_batch(prompts, params=None, *, max_batch_size=None, prefix_cache=None) -> Iterator[BatchEvent]`

Drive a `ContinuousBatcher` and yield `BatchEvent` values.

- **`prompts: Sequence[str]`** — empty strings are skipped silently
  (their `req_index` stays mapped to the original list position).
- **`params`** — single `SamplingParams` (homogeneous batch, P-2 case) or
  a list of length `len(prompts)`. P-2 requires all elements equal; a
  heterogeneous list raises `NotImplementedError` pointing at P-3.
- **`max_batch_size`** — cap on **active physical rows**, not queue
  length. Defaults to non-empty prompt count. Callers testing
  queue-bounded admission pass e.g. `max_batch_size=4` with 8 prompts.
- **`prefix_cache`** — optional `RadixPrefixCache` for shared-prefix
  reuse. Callers own the cache's lifetime so it persists across multiple
  `generate_batch` invocations. When `None`, behaviour is bit-identical
  to 16c.1 (invariant S-6).

Internal flow: tokenize + skip empties → construct `ContinuousBatcher` →
first `effective_batch_size` admits pre-step → the rest queue via
`add_request` (hits the mid-run admission path) → drain loop on
`batcher.has_work()`.

`_resolve_batch_params` (module-level helper) validates / reduces the
`params` union type.

---

## `silica.bench`

P-4 unified benchmark harness. The runner walks a scenario catalogue,
holds an oracle stack (smoke / B=1 parity / teacher-forced argmax /
PPL / storage / admission-headroom), and emits JSONL + Markdown
reports. The harness drives `Engine` directly — no special path; the
same configuration produced by the CLI runs the bench.

Re-exports (via `silica/bench/__init__.py`):

| Name | Origin | Kind |
| --- | --- | --- |
| `Scenario`, `ScenarioResult`, `OracleKind`, `OracleFn`, `Workload`, `VqbenchXcheckSpec`, `hf_cache_path_for_repo` | scenario | public |
| `BUILTIN_SCENARIOS`, `get_scenario`, `list_scenario_ids` | scenarios | public |
| `BenchRunner`, `EngineFactory`, `DirectBatchedReferenceFn`, `TeacherForcedSilicaFn`, `TeacherForcedReferenceFn` | runner | public |
| `VqbenchBaselineResult`, `default_reproduce_script_path`, `parse_headline_row`, `run_vqbench_baseline` | vqbench_baseline | public |
| `render_markdown_report` | report | public |

Oracle implementations are an internal concern of the runner and are
**not** re-exported at the package level; authors of new oracles
import from `silica.bench.oracles` directly.

### `silica.bench.scenario`

#### `OracleKind` *(public, str enum)*

How a scenario decides pass / fail. Values: `SMOKE`,
`B1_PARITY_VS_SINGLE`, `BGT1_DIRECT_BATCHED_REFERENCE`,
`TEACHER_FORCED_ARGMAX`, `DECODE_TOK_S_WITH_PREFIX_HIT`, `PPL`,
`STORAGE`, `ADMISSION_HEADROOM`.

#### `Workload` *(public, frozen dataclass)*

Prompt set + decoding parameters. Fields: `name`, `prompts`,
`max_tokens`, `max_batch_size=1`, `prefix_cache=False`,
`temperature=0.0`, `top_p=1.0`, `kv_codec=None`.

#### `VqbenchXcheckSpec` *(public, frozen dataclass)*

Declarative spec for a `--vqbench-xcheck` cross-check arm. Fields:
`script_path`, `method`, `bits`, `extra_args=()`. The runner
materialises this into a subprocess invocation when
`bench.py --vqbench-xcheck` is set.

#### `Scenario` *(public, frozen dataclass)*

One bench row. Fields: `id`, `repo`, `workload`,
`oracle=OracleKind.SMOKE`, `oracle_config={}`, `gate_env_var=None`,
`description=""`, `vqbench_xcheck=None`. Scenarios with a non-None
`gate_env_var` skip when the named env var is unset (cache-presence
gates for HF-cache-bound rows).

#### `ScenarioResult` *(public, dataclass)*

Outcome of running one scenario. Fields: `scenario_id`, `status`,
`reason`, `ttft_ms`, `prefill_tok_s`, `decode_tok_s`, `resident_mb`,
`peak_memory_mb`, `total_tokens`, `wall_s`, `metadata`. Status
values: `"ok"`, `"failed"`, `"skipped"`. The runner dumps these as
JSONL rows and `render_markdown_report` aggregates them.

#### `OracleFn` *(public type alias)*

`Callable[[Scenario, Any, Any], tuple[bool, str | None, dict[str, Any]]]`.
Oracle functions return `(passed, reason_if_failed, extra_metadata)`.

#### `hf_cache_path_for_repo(repo: str) -> Path` *(public function)*

Derive the HF hub cache directory for `repo`. Used by
`gate_env_var=None` scenarios that gate on cache-presence by
checking whether the directory exists.

### `silica.bench.scenarios`

Module-level scenario catalogue.

#### `BUILTIN_SCENARIOS: dict[str, Scenario]` *(public, constant)*

Canonical map of scenario id → `Scenario`. Curated to cover the
P-4 / P-5 acceptance gates and the Qwen3-0.6B / Qwen3.5 dev loops.

#### `get_scenario(scenario_id: str) -> Scenario` *(public function)*

Look up a scenario by id; raises `KeyError` on unknown id.

#### `list_scenario_ids() -> list[str]` *(public function)*

Sorted list of known scenario ids — used by `bench.py --list`.

### `silica.bench.runner`

#### `BenchRunner` *(public class)*

Drives scenarios and emits `ScenarioResult` rows.

Constructor: `BenchRunner(engine_factory: EngineFactory | None = None,
peak_memory_fn=None)`. Default `engine_factory` builds an `Engine`
from the scenario's `repo`; tests inject mocks via the constructor.

Key method: **`run(scenarios, output_path, *, report_md=None,
all_kv_codecs=False, vqbench_xcheck=False, codec_overrides=None,
seeds=(42,)) -> list[ScenarioResult]`** — runs each scenario, writes
JSONL to `output_path`, optionally writes a Markdown report.
`all_kv_codecs` expands every PPL / storage / admission-headroom row
into the codec catalogue.

#### `EngineFactory`, `DirectBatchedReferenceFn`, `TeacherForcedSilicaFn`, `TeacherForcedReferenceFn` *(public type aliases)*

Type aliases capturing the runner's pluggable seams. See the runner
source for signatures; all four are wired to default implementations
inside `BenchRunner.__init__`.

### `silica.bench.codec_registry`

#### `CodecSpec` *(public, frozen dataclass)*

Registry entry for one named codec configuration. Fields: `id`,
`family`, `bits_per_value`, `k_supported`, `v_supported`,
`requires_fit`, `payload_packed`, `production_recommended`,
`factory: Callable[..., VectorCodec]`. The `factory` builds a side-
level codec for either K or V given a `KVLayout`.

#### `get_codec_spec(codec_id: str) -> CodecSpec` *(public function)*

Look up a codec spec by id. Raises `KeyError` on unknown id.

#### `list_codec_ids() -> list[str]` *(public function)*

Sorted ids in registry order.

### `silica.bench.oracles`

Module-level oracle implementations called by `BenchRunner` based on
`scenario.oracle`. All conform to `OracleFn`.

- **`smoke_oracle`** — did the workload emit at least one valid
  token?
- **`b1_parity_oracle`** — `B=1` parity: batched token stream must
  equal the single-request reference token-by-token.
- **`bgt1_direct_batched_reference_oracle`** — `B>1` parity: silica
  per-row output equals a direct mlx-lm batched reference call.
- **`teacher_forced_argmax_oracle`** — silica adapter path's
  positional-argmax outputs must match a teacher-forced reference at
  every position.
- **`decode_tok_s_with_prefix_hit_oracle`** — P-5-A.0 prefix-hit
  decode-throughput gate.
- **`ppl_oracle`** — P-5-C end-to-end PPL gate (vqbench-aligned
  oracle path).
- **`storage_oracle`** — P-5-C resident-bytes invariant gate.
- **`admission_headroom_oracle`** — P-5-C admission-headroom gate
  proving Principle 8 (codec on → more rows admitted at the same
  cap).

### `silica.bench.ppl_oracle`

Teacher-forced perplexity primitives.

- **`teacher_forced_chunked_nll(...)`** — cumulative teacher-forced
  NLL over a token sequence.
- **`teacher_forced_chunked_nll_with_codec(...)`** — same, but
  routed through a `RadixPrefixCache` so codec encode / decode is
  exercised on every chunk.
- **`teacher_forced_chunked_nll_with_codec_pre_rope(...)`** — P-5-F
  pre-RoPE production-store oracle.
- **`teacher_forced_chunked_nll_with_codec_pre_norm(...)`** — P-5-F
  (3b) production-store oracle: pre-norm K via projection-output
  capture.
- **`teacher_forced_chunked_nll_vqbench_aligned(...)`** — vqbench-
  style pre-RoPE codec injection (D.2a oracle); the (4-b) PPL gate
  measures against this path.
- **`perplexity_from_nll(nll_sum, num_tokens) -> float`** — convert
  cumulative NLL + token count to perplexity.

### `silica.bench.report`

#### `render_markdown_report(results) -> str` *(public function)*

Return a Markdown report covering `results`. Used both by the runner
(via `--report-md` flag) and by callers building ad-hoc reports.

### `silica.bench.wikitext`

WikiText loading helpers used by the PPL oracles.

- **`load_wikitext_text(path) -> str`** — read a pre-extracted
  WikiText text file into a single string.
- **`tokenize_for_ppl(text, tokenizer) -> mx.array`** — tokenize
  into the `(1, seq_len)` int32 shape the PPL oracles expect.

### `silica.bench.vqbench_baseline`

Drives the vqbench reproduce-script path so silica's PPL numbers are
measured against the same harness vqbench cites in its public REPORT.

#### `VqbenchBaselineResult` *(public, dataclass)*

One vqbench reproduce-script invocation outcome. Fields: `status`,
`reason`, `script`, `python_executable`, `script_args`, `model`,
`method`, `bits`, `ppl_fp16`, `ppl_quant`, `delta_ppl`, `delta_pct`,
`wall_s`, `returncode`, `stdout_tail`.

#### `parse_headline_row(stdout: str) -> dict[str, Any] | None` *(public function)*

Extract the PPL headline fields from a reproduce-script stdout. Used
by `run_vqbench_baseline` and exposed for tests.

#### `run_vqbench_baseline(script, *, python_executable, script_args, model, method, bits, subprocess_runner=subprocess.run) -> VqbenchBaselineResult` *(public function)*

Run `script` in a subprocess; parse its PPL headline. The
`subprocess_runner` seam is for tests.

#### `default_reproduce_script_path() -> Path` *(public function)*

Return the checked-in reproduce script path (relative to the repo
root). Used as the default value for the
`bench.py --vqbench-xcheck` command.

---

## `silica.chat`

Chat session abstraction (multi-turn loop with prefix-cache reuse)
plus the bundled `prompt_toolkit` REPL CLI.

Re-exports (via `silica/chat/__init__.py`):

| Name | Origin | Kind |
| --- | --- | --- |
| `ChatSession`, `TurnMetrics` | session | public |

The CLI subpackage `silica.chat.cli` is intentionally not re-exported
at `silica.chat`; it is a separate namespace with its own internal
layering. Programmatic users build on `ChatSession` directly.

### `silica.chat.session`

#### `TurnMetrics` *(public, dataclass)*

One chat-turn outcome. Fields: `reply`, `prompt_tokens`,
`output_tokens`, `finish_reason`, `ttft_ms`, `prefill_tok_s`,
`decode_tok_s`, `resident_mb`, `peak_memory_mb`, `logical_kv_bytes`,
`wall_s`, `prefix_hit_blocks`, `prefix_hit_tokens`,
`prefix_store_resident_bytes`, `prefix_store_logical_bytes`. The
chat CLI's toolbar reads these; `/showcase` aggregates them across a
session.

#### `ChatSession` *(public class)*

Stateful multi-turn chat loop. Holds a conversation history (list of
`{role, content}` messages), an optional `RadixPrefixCache`, and the
adapter / engine pair. Each `chat()` call rebuilds the prompt via
`tokenizer.apply_chat_template`, runs through `Engine.generate_batch`
with `prefix_cache=self._prefix_cache`, and returns a `TurnMetrics`.

Constructor:

```python
ChatSession(
    adapter,
    engine,
    *,
    system_prompt: str | None = None,
    prefix_cache: RadixPrefixCache | None = None,
    reset_peak_memory: Callable[[], None] | None = None,
    read_peak_memory_mb: Callable[[], float | None] | None = None,
    clock: Callable[[], float] = time.perf_counter,
)
```

Key methods:

- **`chat(message, params=None, stream_to=None) -> TurnMetrics`** —
  one turn; `stream_to` is an optional `Callable[[str], None]`
  invoked on each decoded chunk.
- **`reset() -> None`** — clear conversation history and reset the
  prefix cache (if attached).
- **`replace_messages(messages) -> None`** — wholesale swap of the
  history, used by `/load` in the chat CLI.
- **`set_prefix_cache(cache) -> None`** — attach / detach a prefix
  cache mid-session.

Properties: `messages`, `eos_token_ids`, `prefix_cache`.

### `silica.chat.cli`

Bundled `prompt_toolkit`-driven REPL — design contract in
`plans/CHAT_CLI_OPENING.md`. The package is layered:

- **`silica.chat.cli.state`** — `ChatCliState` dataclass holding
  live engine metrics, `StreamState` enum, conversation log,
  configuration. All other CLI code reads / writes through this
  state.
- **`silica.chat.cli.palette`** — `Palette` + `PaletteMode` enum
  (`PLAIN` / `EIGHT` / `TRUE_COLOR`) + capability detection
  (24-bit / 8-colour / `NO_COLOR` fallback).
- **`silica.chat.cli.toolbar`** — pure-function formatter from
  `ChatCliState` to the bottom-of-terminal status line.
  `render_toolbar(state, *, palette=None) -> str`,
  `render_codec_hint(state, *, palette=None, threshold_mb=200.0)
  -> str | None` (codec recommendation when prefix store crosses a
  threshold under fp16), `render_showcase(state, *, palette=None)
  -> str` (the multi-line `/showcase` session report).
- **`silica.chat.cli.thinking_parser`** — incremental parser that
  separates `<think>` tag content from reply content during
  streaming. Emits `ThinkChunk` and `ReplyChunk` events.
- **`silica.chat.cli.code_fence`** — incremental parser that
  isolates fenced code blocks for pygments highlighting. Emits
  `PlainText`, `EnterFence`, `ExitFence` events.
- **`silica.chat.cli.commands`** — slash-command dispatch
  (`/reset`, `/stats`, `/save`, `/load`, `/model`, `/showcase`,
  `/exit`).
- **`silica.chat.cli.config`** — `ChatCliConfig` pydantic model
  + `--system` / `--kv-codec` parsing.
- **`silica.chat.cli.persistence`** — JSON session serialization
  with `SCHEMA_VERSION = 1`. `serialize_session(...)`,
  `save_session(path, ...)`, `load_session(path) -> dict`.
  Permissive forward-compat loading: missing optional fields
  default; unknown fields ignored.
- **`silica.chat.cli.app`** — the prompt_toolkit application shell.
  Owns the event loop, redraws the toolbar at decode boundaries,
  routes input through commands / fence / thinking parsers,
  emits coloured output via the palette.

The `state` / `palette` / `toolbar` / `thinking_parser` /
`code_fence` / `persistence` modules are pure-Python and unit-tested
without prompt_toolkit. Only `app` carries the event-loop wiring.

---

## `silica.server`

### `silica.server.cli`

`silica` console-script entry point (also reachable via `python -m
silica …`). Two subcommands today: `run` (one-shot single-prompt
generation; the P-7+ HTTP server lands here too) and `chat` (drops
into the `silica.chat.cli` REPL). Bare `silica …` is rewritten to
`silica chat …` at parse time so the chat REPL is the default.

#### `build_parser() -> argparse.ArgumentParser`

Construct the argparse parser. Subcommands:

- **`run`** — one-shot generation. Arguments: `--model` (HF repo id,
  required), `--prompt` (required), `--max-tokens=64`,
  `--temperature=0.0` (0.0 = greedy), `--top-p=None`, `--top-k=None`,
  `--seed=None`.
- **`chat`** — start the chat REPL. Arguments: `--model`,
  `--system`, `--kv-codec`, `--max-tokens`, plus the `--top-*` /
  `--temperature` / `--seed` shared with `run`.

#### `main(argv: Sequence[str] | None = None) -> int`

Parse + dispatch. Calls `_preprocess_argv(argv)` first so bare
invocations (`silica --model …`) become `silica chat --model …`.
`run` calls `_run`; `chat` calls `silica.chat.cli.app.run_app`.
Returns shell exit code.

#### `_preprocess_argv(argv) -> list[str]` *(internal)*

C-7 bare-`silica` argv preprocessor (claude-style). When the first
positional token is not a known subcommand and not a top-level flag
(`--help` / `-h` / `--version`), prepend `chat` so the parser sees
the chat subcommand. The known-subcommand set is the module-level
`_KNOWN_SUBCOMMANDS = frozenset({"run", "chat"})`. Empty argv
becomes `["chat"]`.

#### `_run(args) -> int` *(internal)*

`adapter_for_repo(args.model)` → `Engine(adapter, kv)` → `SamplingParams`
with tokenizer EOS wired into `stop_token_ids` → exhaust
`engine.generate` → `print(prompt + tokenizer.decode(generated))` →
`_print_metrics` to stderr.

#### `_print_metrics(snapshot) -> None` *(internal)*

One-line stderr dump: `[metrics] ttft=…ms prefill=…tok/s decode=…tok/s resident=…MB`,
skipping `None` fields.

---

## See also

- [`PLAN.md`](../plans/PLAN.md) — phases, decisions, open questions.
- [`P2_OPENING.md`](../plans/P2_OPENING.md) — architectural opening doc for the
  `ContinuousBatcher` / `RadixPrefixCache` / `MemoryBudgeter` stack.
- [`P2_UNIT_16C_PREP.md`](../plans/P2_UNIT_16C_PREP.md), [`P2_UNIT_16C_2_PREP.md`](../plans/P2_UNIT_16C_2_PREP.md),
  [`P2_UNIT_16D_PREP.md`](../plans/P2_UNIT_16D_PREP.md) — per-unit prep docs with
  the invariant tables (S-1..S-7, B-1..B-9, L-1..L-3) that the batcher's
  tests enforce.
- [`P5_OPENING.md`](../plans/P5_OPENING.md) — payload schemas, codec
  catalogue, scalar-equivalence invariant for `silica.vq` /
  `silica.kvcache.codec`.
- [`P5_F_OPENING.md`](../plans/P5_F_OPENING.md) — pre-RoPE production
  routing for the (3b) projection-output capture path used by
  `silica.models.pre_norm_capture`.
- [`P3_MOE_SURVEY.md`](../plans/P3_MOE_SURVEY.md) — Qwen3.5-MoE +
  Gemma4-MoE adapter design (`silica.models.qwen3_5_moe`,
  `silica.models.gemma4_moe`).
- [`CHAT_CLI_OPENING.md`](../plans/CHAT_CLI_OPENING.md) — design
  contract for `silica.chat.cli` (toolbar field set, palette mode
  detection, thinking parser, persistence schema).
- `tests/` — the authoritative behaviour spec. `test_batcher.py`,
  `test_memory_budgeter.py`, `test_p2_batched_parity.py` are the
  high-signal entry points.
