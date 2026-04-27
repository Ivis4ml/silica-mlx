# P-5-A.0.4 — store migration: implementation checklist

| Field        | Value                                                                  |
| ------------ | ---------------------------------------------------------------------- |
| Version      | v1.0.0 (pre-implementation checklist)                                  |
| Last updated | 2026-04-21                                                             |
| Status       | design locked; implementation starts after reviewer ack                |
| Sub-unit     | P-5-A.0.4 (the fourth and final P-5-A.0 sub-unit)                      |
| Scope        | migrate `SyntheticPrefixBlockStore` + `silica.kvcache.codec` from the  |
|              | pre-P-5 pair-level API to the P-5 side-level `VectorCodec[P]` API;     |
|              | add `RadixPrefixCache.store` property; update four test modules        |
|              | (`test_prefix_store.py` — existing, expand with non-gated cases;       |
|              | `test_kvcodec.py` + `test_interfaces.py` — migrate to side-level;      |
|              | `test_kvcodec_integration.py` — counter refactor); sync the affected   |
|              | docs (PLAN I-3 definition + Q-008 resolution, README terminology,      |
|              | `plans/P5_OPENING.md` cross-reference)                                  |
| Predecessors | P-5-A.0.1 (`8f7e2c6`), P-5-A.0.2 (`c12b2fd`), P-5-A.0.3 (`49e99ae`)    |
| Reference    | `plans/P5_OPENING.md` §4.1, §4.3, §4.4, §4.7                            |

This checklist locks the landing details that the P-5 opening doc left
generic. The opening pinned *what* side-level looks like; this document
pins *how* the migration commit shapes constructor rules, internal
storage, call counts, byte accounting, and test rewrites so the diff
does not drift during implementation.

---

## 1. `SyntheticPrefixBlockStore` constructor rules

Signature:

```python
def __init__(
    self,
    *,
    block_size: int,
    k_codec: VectorCodec | None = None,
    v_codec: VectorCodec | None = None,
    codec: VectorCodec | None = None,
) -> None: ...
```

Resolution rules, evaluated in this order (first match wins; all other
combinations raise `ValueError`):

| `codec`   | `k_codec` | `v_codec` | Result                                                                                                             |
| --------- | --------- | --------- | ------------------------------------------------------------------------------------------------------------------ |
| `None`    | `None`    | `None`    | **pass-through store** — no payload wrapping; `_detached` holds raw `(k, v)`. Byte-for-byte match with pre-P-5.    |
| `X`       | `None`    | `None`    | `self._k_codec = self._v_codec = X` (the shorthand).                                                              |
| `None`    | `K`       | `V`       | split: `self._k_codec = K`, `self._v_codec = V`. Both must be non-`None`.                                         |
| `X`       | any non-`None` | any       | **raise** — `codec=` shorthand conflicts with explicit side kwargs.                                              |
| any       | `K`       | `None`    | **raise** — split form is "both or neither" (symmetric V=`None` case also raises).                                |

Block-size precondition: **after** constructor resolution, every
effective codec must satisfy `codec.block_size == store.block_size`.
Applies uniformly:

- Shorthand path (`codec=X`): assert `X.block_size == block_size` before
  assigning `self._k_codec = self._v_codec = X`.
- Split path (`k_codec=K, v_codec=V`): assert `K.block_size == block_size`
  **and** `V.block_size == block_size`.
- Pass-through path (all three codec kwargs `None`): precondition is
  trivially satisfied — no codec to check.

Matches the pre-P-5 rule the current store enforces on its single
`codec` kwarg; the only change is covering both sides in the split case
and covering the shorthand case explicitly (rather than assuming it is
obvious).

## 2. Side-level call counts

Every `register_detached(block_id, per_layer_kv)` call iterates over
`num_layers` per-layer `(K, V)` tuples:

- **Two** codec calls per layer per block: one `k_codec.encode_tensor(K)`
  and one `v_codec.encode_tensor(V)`. Not one.
- Total per `register_detached`: `2 × num_layers` encode calls.
- Across a reclaim that detaches `num_reclaim_blocks` blocks:
  `2 × num_layers × num_reclaim_blocks`.

Symmetric for `fetch_detached(block_id)` on the decode side.

Pass-through path (`codec=None, k_codec=None, v_codec=None`) calls zero
codec methods — `_detached` stores raw tuples; `fetch_detached` returns
them without codec involvement.

Counter regression-lock split across two test files:

- `tests/test_kvcodec_integration.py` uses a single shared
  `_CountingIdentityCodec` via the `codec=` shorthand; asserts total
  side calls = `2 × num_layers × num_reclaim_blocks` (K + V combined
  through one counter).
- `tests/test_prefix_store.py` uses split-mode kwargs with two distinct
  counting codec instances (`k_codec=_CountingCodec("k")`,
  `v_codec=_CountingCodec("v")`); asserts each codec saw exactly the
  expected side's tensors by `is`-identity. This is where K/V dispatch
  independence (the Q-008 resolution) is regression-locked.

## 3. Side-level `IdentityCodec` semantics

New signature (replaces pre-P-5 pair-level class under the same module
path `silica.kvcache.codec.IdentityCodec`):

```python
class IdentityCodec:
    def __init__(
        self,
        *,
        block_size: int,
        n_kv_heads: int,
        head_dim: int,
        dtype: mx.Dtype = mx.float16,
    ) -> None: ...

    def encode_tensor(self, x: mx.array) -> RawFp16Payload:
        return RawFp16Payload(resident_bytes=int(x.nbytes), t=x)

    def decode_tensor(self, payload: RawFp16Payload) -> mx.array:
        return payload.t   # no defensive copy — identity by reference

    def logical_bytes(self, num_tokens: int) -> int:
        return num_tokens * self._n_kv_heads * self._head_dim * self.dtype.size
    #      ^ one side only

    def resident_bytes(self, num_blocks: int) -> int:
        return num_blocks * self.block_size * self._n_kv_heads * self._head_dim * self.dtype.size
    #      ^ one side only
```

Invariants regression-locked in the migrated integration test:

- `payload = codec.encode_tensor(x)` has `payload.t is x` — payload field
  held by reference, no copy.
- `codec.decode_tensor(payload) is payload.t` — identity round-trip by
  tensor reference.
- `codec=None` path and `codec=IdentityCodec(...)` path produce
  byte-identical output token streams on the same prompt (the existing
  `paired_run` assertion carries forward under the new API).

Breaking signature changes vs pre-P-5:

- Drops `k_dtype` / `v_dtype` — side-level has a single `dtype`.
- Callers that used `IdentityCodec(k_dtype=..., v_dtype=...)` either
  collapse to one `codec=` shorthand (when K and V share `dtype`) or
  construct two explicit side instances via `k_codec=` / `v_codec=`.

## 4. Byte accounting

- `SyntheticPrefixBlockStore.resident_bytes()` signature unchanged
  (no args, returns total `int`). Internal sum changes from
  `sum(cb.resident_bytes for cb in coded_tuple)` to
  `sum(layer.k_resident + layer.v_resident for layer in layer_list)`
  where `k_resident` / `v_resident` come from the per-side `CodedPayload`
  (codec path) or from the raw tensor `.nbytes` (pass-through path).
- `IdentityCodec.resident_bytes(num_blocks)` returns **one side only**.
  This is the breaking change from the pre-P-5 class that summed K+V
  internally. Callers that want K+V must sum two codec-level calls, or
  — more natural — read `store.resident_bytes()` which already sums both
  sides across all stored blocks.
- Same per-side semantics for `logical_bytes(num_tokens)`.
- Pass-through path's `resident_bytes()`: raw `(k, v)` with no payload
  wrapper, sum is `sum(k.nbytes + v.nbytes for (k, v) in layer_list)`.
  Byte-identical to pre-P-5.

## 5. `_detached` internal shape

Private dataclass, not exported:

```python
@dataclass(frozen=True, slots=True)
class _DetachedLayer:
    """One layer's K/V payloads for one detached block. Internal to
    SyntheticPrefixBlockStore."""
    k: CodedPayload | mx.array   # mx.array on the pass-through path
    v: CodedPayload | mx.array
```

`self._detached: dict[int, tuple[_DetachedLayer, ...]]`.

- Codec paths: `.k` / `.v` are concrete `CodedPayload` subclasses
  (`RawFp16Payload` for `IdentityCodec`, `BlockTQPayload` for BlockTQ,
  etc.).
- Pass-through path: `.k` / `.v` are raw `mx.array`. `resident_bytes()`
  branches on `isinstance(layer.k, mx.array)` to pick `.nbytes` vs
  `.resident_bytes`.

Rejected alternative: a bare `tuple[tuple[CodedPayload | mx.array, ...],
...]`. Unreadable at call sites, no comment anchor for the pass-through
discriminant.

## 6. `RadixPrefixCache.store` property

Read-only, no setter. Return type is the non-optional
`PrefixBlockStore` — the `RadixPrefixCache` constructor already requires
`store: PrefixBlockStore` and stores it as `self._store` without a
`None` branch, so the property must reflect that:

```python
@property
def store(self) -> PrefixBlockStore:
    return self._store
```

Consumer: `MemoryBudgeter.__init__` reads `prefix_cache.store` in
P-5-A.2. Nothing in P-5-A.0 consumes it; the property lands here so
P-5-A.2 does not touch `silica/kvcache/prefix.py` again. Does not widen
the mutation surface of `RadixPrefixCache`.

**`resident_bytes()` is not on the `PrefixBlockStore` Protocol**, and
this is deliberate per `silica/kvcache/store.py:353-358` — the paged
backend does not track per-layer physical K/V residency, and adding
`resident_bytes()` to the Protocol would force `PagedPrefixBlockStore`
to choose between a wrong number and a `NotImplementedError` at the
Protocol boundary.

P-5-A.2's `MemoryBudgeter` must therefore not assume
`prefix_cache.store.resident_bytes()` exists on every concrete store.
The consumer uses a small structural capability check — either a
dedicated `SupportsResidentBytes` `Protocol` or `hasattr(store,
"resident_bytes")` with a `callable(...)` guard — and falls back to
zero prefix residency when the capability is absent (treated as the
paged-backend / capability-absent case). This belongs in P-5-A.2; the
note here is a forward-pointer so the capability boundary is not
accidentally broken when Unit 4 lands the `store` property.

## 7. Test migration

Four test modules change. The integration test (`test_kvcodec_integration.py`)
is Qwen3-0.6B cache-presence gated, so it cannot be the sole guardrail for
store-level semantics; a dedicated non-gated store-level file carries the
constructor / split-dispatch / per-side accounting invariants.

**`tests/test_prefix_store.py`** (existing file, 286 lines; add non-gated
cases for the P-5 side-level API on top of the current pre-P-5 coverage,
~15 new tests) — store-level unit coverage for the API surface that
Unit 4 introduces:

- Constructor rejections:
  - `SyntheticPrefixBlockStore(codec=X, k_codec=Y)` raises `ValueError`.
  - `SyntheticPrefixBlockStore(codec=X, v_codec=Y)` raises `ValueError`.
  - `SyntheticPrefixBlockStore(k_codec=K, v_codec=None)` and the
    symmetric V-only form raise `ValueError` with a "both or neither"
    message.
  - `codec.block_size != store.block_size` raises (symmetric for
    `k_codec.block_size` and `v_codec.block_size`).
- Pass-through path (`codec=None, k_codec=None, v_codec=None`):
  - `register_detached` → `fetch_detached` returns tensors whose
    `is`-identity equals the originals (no codec involvement, no copy).
  - `resident_bytes()` sums `sum(k.nbytes + v.nbytes for (k, v) in
    layer_list)` across all detached blocks; arithmetic compared
    against a hand-computed reference.
- Shorthand path (`codec=IdentityCodec(...)`):
  - Both sides route through the same codec instance (verified via a
    single counting codec that tallies total encode / decode calls).
  - `fetch_detached(bid)[layer_i][0] is` the original K tensor (identity
    reference round-trip through `RawFp16Payload`).
- **Split path** (`k_codec=_CountingCodec("k"), v_codec=_CountingCodec("v")`
  — two distinct instances, each tagged with an id):
  - After a `register_detached(bid, per_layer_kv)` call, `k_codec.seen_tensors`
    is exactly the K tensors (by `is`-identity) and `v_codec.seen_tensors`
    is exactly the V tensors. Cross-contamination (K going through V
    codec) would fail this assertion immediately.
  - Regression-locks the Q-008 resolution: independent K/V codecs are
    strictly independent on the dispatch path.
- Per-side `resident_bytes()`:
  - In pass-through, sum of raw `.nbytes` across K and V sides.
  - Under codec path, sum of `k_payload.resident_bytes + v_payload.resident_bytes`
    across layers and blocks.
  - Both arithmetics compared against hand-computed references.

**`tests/test_interfaces.py`** — row for `I-3 KVCodec` becomes
`I-3 VectorCodec`; the frozen attribute tuple changes from
`("block_size", "k_dtype", "v_dtype")` to `("block_size", "dtype")`;
`IdentityCodec(...)` factory drops `k_dtype` / `v_dtype` kwargs. Both
the label / factory and the attribute-set change must land together —
changing only the label leaves the attribute freeze covering the wrong
surface.

**`tests/test_kvcodec.py`** — full rewrite against the side-level
Protocol:

- `encode_tensor(x) -> RawFp16Payload` replaces `encode_block(k, v) ->
  CodedBlock`.
- `decode_tensor(payload) is payload.t` (identity by reference) replaces
  the two-tuple roundtrip.
- `resident_bytes(num_blocks)` and `logical_bytes(num_tokens)` are
  per-side; tests read `n_kv_heads * head_dim * dtype.size`, not `2 × ...`.
- Drops `CodedBlock`-specific tests (the dataclass is gone).
- Keeps D-012 idempotency and linear-scaling tests.
- End state: ~12–15 tests (down from current 15), equivalent invariant
  coverage shifted to side-level semantics.

**`tests/test_kvcodec_integration.py`** — targeted rewrite:

- `_CountingIdentityCodec` implements `VectorCodec[RawFp16Payload]` not
  pair-level `KVCodec`. Two counters: `encode_tensor_calls` and
  `decode_tensor_calls`, tallying total side invocations (K+V combined
  when one counter instance is shared via the `codec=` shorthand).
- Counter assertions double: what previously said `encode_block_calls ==
  num_layers × num_blocks` becomes `encode_tensor_calls == 2 × num_layers
  × num_blocks` (K-side + V-side flow through the same shared counter).
- Store construction switches from `codec=_CountingIdentityCodec(...)`
  to `codec=_CountingIdentityCodec(...)` (shorthand; semantics preserved —
  single counter counting K+V flow for the existing integration tests).
- K/V dispatch independence is NOT asserted here — that is the
  split-mode test in `tests/test_prefix_store.py`. The integration test
  only needs total-side-call counting to cover the Qwen3-0.6B reclaim
  path's invocation shape.
- `paired_run` byte-identity test continues to pass — `codec=None` path
  and `codec=IdentityCodec(...)` path still produce identical token
  streams on the same prompt.
- Defensive `len(store._detached) == len(store.live_block_ids())`
  asserts at Section 2 and Section 4 continue to pass unchanged —
  `_detached` is still a `dict` keyed by block_id, only the value shape
  changes (`tuple[CodedBlock, ...]` → `tuple[_DetachedLayer, ...]`).

## 8. Pre-P-5 type retirement

Deletions from `silica/kvcache/codec.py`:

- `CodedBlock` dataclass
- `KVCodec` Protocol
- pre-P-5 pair-level `IdentityCodec` class (replaced in place by the
  side-level class under the same name)

Deletions from `silica/kvcache/__init__.py`:

- `CodedBlock` and `KVCodec` from the import list and `__all__`.
- Keep `IdentityCodec` (name preserved; semantics now side-level).
- Drop the "two-generation" transitional docstring paragraph added in
  P-5-A.0.1; single generation again.

Non-test / non-module docstring references found in grep:

- `silica/models/adapter.py:93` — docstring mentions "consumed by
  KVCodec"; update to "consumed by VectorCodec" for consistency, no
  behaviour change.
- `silica/models/capabilities.py:23` — docstring mentions
  `kv_codec_compatible` (external capability field name). Leave the
  field name untouched; update only surrounding prose if reading as
  stale.

## 8a. Documentation sync (lands in this commit)

Deleting `KVCodec` / `CodedBlock` from the public API is an
interface-layer semantic change, not just a code-level migration. The
user-facing docs that reference the pre-P-5 names must also flip so the
project state stays internally consistent.

- **`plans/PLAN.md`** — PLAN §6 is the canonical frozen-interfaces table;
  its I-3 definition cannot diverge from the code. Unit 4 makes two
  in-place amendments plus a Changelog entry:

  1. **§6 I-3 definition (in-place rewrite).** Currently at
     `plans/PLAN.md:283`, reads:
     ```python
     class KVCodec(Protocol):
         def encode_block(self, k, v) -> CodedBlock: ...
         def decode_block(self, block) -> tuple[K, V]: ...
     ```
     Rewrite to the side-level `VectorCodec[P]` shape: one tensor in,
     one `CodedPayload` subclass out; pair dispatch documented as a
     store-level concern (`SyntheticPrefixBlockStore(k_codec=,
     v_codec=)`). Retain the historical pair-level sketch as a short
     parenthetical note so readers of the P-4.5 record have context,
     but the active table reflects the current code.

  2. **§10 Q-008 (in-place resolution).** Currently at
     `plans/PLAN.md:959` with three old option labels — A "hidden inside
     codec ctor" / B "split into Key+Value Protocols" / C "`from_pair`
     class method". Unit 4 ships a **fourth path that supersedes all
     three**: side-level `VectorCodec[P]` Protocol operating on a
     single tensor, plus store-level `k_codec` / `v_codec` kwargs on
     `SyntheticPrefixBlockStore` carrying the K/V pair dispatch.
     Q-008 resolution text should say exactly this, and mark the old
     A / B / C option labels as superseded — not "Option A resolved"
     (that label meant something different and would mislead readers
     who cross-reference the pre-P-5 text).

  3. **Changelog (v1.7.x) entry.** Short summary pointing at §6 and
     §10 amendments plus the code drop (`CodedBlock` / `KVCodec` names
     retired). Two sentences.

  Other pre-P-5 references in PLAN.md (e.g. D-003 rationale at line
  679, D-012 definition at line 798, P-4.5-C historical entries at
  lines 503-532) are historical records describing past state and
  stay as written; the §6 table update plus Q-008 resolution are the
  load-bearing active edits.
- **`README.md`** — grep for `I-3 KVCodec`, `encode_block`,
  `decode_block`, `CodedBlock`. Flip them to side-level terms.
  Stale P-4.5 / P-5 status lines may also need refreshing; scope the
  edit to keep the README truthful about the current state but avoid
  broader reorganization (that is README-maintenance work, not part of
  this unit).
- **`plans/P5_OPENING.md`** — one small clarification. The `_detached`
  shape sketch around line 770 describes the internal storage as
  `tuple[tuple[CodedPayload, CodedPayload], ...]`. Unit 4 refines this
  to a `_DetachedLayer` dataclass (see §5 above). Add a single
  cross-reference sentence:
  > "Unit 4 checklist (`plans/P5_A_U4_STORE_MIGRATION.md` §5) refines
  > this to a private `_DetachedLayer` dataclass for readability; the
  > behaviour is the same."
  No behavioural changes to the opening.

Scope guard: no other doc edits land in this commit. PLAN `Status` line
flips (P-5-A.0 → complete, P-5-A.1 → active) are the next commit's job.

## 9. Non-goals (explicitly out of scope for this unit)

- `MemoryBudgeter` three-mode accounting (`account_prefix_residency`
  flag, honest mode-B / mode-C residency subtraction,
  `resident_bytes_for_block_ids` helper) — P-5-A.2.
- `RadixPrefixCache.store` property's budgeter consumer wiring —
  P-5-A.2. The property itself lands here.
- Any bench-row work, `--kv-codec` CLI flag, PPL oracle — P-5-A.3 /
  P-5-C.
- `PagedPrefixBlockStore` codec hook — remains `NotImplementedError`
  stub (D-003).

## 10. Acceptance gates for the commit

- `uv run pytest tests/ -q` — full suite green, 921 → roughly same
  count (a few more K-side / V-side counter asserts may add tests; no
  existing test is removed without a side-level replacement covering
  the same invariant).
- `uv run mypy silica tests/` — clean.
- `uv run ruff check silica tests/` — clean.
- `tests/test_kvcodec_integration.py` byte-identity assertion
  (`paired_run`-equivalent) passes under the new codec API.
- Constructor rejection cases (`codec=X, k_codec=Y` combination;
  mixed-None split-kwargs) each raise `ValueError` at construction;
  regression tests added.

---

## 11. Estimated diff size

- Production code: ~200 lines
  - `silica/kvcache/codec.py`: ~−80 lines (drop pre-P-5 types),
    ~+30 lines (side-level `IdentityCodec`)
  - `silica/kvcache/store.py`: ~+60 net (constructor rules, per-side
    dispatch, `_DetachedLayer` dataclass)
  - `silica/kvcache/prefix.py`: ~+5 lines (`store` property)
  - `silica/kvcache/__init__.py`: ~−8 lines (drop retired exports)
- Test rewrites + additions: ~300 lines
  - `tests/test_prefix_store.py`: **existing file, add ~150 lines / ~15
    new tests** (constructor rejections, pass-through identity,
    shorthand dispatch, split-mode dispatch independence, per-side
    `resident_bytes` arithmetic). Pre-P-5 cases either stay (if they
    still make sense under side-level) or migrate to the new signature.
  - `tests/test_kvcodec.py`: full rewrite, ~net zero line count.
  - `tests/test_kvcodec_integration.py`: counter refactor, ~+30 lines
    (total-side-call tally under shorthand); split-mode cases live
    in `test_prefix_store.py`.
  - `tests/test_interfaces.py`: class rename + attribute-tuple change
    from `("block_size", "k_dtype", "v_dtype")` to `("block_size",
    "dtype")`.
- Docstring touch-ups in `silica/models/adapter.py` and
  `silica/models/capabilities.py`: ~2-line change.
- Doc sync (§8a): `plans/PLAN.md` Changelog entry (~10 lines),
  `README.md` terminology flips (~5-10 lines spread across the doc),
  `plans/P5_OPENING.md` one-sentence cross-reference near line 770.

Total: ~550-600 lines diff. One commit — interface + store + test
migration + doc sync are tightly coupled; splitting leaves the tree
in a non-compiling intermediate state or in a state where public names
are deleted without README / PLAN reflecting the change.
