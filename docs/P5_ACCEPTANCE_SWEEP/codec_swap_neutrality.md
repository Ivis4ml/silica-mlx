# P-5 Acceptance (1) — Codec-swap neutrality by inspection

**Gate (PLAN.md §7 P-5 Acceptance item 1):** "Switching the codec
requires no change to the scheduler or model adapter."

**Inspection date:** 2026-04-24 (commit `ec8ed27` base, v1.7.3).
**Method:** static grep of the silica source tree (`silica/**`) for
runtime codec-type dispatch and for codec imports from scheduler /
model-adapter subsystems. Complements the runtime evidence in
`silica/bench/runner.py` (which swaps codecs per scenario through
the `CodecSpec` registry without touching scheduler or adapter
code paths) and `tests/test_kvcodec_integration.py` (which
end-to-end exercises BlockTQ / IdentityCodec on the same scheduler
instance).

## (1.1) Zero runtime codec-type dispatch across `silica/**`

Grep pattern (matches `isinstance`, `type(x) ==`, and
`__class__.__name__` forms that would branch on concrete codec
classes):

```text
isinstance\(.*[Cc]odec|isinstance\(.*[Pp]ayload|__class__\.__name__|type\(.*[Cc]odec\)
```

Scope: `silica/scheduler`, `silica/models`, `silica/engine`,
`silica/kvcache`, `silica/weights`, `silica/core`, `silica/mlx`,
`silica/llm`, `silica/speculative`, `silica/chat`, `silica/server`.

**Result: zero matches.** No file in any of these subsystems
branches on a concrete codec class name at runtime. All codec
polymorphism is carried by the `VectorCodec[P]` Protocol (duck-typed
`encode_tensor` / `decode_tensor` / `logical_bytes` /
`resident_bytes` method surface) and by the `SyntheticPrefixBlockStore`
wrapper that holds the codec opaquely.

## (1.2) Zero concrete-codec imports from scheduler / adapter subsystems

Grep pattern (inbound imports from `silica.vq.*` or the codec
Protocol module):

```text
^from silica\.vq|^import silica\.vq|^from silica\.kvcache\.codec
```

Scope: `silica/scheduler`, `silica/models`, `silica/engine`,
`silica/weights`.

**Result: zero matches.** Neither scheduler nor model adapter nor
engine nor weight-provider subsystems import the `silica.vq`
codec families (`BlockTurboQuantMSE` / `TurboQuantMSE` /
`RaBitQ1Bit` / `ExtRaBitQ`) or the `VectorCodec[P]` Protocol module
`silica.kvcache.codec` directly. Concrete codec types never cross
these subsystem boundaries.

Actual inbound imports from `silica.kvcache` (canonical `grep -nE
"from silica\.kvcache" silica/engine/**/*.py silica/scheduler/**/*.py
silica/models/**/*.py silica/weights/**/*.py`):

- `silica/engine/__init__.py:53` — `from silica.kvcache.manager import KVHandle, KVManager`
- `silica/engine/__init__.py:54` — `from silica.kvcache.prefix import RadixPrefixCache`
- `silica/scheduler/batcher.py:53` — `from silica.kvcache.prefix import RadixPrefixCache`
- `silica/scheduler/budget.py:62` — `from silica.kvcache.prefix import RadixPrefixCache`
- `silica/models/adapter.py:47` — `from silica.kvcache.manager import KVHandle`
- `silica/models/factory.py:26` — `from silica.kvcache.simple import SimpleKVCache`
- `silica/models/{gemma4, qwen3, qwen3_5, gemma4_moe, qwen3_5_moe}.py` — `from silica.kvcache.manager import KVHandle` and/or `from silica.kvcache.simple import SimpleKVCache`
- `silica/weights/**` — no imports from `silica.kvcache.*`

All five import targets (`RadixPrefixCache`, `KVHandle`, `KVManager`,
`SimpleKVCache`, plus the module paths themselves) are
codec-agnostic container / manager types. None of them is a codec
class or a codec-Payload class, and none re-exports one.

## (1.3) Codec method surface the scheduler actually calls

Every byte-accounting call the scheduler makes goes through the
store (`PrefixBlockStore` / `SyntheticPrefixBlockStore`), which
exposes a polymorphic method surface whose return values are plain
`int` bytes:

- `silica/scheduler/budget.py:149` — `prefix_cache.store.resident_bytes()`
- `silica/scheduler/budget.py:156` — `store.resident_bytes_per_block()`

The store delegates internally to whichever codec it holds (via
`codec.resident_bytes(num_blocks)` / `codec.logical_bytes(num_blocks)`
if the codec exposes them, otherwise via a pass-through fp16 byte
formula — `silica/kvcache/store.py`). The scheduler never observes
the codec type; it observes only the integer residency report.

`silica/engine/__init__.py:175-176` similarly reads scalar
`budget.resident_bytes` / `budget.logical_bytes` for metrics
emission, not any codec-typed payload.

## (1.4) Mentions of concrete codec class names and the codec noun — classified

Two independent greps, each exactly reproducible. Scope for both:
`silica/scheduler/**` and `silica/models/**`.

**Grep A — concrete codec / payload class names.** Pattern:

```text
IdentityCodec|BlockTurboQuantMSE|BlockTQCodec|TurboQuantMSE|RaBitQ1Bit|ExtRaBitQ
```

Result: **3 hits across 2 files**, all docstring references:

| file:line | hit | classification |
| --- | --- | --- |
| `silica/scheduler/budget.py:152` | ``Under ``IdentityCodec`` `` | docstring — `account_prefix_residency=True` mode-B semantics |
| `silica/scheduler/budget.py:293` | `fp16 under IdentityCodec, compressed under` | docstring — `_prefix_resident_bytes()` return semantics |
| `silica/models/qwen3.py:159` | ``BlockTurboQuantMSE.encode_tensor`` to reject legitimate | docstring — historical note on the pre-P-5-A.3c bf16 dtype bug fix |

**Grep B — `VectorCodec` Protocol name + lowercase `codec` / `codecs`
word.** Pattern:

```text
VectorCodec|\bcodec\b|\bcodecs\b
```

Result: **9 hits across 3 files**, all docstring references:

| file:line | hit | classification |
| --- | --- | --- |
| `silica/scheduler/budget.py:154` | `codecs it is honest compressed bytes (§4.7 mode C)` | docstring — companion to budget.py:152, explains mode-C |
| `silica/scheduler/budget.py:165` | `bound at fp16 regardless of codec` | docstring — active-KV upper-bound invariant |
| `silica/scheduler/budget.py:420` | `reflecting codec compression` | docstring — `_store_resident_bytes_per_block()` semantics |
| `silica/scheduler/budget.py:424` | `synthetic pass-through path — no codec` | docstring — pass-through fallback |
| `silica/scheduler/budget.py:429` | `using honest per-block bytes under a compressed codec` | docstring — companion to 420 |
| `silica/scheduler/budget.py:433` | `at fp16 regardless of codec` | docstring — active-KV invariant restatement |
| `silica/models/adapter.py:93` | ``consumed by ``VectorCodec`` `` | docstring on `KVLayout` — describes downstream consumer, not an import |
| `silica/models/qwen3.py:160` | `bf16 K/V on the prefix-cache codec path` | docstring — companion to qwen3.py:159 |
| `silica/models/qwen3.py:192` | `codec construction must use the same` (dtype) | docstring — records that the layer-dtype inference feeds codec factories downstream |

Total (Grep A ∪ Grep B): 12 docstring lines across 3 files. None is
a runtime branch. Removing every mention would not change behaviour;
they are reader-facing semantic notes. A future rename of
`IdentityCodec` or `BlockTurboQuantMSE` would require a doc-comment
sweep in these files but would not require any code change under the
(1) gate.

## (1.5) Codec-swap behavioural evidence (external, cross-referenced)

In addition to the static inspection above, the following already-
landed artifacts exercise codec swap on a live scheduler instance
without any scheduler or model-adapter source change. The two
subsystems exercised in these tests are the ones the gate covers;
concrete codec classes are named only in the bench harness / codec
registry / codec-family unit tests.

- **`tests/test_kvcodec_integration.py`** (P-4.5-C.1) — drives
  `ContinuousBatcher` + `RadixPrefixCache` end-to-end through
  `SyntheticPrefixBlockStore(codec=...)` with `IdentityCodec` and a
  local `_CountingIdentityCodec` Protocol-conforming wrapper, plus
  the `codec=None` pass-through path. BlockTQ is **not** exercised
  in this file; it does, however, prove that two
  Protocol-conforming codecs can be swapped in on the same
  scheduler instance with zero scheduler source change — which is
  exactly the neutrality claim (1) asserts.
- **`tests/test_bench_workload_kv_codec.py`** — verifies that
  `kv_codec="block_tq_b64_b4"` resolves to a live
  `BlockTurboQuantMSE` installed on the bench runner's `Workload`
  (test function docstring at line 213:
  `test_maybe_build_prefix_cache_block_tq_installs_block_tq_codec`,
  "`kv_codec=\"block_tq_b64_b4\"` installs `BlockTurboQuantMSE`").
  This binds the string-id codec-swap path end-to-end through
  `ContinuousBatcher` without any scheduler / adapter code change.
- **`tests/test_prefix_hit_decode_speed_gate.py`** (P-5-A.3c) —
  drives the `qwen3-0.6b-prefix-hit-decode-{fp16, block-tq-b64-b4}`
  paired rows through the `ContinuousBatcher` + `RadixPrefixCache`
  path; `ContinuousBatcher` source is unchanged between the fp16
  and BlockTQ arms, and the file's module docstring documents
  `BlockTurboQuantMSE` explicitly as the P-5-A.3c-landed BlockTQ
  codec on the production hot path.
- **`silica/bench/runner.py::_build_prefix_cache`** — constructs a
  `SyntheticPrefixBlockStore(block_size, k_codec=..., v_codec=...)`
  from a scenario-level `CodecSpec` registry entry. The runner
  swaps codecs per scenario row without importing or branching on
  any concrete codec class.
- **Scenario coverage** — `silica/bench/scenarios.py` declares
  `qwen3-0.6b-wikitext-ppl-fp16` (IdentityCodec via the
  pass-through store path) and
  `qwen3-0.6b-wikitext-ppl-{tq-mse-b4, block-tq-b64-b4,
  ext-rabitq-b4}` + `-vqbench-aligned` plus the three
  prefix-hit decode rows; each row uses the same `BenchRunner`
  code path with a different `CodecSpec`. `CodecSpec.factory` is
  the only hook that names a concrete codec class, and it lives
  in `silica/bench/codec_registry.py` — never in `silica/scheduler`
  or `silica/models`.

## Conclusion

**Acceptance (1) is structurally satisfied.** Switching between
`IdentityCodec`, `BlockTurboQuantMSE`, `TurboQuantMSE`,
`RaBitQ1Bit`, and `ExtRaBitQ` does not require — and the tree does
not contain — any change in `silica/scheduler/**` or
`silica/models/**`. The neutrality is enforced at the import graph
level (neither subsystem imports the codec families) and at the
runtime-dispatch level (zero `isinstance` / type-name branches).
Docstring mentions of concrete codec class names are
non-dispatching reader notes and do not affect the gate.

Ready to flip PLAN.md §7 P-5 Acceptance item (1) `[ ]` → `[x]` at
v1.7.4.
