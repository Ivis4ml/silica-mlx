# P-3 Gemma4 — mlx-lm Source Survey (P-3-D0.2)

| Field         | Value                                                             |
| ------------- | ----------------------------------------------------------------- |
| Unit          | P-3-D0.2 (mlx-lm Gemma4 source + Gemma4-31B-4bit config audit)    |
| Prerequisite  | P-3-D0 probe (commit `8718bcd`); `mlx-community/gemma-4-31b-4bit` load succeeds |
| Targets       | Gemma4-31B (primary); 26B-MoE / smaller Gemma4 variants are noted |
| Date          | 2026-04-20                                                        |
| Source commit | `8718bcd` (probe-only, no silica/ changes yet)                    |

## 1. Purpose

P-3-D opened with a load probe that confirmed (i) mlx-lm does ship a Gemma4
text model, (ii) `adapter_for_repo` rejects the `gemma4` `model_type`
because no Silica adapter is registered yet, and (iii) the text_config
exposes ~35 fields — well beyond what `Qwen3_5Adapter` needs. Before
writing a `Gemma4Adapter` we need to know exactly what mlx-lm's
`Gemma4TextModel.make_cache` returns, how its forward dispatches per
layer, and whether Silica's existing `KVLayout` / `ModelCapabilities` /
`SimpleKVCache` surfaces cover the architecture without changes. This
document is that audit, with direct citations into the mlx-lm sources
read locally on 2026-04-20.

It does **not** propose new interfaces — those go into D1+. The goal is
that by the time D1 runs, no follow-up "wait, how does sharedkv work?"
question lands mid-implementation.

## 2. Scope

Sources read under `.venv/lib/python3.13/site-packages/mlx_lm/models/`:

- `gemma4.py` (92 lines) — outer multimodal `Model` wrapper; delegates
  to `gemma4_text.Model`.
- `gemma4_text.py` (666 lines) — `ModelArgs`, `Attention`,
  `DecoderLayer`, `Gemma4TextModel`, `Model`, `make_cache`, `sanitize`.
- `cache.py` — `KVCache`, `RotatingKVCache`, `BatchKVCache` (read during
  C0 for DeltaNet; re-consulted here for the rotating variants).

Plus a runtime one-shot dump of `Gemma4-31B-4bit`'s `text_config` values
(taken with the repo already downloaded, so the cost was only a model
instantiation).

## 3. Findings

### 3.1 Outer `Model` is a multimodal shell; text path is the only one Silica uses

`gemma4.py:32-53`. `Model.__init__` wraps a `gemma4_text.Model` under
`self.language_model`; `Model.__call__` forwards. The only extra
argument on the outer call is `per_layer_inputs` (optional, used by
2B/4B variants with `hidden_size_per_layer_input > 0`, **zero on 31B**).
`Model.sanitize` strips `vision_tower.*`, `multi_modal_projector.*`,
`audio_tower.*`, `embed_audio.*`, `embed_vision.*` — mlx-lm already
removes the multimodal heads during load, so Silica does **not** need
its own sanitizer for that.

### 3.2 `make_cache` returns a heterogeneous list keyed on `layer_types`

`gemma4_text.py:653-666`:

```python
def make_cache(self):
    first_kv_shared = self.args.num_hidden_layers - self.args.num_kv_shared_layers
    caches = []
    for i in range(first_kv_shared):
        if self.args.layer_types[i] == "full_attention":
            caches.append(KVCache())
        else:
            caches.append(
                RotatingKVCache(
                    max_size=self.args.sliding_window,
                    keep=0,
                )
            )
    return caches
```

Two things worth pinning:

1. **Cache list length** is `num_hidden_layers - num_kv_shared_layers`,
   **not** `num_hidden_layers`. For Gemma4-31B
   `num_kv_shared_layers == 0` so length is 60, matching layers. For
   other variants (the Gemma config surface supports shared KV) the
   forward pads with `None` and routes shared-KV layers through
   `previous_kvs` (see §3.5) — so any adapter that iterates
   `zip(self._model.layers, cache_list)` must tolerate a short cache
   list.
2. **Cache type depends on `layer_types[i]`** — `full_attention`
   → `KVCache` (auto-growing, non-rotating); `sliding_attention`
   → `RotatingKVCache(max_size=sliding_window, keep=0)`. The sliding
   cache keeps at most `sliding_window` tokens (1024 on 31B) and drops
   older positions on rotation.

### 3.3 Forward dispatches per-layer by layer_type; masks are per-type

`gemma4_text.py:508-566`. `Gemma4TextModel.__call__`:

- Embeds the input tokens, scales by `hidden_size**0.5`, and walks
  `zip(self.layers, cache, masks, self.previous_kvs, per_layer_inputs)`.
- `masks = self._make_masks(h, cache)` (lines 494-506) builds **per-type**
  masks once and reuses them: full layers get a standard causal mask;
  sliding layers get `create_attention_mask(h, c, window_size=self.window_size)`.
- The layer module itself (`DecoderLayer.__call__`, lines 324-380) reads
  `self.layer_type` indirectly via `self.self_attn.is_sliding` / per-layer
  RoPE and does not require the caller to tag the call.
- **Call signature** for each layer: `layer(h, mask, c, per_layer_input=..., shared_kv=..., offset=...)`. The last three are Gemma4-specific — they default to `None` when per-layer-input / kv-sharing are not active, which is exactly the 31B case.

Implication: a Silica-side `Gemma4Adapter.prefill` that passes the
cache list verbatim through `mlx.runner.forward(model, tokens, cache_list)`
will work for the 31B / 26B variants (no per-layer-input, no shared-kv).
The smaller variants (2B / 4B with per-layer inputs) would need the
adapter to thread `per_layer_inputs` through — deferred until we actually
target one of them.

### 3.4 Two distinct KV shapes per attention kind

`gemma4_text.py:176-224`. `Attention.__init__` branches:

- `sliding_attention`: `head_dim = config.head_dim` (256 on 31B),
  `n_kv_heads = config.num_key_value_heads` (16 on 31B). Full K and V
  projections.
- `full_attention` with `attention_k_eq_v=True` (31B): `head_dim =
  config.global_head_dim` (512 on 31B), `n_kv_heads =
  config.num_global_key_value_heads` (4 on 31B). **K == V** — the same
  projected tensor backs both (no `v_proj` weight), though the cache
  still stores a `(keys, values)` tuple by reference identity.

Per-token KV bytes, bf16, Gemma4-31B, single request:

| Layer kind | tensors | n_kv × head_dim | bytes/token/layer | layer count | total bytes/token |
| --- | ---: | ---: | ---: | ---: | ---: |
| `sliding_attention` | 2 (K, V) | 16 × 256 | 8 192 | 50 | 409 600 |
| `full_attention` (k_eq_v) | 1 (K = V) | 4 × 512 | 4 096 | 10 | 40 960 |

Sliding layers cap at `sliding_window=1024` tokens by rotation; full
layers grow unboundedly with context. This matters for
`MemoryBudgeter.bytes_per_token` — §4 returns to it.

### 3.5 Shared KV is declared but inactive at 31B

`gemma4_text.py:424-433`. `Gemma4TextModel` builds a
`self.previous_kvs` map so the last `num_kv_shared_layers` layers
re-use KV from an earlier same-type layer. At 31B `num_kv_shared_layers
== 0` so `previous_kvs` is the identity. Smaller variants may enable
it; the surface (short cache list + `None` padding in the forward)
means a Silica adapter targeting 31B does not have to implement
anything, but must not assume `len(make_cache()) == num_hidden_layers`.

### 3.6 RoPE + softcap + mask are handled inside mlx-lm

- RoPE (`gemma4_text.py:214-224`) is per-layer-type: sliding uses
  `rope_theta=10000.0` default, full uses `rope_theta=1e6` with
  `partial_rotary_factor=0.25` on 31B. `initialize_rope` builds the
  rotation tables inside each `Attention`; no Silica-side RoPE work.
- Logit softcap (`gemma4_text.py:597-598`) is applied inside
  `Model.__call__` after `lm_head`. `final_logit_softcapping=30.0` on
  31B. Silica sees the already-softcapped logits — `Sampler` consumes
  them unchanged.
- Attention mask is built inside `_make_masks`; the adapter passes
  `cache_list` through and mlx-lm sizes the mask against
  `cache.offset`.

### 3.7 `capabilities()` for Gemma4-31B

`attention_pattern().per_layer` is built by mapping `layer_types[i]`:

- `"sliding_attention"` → `AttentionKind.SLIDING`
- `"full_attention"` → `AttentionKind.GLOBAL`

Giving `attention_kinds = frozenset({SLIDING, GLOBAL})`,
`has_recurrent_state = False`, `has_moe = False`. A `Gemma4Adapter`
built this way would fail the C3c capability gate — current supported
set is `{GLOBAL, HYBRID_DELTANET}` — so **batched execution stays
locked until Silica grows sliding-window batching** (PLAN §Q-013;
mlx-lm has no `BatchRotatingKVCache`, only `BatchKVCache`).

## 4. Concrete Gemma4-31B-4bit values

Extracted 2026-04-20 from `model.language_model.args.text_config` (full
dump recorded in chat):

| Field | Value |
| --- | --- |
| `model_type` (outer) | `"gemma4"` |
| `model_type` (text_config) | `"gemma4_text"` |
| `num_hidden_layers` | 60 |
| `layer_types` | `[S×5, F, S×5, F, …]` — 50 sliding + 10 full, strict 5:1 pattern |
| `hidden_size` | 5376 |
| `intermediate_size` | 21504 |
| `num_attention_heads` | 32 |
| `num_key_value_heads` (sliding) | 16 |
| `num_global_key_value_heads` (full) | 4 |
| `head_dim` (sliding) | 256 |
| `global_head_dim` (full) | 512 |
| `sliding_window` | 1024 |
| `sliding_window_pattern` | 6 (5 sliding + 1 full) |
| `attention_k_eq_v` | True (only on full layers) |
| `num_kv_shared_layers` | 0 |
| `hidden_size_per_layer_input` | 0 (inactive) |
| `num_experts` | None (dense) |
| `enable_moe_block` | False |
| `use_double_wide_mlp` | False |
| `use_bidirectional_attention` | `"vision"` (text path stays causal) |
| `final_logit_softcapping` | 30.0 |
| `tie_word_embeddings` | True |
| `vocab_size` | 262144 |
| `max_position_embeddings` | 262144 |
| `dtype` | bf16 |

## 5. Implications for Silica (D-series scope)

### 5.1 `KVLayout` needs a per-kind view before MoE / bench work

Current `silica/models/adapter.py::KVLayout` is `(num_layers, n_kv_heads,
head_dim, dtype)` — a single shape covers all layers. Gemma4-31B has
**two KV shapes coexisting in one model**. For the single-request path
this is a cosmetic issue (`make_cache` produces correct cache objects
per layer and the adapter forwards them verbatim); for
`MemoryBudgeter.bytes_per_token` and the P-4 bench's KV-residency
accounting it is a correctness issue — `2 × n_kv_heads × head_dim × dtype_size`
would under- or over-count depending on which kind's numbers got
picked.

Options, in ascending invasiveness:

- **(a) First-kind-wins summary + caveat comment.** The adapter
  populates `KVLayout` with the *sliding* layer's shape (majority, 50
  of 60 layers on 31B); `MemoryBudgeter` lives with a ~20 % over-count
  from ignoring the full layers' smaller-kv footprint. This matches
  today's `Qwen3_5Adapter._build_kv_layout` approach of "read the
  first full-attention layer, ignore DeltaNet layers". Smallest
  change; caveat documented in code.
- **(b) Aggregate KV bytes.** `KVLayout` grows a
  `bytes_per_token_global: int` scalar (sum across layers, independent
  of per-kind composition). `MemoryBudgeter.bytes_per_token` consumes
  it directly. Preserves the single-`KVLayout` shape but stops pretending
  one `(n_kv_heads, head_dim)` covers everything.
- **(c) Per-kind decomposition.** `KVLayout` gains
  `n_kv_heads_by_kind` and `head_dim_by_kind` dicts. Most explicit;
  opens the door to per-kind scheduling later (e.g. sliding layers'
  bounded residency). Most invasive on consumers.

Decision deferred until D1 picks a path — (a) is the "smallest viable
next commit", (b) is the "correct for bench" option. (c) should be
gated on Q-013 landing anyway.

### 5.2 Single-request `Gemma4Adapter` is a small new file

Modeled on `silica/models/qwen3.py`:

- Store `model`, `tokenizer`, `kv_manager` (a `SimpleKVCache` wrapping
  `model.make_cache()`).
- `build` / `kv_layout` / `attention_pattern` / `capabilities` /
  `tokenizer` / `prefill` / `decode_step` / `make_batch_cache` — all
  minimal; `make_batch_cache` is dead code today because the gate
  rejects `SLIDING`, but it needs to exist for Protocol symmetry once
  C3c-style gate lifting lands for sliding.
- `attention_pattern()`: map `text_config["layer_types"]` to `SLIDING`
  / `GLOBAL`.
- `capabilities()`: `capabilities_from_attention_pattern(pattern)` —
  the helper handles `frozenset({SLIDING, GLOBAL})` and returns
  `has_recurrent_state=False` correctly.
- `_build_kv_layout`: option (a) summary from the first sliding
  layer's values; noted in comment.
- `factory._ADAPTERS["gemma4"] = _build_gemma4` — analogous to
  `_build_qwen3_5`.

Estimated ~150 lines + ~80 lines of tests, comparable to Qwen3Adapter
at introduction.

### 5.3 Batched path is blocked on Q-013 / sliding-window-batched KV cache

> **Correction (P-3-D2.0, 2026-04-20).** The first sentence below —
> "mlx-lm ships `BatchKVCache` ... but **no `BatchRotatingKVCache`**"
> — is **wrong**. `mlx_lm/models/cache.py` defines a complete
> `BatchRotatingKVCache` (lines 1100-1444) with the full batched
> surface (`update_and_fetch` / `prepare` / `finalize` / `filter` /
> `extend` / `extract` / `merge` / `make_mask`). See
> `docs/P3_BATCH_ROTATING_KV_SURVEY.md` for the authoritative audit
> and the revised D2 / D3 scope (which shrinks from "implement the
> container" to "wire the factory + lift the gate"). The rest of §5.3
> is preserved below as the historical reasoning; the §5.4 roadmap is
> superseded by `P3_BATCH_ROTATING_KV_SURVEY.md` §6.

mlx-lm ships `BatchKVCache` (adopted by C3a/b/c) but **no
`BatchRotatingKVCache`**. A Gemma4 batched admission would have to:

- Either keep `KVCache` / `RotatingKVCache` single-request shapes and
  drive `ContinuousBatcher.step()` at B=1 only — i.e. "multi-request
  via serialisation". Pointless.
- Or implement the sliding variant of `BatchKVCache` with per-row
  rotation + left-padding alignment. This is the PLAN §Q-013 item; no
  prior work lands it today.

`ArraysCache` (DeltaNet's path) does not help here — it is for
recurrent state, not rotating windows.

Therefore D-series should land D1 (single-request `Gemma4Adapter`)
first as the minimal utility ("can we load and run 31B?"), and **treat
Q-013 / sliding batched as its own major unit** (call it D2 or a new
letter) before attempting a C3c-style gate lift for `SLIDING`.

### 5.4 Order of P-3-D sub-units (recommendation, not commitment)

| Sub-unit | Scope | Depends on |
| --- | --- | --- |
| **D1** | Single-request `Gemma4Adapter` on Gemma4-31B (single-request smoke + factory registration + capabilities test). KVLayout option (a). Batched deferred. | D0.2 survey (this doc). |
| **D2** | `BatchRotatingKVCache` research + implementation (may overflow into a standalone mini-phase). | D1 green; PLAN §Q-013. |
| **D3** | C3c-style gate lift for `SLIDING` + hybrid batched smoke on Gemma4-31B. | D2. |
| **D4** | `KVLayout` / `MemoryBudgeter` correctness on per-kind KV shapes (option (b) or (c)). | D3 so bench reads real batched figures. |
| **D5** | Gemma4-26B MoE variant (`num_experts > 0`, `enable_moe_block=True`). | D1 + P-3-E MoE prerequisites (expert routing in adapter). |

## 6. Open items (P-3-D local)

These are scoped to the D track; do not enter the global `Q-NNN`
space (same convention as P-3-C `C-open-*`).

- **D-open-1** — Which `KVLayout` variant ((a) first-kind summary,
  (b) aggregate bytes, (c) per-kind decomposition) does D1 commit to?
  Leaning (a) for speed; (b) if bench starts reading
  `MemoryBudgeter.bytes_per_token` on Gemma4 before D4.
- **D-open-2** — Does `Gemma4Adapter` target **all** Gemma4 variants
  (1B / 2B / 4B / 26B-MoE / 31B) or only 31B? Per-layer-input support
  (2B / 4B) is separate code; MoE (26B) overlaps P-3-E. D1's first
  commit should be 31B-only, with explicit `NotImplementedError` when
  `hidden_size_per_layer_input > 0` or `enable_moe_block is True`.
- **D-open-3** — `num_kv_shared_layers > 0` variants: adapter should
  tolerate `len(make_cache()) < num_hidden_layers`. Is there an
  existing Gemma4 variant where this is non-zero, and does any current
  adapter (`Qwen3*`) implicitly assume equal length somewhere Silica
  can break? Spot-check during D1.

## 7. References

- mlx-lm `gemma4.py` — outer multimodal wrapper
- mlx-lm `gemma4_text.py:15-70` (ModelArgs / post_init), 176-267
  (Attention), 270-380 (DecoderLayer), 383-566 (Gemma4TextModel),
  569-666 (Model, make_cache, sanitize).
- mlx-lm `cache.py` — `RotatingKVCache` (§`KVCache`-adjacent); no
  `BatchRotatingKVCache` present.
- Silica — `silica/models/qwen3.py` (adapter template),
  `silica/models/adapter.py::KVLayout`,
  `silica/scheduler/batcher.py::_SUPPORTED_ATTENTION_KINDS`.
- PLAN — §Q-013 (sliding-window batching), §P-3 deliverables, D-015
  (attention kinds + per-layer routing).
- P-3-D0 probe output (2026-04-20, commit `8718bcd`) — confirms
  `model_type='gemma4'`, 60 layers, two KV shapes, dense (no MoE).
