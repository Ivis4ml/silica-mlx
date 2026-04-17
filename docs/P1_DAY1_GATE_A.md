# P-1 Day-1 Gate A — D-010 external cache injection

**Date:** 2026-04-16
**mlx-lm version:** 0.31.2
**Result:** **ACCEPTS** — clean external-cache injection. R-6 does not trigger.
**Probe:** `scripts/probe_mlx_lm_cache.py` (exit 0).

## Question

PLAN.md §7 P-1 / D-010 requires a day-1 smoke test before any other P-1
deliverable: does `mlx_lm.generate.generate_step(prompt_cache=...)` accept an
externally constructed cache object? The answer determines whether Silica's
`SimpleKVCache` (P-1) can inject directly or whether P-1 must monkey-patch
`mlx_lm.models.*` forward paths (risk R-6).

## Source evidence

All line numbers below reference the installed wheel
(`.venv/lib/python3.13/site-packages/mlx_lm/`).

1. **Public knob exists and is typed `Optional[Any]`.**
   `generate.py:315` —
   ```python
   def generate_step(
       ...,
       prompt_cache: Optional[Any] = None,
       ...,
   ) -> Generator[Tuple[mx.array, mx.array], None, None]:
   ```

2. **External cache is routed, not overridden.**
   `generate.py:371–375` constructs a default cache only when none is supplied:
   ```python
   if prompt_cache is None:
       prompt_cache = cache.make_prompt_cache(model, max_kv_size=max_kv_size)
   ```

3. **The cache is passed through to the model forward pass unchanged.**
   `generate.py:388–394` —
   ```python
   def _model_call(input_tokens, input_embeddings):
       if input_embeddings is not None:
           return model(input_tokens, cache=prompt_cache, input_embeddings=...)
       else:
           return model(input_tokens, cache=prompt_cache)
   ```
   No cache adapter / wrapping / copy. mlx-lm does call
   `maybe_quantize_kv_cache(prompt_cache, ...)` (`generate.py:418, 441`) but that
   is a no-op when `kv_bits is None` (`generate.py:299–301`).

4. **Per-layer cache contract is duck-typed, not an ABC.**
   `models/cache.py:127` defines `_BaseCache` as a plain class with defaults.
   mlx-lm only depends on the following method / property surface per entry:
   | Member | Purpose | Used at |
   | --- | --- | --- |
   | `update_and_fetch(keys, values) -> (k, v)` | hot-path accumulate + return full history | attention forward |
   | `state` (getter + setter) | save / load, `mx.eval` sync | `generate.py:442`, `cache.py:43–85` |
   | `meta_state` (getter + setter) | per-class metadata for save / load | `cache.py:54, 78` |
   | `is_trimmable()`, `trim(n)` | speculative rollback / re-admission | `cache.py:88–111` |
   | `size()` | current sequence length | accounting |
   | `nbytes` | memory accounting | `_BaseCache` default raises; subclasses implement |
   | `empty()` | state check | subclasses implement |
   | `make_mask(N, return_array, window_size)` | attention-mask factory | optional, per-layer |
   | `from_state(state, meta_state)` | deserialize | `cache.py:169–175` |

## Probe evidence

`scripts/probe_mlx_lm_cache.py` builds a toy one-layer model and a
`_TracedKVCache(KVCache)` that counts `update_and_fetch` invocations, then runs
`generate_step(prompt=[1,2,3], max_tokens=4, prompt_cache=[traced])`.

Observed on 2026-04-16:

```text
Gate A probe — mlx-lm 0.31.2
  generated tokens: [0, 0, 0, 0]
  TracedKVCache.update_and_fetch calls: 6
  TracedKVCache.offset after generation: 7
RESULT: ACCEPTS — D-010 clean path. R-6 does not trigger.
```

Breakdown — the external cache receives every expected call:
- 1 call during prefill chunk (`prompt[:2]`, 2 tokens)
- 1 call for the final prefill token (`_step(prompt[2:])`, 1 token)
- 4 calls during decode (one per yielded token, `max_tokens = 4`)
- Total `offset = 2 + 1 + 4 = 7`, matching the observed value.

## Implications for Silica

1. **P-1 `SimpleKVCache` is a clean duck-type.** It conforms to the surface
   above; no inheritance from mlx-lm classes, so D-009 (MLX-native, no PyTorch
   runtime dep) is preserved.
2. **D-012 `resident_bytes` maps 1:1 to `nbytes`.** The canonical measurement
   adopted in P-0 `IdentityCodec.resident_bytes` already matches mlx-lm's
   accounting — no reconciliation needed at P-1.
3. **D-015 `StateDelta.recurrent_bytes()` is orthogonal to the KV path.**
   mlx-lm's `_BaseCache` contract covers KV-attention layers only; recurrent
   (DeltaNet) state remains adapter-owned per D-015, carried through the
   `prefill` / `decode_step` return tuple. The two ownership domains do not
   overlap.
4. **`is_trimmable()` / `trim(n)` satisfies the P-7 speculative rollback API.**
   Silica's `KVManager.rollback` can delegate to mlx-lm's trim semantics on the
   `SimpleKVCache` path; no parallel implementation required for P-1.

## Side finding relevant to Gate B

`mlx_lm/models/qwen3_5.py:304–305` already ships a hybrid-DeltaNet cache
factory matching D-015's `AttentionPattern.HYBRID_DELTANET`:

```python
def make_cache(self):
    return [ArraysCache(size=2) if l.is_linear else KVCache() for l in self.layers]
```

This removes the "mlx-lm does not carry Qwen3.5 forward cleanly" arm of R-8 at
the source level. Gate B remains required to confirm the runtime behaviors: (a)
checkpoint loads text-only, (b) MTP head can be disabled, (c) tokenizer
round-trips match the HF reference.

## Decision log update

- **D-010 status:** day-1 smoke test passed at P-1 entry. Silica P-1 proceeds
  on the clean-injection path.
- **R-6 status:** does not trigger; P-1 cost estimate unchanged.
- **R-8 partial status:** "mlx-lm Qwen3.5 support gap" sub-risk reduced — the
  model module exists and factories are in place; remaining verification is
  Gate B (runtime load / MTP disable / tokenizer parity).
