# P-1 Acceptance — mlx-lm greedy parity

**Date:** 2026-04-16
**Model:** Qwen/Qwen3.5-0.8B
**mlx-lm version:** 0.31.2
**Result:** **PASS** — 32 tokens, greedy argmax, identical to mlx-lm reference.
**Probe:** `scripts/acceptance_p1_mlx_lm_parity.py` (exit 0).

## Acceptance criterion (PLAN.md §7 P-1)

> "Greedy decoding is token-for-token identical to the mlx-lm reference
> implementation (fixed seed, same model)."

## Methodology

Both paths share the same loaded model (`mlx_lm.load(REPO)`) to isolate any
divergence to Silica's orchestration. MLX model forwards are cache-scoped —
mutation lives inside the cache list entries, not on the module itself — so
running the reference first does not contaminate the candidate.

| Leg | Path under test | Cache |
| --- | --- | --- |
| A (reference) | `mlx_lm.generate.generate_step(prompt_arr, model, max_tokens=32)` | mlx-lm's default `make_prompt_cache(model)` |
| B (candidate) | `silica.engine.Engine.generate(prompt, SamplingParams(temperature=0.0, max_tokens=32))` wiring `Qwen3Adapter` + `SimpleKVCache` + P-0 `Sampler` greedy fast path | `SimpleKVCache.from_model(model)` |

Prompt: `"The capital of France is"` (fixed).

## Observed (cache-hit run)

```text
P-1 acceptance — mlx-lm parity (Qwen/Qwen3.5-0.8B)
  prompt: 'The capital of France is'
  max_tokens: 32
Running mlx-lm reference...
Running Silica candidate...
  reference:  [11751, 13, 198, 760, 6511, 314, 9338, 369, 11751, 13, 198, 760,
               6511, 314, 9338, 369, 11751, 13, 198, 760, 6511, 314, 9338, 369,
               11751, 13, 198, 760, 6511, 314, 9338, 369]
  candidate:  [11751, 13, 198, 760, 6511, 314, 9338, 369, 11751, 13, 198, 760,
               6511, 314, 9338, 369, 11751, 13, 198, 760, 6511, 314, 9338, 369,
               11751, 13, 198, 760, 6511, 314, 9338, 369]
  decoded ref: ' Paris.\nThe capital of France is Paris.\nThe capital of France
               is Paris.\nThe capital of France is Paris.\nThe capital of France is'
RESULT: PASS — token-for-token match (P-1 acceptance satisfied).
```

## Why this match is non-trivial

Silica's greedy path and mlx-lm's default sampler both compute
`mx.argmax(logits, axis=-1)`, but the logit-producing chain is **not** the
same:

- **mlx-lm's `generate_step`** runs a single prefill-then-decode loop
  internally, keeping the model's reference to its own cache and managing
  chunking via `prefill_step_size`.
- **Silica's Engine** is a user-land loop: `adapter.prefill` (which calls
  `silica.mlx.runner.forward`) for the prompt, then a Python `while` loop
  calling `adapter.decode_step` per step. Every cache read / write goes
  through `SimpleKVCache` → `cache_list(req_id)` → mlx-lm's per-layer
  `_BaseCache.update_and_fetch`.

Token-for-token match under two independent control flows confirms that:
- `cache_list(req_id)` hands the same list object mlx-lm would have built
  internally (D-010 clean injection — verified at Gate A, now verified at
  model scale).
- `silica.mlx.runner.forward` performs no spurious shape manipulation — the
  `(1, T, V)` → `(V,)` slicing is consistent with mlx-lm's `logits[:, -1, :]`.
- Silica's `Sampler` greedy fast path (`temperature <= 0`) reproduces the
  reference sampler exactly.
- `Qwen3Adapter`'s per-layer dispatch lines up with the heterogeneous
  `18 ArraysCache + 6 KVCache` list — any cross-wiring between DeltaNet and
  full-attention layers would immediately desynchronise the cache contents.

## Decision log update

- **PLAN.md §7 P-1 Acceptance criterion #2** (greedy parity with mlx-lm):
  **satisfied** at 2026-04-16 for Qwen3.5-0.8B / 32 tokens / seed-free greedy.
- **D-004** (borrow mlx-lm loader) and **D-010** (borrow mlx-lm forward, own
  the cache) are jointly validated at full model scale.
- Remaining P-1 acceptance items: #1 "generates text reliably" (covered by
  CLI end-to-end); #3 "profiler produces TTFT / decode tok-s / resident MB"
  (deferred — profiler hooks exist in P-0 but the CLI does not yet emit them).
