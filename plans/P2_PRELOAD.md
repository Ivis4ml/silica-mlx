# P-2 Preload — Qwen3-0.6B as the dev-loop model

**Date:** 2026-04-17
**mlx-lm version:** 0.31.2
**Repo:** `Qwen/Qwen3-0.6B`
**Result:** **PASS** — all four sub-checks green.
**Probe:** `scripts/probe_p2_preload.py` (exit 0).

## Why a preload check

P-2 opening v2.1 fixed Qwen3-0.6B (plain KV attention) as the batched
dev-loop model, with Qwen3.5-0.8B's hybrid DeltaNet path deferred to P-3.
Unit 16a's P-1 oracle-parity test needs a concrete baseline — "batched
output equals P-1 single-request output on this fixed prompt" — so this
probe establishes that baseline the same way Gate B did for P-1.

## Checkpoint facts

| Property | Value |
| --- | --- |
| Total repo size on HF | 1.52 GB |
| `model_type` | `qwen3` |
| `architectures` | `Qwen3ForCausalLM` |
| `num_hidden_layers` | 28 |
| `num_attention_heads` | 16 |
| `num_key_value_heads` | 8 (GQA) |
| `hidden_size` | 1024 |
| `head_dim` | **128** (NOT `hidden_size / num_heads = 64`) |
| `vocab_size` | 151936 (tokenizer exposes 151643) |
| `sliding_window` | None (`use_sliding_window = False`) |
| `text_config` / `vision_config` | absent |
| `mtp_*` | absent |

## Naming-convention drift from Qwen3.5 (adapter fix)

mlx-lm uses **two different naming schemes** for attention-related state:

| Concept | Qwen3.5 (hybrid) | Plain Qwen3 |
| --- | --- | --- |
| KV head count | `self_attn.num_key_value_heads` | `self_attn.n_kv_heads` |
| `head_dim` location | `self_attn.head_dim` | `model.args.head_dim` |
| Top-level model args | `args.text_config` (nested dict) | `args.*` (flat) |

This naming drift motivated splitting the single ``Qwen3Adapter`` class
(which covered both Qwen3.5 and Qwen3) into **per-generation adapters**
on the same day as this preload:

- ``silica.models.qwen3.Qwen3Adapter`` — plain Qwen3 only, reads
  ``self_attn.n_kv_heads`` + flat ``args.hidden_size`` + ``args.head_dim``.
- ``silica.models.qwen3_5.Qwen3_5Adapter`` — Qwen3.5 hybrid only, reads
  the nested names.
- ``silica.models.factory.adapter_for_repo(repo)`` — dispatches by
  ``model.model_type``; used by the CLI.

The split ensures adding a new family (Qwen4, DeepSeek, Kimi, GLM,
MiniMax) is a new module + one dict registration, never a conditional
inside an existing adapter. Unit tests live in two files
(``tests/test_qwen3_adapter.py`` and ``tests/test_qwen3_5_adapter.py``)
plus a dispatcher test (``tests/test_models_factory.py``).

## Probe output

```text
P-2 preload probe — mlx-lm 0.31.2, repo Qwen/Qwen3-0.6B
Loading model (first run downloads ~1.52 GB to ~/.cache/huggingface)...
  adapter: qwen3 / 28 layers / vocab=151643

(a) ModelConfig + KVLayout populated from plain-Qwen3 args:
    PASS
(b) AttentionPattern is all GLOBAL (pure KV, no hybrid layers):
    PASS — 28/28 GLOBAL; 16a scope-guard will accept
(c) Tokenizer parity with HF AutoTokenizer:
    PASS
(d) P-1 Engine.generate greedy baseline (oracle for Unit 16a):
    tokens: [12095, 13, 576, 6722, 315, 15344, 374, 21718, 13,
             576, 6722, 315, 17689, 374, 24081, 13]
    decoded: ' Paris. The capital of Italy is Rome. The capital of
              Spain is Madrid.'
    PASS

RESULT: PASS — Qwen3-0.6B is the P-2 dev-loop model. Unit 16a unblocked.
```

## Unit 16a oracle specification

Unit 16a's acceptance test uses this **exact** config to assert P-1
parity:

```python
REPO = "Qwen/Qwen3-0.6B"
PROMPT = "The capital of France is"
params = SamplingParams(temperature=0.0, max_tokens=16)

# Oracle (recorded above, must byte-match):
EXPECTED = [
    12095, 13, 576, 6722, 315, 15344, 374, 21718, 13,
    576, 6722, 315, 17689, 374, 24081, 13,
]
```

The 16a test runs:

1. `adapter, kv = Qwen3Adapter.from_hf_repo(REPO)` then
   `Engine(adapter, kv).generate(PROMPT, params)` — the P-1 path; result
   must equal `EXPECTED`.
2. `Engine(adapter, kv).generate_batch([PROMPT], params)` — the new P-2
   batched path at B=1; result must equal `EXPECTED`.
3. Both must agree. Divergence is a bug in 16a's batched scaffolding.

Storing `EXPECTED` rather than recomputing P-1 at test time is a
deliberate choice: it catches regressions in **either** path. If a
future mlx-lm upgrade silently changes Qwen3's greedy output, we notice
at this file rather than downstream.

## Decision log update

- **P-2 preload:** PASS at 2026-04-17. Qwen3-0.6B confirmed as the
  dev-loop model; Qwen3.5-0.8B single-request path (P-1) untouched and
  remains green; Qwen3.5-0.8B batched path (P-3) still deferred.
- **`Qwen3Adapter`:** now plain-Qwen3 only; `Qwen3_5Adapter` owns the
  hybrid Qwen3.5 path in a separate module. Each family has its own
  unit-test file + the factory has its own dispatch test.
- **Unit 16a:** unblocked; oracle token sequence recorded above.

## Side observation

Qwen3-0.6B's `head_dim = 128` is larger than `hidden_size /
num_attention_heads = 64`. That means Q/K/V projections expand to
`16 * 128 = 2048` (2× hidden) before the attention op, then
`o_proj: 2048 → 1024` contracts back. The probe caught this because
the adapter correctly reads `model.args.head_dim`, but worth flagging
for P-2 `MemoryBudgeter.bytes_per_token` computation: it is
`2 * n_layers * n_kv_heads * head_dim * 2 bytes`
`= 2 * 28 * 8 * 128 * 2 = 114688 bytes / token` for Qwen3-0.6B in fp16,
not the naive `2 * 28 * 8 * 64 * 2 = 57344`. Using the wrong value
would over-admit by 2× on this model.
