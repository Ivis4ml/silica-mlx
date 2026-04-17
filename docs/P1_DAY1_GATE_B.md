# P-1 Day-1 Gate B — D-014 Qwen3.5-0.8B text-only load probe

**Date:** 2026-04-16
**mlx-lm version:** 0.31.2
**Repo:** `Qwen/Qwen3.5-0.8B`
**Result:** **PASS** — all four sub-checks green. R-8 does not trigger.
**Probe:** `scripts/probe_qwen3_5_load.py` (exit 0).

## Question

PLAN.md §7 P-1 / D-014 requires Qwen3.5-0.8B to load under three simultaneous
constraints: (a) multimodal heads skipped, (b) MTP disabled so decode is
one-token-per-step, (c) tokenizer round-trip parity with the HF reference.
Failure of any sub-check triggers risk R-8 (monkey-patch the load path, or
fall back to Qwen3-0.6B and shift DeltaNet to P-3).

## Checkpoint facts

| Property | Value |
| --- | --- |
| Total repo size on HF | 1.77 GB |
| Safetensors shards | 1 (single file) |
| `config.json` → `model_type` | `qwen3_5` |
| `config.json` → `architectures` | `Qwen3_5ForConditionalGeneration` |
| `config.json` → `vision_config` | 14-key dict (vision encoder present) |
| `config.json` → `text_config.mtp_num_hidden_layers` | 1 (MTP head present) |
| `config.json` → `tie_word_embeddings` | True |
| Post-load parameter tensors | 320 |
| Post-load total params | 0.752 B |
| Post-load resident bytes | 1.50 GB |
| Post-load top-level prefixes | `language_model.*` only |
| Layer count | 24 |
| Layer breakdown | 18 DeltaNet (linear) + 6 full attention |
| Tokenizer vocab_size | 248044 |

## Source evidence (why mlx-lm handles each constraint)

### (a) Multimodal heads skipped

`mlx_lm/models/qwen3_5.py:384-398` — `Model.sanitize()` drops any weight key
starting with `vision_tower` or `model.visual`:

```python
def sanitize(self, weights):
    sanitized = {}
    for key, value in weights.items():
        if key.startswith("vision_tower") or key.startswith("model.visual"):
            continue
        ...
```

The `Model` class has no `vision_tower` or `visual` attribute, so the filter
is structural, not optional.

### (b) MTP head dropped + norm compensation

`mlx_lm/models/qwen3_5.py:307-331` — `TextModel.sanitize()`:

```python
has_mtp_weights = any("mtp." in k for k in weights)
has_unsanitized_conv1d = any(
    "conv1d.weight" in k and v.shape[-1] != 1 for k, v in weights.items()
)
should_shift_norm_weights = has_mtp_weights or has_unsanitized_conv1d
weights = {k: v for k, v in weights.items() if "mtp." not in k}
...
for k, v in weights.items():
    ...
    if should_shift_norm_weights and any(k.endswith(sfx) for sfx in norm_keys):
        if v.ndim == 1:
            weights[k] = v + 1.0
```

Two things worth naming:

1. All `mtp.*` keys are filtered — the `Model` class has no MTP head, so
   `decode_step` yields exactly one token.
2. When MTP weights are present in the checkpoint, `RMSNorm.weight` values are
   shifted by +1.0. This compensates for Qwen3.5's MTP-trained norm
   initialization: the main-head norm weights are stored as `weight - 1.0`
   relative to an "MTP-absent" baseline, and sanitize restores the baseline.

   **Implication for Silica:** any P-3 loader that bypasses mlx-lm's
   `sanitize` on a Qwen3.5 checkpoint will produce garbage logits. Borrowing
   mlx-lm's weight loader (D-004 / D-010) is not merely a convenience — it is
   load-correctness.

### (c) Tokenizer parity

`mlx_lm/tokenizer_utils.py` wraps `transformers.AutoTokenizer`. Parity with
the HF reference is by construction; the probe verifies the round-trip on a
mixed English / Chinese fixture.

## Probe output (cache-hit run)

```text
Gate B probe — mlx-lm 0.31.2, repo Qwen/Qwen3.5-0.8B
Loading model (first run downloads ~1.77 GB to ~/.cache/huggingface)...
  model type: mlx_lm.models.qwen3_5.Model
  num layers (model.layers): 24
  tokenizer type: TokenizerWrapper
  config has vision_config: True
  config has text_config.mtp_num_hidden_layers: 1

(a) No vision-prefixed parameters:
    PASS (sanitize() dropped vision weights despite vision_config in json)
(b.1) No MTP-prefixed parameters:
    PASS
(b.2) generate_step yields exactly one token per iteration:
    PASS
(c) Tokenizer parity with HF AutoTokenizer:
    PASS

RESULT: PASS — Qwen3.5-0.8B loads text-only, MTP disabled, tokenizer matches HF.
```

## Implications for Silica

1. **18:6 DeltaNet-to-full-attention ratio** is the concrete instance behind
   `AttentionKind.HYBRID_DELTANET` (D-015). Only 6 of 24 layers carry paged KV;
   the remaining 18 carry DeltaNet recurrent state via `StateDelta`. For the
   P-2 `MemoryBudgeter`, this means the KV-cache term in the memory model is
   `6/24 = 25%` of a pure-attention 24-layer baseline — a significant headroom
   that should be reflected in scheduler capacity estimates.

2. **P-1 `Qwen3Adapter` delegates loading to mlx-lm, not a parallel loader.**
   The MTP norm-shift (+1.0) is a non-obvious correctness detail that would
   silently regress under any re-implementation; D-004's "borrow the weight
   loader" is load-correctness-critical, not ergonomic.

3. **Tokenizer parity is structural, not tested.** Since `TokenizerWrapper`
   thinly wraps `transformers.AutoTokenizer`, regression risk lives in
   `tokenizer_utils.py` (chat template handling, special-token injection), not
   in the BPE layer itself. P-1 acceptance test "greedy decoding token-for-
   token identical to mlx-lm" suffices for tokenizer parity.

4. **Multimodal tokens remain in the vocab.** `vocab_size = 248044` base BPE,
   `config.vocab_size = 248320`; the gap includes `image_token_id = 248056`,
   `video_token_id = 248057`, `vision_start/end_token_id`. Silica's text-only
   generation can safely produce these ids if a prompt elicits them — they
   decode to the raw `<|image_pad|>` etc. strings without crashing.

## Decision log update

- **D-014 status:** day-1 gate passed at P-1 entry. P-1 proceeds on Qwen3.5-0.8B.
- **R-8 status:** does not trigger. mlx-lm 0.31.2 carries Qwen3.5 forward
  cleanly with multimodal + MTP handled in `sanitize()`.
- **D-015 empirical grounding:** HYBRID_DELTANET is 18:6 on this checkpoint;
  record in the DeltaNet state store design review.
