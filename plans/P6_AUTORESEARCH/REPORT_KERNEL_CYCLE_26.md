# P-6 Autoresearch Kernel Cycle 26 — bf16 FA-Decode Repair

## Hypothesis

`flash_attention_decode_v8/v10` had Python-side bf16 dtype branches, but
the Metal source still vector-loaded inputs as `half4`. In production,
`shadow_install` also rejected bf16 queries, so Qwen3.5-27B bf16
activations fell back to MLX SDPA. A bf16-native vector source should make
the v10 flag actually fire and recover the intended attention-kernel lever.

Lever family: kernel fusion / bandwidth utilisation.

## Pass / Fail Thresholds

- Correctness: v8/v10 bf16 max-abs error vs `mx.fast.scaled_dot_product_attention`
  under `5e-3`.
- Microbench: bf16 v10 should beat MLX SDPA by at least 1.2x on production
  B=52 attention shapes.
- E2E keep: B=52 must clear the strict-envelope running-best threshold,
  currently `207.7 tok/s` on repeated runs.

## Change

- Added bf16-specific vectorized Metal sources for v8 and v10:
  `bfloat4` HBM/TG loads, fp32 dot/accumulate via `float4(...)`, bf16 output.
- Changed v10 shadow eligibility from fp16-only to fp16-or-bf16.
- Added bf16 correctness coverage for v8/v10 at T_kv 128 and 512.

## Commands

```bash
python -m pytest tests/test_p2_preload_parity.py \
  tests/test_flash_attention_decode.py \
  tests/test_fa_decode_shadow_install.py -q

SILICA_REAL_QWEN3_5_27B=1 \
SILICA_USE_FA_DECODE_V10=1 \
SILICA_USE_BF16_DELTANET_STATE=1 \
python scripts/bench.py --scenario qwen3.5-27b-warm-decode-b52 \
  --out plans/P6_AUTORESEARCH/c26_b52_bf16_fa_run1.jsonl
```

## Results

Focused tests: **55 passed**.

Instrumentation on a 2-token 27B decode after the patch:

```text
fa_v10_calls 16
```

Microbench, B=52 bf16 gated production shapes:

| T_kv | MLX p50 ms | Silica p50 ms | Speedup | Max abs |
| ---: | ---: | ---: | ---: | ---: |
| 128 | 0.673 | 0.314 | 2.14x | 2.44e-4 |
| 256 | 0.767 | 0.470 | 1.63x | 1.22e-4 |
| 512 | 0.963 | 0.702 | 1.37x | 1.22e-4 |
| 1024 | 1.520 | 1.190 | 1.28x | 6.10e-5 |

Real B=52 E2E with bf16 FA path:

| Run | decode_tok_s | Peak GB |
| ---: | ---: | ---: |
| 1 | 204.0 | 35.52 |
| 2 | 190.0 | 35.52 |

## Interpretation

The kernel repair is real at the microbench layer and fixes a shadow-path
bug: v10 now fires on production bf16 Qwen3.5 decode. However, the E2E
result did not reproduce a strict-envelope running-best keep in the current
machine state. The two B=52 E2E runs are too noisy and below the 207.7 tok/s
keep threshold.

This patch is therefore a **kernel/correctness keep** but **not** a
primary-metric keep. It should remain shadow-gated by
`SILICA_USE_FA_DECODE_V10=1`.

## Decision

**diagnostic for E2E, keep for kernel correctness/microbench.** Next clean
step is to re-run B=52 after a cool-down / clean process baseline, then only
proceed to the 40 GB cliff if the strict-envelope line is stable.
