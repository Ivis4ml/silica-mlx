# D-023 B=1 sweep — aggregated report

## Toolchain (runtime stack divergence)

| Item | Value |
| --- | --- |
| `venv` | `~/.cache/silica-d023-mtp/.venv (isolated; project venv untouched)` |
| `mlx` | `0.31.2` |
| `mlx-lm` | `0.31.3` |
| `mlx-metal` | `0.31.2` |
| `mlx-vlm` | `0.5.0` |
| `transformers` | `5.8.0` |
| silica project pin | `mlx==0.31.1 / mlx-lm==0.31.2 / mlx-metal==0.31.1` (untouched) |

> External spike stack != silica pinned stack. Spike result informs D-023 decision only and does NOT constitute a silica runtime attestation. tests/test_p2_preload_parity.py remains anchored on the silica project pin.


**Sessions aggregated:** 20260506_175802

## Per-prompt off-spec baseline

| prompt | mean gen_tps | sigma | n | mean prompt_tok | mean gen_tok |
| --- | --- | --- | --- | --- | --- |
| `bst` | 14.86 | 0.01 | 3 | 31.0 | 200.0 |
| `creative_scene` | 14.87 | 0.01 | 3 | 37.0 | 200.0 |
| `factorial` | 14.74 | 0.25 | 3 | 34.0 | 200.0 |
| `factual_explain` | 14.85 | 0.00 | 3 | 52.0 | 200.0 |

## Per-(prompt, block_size) on-spec measurements

| prompt | block | k_cand | gen_tps mean ± sigma | n | speedup vs off | accept_rate mean | mean rounds | mean gen_tok |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `bst` | 2 | 1 | 20.90 ± 0.25 | 3 | 1.406x | 70.09% | 117.0 | 200.0 |
| `bst` | 3 | 2 | 21.08 ± 0.55 | 3 | 1.418x | 59.89% | 91.0 | 200.0 |
| `bst` | 6 | 5 | 14.82 ± 0.07 | 3 | 0.997x | 36.34% | 71.0 | 200.0 |
| `bst` | 9 | 8 | 11.70 ± 0.01 | 3 | 0.787x | 31.92% | 56.0 | 200.0 |
| `creative_scene` | 2 | 1 | 19.84 ± 0.00 | 3 | 1.335x | 60.48% | 124.0 | 200.0 |
| `creative_scene` | 3 | 2 | 17.93 ± 0.10 | 3 | 1.206x | 42.13% | 108.0 | 200.0 |
| `creative_scene` | 6 | 5 | 11.57 ± 0.02 | 3 | 0.778x | 23.96% | 91.0 | 200.0 |
| `creative_scene` | 9 | 8 | 7.69 ± 0.00 | 3 | 0.517x | 16.57% | 86.0 | 200.0 |
| `factorial` | 2 | 1 | 21.99 ± 0.60 | 3 | 1.492x | 86.92% | 107.0 | 200.0 |
| `factorial` | 3 | 2 | 24.39 ± 1.08 | 3 | 1.655x | 80.92% | 76.0 | 200.0 |
| `factorial` | 6 | 5 | 22.01 ± 0.60 | 3 | 1.493x | 66.52% | 46.0 | 200.0 |
| `factorial` | 9 | 8 | 16.32 ± 0.30 | 3 | 1.107x | 51.60% | 39.0 | 200.0 |
| `factual_explain` | 2 | 1 | 22.30 ± 0.03 | 3 | 1.501x | 81.82% | 110.0 | 200.0 |
| `factual_explain` | 3 | 2 | 22.82 ± 0.06 | 3 | 1.536x | 67.06% | 85.0 | 200.0 |
| `factual_explain` | 6 | 5 | 17.89 ± 0.01 | 3 | 1.204x | 47.46% | 59.0 | 200.0 |
| `factual_explain` | 9 | 8 | 11.71 ± 0.00 | 3 | 0.788x | 32.14% | 56.0 | 200.0 |

## Decision row per prompt (highest gen_tps)

| prompt | decision block | k_cand | gen_tps | speedup vs off | accept_rate | gate (>=1.3x B=1)? |
| --- | --- | --- | --- | --- | --- | --- |
| `bst` | 3 | 2 | 21.08 | 1.418x | 59.89% | **PASS** |
| `creative_scene` | 2 | 1 | 19.84 | 1.335x | 60.48% | **PASS** |
| `factorial` | 3 | 2 | 24.39 | 1.655x | 80.92% | **PASS** |
| `factual_explain` | 3 | 2 | 22.82 | 1.536x | 67.06% | **PASS** |

## block_size=9 stress check

Comparison: best on-spec (decision row) vs block_size=9 to detect any throughput regression or accept-rate cliff at long drafts.

| prompt | best_block | best_tps | block=9 tps | block=9 speedup vs off | block=9 accept_rate | regression vs decision row? |
| --- | --- | --- | --- | --- | --- | --- |
| `bst` | 3 | 21.08 | 11.70 | 0.787x | 31.92% | YES (-44.5%) |
| `creative_scene` | 2 | 19.84 | 7.69 | 0.517x | 16.57% | YES (-61.2%) |
| `factorial` | 3 | 24.39 | 16.32 | 1.107x | 51.60% | YES (-33.1%) |
| `factual_explain` | 3 | 22.82 | 11.71 | 0.788x | 32.14% | YES (-48.7%) |

## Accept-rate by prompt type

Heuristic categorization: `factorial`, `bst` are code/template; `creative_scene`, `factual_explain` are natural-language. Lower accept-rate on natural prompts would suggest the high template rate is template-specific rather than fundamental to the pairing.

| block | code mean | natural mean | gap |
| --- | --- | --- | --- |
| 2 | 78.50% | 71.15% | 7.35% |
| 3 | 70.41% | 54.59% | 15.81% |
| 6 | 51.43% | 35.71% | 15.72% |
| 9 | 41.76% | 24.36% | 17.40% |
