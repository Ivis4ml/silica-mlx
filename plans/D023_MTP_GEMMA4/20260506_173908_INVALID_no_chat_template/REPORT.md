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


**Sessions aggregated:** 20260506_173908

## Per-prompt off-spec baseline

| prompt | mean gen_tps | sigma | n | mean prompt_tok | mean gen_tok |
| --- | --- | --- | --- | --- | --- |
| `bst` | 14.90 | 0.02 | 3 | 18.0 | 200.0 |
| `creative_scene` | 14.92 | 0.04 | 3 | 24.0 | 200.0 |
| `factorial` | 14.81 | 0.17 | 3 | 21.0 | 200.0 |
| `factual_explain` | 15.07 | 0.21 | 3 | 39.0 | 200.0 |

## Per-(prompt, block_size) on-spec measurements

| prompt | block | k_cand | gen_tps mean ± sigma | n | speedup vs off | accept_rate mean | mean rounds | mean gen_tok |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `bst` | 2 | 1 | 13.69 ± 0.19 | 3 | 0.919x | 11.17% | 179.0 | 200.0 |
| `bst` | 3 | 2 | 10.75 ± 0.22 | 3 | 0.721x | 5.90% | 178.0 | 200.0 |
| `bst` | 6 | 5 | 6.04 ± 0.04 | 3 | 0.405x | 2.74% | 175.0 | 200.0 |
| `bst` | 9 | 8 | 3.53 ± 0.01 | 3 | 0.237x | 0.73% | 188.0 | 200.0 |
| `creative_scene` | 2 | 1 | 24.63 ± 0.13 | 3 | 1.651x | 100.00% | 100.0 | 200.0 |
| `creative_scene` | 3 | 2 | 29.06 ± 0.12 | 3 | 1.948x | 98.51% | 67.0 | 200.0 |
| `creative_scene` | 6 | 5 | 30.58 ± 0.09 | 3 | 2.050x | 94.29% | 35.0 | 200.0 |
| `creative_scene` | 9 | 8 | 28.39 ± 0.06 | 3 | 1.903x | 95.65% | 23.0 | 200.0 |
| `factorial` | 2 | 1 | 24.10 ± 0.63 | 3 | 1.627x | 100.00% | 100.0 | 200.0 |
| `factorial` | 3 | 2 | 28.24 ± 0.93 | 3 | 1.906x | 99.25% | 67.0 | 200.0 |
| `factorial` | 6 | 5 | 30.70 ± 1.03 | 3 | 2.072x | 97.65% | 34.0 | 200.0 |
| `factorial` | 9 | 8 | 28.83 ± 0.63 | 3 | 1.946x | 96.20% | 23.0 | 200.0 |
| `factual_explain` | 2 | 1 | 24.73 ± 0.38 | 3 | 1.642x | 98.02% | 101.0 | 200.0 |
| `factual_explain` | 3 | 2 | 20.07 ± 0.22 | 3 | 1.332x | 52.04% | 98.0 | 200.0 |
| `factual_explain` | 6 | 5 | 11.44 ± 0.11 | 3 | 0.759x | 22.55% | 94.0 | 200.0 |
| `factual_explain` | 9 | 8 | 6.99 ± 0.07 | 3 | 0.464x | 13.54% | 96.0 | 200.0 |

## Decision row per prompt (highest gen_tps)

| prompt | decision block | k_cand | gen_tps | speedup vs off | accept_rate | gate (>=1.3x B=1)? |
| --- | --- | --- | --- | --- | --- | --- |
| `bst` | 2 | 1 | 13.69 | 0.919x | 11.17% | **FAIL** |
| `creative_scene` | 6 | 5 | 30.58 | 2.050x | 94.29% | **PASS** |
| `factorial` | 6 | 5 | 30.70 | 2.072x | 97.65% | **PASS** |
| `factual_explain` | 2 | 1 | 24.73 | 1.642x | 98.02% | **PASS** |

## block_size=9 stress check

Comparison: best on-spec (decision row) vs block_size=9 to detect any throughput regression or accept-rate cliff at long drafts.

| prompt | best_block | best_tps | block=9 tps | block=9 speedup vs off | block=9 accept_rate | regression vs decision row? |
| --- | --- | --- | --- | --- | --- | --- |
| `bst` | 2 | 13.69 | 3.53 | 0.237x | 0.73% | YES (-74.2%) |
| `creative_scene` | 6 | 30.58 | 28.39 | 1.903x | 95.65% | YES (-7.2%) |
| `factorial` | 6 | 30.70 | 28.83 | 1.946x | 96.20% | YES (-6.1%) |
| `factual_explain` | 2 | 24.73 | 6.99 | 0.464x | 13.54% | YES (-71.8%) |

## Accept-rate by prompt type

Heuristic categorization: `factorial`, `bst` are code/template; `creative_scene`, `factual_explain` are natural-language. Lower accept-rate on natural prompts would suggest the high template rate is template-specific rather than fundamental to the pairing.

| block | code mean | natural mean | gap |
| --- | --- | --- | --- |
| 2 | 55.59% | 99.01% | -43.42% |
| 3 | 52.58% | 75.27% | -22.70% |
| 6 | 50.19% | 58.42% | -8.22% |
| 9 | 48.46% | 54.60% | -6.13% |
