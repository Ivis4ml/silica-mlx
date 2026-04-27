# silica-mlx bench report

Generated: 2026-04-27T13:36:13

Scenarios: total=1 Runs: total=1 ok=1 skipped=0 failed=0

## Results

| id | codec | runs | ok | skipped | failed | ttft_ms | decode_tok_s | resident_mb | peak_mb | wall_s | tokens | vqbench_gap |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| gemma4-moe-26b-a4b-warm-decode-b1 |  | 1 | 1 | 0 | 0 | 454.0 | 68.6 | 149.3 | 14724.4 | 8.482 | 384 |  |

## Scenario details

### `gemma4-moe-26b-a4b-warm-decode-b1` (seed=0)

- repo: `mlx-community/gemma-4-26b-a4b-4bit`
- oracle: `warm_decode`
- gate: `SILICA_REAL_GEMMA4_MOE`
- workload: `max_batch_size=1`, `max_tokens=384`, `prompts=1`
- status: **ok**

P-6 second MoE family baseline on gemma-4-26b-a4b-4bit (active 4B). Different shape from Qwen3.5-MoE: 128 experts × top-8 (vs 256 × 8), always-on dense MLP plus ungated top-k experts (Gemma4-MoE pattern, see plans/P3_MOE_SURVEY.md §3.2). Tests whether the WARM_DECODE oracle is robust across MoE routing variants. Dual-gated on SILICA_REAL_GEMMA4_MOE; ~16 GB checkpoint.

Metadata:

```
{
  "aggregate_overlap_decodes": 352,
  "aggregate_overlap_window_ms": 5130.018374999054,
  "codec_id": null,
  "decode_tok_s_warm_aggregate": 68.61573863272272,
  "decode_tok_s_warm_per_row_mean": 68.42080755706158,
  "measurement_steps_min": 64,
  "rows": [
    {
      "cold_ttft_ms": 454.758333042264,
      "decode_interval_ms_mean": 14.61543696580927,
      "decode_interval_ms_std": 0.45410219544720715,
      "decode_interval_rel_std": 0.031070038925932523,
      "decode_tok_s_warm": 68.42080755706158,
      "measurement_steps": 351.0,
      "row": 0,
      "row_first_meas_ms": 924.6112500550225,
      "row_last_ms": 6054.629625054076,
      "warmup_steps_used": 32.0
    }
  ],
  "seed": 0,
  "warmup_min_steps": 32,
  "warmup_rel_std_threshold": 0.05,
  "warmup_rolling_window": 16
}
```
