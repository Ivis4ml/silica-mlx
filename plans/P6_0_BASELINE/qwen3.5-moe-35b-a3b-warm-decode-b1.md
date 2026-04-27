# silica-mlx bench report

Generated: 2026-04-27T13:34:33

Scenarios: total=1 Runs: total=1 ok=1 skipped=0 failed=0

## Results

| id | codec | runs | ok | skipped | failed | ttft_ms | decode_tok_s | resident_mb | peak_mb | wall_s | tokens | vqbench_gap |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| qwen3.5-moe-35b-a3b-warm-decode-b1 |  | 1 | 1 | 0 | 0 | 2732.1 | 76.0 | 80.1 | 19862.1 | 12.495 | 384 |  |

## Scenario details

### `qwen3.5-moe-35b-a3b-warm-decode-b1` (seed=0)

- repo: `mlx-community/Qwen3.5-35B-A3B-4bit`
- oracle: `warm_decode`
- gate: `SILICA_REAL_QWEN3_5_MOE`
- workload: `max_batch_size=1`, `max_tokens=384`, `prompts=1`
- status: **ok**

P-6 MoE B=1 baseline on Qwen3.5-35B-A3B-4bit (active 3B). Bandwidth math (P6_OPENING §1.2) gives a ceiling of ~200 tok/s on M5 Pro 48 GB; the published vllm-mlx number for the same family on M4 Max is 127.7 tok/s, so a B=1 result in the 40-80 tok/s range is consistent with the published reference. Dual-gated on SILICA_REAL_QWEN3_5_MOE; ~20 GB checkpoint.

Metadata:

```
{
  "aggregate_overlap_decodes": 352,
  "aggregate_overlap_window_ms": 4631.167208077386,
  "codec_id": null,
  "decode_tok_s_warm_aggregate": 76.00675686813123,
  "decode_tok_s_warm_per_row_mean": 75.79082858157403,
  "measurement_steps_min": 64,
  "rows": [
    {
      "cold_ttft_ms": 2732.737333048135,
      "decode_interval_ms_mean": 13.194208570021043,
      "decode_interval_ms_std": 0.684087276646368,
      "decode_interval_rel_std": 0.05184754151914069,
      "decode_tok_s_warm": 75.79082858157403,
      "measurement_steps": 351.0,
      "row": 0,
      "row_first_meas_ms": 3283.721624989994,
      "row_last_ms": 7914.88883306738,
      "warmup_steps_used": 32.0
    }
  ],
  "seed": 0,
  "warmup_min_steps": 32,
  "warmup_rel_std_threshold": 0.05,
  "warmup_rolling_window": 16
}
```
