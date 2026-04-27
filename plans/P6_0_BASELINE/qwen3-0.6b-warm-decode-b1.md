# silica-mlx bench report

Generated: 2026-04-27T13:32:45

Scenarios: total=1 Runs: total=1 ok=1 skipped=0 failed=0

## Results

| id | codec | runs | ok | skipped | failed | ttft_ms | decode_tok_s | resident_mb | peak_mb | wall_s | tokens | vqbench_gap |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| qwen3-0.6b-warm-decode-b1 |  | 1 | 1 | 0 | 0 | 39.2 | 161.2 | 58.7 | 1475.3 | 2.158 | 256 |  |

## Scenario details

### `qwen3-0.6b-warm-decode-b1` (seed=0)

- repo: `Qwen/Qwen3-0.6B`
- oracle: `warm_decode`
- gate: `(cache-only)`
- workload: `max_batch_size=1`, `max_tokens=256`, `prompts=1`
- status: **ok**

P-6.0 cache-only baseline on plain Qwen3-0.6B. Validates the WARM_DECODE oracle end-to-end without requiring any 16+ GB checkpoint download — every dev box that has run the older `qwen3-0.6b-smoke` scenarios already has this checkpoint pulled. The 256-token max_tokens leaves ~207 decodes after the default 32 + 16 warm-up window, well above the measurement_steps_min=64 floor.

Metadata:

```
{
  "aggregate_overlap_decodes": 224,
  "aggregate_overlap_window_ms": 1389.8869579425082,
  "codec_id": null,
  "decode_tok_s_warm_aggregate": 161.16418584975716,
  "decode_tok_s_warm_per_row_mean": 160.44470287721362,
  "measurement_steps_min": 64,
  "rows": [
    {
      "cold_ttft_ms": 40.128666907548904,
      "decode_interval_ms_mean": 6.232676941446225,
      "decode_interval_ms_std": 0.3823417715878166,
      "decode_interval_rel_std": 0.061344711939954714,
      "decode_tok_s_warm": 160.44470287721362,
      "measurement_steps": 223.0,
      "row": 0,
      "row_first_meas_ms": 233.37741696741432,
      "row_last_ms": 1623.2643749099225,
      "warmup_steps_used": 32.0
    }
  ],
  "seed": 0,
  "warmup_min_steps": 32,
  "warmup_rel_std_threshold": 0.05,
  "warmup_rolling_window": 16
}
```
