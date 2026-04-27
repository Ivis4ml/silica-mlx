# silica-mlx bench report

Generated: 2026-04-27T13:33:07

Scenarios: total=1 Runs: total=1 ok=1 skipped=0 failed=0

## Results

| id | codec | runs | ok | skipped | failed | ttft_ms | decode_tok_s | resident_mb | peak_mb | wall_s | tokens | vqbench_gap |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| qwen3.5-0.8b-warm-decode-b1 |  | 1 | 1 | 0 | 0 | 43.3 | 123.3 | 25.8 | 1813.7 | 3.059 | 256 |  |

## Scenario details

### `qwen3.5-0.8b-warm-decode-b1` (seed=0)

- repo: `Qwen/Qwen3.5-0.8B`
- oracle: `warm_decode`
- gate: `(cache-only)`
- workload: `max_batch_size=1`, `max_tokens=256`, `prompts=1`
- status: **ok**

P-6.0 cache-only baseline on Qwen3.5-0.8B (hybrid DeltaNet). Validates the WARM_DECODE oracle on the hybrid DeltaNet + GQA architecture that Qwen3.5-27B inherits. Dense-27B behaves like a wider/deeper version of this model on the warm-decode path, so a working measurement here is a necessary (not sufficient) condition for the gated 27B row to produce credible numbers.

Metadata:

```
{
  "aggregate_overlap_decodes": 224,
  "aggregate_overlap_window_ms": 1816.9704580213875,
  "codec_id": null,
  "decode_tok_s_warm_aggregate": 123.28213648774873,
  "decode_tok_s_warm_per_row_mean": 122.73176980699985,
  "measurement_steps_min": 64,
  "rows": [
    {
      "cold_ttft_ms": 43.88808296062052,
      "decode_interval_ms_mean": 8.147849587539854,
      "decode_interval_ms_std": 0.5339038407424114,
      "decode_interval_rel_std": 0.06552696328107074,
      "decode_tok_s_warm": 122.73176980699985,
      "measurement_steps": 223.0,
      "row": 0,
      "row_first_meas_ms": 292.8921669954434,
      "row_last_ms": 2109.862625016831,
      "warmup_steps_used": 32.0
    }
  ],
  "seed": 0,
  "warmup_min_steps": 32,
  "warmup_rel_std_threshold": 0.05,
  "warmup_rolling_window": 16
}
```
