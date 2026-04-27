# silica-mlx bench report

Generated: 2026-04-27T13:32:57

Scenarios: total=1 Runs: total=1 ok=1 skipped=0 failed=0

## Results

| id | codec | runs | ok | skipped | failed | ttft_ms | decode_tok_s | resident_mb | peak_mb | wall_s | tokens | vqbench_gap |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| qwen3-0.6b-warm-decode-b2 |  | 1 | 1 | 0 | 0 | 53.5 | 208.7 |  | 1763.0 | 2.966 | 512 |  |

## Scenario details

### `qwen3-0.6b-warm-decode-b2` (seed=0)

- repo: `Qwen/Qwen3-0.6B`
- oracle: `warm_decode`
- gate: `(cache-only)`
- workload: `max_batch_size=2`, `max_tokens=256`, `prompts=2`
- status: **ok**

P-6.0 cache-only B>1 path validation. Same prompt replicated twice through `Engine.generate_batch` exercises the per-row timestamp collector (`_collect_warm_decode_batched`) and the oracle's aggregate-window math without the 16 GB load cost real-model batched scenarios pay.

Metadata:

```
{
  "aggregate_overlap_decodes": 446,
  "aggregate_overlap_window_ms": 2136.870624963194,
  "codec_id": null,
  "decode_tok_s_warm_aggregate": 208.71642615597375,
  "decode_tok_s_warm_per_row_mean": 104.35814592817653,
  "measurement_steps_min": 64,
  "rows": [
    {
      "cold_ttft_ms": 53.53783303871751,
      "decode_interval_ms_mean": 9.582381165636528,
      "decode_interval_ms_std": 0.2948930922175726,
      "decode_interval_rel_std": 0.030774510752618736,
      "decode_tok_s_warm": 104.35819476542113,
      "measurement_steps": 223.0,
      "row": 0,
      "row_first_meas_ms": 352.23341605160385,
      "row_last_ms": 2489.1044159885496,
      "warmup_steps_used": 32.0
    },
    {
      "cold_ttft_ms": 53.54529095347971,
      "decode_interval_ms_mean": 9.582390134314684,
      "decode_interval_ms_std": 0.294896791751423,
      "decode_interval_rel_std": 0.030774868025399336,
      "decode_tok_s_warm": 104.35809709093192,
      "measurement_steps": 223.0,
      "row": 1,
      "row_first_meas_ms": 352.23379102535546,
      "row_last_ms": 2489.10679097753,
      "warmup_steps_used": 32.0
    }
  ],
  "seed": 0,
  "warmup_min_steps": 32,
  "warmup_rel_std_threshold": 0.05,
  "warmup_rolling_window": 16
}
```
