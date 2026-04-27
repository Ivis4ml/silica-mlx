# silica-mlx bench report

Generated: 2026-04-27T13:35:47

Scenarios: total=1 Runs: total=1 ok=1 skipped=0 failed=0

## Results

| id | codec | runs | ok | skipped | failed | ttft_ms | decode_tok_s | resident_mb | peak_mb | wall_s | tokens | vqbench_gap |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| gemma4-31b-warm-decode-b1 |  | 1 | 1 | 0 | 0 | 949.7 | 13.6 | 597.0 | 17930.0 | 32.207 | 384 |  |

## Scenario details

### `gemma4-31b-warm-decode-b1` (seed=0)

- repo: `mlx-community/gemma-4-31b-4bit`
- oracle: `warm_decode`
- gate: `SILICA_REAL_GEMMA4_31B`
- workload: `max_batch_size=1`, `max_tokens=384`, `prompts=1`
- status: **ok**

P-6 dense secondary baseline. Gemma4-31B-4bit is the second dense production target (PLAN.md §3.4); the §6 acceptance gate sets ≥55 tok/s here. Hybrid sliding + full attention layout (50 sliding + 10 full) makes the warm-decode path different from Qwen3.5-27B's hybrid DeltaNet — running both rows pins whether the WARM_DECODE oracle is robust across attention-pattern variants. Dual-gated on SILICA_REAL_GEMMA4_31B; ~18 GB checkpoint.

Metadata:

```
{
  "aggregate_overlap_decodes": 352,
  "aggregate_overlap_window_ms": 25824.110832996666,
  "codec_id": null,
  "decode_tok_s_warm_aggregate": 13.630672601909424,
  "decode_tok_s_warm_per_row_mean": 13.591949100199454,
  "measurement_steps_min": 64,
  "rows": [
    {
      "cold_ttft_ms": 950.2115000505,
      "decode_interval_ms_mean": 73.57296533617284,
      "decode_interval_ms_std": 2.1207252187134293,
      "decode_interval_rel_std": 0.028824789228262284,
      "decode_tok_s_warm": 13.591949100199454,
      "measurement_steps": 351.0,
      "row": 0,
      "row_first_meas_ms": 3356.8610000656918,
      "row_last_ms": 29180.97183306236,
      "warmup_steps_used": 32.0
    }
  ],
  "seed": 0,
  "warmup_min_steps": 32,
  "warmup_rel_std_threshold": 0.05,
  "warmup_rolling_window": 16
}
```
