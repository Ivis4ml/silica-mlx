# silica-mlx bench report

Generated: 2026-04-27T13:33:51

Scenarios: total=1 Runs: total=1 ok=1 skipped=0 failed=0

## Results

| id | codec | runs | ok | skipped | failed | ttft_ms | decode_tok_s | resident_mb | peak_mb | wall_s | tokens | vqbench_gap |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| qwen3.5-27b-warm-decode-b1 |  | 1 | 1 | 0 | 0 | 1919.6 | 16.1 | 204.3 | 15731.2 | 29.624 | 384 |  |

## Scenario details

### `qwen3.5-27b-warm-decode-b1` (seed=0)

- repo: `mlx-community/Qwen3.5-27B-4bit`
- oracle: `warm_decode`
- gate: `SILICA_REAL_QWEN3_5_27B`
- workload: `max_batch_size=1`, `max_tokens=384`, `prompts=1`
- status: **ok**

**P-6 dense primary baseline.** Sustained warm-start decode_tok_s on Qwen3.5-27B-4bit; the §6 dense-primary acceptance gate (≥60 tok/s) compares the highest-performing Track A/B/C/D combination's measurement against the number this scenario establishes. Bandwidth math (P6_OPENING §1.2) predicts a ceiling of ~22.7 tok/s with a realistic upper bound near 20 tok/s; if this row comes in materially below that, the phase pauses for a re-target Decision Log entry before any Track work begins (Q-C resolution, P6_OPENING §11). Dual-gated on SILICA_REAL_QWEN3_5_27B because the checkpoint is ~16 GB on disk and peak device memory during the forward is ~30 GB on M5 Pro 48 GB.

Metadata:

```
{
  "aggregate_overlap_decodes": 352,
  "aggregate_overlap_window_ms": 21929.830041946843,
  "codec_id": null,
  "decode_tok_s_warm_aggregate": 16.05119598860105,
  "decode_tok_s_warm_per_row_mean": 16.00559599999707,
  "measurement_steps_min": 64,
  "rows": [
    {
      "cold_ttft_ms": 1920.2463330002502,
      "decode_interval_ms_mean": 62.478148267654824,
      "decode_interval_ms_std": 2.0750766499548754,
      "decode_interval_rel_std": 0.03321283852820508,
      "decode_tok_s_warm": 16.00559599999707,
      "measurement_steps": 351.0,
      "row": 0,
      "row_first_meas_ms": 4024.4247500086203,
      "row_last_ms": 25954.254791955464,
      "warmup_steps_used": 32.0
    }
  ],
  "seed": 0,
  "warmup_min_steps": 32,
  "warmup_rel_std_threshold": 0.05,
  "warmup_rolling_window": 16
}
```
