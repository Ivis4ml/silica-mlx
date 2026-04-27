# silica-mlx bench report

Generated: 2026-04-27T13:35:02

Scenarios: total=1 Runs: total=1 ok=1 skipped=0 failed=0

## Results

| id | codec | runs | ok | skipped | failed | ttft_ms | decode_tok_s | resident_mb | peak_mb | wall_s | tokens | vqbench_gap |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| qwen3.5-moe-35b-a3b-warm-decode-b2 |  | 1 | 1 | 0 | 0 | 2055.7 | 120.9 |  | 20156.2 | 12.487 | 768 |  |

## Scenario details

### `qwen3.5-moe-35b-a3b-warm-decode-b2` (seed=0)

- repo: `mlx-community/Qwen3.5-35B-A3B-4bit`
- oracle: `warm_decode`
- gate: `SILICA_REAL_QWEN3_5_MOE`
- workload: `max_batch_size=2`, `max_tokens=384`, `prompts=2`
- status: **ok**

**P-6 MoE stretch validator — primary B=2 measurement.** Qwen3.5-35B-A3B-4bit at B=2 (the largest MoE batch validated on M5 Pro 48 GB to date — see tests/test_p3_qwen3_5_moe_batched_parity.py at v1.7.9). The §6 stretch gate is ≥100 tok/s aggregate; the B=2 row is the primary path because B=4 has not been validated to fit the 48 GB envelope (peak at B=2 was already ~30 GB on the test). If this row clears 100 tok/s aggregate the phase exits the MoE stretch validator successfully whether or not B=4 also lands.

Metadata:

```
{
  "aggregate_overlap_decodes": 702,
  "aggregate_overlap_window_ms": 5805.079207988456,
  "codec_id": null,
  "decode_tok_s_warm_aggregate": 120.92858251339058,
  "decode_tok_s_warm_per_row_mean": 60.4642734612642,
  "measurement_steps_min": 64,
  "rows": [
    {
      "cold_ttft_ms": 2055.672625079751,
      "decode_interval_ms_mean": 16.538690051270855,
      "decode_interval_ms_std": 0.5439842475772496,
      "decode_interval_rel_std": 0.03289161631851545,
      "decode_tok_s_warm": 60.46428084086131,
      "measurement_steps": 351.0,
      "row": 0,
      "row_first_meas_ms": 2751.8682080553845,
      "row_last_ms": 8556.948416051455,
      "warmup_steps_used": 32.0
    },
    {
      "cold_ttft_ms": 2055.6763330241665,
      "decode_interval_ms_mean": 16.538694088328683,
      "decode_interval_ms_std": 0.5439749012251271,
      "decode_interval_rel_std": 0.032891043169424664,
      "decode_tok_s_warm": 60.46426608166709,
      "measurement_steps": 351.0,
      "row": 1,
      "row_first_meas_ms": 2751.869208062999,
      "row_last_ms": 8556.950833066367,
      "warmup_steps_used": 32.0
    }
  ],
  "seed": 0,
  "warmup_min_steps": 32,
  "warmup_rel_std_threshold": 0.05,
  "warmup_rolling_window": 16
}
```
