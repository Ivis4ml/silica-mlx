# silica-mlx bench report

Generated: 2026-04-29T11:07:58

Scenarios: total=1 Runs: total=1 ok=1 skipped=0 failed=0

## Results

| id | codec | runs | ok | skipped | failed | ttft_ms | decode_tok_s | resident_mb | peak_mb | wall_s | tokens | vqbench_gap |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| qwen3.5-moe-35b-a3b-warm-decode-b3 |  | 1 | 1 | 0 | 0 | 2472.7 | 163.5 |  | 20415.8 | 13.596 | 1152 |  |

## Scenario details

### `qwen3.5-moe-35b-a3b-warm-decode-b3` (seed=0)

- repo: `mlx-community/Qwen3.5-35B-A3B-4bit`
- oracle: `warm_decode`
- gate: `SILICA_REAL_QWEN3_5_MOE`
- workload: `max_batch_size=3`, `max_tokens=384`, `prompts=3`
- status: **ok**

**P-6.0.5 sub-unit 3 (D-021 step 3) — MoE 35B-A3B B=3 saturation row.** Intermediate row between B=2 (120.93 tok/s aggregate, 59.1% bandwidth utilisation, 19.68 GB peak — see plans/P6_0_BASELINE/REPORT.md §1) and B=4 (opt-in, OOM-flagged at v1.7.13). Tells whether MoE saturates before B=4: if B=3 lifts aggregate above ~150 tok/s, the (2b) stretch can be reframed to a per-row quantity; if B=3 stalls near B=2, MoE acceptance shape is set by B=2. Workload mirrors the B=1 / B=2 / B=4 rows on this checkpoint exactly (same prompt, ``max_tokens=384``) so per-token throughput is comparable across batch sizes. Dual-gated on SILICA_REAL_QWEN3_5_MOE — same checkpoint as the existing MoE rows, single-toggle per checkpoint. See plans/P6_0_5_OPENING.md §3.3.

Metadata:

```
{
  "aggregate_overlap_decodes": 1052,
  "aggregate_overlap_window_ms": 6432.538916124031,
  "codec_id": null,
  "decode_tok_s_warm_aggregate": 163.5435111574715,
  "decode_tok_s_warm_per_row_mean": 54.566305796216334,
  "measurement_steps_min": 64,
  "rows": [
    {
      "cold_ttft_ms": 2472.6711658295244,
      "decode_interval_ms_mean": 18.326326327775195,
      "decode_interval_ms_std": 0.949077983737475,
      "decode_interval_rel_std": 0.051787683290298175,
      "decode_tok_s_warm": 54.56630980560518,
      "measurement_steps": 351.0,
      "row": 0,
      "row_first_meas_ms": 3222.201749915257,
      "row_last_ms": 9654.74229096435,
      "warmup_steps_used": 32.0
    },
    {
      "cold_ttft_ms": 2472.673665964976,
      "decode_interval_ms_mean": 18.326328823242093,
      "decode_interval_ms_std": 0.9490762244457971,
      "decode_interval_rel_std": 0.05178758024041047,
      "decode_tok_s_warm": 54.56630237539801,
      "measurement_steps": 351.0,
      "row": 1,
      "row_first_meas_ms": 3222.2026658710092,
      "row_last_ms": 9654.744082828984,
      "warmup_steps_used": 32.0
    },
    {
      "cold_ttft_ms": 2472.6742908824235,
      "decode_interval_ms_mean": 18.326327872019462,
      "decode_interval_ms_std": 0.9490793437339284,
      "decode_interval_rel_std": 0.05178775313645772,
      "decode_tok_s_warm": 54.566305207645804,
      "measurement_steps": 351.0,
      "row": 2,
      "row_first_meas_ms": 3222.203374840319,
      "row_last_ms": 9654.74445791915,
      "warmup_steps_used": 32.0
    }
  ],
  "seed": 0,
  "warmup_min_steps": 32,
  "warmup_rel_std_threshold": 0.05,
  "warmup_rolling_window": 16
}
```

## Interpretation (P-6.0.5 §1 Q2 — does MoE saturate before B=4?)

B=3 measured **163.5 tok/s aggregate** (54.5 tok/s/row × 3 rows),
a **1.352× speedup** over the B=2 baseline of 120.93 tok/s — that
is **90.2% of ideal linear** B=2→B=3 scaling. Bandwidth
utilisation continues to climb: 37.1% (B=1) → 59.1% (B=2) →
**79.9% (B=3)** against the 204.7 tok/s ceiling. This is the
"unsaturated" arm §1 Q1 hypothesised for dense — but it shows up
on **MoE** instead, because at B=1 the MoE active-3B / router /
expert-gather path spends only ~37% of bandwidth, leaving slack
for batch parallelism to consume. Per-row throughput drops 60.5
→ 54.5 (−10%) consistent with router diversity: at B=3 more
distinct experts are touched per step than at B=2, so bytes/step
grows modestly even as the weight-read amortises across rows.
Peak memory rose 19.68 → 20.42 GB (+0.74 GB for the third batch
lane), so the per-row memory cost is roughly linear and B=4
projects to ~21.2 GB — well within the 48 GB envelope on a fresh
boot. The §6(2) ≥100 tok/s gate is exceeded by **64%** at B=3
(163.5 vs 100), so the (2b) stretch reframing branch becomes
viable: ≥150 tok/s aggregate is already met at B=3, and the
≥100 tok/s **per-row** variant is unreachable on this checkpoint
(per-row falls with B, not rises). **Decision Gate 1 reads this
row as:** MoE has not saturated at B=3 — both B=4 attempt (sub-unit
4) and the aggregate-anchored stretch reframing are warranted.
The opt-in B=4 row will tell whether the curve is still rising
or has flattened by B=4; the OOM risk is now low (projected ~21
GB peak).
