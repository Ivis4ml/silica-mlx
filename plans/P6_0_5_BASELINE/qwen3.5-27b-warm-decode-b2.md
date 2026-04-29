# silica-mlx bench report

Generated: 2026-04-29T11:03:30

Scenarios: total=1 Runs: total=1 ok=1 skipped=0 failed=0

## Results

| id | codec | runs | ok | skipped | failed | ttft_ms | decode_tok_s | resident_mb | peak_mb | wall_s | tokens | vqbench_gap |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| qwen3.5-27b-warm-decode-b2 |  | 1 | 1 | 0 | 0 | 1914.8 | 31.2 |  | 16259.2 | 30.194 | 768 |  |

## Scenario details

### `qwen3.5-27b-warm-decode-b2` (seed=0)

- repo: `mlx-community/Qwen3.5-27B-4bit`
- oracle: `warm_decode`
- gate: `SILICA_REAL_QWEN3_5_27B`
- workload: `max_batch_size=2`, `max_tokens=384`, `prompts=2`
- status: **ok**

**P-6.0.5 sub-unit 1 (D-021 step 3) — dense 27B B=2 batch-scaling row.** Reads against the 27B B=1 baseline at 16.05 tok/s @ 70.6% bandwidth utilisation (see plans/P6_0_BASELINE/REPORT.md §1). If aggregate tok/s rises with B (utilisation rises toward 90%+), KV / activation traffic was the slack and Track C ROI estimates must be adjusted; if aggregate stalls, the chip is bandwidth-bound even with batch and (1b) ≥60 tok/s is harder to reach. The workload shape and ``max_tokens=384`` mirror ``qwen3.5-27b-warm-decode-b1`` exactly so per-token throughput is comparable across batch sizes. Dual-gated on SILICA_REAL_QWEN3_5_27B — same checkpoint as the existing B=1 / 4K / 8K rows, single-toggle per checkpoint. See plans/P6_0_5_OPENING.md §3.1.

Metadata:

```
{
  "aggregate_overlap_decodes": 702,
  "aggregate_overlap_window_ms": 22483.569375006482,
  "codec_id": null,
  "decode_tok_s_warm_aggregate": 31.222800450019633,
  "decode_tok_s_warm_per_row_mean": 15.611399400436145,
  "measurement_steps_min": 64,
  "rows": [
    {
      "cold_ttft_ms": 1914.8325000423938,
      "decode_interval_ms_mean": 64.05575510560193,
      "decode_interval_ms_std": 0.34403441865256595,
      "decode_interval_rel_std": 0.005370858841417026,
      "decode_tok_s_warm": 15.611399761838824,
      "measurement_steps": 351.0,
      "row": 0,
      "row_first_meas_ms": 4031.1584169976413,
      "row_last_ms": 26514.728459063917,
      "warmup_steps_used": 32.0
    },
    {
      "cold_ttft_ms": 1914.8425839375705,
      "decode_interval_ms_mean": 64.05575807137346,
      "decode_interval_ms_std": 0.344047785902248,
      "decode_interval_rel_std": 0.005371067274215947,
      "decode_tok_s_warm": 15.611399039033468,
      "measurement_steps": 351.0,
      "row": 1,
      "row_first_meas_ms": 4031.1590840574354,
      "row_last_ms": 26514.73016710952,
      "warmup_steps_used": 32.0
    }
  ],
  "seed": 0,
  "warmup_min_steps": 32,
  "warmup_rel_std_threshold": 0.05,
  "warmup_rolling_window": 16
}
```

## Interpretation (P-6.0.5 §1 Q1 — does dense 27B scale with batch?)

B=2 measured **31.22 tok/s aggregate** (15.61 tok/s/row × 2 rows),
a **1.945× speedup** over the B=1 baseline of 16.05 tok/s — within
2.7% of ideal linear scaling. Per-row throughput dropped from
16.05 → 15.61 (−2.7%), indicating modest KV / activation traffic
surfacing on top of the still-dominant weight-read cost. Bandwidth
utilisation does **not** jump toward 90% the way §1 Q1's
"unsaturated" arm predicted; the chip remains bandwidth-bound
even with batch — the slight per-row degradation is the give-up
signal, not utilisation gain. For the §6(1) ≥60 tok/s gate, B=2
closes **28.8 of the 43.95 tok/s gap** (52% of the gate from
batching alone), leaving roughly another 2× to find from Track C.
Peak memory rose 15.36 → 16.26 GB (+0.9 GB for the second batch
lane), so the §6(4) 36 GB / 4K-context headroom is unaffected at
B=2. **Decision Gate 1 reads this row as:** dense 27B is
bandwidth-bound; (1b) ≥60 tok/s requires the C.4 / C.5 upper-band
landings, since A+B+C.1 stacked together (PLAN.md §1.3
arithmetic) projects to ~31–42 tok/s and that ceiling now has
empirical anchoring rather than back-of-envelope projection.
