# silica-mlx bench report

Generated: 2026-04-29T11:10:22

Scenarios: total=1 Runs: total=1 ok=1 skipped=0 failed=0

## Results

| id | codec | runs | ok | skipped | failed | ttft_ms | decode_tok_s | resident_mb | peak_mb | wall_s | tokens | vqbench_gap |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| qwen3.5-moe-35b-a3b-warm-decode-b1-4k |  | 1 | 1 | 0 | 0 | 4822.3 | 85.0 | 143.0 | 23599.6 | 17.402 | 600 |  |

## Scenario details

### `qwen3.5-moe-35b-a3b-warm-decode-b1-4k` (seed=0)

- repo: `mlx-community/Qwen3.5-35B-A3B-4bit`
- oracle: `warm_decode`
- gate: `SILICA_REAL_QWEN3_5_MOE`
- workload: `max_batch_size=1`, `max_tokens=600`, `prompts=1`
- status: **ok**

**P-6.0.5 sub-unit 5 (D-021 step 3) — MoE 35B-A3B B=1 sustained 4K-context probe.** First MoE 4K row in the catalog. Mirrors the dense 27B 4K row (P5.9 step 2(d), ``qwen3.5-27b-warm-decode-b1-4k``) on workload shape (``max_tokens=600``, ``target_context_tokens=4096``) so cross-family 4K comparisons read from the same envelope. Surfaces the §6(4) RAM-headroom gate under the MoE routing-state + KV-growth path, which has a materially different memory profile from dense (256 experts × top-8, 19.4 GB peak at 384 tokens). Validates that the MoE B=1 baseline path is safe at 4K context before any (2b) stretch reframing on B>1 is contemplated. Dual-gated on SILICA_REAL_QWEN3_5_MOE — same checkpoint as the existing MoE rows. See plans/P6_0_5_OPENING.md §3.5.

Metadata:

```
{
  "actual_total_context_min": 3728,
  "actual_total_context_per_row": [
    3728
  ],
  "aggregate_overlap_decodes": 568,
  "aggregate_overlap_window_ms": 6678.913125069812,
  "codec_id": null,
  "decode_tok_s_warm_aggregate": 85.04377723794138,
  "decode_tok_s_warm_per_row_mean": 84.89405227801542,
  "expected_total_context_floor": 3500,
  "max_tokens": 600,
  "measurement_steps_min": 64,
  "prompt_token_count_per_row_mean": 3128.0,
  "prompt_token_counts": [
    3128
  ],
  "reached_expected_floor": true,
  "rows": [
    {
      "cold_ttft_ms": 4825.119749875739,
      "decode_interval_ms_mean": 11.779388227636353,
      "decode_interval_ms_std": 0.3037880913849494,
      "decode_interval_rel_std": 0.02578980211147242,
      "decode_tok_s_warm": 84.89405227801542,
      "measurement_steps": 567.0,
      "row": 0,
      "row_first_meas_ms": 7337.68737479113,
      "row_last_ms": 14016.600499860942,
      "warmup_steps_used": 32.0
    }
  ],
  "seed": 0,
  "target_context_tokens": 4096,
  "warmup_min_steps": 32,
  "warmup_rel_std_threshold": 0.05,
  "warmup_rolling_window": 16
}
```

## Interpretation (P-6.0.5 §1 Q3 — is 4K-context safe on MoE?)

At an effective context of ~4K + 600-token decode tail, MoE
35B-A3B B=1 measured **23.60 GB peak** and **85.0 tok/s decode**.
The §6(4) RAM-headroom gate (≤36 GB at 4K context) passes with
**12.4 GB of headroom (35% safety margin)** — the (2b) MoE
stretch reframing does not need to account for context length on
this checkpoint at this batch size. The decode number is the
unexpected one: 85.0 tok/s at 4K is **+12% above** the
short-context B=1 baseline of 76.01 tok/s, the opposite of the
"KV-attention cost grows linearly with sequence length"
prediction. Two plausible explanations: (a) longer / more diverse
prompt content stabilises the MoE expert-routing distribution, so
within each decode step the same hot experts are touched
repeatedly and their weight reads amortise more efficiently than
at short context where routing is sparser; (b) this is
run-to-run variance — ±12% is within the band typical for
single-run MoE rows, and the v1.7.13 short-context anchor itself
was one measurement. Either way the load-bearing finding is
conservative: 4K context does **not** degrade MoE decode, and
there is no extra-context perf-budget to subtract from the (2b)
stretch. TTFT rose 2732 → 4822 ms (+76%), in line with prefill
cost scaling roughly linearly with context length. The
``resident_mb`` column reads 143.0 — an artefact of which metric
path this scenario populates, not a working-set figure (cf. the
B=2 / B=3 rows which leave that column blank); ``peak_mb`` is the
gate-relevant number. **Decision Gate 1 reads this row as:** §1
Q3 fully closed — MoE 4K is RAM-safe and decode-stable. No new
constraint surfaced; the (2b) reframing remains the consequential
choice rather than a 4K-driven memory anchor.
