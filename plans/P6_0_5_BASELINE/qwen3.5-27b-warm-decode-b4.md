# silica-mlx bench report

Generated: 2026-04-29T11:44:31

Scenarios: total=1 Runs: total=1 ok=1 skipped=0 failed=0

## Results

| id | codec | runs | ok | skipped | failed | ttft_ms | decode_tok_s | resident_mb | peak_mb | wall_s | tokens | vqbench_gap |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| qwen3.5-27b-warm-decode-b4 |  | 1 | 1 | 0 | 0 | 1478.2 | 42.4 |  | 17101.0 | 40.124 | 1536 |  |

## Scenario details

### `qwen3.5-27b-warm-decode-b4` (seed=0)

- repo: `mlx-community/Qwen3.5-27B-4bit`
- oracle: `warm_decode`
- gate: `SILICA_REAL_QWEN3_5_27B`
- workload: `max_batch_size=4`, `max_tokens=384`, `prompts=4`
- status: **ok**

**P-6.0.5 sub-unit 2 (D-021 step 3) — dense 27B B=4 opt-in (OOM risk).** Same checkpoint as the existing B=1 / B=2 dense rows but pushes 4x batched activation state on top of a 13.5 GB weight footprint, uncomfortably close to the 48 GB system envelope. **B=4 has not been validated on real hardware**: dense Qwen3.5-27B has only been exercised at B=1 (P-6.0 baseline) and B=2 (P-6.0.5 sub-unit 1). If this scenario OOMs the user sees an opaque MLX error mid-decode; treat it as opt-in stretch and fall back to the B=2 row for the §6 batch-scaling reading. Run sequence: validate B=2 first (sub-unit 1); only then run B=4 on a freshly booted Mac with no other GPU consumers, ideally with ``mx.metal.set_memory_limit`` set to ~42 GB to fail fast rather than letting macOS swap. Workload shape and ``max_tokens=384`` mirror the B=1 / B=2 rows on this checkpoint exactly so per-token throughput is comparable across batch sizes. Dual-gated on SILICA_REAL_QWEN3_5_27B — same checkpoint as the existing dense 27B rows. See plans/P6_0_5_OPENING.md §3.2.

Metadata:

```
{
  "aggregate_overlap_decodes": 1402,
  "aggregate_overlap_window_ms": 33083.193874917924,
  "codec_id": null,
  "decode_tok_s_warm_aggregate": 42.378012392054096,
  "decode_tok_s_warm_per_row_mean": 10.609615643220877,
  "measurement_steps_min": 64,
  "rows": [
    {
      "cold_ttft_ms": 1478.1862499658018,
      "decode_interval_ms_mean": 94.25411846998156,
      "decode_interval_ms_std": 1.0486963295071632,
      "decode_interval_rel_std": 0.01112626531901793,
      "decode_tok_s_warm": 10.609615964086323,
      "measurement_steps": 351.0,
      "row": 0,
      "row_first_meas_ms": 4605.325333075598,
      "row_last_ms": 37688.520916039124,
      "warmup_steps_used": 32.0
    },
    {
      "cold_ttft_ms": 1478.191165952012,
      "decode_interval_ms_mean": 94.25412120225911,
      "decode_interval_ms_std": 1.0487053344169874,
      "decode_interval_rel_std": 0.011126360535117395,
      "decode_tok_s_warm": 10.609615656530377,
      "measurement_steps": 351.0,
      "row": 1,
      "row_first_meas_ms": 4605.326541000977,
      "row_last_ms": 37688.523082993925,
      "warmup_steps_used": 32.0
    },
    {
      "cold_ttft_ms": 1478.1919999513775,
      "decode_interval_ms_mean": 94.25412262710304,
      "decode_interval_ms_std": 1.0487066065419095,
      "decode_interval_rel_std": 0.011126373863675973,
      "decode_tok_s_warm": 10.609615496144325,
      "measurement_steps": 351.0,
      "row": 2,
      "row_first_meas_ms": 4605.326832970604,
      "row_last_ms": 37688.523875083774,
      "warmup_steps_used": 32.0
    },
    {
      "cold_ttft_ms": 1478.1924579292536,
      "decode_interval_ms_mean": 94.2541229826507,
      "decode_interval_ms_std": 1.0487057183777564,
      "decode_interval_rel_std": 0.011126364398624675,
      "decode_tok_s_warm": 10.60961545612248,
      "measurement_steps": 351.0,
      "row": 3,
      "row_first_meas_ms": 4605.3270411212,
      "row_last_ms": 37688.524208031595,
      "warmup_steps_used": 32.0
    }
  ],
  "seed": 0,
  "warmup_min_steps": 32,
  "warmup_rel_std_threshold": 0.05,
  "warmup_rolling_window": 16
}
```

## Interpretation (P-6.0.5 §1 Q1 — does dense 27B scale with batch? — opt-in B=4 closure)

The B=4 row completed without OOM (peak 17.10 GB, well within
the 36 GB envelope) and pins the dense batch-scaling curve to
B=4. **This row was run twice** (see the JSONL — two `status=ok`
rows): 41.97 / 42.38 tok/s, σ = 0.21, 0.5% rel-std. The auto-
rendered Results table at the top of this file shows run 2 of 2
(`--report-md` is overwrite, `--out` is append). The
interpretation table below uses the **2-run mean of 42.17 ± 0.21
tok/s**:

| row | aggregate tok/s | per-row tok/s | scaling efficiency | bandwidth util |
| --- | ---: | ---: | ---: | ---: |
| B=1 (P-6.0 baseline) | 16.05 | 16.05 | — | 79.1% |
| B=2 (P-6.0.5 Unit 1) | 31.22 | 15.61 | **97% of 2×** | 76.9% |
| **B=4 (this row, 2-run mean)** | **42.17 ± 0.21** | **10.54** | **66% of 4×** | **52.0%** |

Bandwidth utilisation here uses the corrected **15.13 GB**
weight footprint from `mlx.utils.tree_flatten(model.parameters())`,
not the v1.7.13 P-6.0 anchor of 13.5 GB; absolute % values shift
~+12% but ratios are preserved (see
`target_verify_microbench.md` "Weight-footprint reconciliation"
for the full provenance).

**The "utilisation jumps to >90%" arm of §1 Q1 is decisively
rejected** for dense 27B: bandwidth utilisation does not climb
with batch — it **drops** from 79.1% (B=1) to 52.0% (B=4). The
B=2 row already showed near-perfect linear scaling at slightly
lower per-row efficiency, and the B=4 row now shows the curve
bending sharply: from B=2 → B=4, scaling efficiency falls from
97% to 66%, and per-row throughput drops from 15.61 to 10.54
tok/s (−32%). The mechanism is KV / activation traffic surfacing
as bytes/step grows with batch — at B=4 the weight read is no
longer the only meaningful bandwidth consumer, so the
ceiling-tok/s formula (which assumes weights amortise across
batch) over-estimates the achievable aggregate.

**§6(1) ≥60 tok/s gate — the load-bearing finding.** Linear
extrapolation from the B=1 anchor predicted B=4 = 64.2 tok/s
(just over the 60 gate); the measured 42.17 ± 0.21 tok/s
(2-run) is **70.3% of the gate, with a 17.83 tok/s residual
gap**. The
**(1b) batch-only path to 60 tok/s is dead** — beyond B=4, per-row
degradation accelerates and memory grows, and B=8 on 48 GB is
infeasible without aggressive tricks (no 4× headroom). Combined
with Unit 7's verify-k ceiling of 2.93× over autoregressive,
§6(1) is reachable only through the **composite Track A + B +
C.4-or-C.5 stack**, not through any single-track or batch-only
path. PLAN.md §1.3's mid-band stack arithmetic
(A 1.05–1.15× × B 1.30× × C.4 2.0–4.0×) projects 16.05 → 33–96
tok/s across that band; this row anchors the lower-band reality
empirically and constrains where the stack must land
collectively to clear 60.

**Memory observations** (sanity-checking the §6(4) RAM gate):
peak rose 16.26 → 17.10 GB from B=2 → B=4 (+0.84 GB for two
extra batch rows = +0.42 GB/row), about half the per-row marginal
of B=1 → B=2 (+0.91 GB/row). The pattern is consistent with some
allocator buffers being amortised rather than scaling strictly
linearly with batch. §6(4) 36 GB / 4K-context headroom is
unaffected at any feasible dense batch size.

**TTFT note**: 1478.2 ms is **lower** than the B=1 / B=2 cold-call
TTFT (~1920 ms). This is not a B=4 property — it reflects warm
OS-page-cache + MLX-kernel-cache state from the prior runs in
the same shell session (Units 1, 6, and 7 all preceded this
row). The decode_tok_s number is unaffected because the
warm-decode oracle discards the first 32 steps as warmup before
measurement.

**Decision Gate 1 reads this row as:** §1 Q1 fully closed — dense
27B is bandwidth-bound at B=1 and KV-traffic-bound at B≥4; the
batch-only ceiling sits well below the §6(1) 60 tok/s gate
(42.17 ± 0.21 < 60). Combined with §1 Q4 (Unit 7 verify-k 2.93×
target-side ceiling),
the decision space narrows to **(1a) lower the gate to a band
reachable from the composite stack** (e.g. 40 tok/s, which A+B
alone may approach) **or (1b) accept that the 60 tok/s gate
requires C.4 / C.5 to land in the upper half of their MLX
conservative bands, on top of A and B**. Pure batch scaling does
not get there.
