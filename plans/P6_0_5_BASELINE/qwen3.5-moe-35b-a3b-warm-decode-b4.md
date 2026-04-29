# silica-mlx bench report

Generated: 2026-04-29T11:56:36

Scenarios: total=1 Runs: total=1 ok=1 skipped=0 failed=0

## Results

| id | codec | runs | ok | skipped | failed | ttft_ms | decode_tok_s | resident_mb | peak_mb | wall_s | tokens | vqbench_gap |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| qwen3.5-moe-35b-a3b-warm-decode-b4 |  | 1 | 1 | 0 | 0 | 981.3 | 188.5 |  | 20619.7 | 11.763 | 1536 |  |

## Scenario details

### `qwen3.5-moe-35b-a3b-warm-decode-b4` (seed=0)

- repo: `mlx-community/Qwen3.5-35B-A3B-4bit`
- oracle: `warm_decode`
- gate: `SILICA_REAL_QWEN3_5_MOE`
- workload: `max_batch_size=4`, `max_tokens=384`, `prompts=4`
- status: **ok**

**P-6 MoE stretch — opt-in B=4 (OOM risk).** Same checkpoint as the B=2 primary row but pushes batched activation state into the 40 GB+ regime, uncomfortably close to the 48 GB system envelope. **B=4 has not been validated on real hardware**: existing MoE batched tests (tests/test_p3_qwen3_5_moe_batched_parity.py at v1.7.9) pin B=2 only, with a `_release_mlx_state` between forwards. If this scenario OOMs the user sees an opaque MLX error mid-decode; treat it as opt-in stretch and fall back to the B=2 row for the §6 gate. Run sequence: validate B=2 first; only then run B=4 on a freshly booted Mac with no other GPU consumers, ideally with `mx.metal.set_memory_limit` set to ~42 GB to fail fast rather than letting macOS swap.

Metadata:

```
{
  "aggregate_overlap_decodes": 1402,
  "aggregate_overlap_window_ms": 7436.486249999973,
  "codec_id": null,
  "decode_tok_s_warm_aggregate": 188.5298987811623,
  "decode_tok_s_warm_per_row_mean": 47.19969455442517,
  "measurement_steps_min": 64,
  "rows": [
    {
      "cold_ttft_ms": 981.2711669999885,
      "decode_interval_ms_mean": 21.186574430199382,
      "decode_interval_ms_std": 0.462752062186218,
      "decode_interval_rel_std": 0.02184175944585975,
      "decode_tok_s_warm": 47.19970202330577,
      "measurement_steps": 351.0,
      "row": 0,
      "row_first_meas_ms": 1660.2367919999779,
      "row_last_ms": 9096.724416999961,
      "warmup_steps_used": 32.0
    },
    {
      "cold_ttft_ms": 981.27925,
      "decode_interval_ms_mean": 21.18657775213688,
      "decode_interval_ms_std": 0.46276463818986946,
      "decode_interval_rel_std": 0.02184234960472533,
      "decode_tok_s_warm": 47.19969462265513,
      "measurement_steps": 351.0,
      "row": 1,
      "row_first_meas_ms": 1660.2379169999608,
      "row_last_ms": 9096.726708000006,
      "warmup_steps_used": 32.0
    },
    {
      "cold_ttft_ms": 981.2801669999658,
      "decode_interval_ms_mean": 21.18657929629629,
      "decode_interval_ms_std": 0.46276797485350346,
      "decode_interval_rel_std": 0.02184250550226396,
      "decode_tok_s_warm": 47.199691182559796,
      "measurement_steps": 351.0,
      "row": 2,
      "row_first_meas_ms": 1660.2380420000031,
      "row_last_ms": 9096.727375,
      "warmup_steps_used": 32.0
    },
    {
      "cold_ttft_ms": 981.2804169999936,
      "decode_interval_ms_mean": 21.1865796524216,
      "decode_interval_ms_std": 0.46276682440503847,
      "decode_interval_rel_std": 0.021842450834301835,
      "decode_tok_s_warm": 47.199690389179985,
      "measurement_steps": 351.0,
      "row": 3,
      "row_first_meas_ms": 1660.2381669999886,
      "row_last_ms": 9096.72762499997,
      "warmup_steps_used": 32.0
    }
  ],
  "seed": 0,
  "warmup_min_steps": 32,
  "warmup_rel_std_threshold": 0.05,
  "warmup_rolling_window": 16
}
```

## Interpretation (P-6.0.5 §1 Q2 — does MoE saturate before B=4? — opt-in B=4 closure)

The B=4 row completed without OOM (peak 20.62 GB, far below
the 36 GB envelope) and pins the MoE batch-scaling curve to B=4:

| row | aggregate tok/s | per-row tok/s | scaling efficiency vs B=1 | bandwidth util |
| --- | ---: | ---: | ---: | ---: |
| B=1 (P-6.0 baseline) | 76.01 | 76.01 | — | 37.1% |
| B=2 (P-6.0 baseline) | 120.93 | 60.47 | 79.6% of 2× | 59.1% |
| B=3 (P-6.0.5 Unit 3) | 163.50 | 54.50 | 71.7% of 3× | 79.9% |
| **B=4 (this row)** | **188.50** | **47.13** | **62.0% of 4×** | **92.1%** |

**§1 Q2 closes positively: MoE does not saturate before B=4.**
Bandwidth utilisation climbs monotonically from 37% → 92% across
B=1 → B=4, exactly the "≥90% headroom-was-unsaturated" arm that
§1 Q1 hypothesised. The aggregate scaling efficiency does fall
modestly — 80% (B=2) → 72% (B=3) → 62% (B=4) — but the curve
still rises strongly: B=3 → B=4 delivers a 1.153× speedup
(aggregate 163.5 → 188.5), which is **86.5% of the ideal 1.333×
B=3 → B=4 jump**. At 92% bandwidth utilisation, MoE B=4 sits
just below the structural ceiling defined by per-step bytes;
B=5 / B=6 may yield small additional aggregate gains but most
remaining bandwidth has been claimed.

**Cross-family contrast (the load-bearing finding for Decision
Gate 1).** Combining this row with Unit 2's dense 27B B=4:

| family | B=1 util | B=4 util | per-row B=1 → B=4 | scaling regime |
| --- | ---: | ---: | ---: | --- |
| Dense 27B | 79.1% | 52% (drops) | 16.05 → 10.54 (−34%) | KV-traffic-bound at B≥4 |
| MoE 35B-A3B | 37.1% | 92% (climbs) | 76.01 → 47.13 (−38%) | weight-read-bound throughout |

The same hypothesis ("does utilisation climb with batch?")
splits the two families onto opposite arms. Dense had little
bandwidth headroom at B=1 (already 79.1% util on the corrected
15.13 GB weight footprint), so adding batch surfaces KV/activation
traffic faster than it amortises weights. MoE had abundant headroom at
B=1 (37.1% util on a 1.5 GB active-weights-read), so batch
parallelism consumes it cleanly until weight reads saturate
near 92%. Per-row throughput drops by similar percentages
(−34% dense, −38% MoE) but the underlying mechanisms differ:
dense's drop is the bandwidth-bound regime giving up to KV traffic;
MoE's drop is the unsaturated regime aggregating onto a fixed
weight read.

**§6(2) ≥100 tok/s gate** is exceeded at B=4 by **88%** (188.5 vs
100). The (2b) reframing branch:

- **≥150 tok/s aggregate variant** — already cleared at B=3
  (163.5) and exceeded at B=4 (188.5 = 1.26× stretch); a
  reasonable next anchor would be ≥175 or ≥200 tok/s aggregate
  to keep the gate informative beyond B=4.
- **≥100 tok/s per-row variant** — unreachable on this
  checkpoint at any feasible batch (per-row drops with B; B=4
  per-row 47.1 is well short of 100, and lower B is per-row
  76 / 60 / 54).

The aggregate variant is therefore the live (2b) candidate; the
per-row variant should be retired or moved to a different
checkpoint family.

**Memory observations** (sanity-checking §6(4)): peak rose from
20.42 → 20.62 GB across B=3 → B=4 (+0.20 GB for one extra batch
row), substantially smaller than the +0.74 GB B=2 → B=3 marginal
and the dense per-row marginal (+0.42 GB at B=4). The trend is
consistent with allocator amortisation as batch grows; MoE has
substantial headroom remaining (≥15 GB margin to the §6(4) 36 GB
gate at B=4), so B=5 or B=6 is not memory-bound — only
diminishing-returns-bound on bandwidth.

**TTFT note**: 981.3 ms vs the v1.7.13 B=1 baseline 2732 ms is
counter-intuitive on a fresh-boot run — possible contributors are
prompt-length differences vs the v1.7.13 row (the warm-decode
workload is shared across batch sizes here, but the baseline used
a different default prompt) or external cache state that did
not reset across reboot. The decode_tok_s number is unaffected
because the warm-decode oracle discards the first 32 steps as
warmup before measurement.

**Decision Gate 1 reads this row as:** §1 Q2 fully closed — MoE
B=4 is at 92% bandwidth utilisation, not saturated, and OOM-safe.
The (2b) MoE reframing branch is empirically warranted with the
**aggregate variant** as the live candidate; the per-row variant
is structurally unreachable and should be retired. Combined with
Unit 2 (dense 27B B=4 = 42.17 ± 0.21 tok/s, 2-run, batch-only ceiling), the
Decision Gate 1 input set is now complete: **dense's primary gate
needs Track A + B + C**, **MoE's stretch reads as already cleared
on a re-anchored aggregate target**, and the cross-family
mechanism difference (utilisation drop on dense, climb on MoE)
is empirical rather than projected.
