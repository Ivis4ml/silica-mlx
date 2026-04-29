# Target-Verify Microbench Report

- **Repo**: `mlx-community/Qwen3.5-27B-4bit`
- **Host**: `macOS-26.4.1-arm64-arm-64bit-Mach-O`
- **Timestamp**: `2026-04-29T18:20:25Z`

P-6.0.5 sub-unit 7 (D-021 step 3). Each row times one ``forward(model, candidate_arr, cache_list)`` call with the prefix KV pre-primed and untimed. Cache isolation: ``cache_list`` rebuilt fresh per rep via ``mlx_cache.make_prompt_cache(model)``.

| verify_k | cand_tokens | prefix_tokens | p50 (ms) | p95 (ms) | marginal (ms) | peak (MB) | kv bytes (est) |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 1 | 112 | 59.59 | 59.89 | +0.00 | 15447.4 | 58_720_256 |
| 2 | 2 | 112 | 62.07 | 62.26 | +2.48 | 15450.7 | 58_720_256 |
| 4 | 4 | 112 | 89.01 | 89.33 | +29.42 | 15454.7 | 58_720_256 |
| 8 | 8 | 112 | 162.96 | 163.22 | +103.37 | 15464.4 | 58_720_256 |

``marginal = forward_ms_p50[k] - forward_ms_p50[k=1]``. A flat curve (marginal ≈ 0 across k) means the verify forward is bandwidth-bound on weights and the candidate slice adds negligible compute; a steep curve means the candidate slice is in the compute-bound regime and speculative-decoding ROI is sensitive to drafter acceptance rate.

## Interpretation (P-6.0.5 §1 Q4 — what is the verify-k cost curve?)

The measured curve is **neither flat nor linear** — there is a clear
regime transition between k=2 and k=4. Bandwidth utilisation
(estimated as `weight_bytes_per_forward / forward_ms_p50` against
the M5 Pro's 307 GB/s peak), using the runtime-measured weight
footprint **15.13 GB** from `mlx.utils.tree_flatten(model.parameters())`
(see "Weight-footprint reconciliation" note below):

| verify_k | forward_ms_p50 | bandwidth util | per-extra-tok marginal |
| ---: | ---: | ---: | ---: |
| 1 | 59.59 | **82.7%** (≈ warm-decode-b1 79.1%) | — (baseline) |
| 2 | 62.07 | **79.4%** | 2.48 ms/tok |
| 4 | 89.01 | **55.4%** | 9.81 ms/tok |
| 8 | 162.96 | **30.2%** | 14.77 ms/tok |

**At k≤2 the verify forward sits in the same regime as
warm-decode-b1** (79–83% util on the corrected anchor) — adding a
second candidate token costs only 2.48 ms (4% of the k=1 baseline
59.59 ms), because both tokens share the same weight stream.
**At k≥4 the candidate-side compute**
(matmuls + attention + FFN over the k-token slice) **leaves
headroom on bandwidth and becomes the bottleneck**: utilisation
drops from 79% (k=2) to 55% (k=4) and to 30% (k=8).

**Speculative-decoding target-side ceiling.** With all k candidate
tokens accepted (perfect drafter) **and** zero drafter-forward
cost, the speedup over autoregressive equivalent (k × 59.59 ms)
is:

| k | verify (ms) | autoregressive equiv (ms) | target-side ceiling |
| ---: | ---: | ---: | ---: |
| 2 | 62.07 | 119.18 | 1.92× |
| 4 | 89.01 | 238.36 | 2.68× |
| 8 | 162.96 | 476.72 | **2.93×** |

The k=8 ceiling of **2.93×** is the **target-side / zero-drafter-cost
ceiling** for k≤8 linear verify against the dense 27B target on
this hardware: it is the speedup achievable when (a) the drafter
incurs zero forward cost, (b) drafter acceptance is 100% on all
k tokens, and (c) candidate shape stays linear. Real
end-to-end speedup falls below this ceiling by drafter-forward
cost, drafter acceptance probability, and the bonus-token rule;
the table below decomposes those terms. Per
the PLAN.md §1.3 stack, this **constrains the upper end of the
"conservative MLX" bands** for Tracks C.4 / C.5 quoted there:

- **C.4 DFlash** (claimed 6× on GPU; PLAN.md conservative MLX
  band 2.0–4.0×): the upper-band 4.0× is provably unreachable at
  k=8 linear; the realistic landing is 2.0–2.9×.
- **C.5 DDTree** (claimed 8.2× on GPU; PLAN.md conservative MLX
  band 2.5–5.0×): the upper-band 5.0× is unreachable at k=8
  linear; ≥3.0× requires the **tree shape's amortisation**, which
  is structurally different from the linear k slice this row
  measures. C.5's 2.5–3.0× lower band is reachable at k=8 linear
  if drafter acceptance approaches 1.0; intermediate values
  require the tree benefit to be ≥10% of that 2.93× ceiling.

**Break-even drafter acceptance** (verify cost = expected
autoregressive cost saved):

| k | break-even acceptance |
| ---: | ---: |
| 2 | ≈ 50% (verify ≈ 1 autoregressive token) |
| 4 | ≈ 36% |
| 8 | ≈ 33% |

Larger k has *lower* break-even acceptance — i.e. larger draft
windows tolerate weaker drafters — but the marginal benefit per
additional candidate slot decays once the regime turns
compute-bound (k≥4). Engineering implication: **k=4 is the sweet
spot** for linear-shaped speculative on this hardware (acceptance
threshold low at 36%, verify still in transition regime at 55%
util). k=8 is reserved for very-high-acceptance drafters where
the 2.93× ceiling is approachable.

**Memory.** Peak rises only 17 MB across k=1→k=8 (15447 → 15464);
verify-k is essentially free on RAM headroom and does not
constrain the §6(4) gate at any feasible k.

**Decision Gate 1 reads this row as:** §1 Q4 closed with a
**hard target-side / zero-drafter-cost ceiling of 2.93× over
autoregressive for k≤8 linear verify**. Track C.4 / C.5 ROI
estimates that quoted bands above this ceiling require either
tree shape (C.5) or larger k with proportionally better drafter
acceptance; bandwidth-bound projections from the P-6.0 baseline
alone do not get there. The 60 tok/s dense gate (§6(1)) at the
upper realistic spec-decoding band of 2.5× reads to **40 tok/s**
from the 16.05 anchor — short of 60 unless paired with Track A +
Track B contributions (PLAN.md §1.3 arithmetic now has empirical
anchoring). The (1b) ≥60 tok/s gate likely needs the **A + B +
C.4-or-C.5 stack**, not any one track alone.

---

## Weight-footprint reconciliation

Bandwidth utilisation in the tables above uses **15.13 GB** as
the target's weight read per forward, derived from
`mlx.utils.tree_flatten(model.parameters())` totalling 4.20 B
elements / 15,132,806,656 bytes (matches the on-disk HF blob
cache size). The number includes both the 4-bit packed weight
tensors and their per-group `scales` / `zeros` metadata, which
mlx-lm streams alongside the weights at decode.

Two other weight-bytes figures appear in nearby artefacts and
should not be confused with the corrected anchor:

- **`weight_bytes_read_estimate=10,066,329,600` (10.07 GB)** in
  this scenario's `target_verify_microbench.jsonl` rows — an
  analytic fallback computed by
  `silica.bench.microbench.target_verify.estimate_weight_bytes_read`
  via the geometric formula `nl × h² × 12 / 2` because
  `model.args.num_parameters` is unset on this checkpoint. The
  fallback under-counts by both (a) under-estimating
  `num_parameters` and (b) using the naive 0.5 byte/param
  conversion that ignores 4-bit scale / zero metadata. The JSONL
  field is preserved as a reproducibility artefact, not as the
  anchor for the utilisation columns above.
- **13.5 GB** in `plans/P6_0_BASELINE/REPORT.md` — derived from
  the branded "Qwen3.5-27B" name × 0.5 byte/param. Closer to the
  truth than the JSONL fallback by accident (over-counting
  num_parameters offsets under-counting bits/param), but still
  ~12% below the runtime-measured 15.13 GB. The v1.7.13 ratios
  remain valid as relative comparisons; their absolute %
  utilisations read **roughly +12% higher** under the corrected
  anchor (e.g. v1.7.13's 70.6% at B=1 → 79.1%).

The qualitative findings — dense bandwidth utilisation drops
with batch (52% at B=4 corrected vs 79% at B=1 corrected),
verify-k bandwidth utilisation drops with k (30% at k=8 vs 83%
at k=1) — hold across all three weight-footprint estimates;
only the absolute % values shift.
