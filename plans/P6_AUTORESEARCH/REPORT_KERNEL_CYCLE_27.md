# P-6 Autoresearch — twenty-seventh cycle (2026-05-04) — Codex review correction; v10 attribution revised

| Field | Value |
| --- | --- |
| Date | 2026-05-04 |
| Branch | `opus` (merged from `opus-codex`) |
| Status | **Cycle-14's "+5.4 tok/s KEEP from v10+bf16 stack" was attribution error.** Codex review found v10 was NEVER firing in cycles 12/14/15+ E2E (dtype defect: shadow_install checked `mx.float16` but Qwen3.5 decode is bf16). After fix: v10 fires correctly, but at B=52 the E2E delta is +0.5 tok/s — within noise. |
| User authorization | Codex review delivered as opus-codex branch; user requested integration |
| Companion docs | cycles 24-26 reports (Codex), c25_b52_reverify.md, c26_b52_bf16_fa_run1.md |

## TL;DR

Codex (GPT-5.5) ran a comprehensive review of the cycle 1-23 work and
caught a critical defect in my cycle-12 shadow install patch:

```python
# silica/kernels/shadow_install.py (cycle 12 defect):
v10_eligible = (
    use_fa_v10
    and T_q == 1
    and head_dim == 256
    ...
    and queries.dtype == mx.float16   # ← BUG: Qwen3.5 decode is bf16
)
```

The Qwen3.5-27B-4bit model's attention activations are `bfloat16`, not
`float16`. So the v10 FA-decode kernel **never fired** in any of my
cycle 12, 14, 15, 16, 23 E2E measurements. The benches were running the
un-patched mlx fallback every time.

**Implications for the cycle-14 KEEP claim:**
- Cycle 14 reported B=52 v10+bf16 = 206.2 ± 0.5 tok/s as a "3.4σ KEEP"
  over cycle-13 bf16-only = 200.8 ± 1.5
- But v10 was not firing in either measurement
- Both were measuring the same thing (bf16 state alone) with run-to-run noise
- The +5.4 tok/s was within-band variance, not a kernel-attributable lift

## Codex's fixes

Codex landed the following on `opus-codex`:

1. **shadow_install.py**: changed `queries.dtype == mx.float16` to
   `queries.dtype in (mx.float16, mx.bfloat16)` — admits bf16 path
2. **flash_attention_decode_v{8,10}.py**: generates bf16-native Metal
   source via string substitution (`half4` → `bfloat4`,
   `metal::dot(half4, half4)` → `metal::dot(float4(...), float4(...))`).
   Caches kernels separately by dtype.
3. **tests/test_flash_attention_decode.py**: added bf16 correctness
   coverage at production shapes
4. **pyproject.toml + uv.lock**: pinned `mlx==0.31.1`, `mlx-lm==0.31.2`,
   `mlx-metal==0.31.1` (was previously `>=0.22`). Codifies the cycle-11
   determinism finding into the project lock.

Plus 3 cycle reports (24-26) and a `bench_with_mlx_limits.py` for the
40 GB cliff probe (deferred per Codex's recommendation).

## Cycle-27 verification (post-merge)

After merging `opus-codex` into `opus`:

**Test suite**: 55 tests pass on
`pytest tests/test_p2_preload_parity.py tests/test_flash_attention_decode.py
tests/test_fa_decode_shadow_install.py`.

**v10 firing verification**: 4-token Qwen3.5-27B-4bit decode through
the patched model:
- Prefill: v10 calls = 0 (correct, prefill is T_q > 1)
- Decode (T_q = 1): v10 calls = 32 = 16 full-attention layers × 2 decode steps

Confirms v10 IS firing correctly at the bf16 production path now.

**B=52 reverify (5 reps with v10 firing)**:

| Run | tok/s | wall (s) |
| --- | ---: | ---: |
| 1 | 202.8 | 194.7 (outlier wall, thermal hiccup) |
| 2 | 204.6 | 115.6 |
| 3 | 204.9 | 115.3 |
| 4 | 206.2 | 114.0 |
| 5 | 205.1 | 115.1 |
| **mean ± std** | **204.7 ± 1.2** | — |

**B=52 bf16-state-only reverify (v10 disabled, 3 reps)**:

| Run | tok/s |
| --- | ---: |
| 1 | 205.0 |
| 2 | 204.6 |
| 3 | 202.9 |
| **mean ± std** | **204.2 ± 1.1** |

**Δ between bf16+v10 and bf16-alone**: +0.5 tok/s. Pooled σ ≈ 1.6.
Δ/σ ≈ 0.3 — **statistically indistinguishable**.

## Honest revised running-best line

| Configuration | Cycle | tok/s | Status |
| --- | --- | ---: | --- |
| C10 baseline B=48 fp32 | 10 | 193.9 ± 0.6 (n=3) | KEEP |
| ~~C13 bf16 state at B=52~~ | 13 | 200.8 ± 1.5 (n=3) | KEEP — but small σ underestimated noise |
| ~~C14 v10+bf16 stack at B=52~~ | 14 | 206.2 ± 0.5 (n=3) | **REVISED — v10 was not firing; small σ underestimated noise** |
| **C27 bf16 state at B=52 (corrected)** | 27 | **204.2 ± 1.1 (n=3)** | KEEP — within-envelope running-best |
| **C27 bf16 + v10 at B=52 (firing)** | 27 | **204.7 ± 1.2 (n=5)** | within noise of bf16-alone |

The honest within-envelope running-best is **B=52 bf16 ≈ 204.5 tok/s
with run-to-run σ ~1-2 tok/s**, regardless of whether v10 fires.

Hardware ceiling at B=64 likely needs a re-measurement under the same
correction; cycle-14's 232.2 number was also produced with v10 not firing.

## Why v10 doesn't help E2E even when it fires

The bf16 microbench (Codex measured) shows v10 vs mlx SDPA:
- T=128: 2.14×
- T=256: 1.63×
- T=512: 1.37×
- T=1024: 1.28×

These are real kernel-level wins. But at B=52 production decode:
- Attention is ~21.9% of step time per cycle-1 decomposition (at B=4)
- At B=52 the proportion is similar or smaller
- Even a 1.3× attention speedup × 22% step share = ~7% theoretical E2E
- Observed: 0.5 tok/s = 0.25% E2E

The shortfall is consistent with cycle-12's "v10 at B=48 = 0% E2E" finding.
**Kernel-level attention wins do not translate to throughput at this
high-B regime** because the per-step time is dominated by other costs
(DeltaNet 74% + dispatch + scheduler).

## What this means for the autoresearch loop's deliverables

The cycle-14 "v10+bf16 stack KEEP" claim that anchored cycles 14-23 was
attribution-corrupt. The honest replacement:

**Within strict 36 GB envelope**:
- ~~206.2 ± 0.5 tok/s (4.89× cycle-1, attributed to v10+bf16 stack)~~
- **204.5 ± ~1.5 tok/s (4.85× cycle-1, attributed to C12 bf16-state +
  C13 axis-shift; v10 contribution within noise)**

**Within 48 GB hardware ceiling**: cycle-14's 232.2 ± 0.3 at B=64 needs
re-measurement under the corrected v10 path. Likely revises down by
~1-3 tok/s based on the B=52 pattern.

**Stop conditions** per AR.md "Stop conditions" still cleared:
- (1b) ≥60 milestone CLEARED 3.41× (envelope) — still passes by margin
- New running-best ≥3σ above 42.17 with clean attribution: **YES, but
  the lever attribution is C10 axis-shift × C12 bf16 peak-save, NOT
  v10 FA-decode kernel**

## Stability concern

Codex noted that B=52 reverify showed environment-dependent variance:
- uv environment: 185.3 ± 1.9 tok/s
- conda python: 200.1 ± 6.0 tok/s
- After Codex's bf16 fix: 204.0 / 190.0

My cycle-27 in the uv environment (same as Codex's first attempt) shows
204-206 range, n=8 total runs. Codex's 185 result is concerning. Possible
causes:
- Thermal state (long sessions vs cold start)
- Other processes competing for memory bandwidth
- mlx-metal's caching behavior across runs

The honest σ for B=52 is closer to **±5 tok/s** when accounting for
between-session variance, not the ±0.5-1.5 shown in same-session 3-run
batches.

## Files added in cycle 27

- `plans/P6_AUTORESEARCH/REPORT_KERNEL_CYCLE_27.md` — this file
- `/tmp/c27_b52_run{1..5}.jsonl` — bf16+v10 reverify artefacts
- `/tmp/c27_b52_bf16only_run{1..3}.jsonl` — bf16-only baseline artefacts

## Ledger row added cycle 27

- `AR_C27_V10_ATTRIBUTION_CORRECTION` (KEEP-revision) — Codex review found
  v10 dtype defect (shadow_install checked fp16, Qwen3.5 is bf16). v10
  never fired in cycles 12/14/15+. After fix, B=52 bf16+v10 = 204.7 ±
  1.2 (n=5) vs bf16-alone 204.2 ± 1.1 (n=3) — within-noise. Cycle-14's
  "+5.4 tok/s KEEP from v10+bf16 stack" was attribution error; the
  true within-envelope running-best is **204.5 ± ~1.5 tok/s at B=52
  bf16 state alone**, attributed to C10 axis-shift × C12 bf16 peak save.
  v10 has measurable kernel-level wins (1.28-2.14× over mlx in
  microbench) that don't translate to E2E at B=52.

## What cycle 28+ should do

Per Codex's recommendation: **stabilize B=52 reverify before pursuing
the 40 GB cliff probe**.

Specific cycle-28 work:
1. Re-measure B=64 under the corrected v10 path (3-5 reps) to update
   the demonstrated-ceiling attribution
2. Run B=52 across multiple sessions (cold start vs warm) to characterize
   the between-session variance
3. Update `plans/P6_AUTORESEARCH_FINAL_REPORT.md` and AR.md with the
   corrected running-best line
4. Re-render charts with the corrected numbers

Only after the running-best line is genuinely stable should we proceed
to the cache-limit / 40 GB cliff probe.
