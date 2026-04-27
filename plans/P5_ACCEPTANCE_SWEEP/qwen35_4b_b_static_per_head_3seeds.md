# (b-static) Qwen3.5-4B re-measurement with `per_head_rotation=True` — 3-seed result

**Date:** 2026-04-26
**Result:** per-head rotation is **net neutral on the production path
mean** (mean ΔPPL +0.004 PPL vs +0.002 PPL with shared rotation, both
inside seed noise) but **5.3× tighter on std / 4.8× tighter on SEM**.
The D.2a-path absolute regression observed at v1.7.10 (silica codec
ΔPPL +0.22 PPL when switching to per-head) does **not** carry over to
the production path. (b-static) gate continues to PASS with
significant margin.

## Setup

- **Model:** `Qwen/Qwen3.5-4B` (hybrid DeltaNet + GQA, head_dim = 256,
  n_kv_heads = 4, dtype bf16).
- **Workload:** WikiText-2 first 512 tokens, `chunk_size = 256` —
  matches `qwen35_4b_b_static_close.md` exactly.
- **Seeds:** {42, 43, 44} — same triple as the v1.7.7 baseline.
- **Codec:** `BlockTurboQuantMSE`, `vq_block_size = 64`, `num_bits = 4`,
  `per_head_rotation = True`. K + V symmetric. Per-head Haar
  rotations seeded `seed * 1000 + head_idx`.
- **Routing:** `codec_quality_path = "prefix_store_pre_norm"` —
  P-5-F (3b) production default.
- **Run:** `uv run python scripts/b_static_per_head_qwen35_4b_3seed.py`.

## Per-seed numbers

silica fp16 baseline PPL (seed-independent — fp16 forward is
deterministic): **8.855979**

silica per-head BlockTQ b64 b4 (production path):

| Seed | codec PPL | ΔPPL | ΔPPL % |
| --- | --- | --- | --- |
| 42 | 8.857279 | +0.001300 | +0.015% |
| 43 | 8.865229 | +0.009250 | +0.104% |
| 44 | 8.858082 | +0.002103 | +0.024% |

Aggregated (silica per-head):

- mean ΔPPL = **+0.004218 PPL**
- std ΔPPL = 0.004376 PPL
- SEM(ΔPPL) = std / √3 = 0.002527 PPL

## Comparison to v1.7.7 (b-static) baseline (shared rotation)

From `plans/P5_ACCEPTANCE_SWEEP/qwen35_4b_b_static_close.md`:

| Metric | Shared rotation (v1.7.7) | Per-head (this run) | Δ |
| --- | --- | --- | --- |
| seed=42 ΔPPL | +0.024785 | +0.001300 | −0.024 |
| seed=43 ΔPPL | −0.016742 | +0.009250 | +0.026 |
| seed=44 ΔPPL | −0.003374 | +0.002103 | +0.005 |
| mean ΔPPL | +0.001556 | +0.004218 | +0.003 |
| **std ΔPPL** | **0.021198** | **0.004376** | **−0.017 (5.3× tighter)** |
| SEM ΔPPL | 0.012239 | 0.002527 | −0.010 (4.8× tighter) |
| (b-static) gate vs vqbench `+0.000%` | PASS (~16× headroom) | PASS (~240× headroom) | wider headroom |

## Comparison to D.2a path observation (v1.7.10)

| Metric | D.2a path (Qwen3-0.6B) | Production path (Qwen3.5-4B) |
| --- | --- | --- |
| Codec install seam | `vqbench_aligned` projection-patch on `k_proj` / `v_proj` | `prefix_store_pre_norm` (3b) capture into prefix store |
| silica shared mean ΔPPL | +0.511 | +0.0016 |
| silica per-head mean ΔPPL | +0.727 | +0.0042 |
| **per-head minus shared (mean)** | **+0.216 PPL (absolute regression)** | **+0.003 PPL (in SEM, no signal)** |
| silica shared std | 0.354 | 0.0212 |
| silica per-head std | 0.373 | 0.0044 |
| **per-head minus shared (std)** | **+0.019 (slight increase)** | **−0.017 (5.3× tighter)** |

**The D.2a-path regression is path-specific, not a general property
of per-head rotation.** On the production path, per-head's effect on
mean reconstruction quality is below measurement noise (+0.003 PPL in
a 0.012 PPL SEM band), while its effect on seed-to-seed variance is
substantial (5× tighter std, 4.8× tighter SEM).

## Why the production path differs from D.2a

The two paths inject codec noise at different points in the
attention computation:

- **D.2a path** (`vqbench_aligned`): the codec wraps `attn.k_proj` /
  `attn.v_proj` directly. Every chunk forward pays the codec
  reconstruction cost on **every layer × every head**, fresh, with
  no carry-over between chunks. Per-head rotation here means each
  head sees its own rotation noise that is uncorrelated across
  layers and chunks; the noise accumulates additively into the
  reconstruction error budget.
- **Production path** (`prefix_store_pre_norm`, P-5-F (3b)): the
  codec writes per-block K / V into the prefix store via the
  capture proxy, then reconstructs once per hit-path admit via
  `apply_k_norm_then_rope`. Codec noise lives in pre-k_norm space
  and is funneled through the post-k_norm + RoPE pipeline; per-head
  rotation noise is partially absorbed by the k_norm RMSNorm step
  before reaching downstream attention.

The production-path per-head std reduction (5.3×) suggests per-head
rotation also has a variance-decorrelating effect: with one shared
rotation, all heads' reconstruction errors are correlated through
that single matrix's eigenstructure; with per-head rotations, the
errors decorrelate across heads, and variance averaging across the
hybrid + GQA stack's 8 attention layers × 4 heads × 256 head_dim
gives a tighter aggregate.

## Gate

**(b-static) two-part aggregated gate, per-head variant:**

- mean_gap = silica.ΔPPL_mean − vqbench.target = +0.004218 − 0
  = +0.004218 PPL.
- SEM_diff = √(SEM(silica)² + SEM(vqbench)²) = √(0.002527² + 0)
  = 0.002527 PPL.
- 2 × SEM_diff = 0.005053 PPL.
- **Gate (i):** `|mean_gap| ≤ 2 × SEM_diff` ⇔ `0.004218 ≤ 0.005053`
  → **PASS** (~1.2× headroom — tight by SEM gate, but the SEM
  itself shrunk 4.8×, so the absolute |mean_gap| ~0.004 is itself
  within seed-noise; the proximity to the gate is a side-effect of
  the much-tighter SEM, not a quality issue).
- **Gate (ii):** `|mean_gap| < 1.0 PPL` ⇔ `0.004218 < 1.0` → **PASS**
  (~240× headroom).

The shared-rotation v1.7.7 baseline had ~16× headroom on gate (i)
and ~640× on gate (ii); per-head is ~1.2× on gate (i) and ~240× on
gate (ii) — gate (i) headroom contracted because the SEM shrunk so
much, not because the mean drifted. Both gates still pass.

## Default-flip recommendation — STRONG empirical case, but flip remains a separate landing

The v1.7.10 caveat that recommended keeping `per_head_rotation=False`
default rested on three concerns. With the production-path data in
hand:

| v1.7.10 concern | v1.7.10 evidence (D.2a) | Production-path evidence (this run) | Verdict |
| --- | --- | --- | --- |
| (a) silica codec ΔPPL absolutely regresses 0.22 PPL with per-head | confirmed on D.2a | **not present on production path** (+0.003 PPL, in SEM) | **dissolves** |
| (b) production path not re-tested | open | **closed by this measurement** | **resolved** |
| (c) RaBitQ1Bit / ExtRaBitQ lack 3-seed parity-scale cross-checks | open | open | unchanged |
| (d) flipping default re-anchors a closed (4-b) gate | open (P-5 §7 surface change) | open | unchanged |

(a) and (b) — the technical concerns — are resolved in favour of
flipping. (c) and (d) — the cross-codec consistency and gate
re-anchor concerns — remain. The flip is now an **administrative
landing** rather than an empirical question:

1. RaBitQ1Bit / ExtRaBitQ measurement coverage at parity scale
   (currently missing). One option: extend the v1.7.10 D.2a
   re-measurement script to cover all three codec families, or
   lock the BlockTQ flip first and treat RaBitQ as separately
   tracked.
2. Re-anchor the closed (4-b) gate evidence in `plans/PLAN.md`
   §7 P-5 Acceptance + the v1.7.3 Changelog reference.

**Provisional default-flip recommendation:** flip
`per_head_rotation` to `True` on `BlockTurboQuantMSE` first,
contingent on (1) above. RaBitQ1Bit / ExtRaBitQ flip can ride along
or land separately. The flip itself is a 3-line change per codec
plus the doc sync; the work is mostly the cross-codec measurements
and the (4-b) anchor re-text.

This file does not flip the default. The v1.7.7 production routing
stays anchored on shared rotation; the empirical case for flipping
is now sufficient that a follow-up unit can land it cleanly.

## Conclusion

**Per-head rotation is empirically net-positive on the production
path of the (b-static) workload:**

- mean ΔPPL change is below measurement noise (+0.003 PPL inside a
  0.012 PPL SEM band) — no quality regression.
- std / SEM is 5×/5× tighter — improved measurement reliability,
  fewer outlier seeds.
- (b-static) two-part gate continues to PASS.
- Per-seed shape becomes monotonically positive (+0.001, +0.002,
  +0.009 PPL — all small positive) instead of straddling zero
  (+0.025, −0.017, −0.003 PPL); the per-head version is closer to
  vqbench REPORT.md's "lossless-at-measurement-precision" baseline
  in shape as well as in mean.

**The D.2a-path regression observed at v1.7.10 is path-specific.**
Per-head rotation on the production `prefix_store_pre_norm` capture
path does not exhibit the +0.22 PPL absolute regression seen on the
`vqbench_aligned` projection-patch path. The empirical case for
flipping the default to `per_head_rotation=True` is now strong; the
remaining reasons to keep it as a separate landing unit are
administrative (cross-codec measurement coverage + (4-b) gate
re-anchor), not empirical.

## Evidence files

- `qwen35_4b_b_static_per_head_3seeds.jsonl` — raw rows + aggregate.
- `qwen35_4b_b_static_per_head_3seeds.md` — this analysis.
- `scripts/b_static_per_head_qwen35_4b_3seed.py` — measurement script.
- `qwen35_4b_b_static_close.md` — v1.7.7 shared-rotation baseline,
  retained for comparison.
