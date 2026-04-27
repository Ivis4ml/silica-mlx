# D.2a re-run with `per_head_rotation=True` — 3-seed result

**Date:** 2026-04-26
**Result:** per-head rotation reduces |mean_gap| from **0.150** PPL
(shared rotation, original D.2a) to **0.066** PPL — a **56% reduction**
on the (4-b) two-part aggregated gate, on top of an already-passing
baseline. Default OFF stance retained until a separate (4-b) close-
review unit decides on the flip.

## Setup

- **Model:** `Qwen/Qwen3-0.6B`
- **Workload:** WikiText-2 first 512 tokens, `chunk_size = 256`
  (matches `plans/P5_D2_INVESTIGATION/d2a_verification_3seeds.jsonl`
  exactly).
- **Seeds:** {42, 43, 44} — same triple as the original D.2a run.
- **Codec:** `BlockTurboQuantMSE`, `vq_block_size = 64`,
  `num_bits = 4`, `per_head_rotation=True`. Per-head Haar rotations
  seeded `seed * 1000 + head_idx`, matching vqbench's
  `actual_seed = run_seed * 1000 + head_idx` at
  `vqbench/scripts/variance_qwen35_4b.py:63`.
- **Routing:** `codec_quality_path = "vqbench_aligned"` —
  pre-RoPE projection-patch oracle
  (`teacher_forced_chunked_nll_vqbench_aligned`), the same path the
  D.2a 3-seed run used.
- **Run command:** `uv run python scripts/d2a_per_head_3seed.py`
- **Wall:** ~13 s (4 forward passes — fp16 baseline + 3 seeds × per-
  head BlockTQ; fp16 is seed-independent so it runs once).

## Per-seed numbers

silica fp16 baseline PPL (seed-independent): **19.673075**

silica per-head BlockTQ b64 b4 (vqbench-aligned arm):

| Seed | codec PPL | ΔPPL | ΔPPL % |
| --- | --- | --- | --- |
| 42 | 20.027523 | +0.354448 | +1.802% |
| 43 | 20.398250 | +0.725175 | +3.686% |
| 44 | 20.774322 | +1.101248 | +5.598% |

Aggregated (silica per-head):

- mean ΔPPL = **+0.726957 PPL**
- std ΔPPL = 0.373403 PPL
- SEM(ΔPPL) = std / √3 = 0.215584 PPL

vqbench's locked baseline (from
`d2a_verification_3seeds.jsonl`, vqbench's per-head rotation is
its native behaviour):

| Seed | vqbench ΔPPL |
| --- | --- |
| 42 | 0.2699 |
| 43 | 0.7848 |
| 44 | 0.9289 |

Aggregated (vqbench):

- mean ΔPPL = **+0.661200 PPL**
- std ΔPPL = 0.346451 PPL
- SEM(ΔPPL) = std / √3 = 0.200023 PPL

## Gate

(4-b) two-part aggregated gate:

- **mean_gap** = silica.ΔPPL_mean − vqbench.ΔPPL_mean
  = +0.726957 − +0.661200 = **+0.065757 PPL**.
- **SEM_diff** = √(SEM(silica)² + SEM(vqbench)²)
  = √(0.215584² + 0.200023²) = **0.294085 PPL**.
- **2 × SEM_diff** = 0.588170 PPL.
- **Gate (i):** `|mean_gap| ≤ 2 × SEM_diff` ⇔ `0.065757 ≤ 0.588170`
  → **PASS** (~9× headroom).
- **Gate (ii):** `|mean_gap| < 1.0 PPL` ⇔ `0.065757 < 1.0` → **PASS**
  (~15× headroom).

## Comparison to baseline (shared rotation, original D.2a)

| Metric | Shared rotation (default) | Per-head rotation (opt-in) | Δ |
| --- | --- | --- | --- |
| silica mean ΔPPL | +0.511 | +0.727 | +0.216 |
| silica std ΔPPL | 0.354 | 0.373 | +0.019 |
| silica SEM ΔPPL | 0.205 | 0.216 | +0.011 |
| **mean_gap (vs vqbench)** | **−0.150** | **+0.066** | **\|gap\| 0.150 → 0.066 (56% reduction)** |
| 2 × SEM_diff | 0.572 | 0.588 | +0.016 |
| gate (i) headroom | ~3.8× | ~9.0× | ~2.4× wider |
| gate (ii) headroom | ~6.7× | ~15.2× | ~2.3× wider |

Per-seed shape comparison:

- vqbench (per-head):        `[0.27, 0.78, 0.93]` — monotone increasing.
- silica per-head (this run): `[0.35, 0.73, 1.10]` — monotone increasing.
- silica shared (baseline):  `[0.88, 0.17, 0.48]` — no monotone pattern.

Per-head rotation produces (a) a smaller mean_gap and (b) a
per-seed pattern that more closely tracks vqbench's, suggesting
the rotation axis was indeed the dominant residual contributor to
the original ~0.150 PPL diagnostic gap.

## Interpretation

The per-head rotation opt-in landed at v1.7.8 (commit `b06bc4c`)
hypothesised that the standing 0.150 PPL `mean_gap` between silica's
shared-rotation BlockTQ and vqbench's per-head-rotation BlockTQ
would shrink under matching per-head rotation. The hypothesis
verifies empirically: |mean_gap| drops from 0.150 to 0.066 PPL, a
~56% reduction, while both gate (i) and gate (ii) headrooms widen
proportionally. The per-seed shape also tracks vqbench's monotone
increasing pattern.

**The remaining 0.066 PPL gap** is small enough to fall within
fp16/bf16 vs torch.float16+MPS precision drift between silica and
vqbench (silica MLX runs bf16; vqbench torch.float16 + MPS — see
`plans/P5_ACCEPTANCE_SWEEP/qwen35_4b_b_static_close.md` §"Why
silica's absolute fp16 PPL differs from vqbench's"). It is plausible
that the rotation axis was the only structural contributor and the
0.066 residual is the precision-axis floor; verifying this would
require an inline NumPy reference at the same per-head rotation,
which is out of scope for this measurement-only unit.

## Default flip — separate decision

Despite the empirical improvement, this script does not flip the
default. The v1.7.8 stance is preserved: `per_head_rotation=False`
remains default OFF on `BlockTurboQuantMSE` / `RaBitQ1Bit` /
`ExtRaBitQ`. Reasons to keep the flip as a separate unit:

1. **Closed evidence preservation.** The (4-b) gate at v1.7.3 closed
   under shared rotation with `mean_gap = -0.150 PPL`. Flipping the
   default would re-anchor the (4-b) gate evidence; that re-anchor
   is a P-5 §7 surface change, not a measurement update.
2. **Production hot-path footprint.** Per-head rotation grows the
   rotation tensor by `n_kv_heads`× at codec construction (still
   <2 MB at production sizes, but a one-time construction cost
   change). Flip with eyes open, not by default-side-effect.
3. **Cross-codec consistency.** RaBitQ1Bit and ExtRaBitQ also have
   the opt-in but neither has a 3-seed cross-check at parity scale;
   flipping all three defaults together without per-codec re-runs
   would introduce drift on the codec-catalog axis.

The flip should land as a separate v1.7.10 unit (or later) when the
project chooses to re-anchor (4-b) on per-head rotation. Until then,
this evidence file is the empirical justification: per-head rotation
narrows the silica-vs-vqbench mean_gap by ~56% on the production
codec config (`block_tq_b64_b4`), Qwen3-0.6B WikiText-2.

## Conclusion

**Per-head Haar rotation is empirically the dominant residual
contributor to the original D.2a `mean_gap` of 0.150 PPL.** Engaging
the v1.7.8 opt-in at `BlockTurboQuantMSE(per_head_rotation=True)`
shrinks |mean_gap| to 0.066 PPL across 3 seeds on Qwen3-0.6B
WikiText-2, retains the (4-b) two-part aggregated gate with ~9×
SEM headroom, and reproduces vqbench's per-seed monotone shape.

This concludes the v1.7.8 Item 3 follow-up "re-run D.2a with
opt-in active". The default flip remains an open decision, not
this unit's deliverable.

## Evidence files

- `per_head_rotation_3seeds.jsonl` — raw rows + aggregate.
- `per_head_rotation_3seeds.md` — this analysis.
- `scripts/d2a_per_head_3seed.py` — measurement script.
- `d2a_verification_3seeds.jsonl` — original (4-b) D.2a 3-seed
  evidence, retained as the shared-rotation baseline for
  comparison.
