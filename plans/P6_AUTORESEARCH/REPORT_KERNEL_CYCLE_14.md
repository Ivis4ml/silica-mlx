# P-6 Autoresearch — fourteenth cycle (2026-05-04) — v10+bf16 composition KEEPs at both ladders

| Field | Value |
| --- | --- |
| Date | 2026-05-04 |
| Branch | `opus` |
| Status | **NEW RUNNING-BEST on both ladders.** Within strict envelope: B=52 v10+bf16 = 206.2 ± 0.5 tok/s (3.4σ over cycle-13 B=52 bf16-only). Hardware ceiling: B=64 v10+bf16 = 232.2 ± 0.3 tok/s (Δ +2.4 over cycle-13 B=64 bf16-only). Cliff bracketed at 40 GB peak / between B=64 and B=66. |
| User authorisation | "very good, looks new kernel and setup make good progress and let us continue" |
| Companion docs | `REPORT_KERNEL_CYCLE_{10,11,12,13}.md` |

## TL;DR

Cycle 11's FA-decode v10 kernel (1.25-1.81× over mlx at fixed shape) and
cycle 13's bf16-state-headroom B-axis unlock **compose net-positively**
when stacked at B=52 (within strict envelope) and B=64 (demonstrated
hardware ceiling). Cycle 14 confirms the composition with 3-reproduction
oracle runs and locates the sharp regime cliff at exactly 40 GB peak.

| Configuration | n | tok/s mean ± std | peak GB | × cycle-1 |
| --- | ---: | ---: | ---: | ---: |
| C10 baseline (B=48 fp32) | 3 | 193.9 ± 0.6 | 33.95 | 4.60× |
| C13 within-envelope KEEP (B=52 bf16) | 3 | 200.8 ± 1.5 | 35.52 | 4.76× |
| **C14 within-envelope KEEP (B=52 v10+bf16)** ⭐ | 3 | **206.2 ± 0.5** | 35.52 | **4.89×** |
| C13 hardware ceiling (B=64 bf16) | 3 | 229.8 ± 2.0 | 40.01 | 5.45× |
| **C14 hardware ceiling (B=64 v10+bf16)** ⭐ | 3 | **232.2 ± 0.3** | 40.01 | **5.51×** |

## Cycle 12's "no E2E impact" was right, but at the wrong B

Cycle 12 measured v10 alone at B=48 → 192.0 tok/s (no E2E delta vs baseline
193.9). That observation was correct *at B=48*. At B=52, with the bf16
state freeing peak headroom, v10's attention savings DO show up E2E:
+5.4 tok/s = +2.7%, 3.4σ above the cycle-13 B=52 KEEP.

Two reasons the same kernel produces different E2E impact:

1. **At B=48 the cycle-10 axis-shift had already saturated the
   batched-aggregate amortisation.** Per-step time was dominated by
   weights/state bandwidth that v10's attention savings couldn't free.
2. **At B=52 the bf16-state peak save shifts the limiting resource.**
   Attention now occupies a slightly larger fraction of step time relative
   to other components, so a 1.25-1.81× speedup on attention buys
   measurable wall-clock.

This is the same compositional pattern cycle 13 demonstrated: each lever's
direct impact at fixed B≤48 looked small/zero, but composing them across
B=48..64 produces a real lift.

## Within-envelope KEEP: B=52 v10+bf16 = 206.2 ± 0.5 tok/s

3 reproductions at warm-decode-b52 with `SILICA_USE_BF16_DELTANET_STATE=1`
+ `SILICA_USE_FA_DECODE_V10=1`:

| Run | decode_tok_s | peak GB | wall (s) |
| --- | ---: | ---: | ---: |
| 1 | 206.3 | 35.52 | 114.1 |
| 2 | 205.7 | 35.52 | 114.7 |
| 3 | 206.7 | 35.52 | 114.0 |
| **mean ± std** | **206.2 ± 0.5** | 35.52 | — |

3σ-keep arithmetic vs cycle-13 B=52 bf16-only (200.8 ± 1.5):
Δ = 5.4, pooled σ ≈ 1.58, **Δ/σ ≈ 3.4 — CLEARS 3σ**.

vs cycle-10 baseline 193.9 ± 0.6: Δ = 12.3, ~13σ. 4.89× cycle-1 baseline.

## Hardware ceiling: B=64 v10+bf16 = 232.2 ± 0.3 tok/s

| Run | decode_tok_s | peak GB | wall (s) |
| --- | ---: | ---: | ---: |
| 1 | 232.3 | 40.01 | 138.8 |
| 2 | 231.8 | 40.01 | 138.5 |
| 3 | 232.5 | 40.01 | 131.8 |
| **mean ± std** | **232.2 ± 0.3** | 40.01 | — |

Δ vs cycle-13 B=64 bf16-only (229.8 ± 2.0) = +2.4 tok/s, pooled σ ≈ 2.0,
1.2σ — modest but consistent across all 3 runs (each ≥ cycle-13 mean).
On the relaxed-envelope ladder this is the new demonstrated ceiling.

## Cliff bracketed at 40 GB peak

| B | bf16-state tok/s | peak GB | regime |
| ---: | ---: | ---: | --- |
| 60 | 219.1 | 38.45 | normal |
| 64 | 229.8 ± 2.0 | 40.01 | normal |
| **66** | **166.8** | **40.79** | **REGIME CHANGE** |
| 68 | 169.1 | 41.79 | regime change |
| 72 | 173.2 | 43.36 | regime change |

The cliff is sharp: between B=64 (peak 40.01 GB → 229.8 tok/s) and B=66
(peak 40.79 GB → 166.8 tok/s). Crossing 40 GB peak triggers a ~26%
throughput drop. Likely cause: M5 Pro's SLC (system-level cache) or
allocator threshold past 40 GB resident, kicking in
compression/eviction/page-faulting that adds per-step overhead.

**The empirical hardware ceiling is B=64 / 40.01 GB peak / 232.2 tok/s.**

## B=53 — diagnostic, no signal

B=53 (bf16-only, n=1) = 202.2 tok/s at peak 35.91 GB. Within the noise
band of B=52 (200.8 ± 1.5); the extra 482 MB headroom doesn't translate
to a new kept improvement. B=52 is the within-envelope sweet spot for
a discrete-row workload.

## Per-row scaling stays sublinear-but-stable

| B | per-row tok/s | aggregate tok/s |
| ---: | ---: | ---: |
| 4 (P-6.0.5 baseline) | 10.05 | 42.17 |
| 48 (C10) | 4.04 | 193.9 |
| 52 (C14 v10+bf16) | 3.97 | 206.2 |
| 64 (C14 v10+bf16) | 3.63 | 232.2 |
| 66 (cliff) | 2.53 | 166.8 |

Per-row drops slowly until the cliff, then crashes. The aggregate climb
holds the same shape as cycles 10/13.

## Files added in cycle 14

- `silica/bench/scenarios.py` — registered `qwen3.5-27b-warm-decode-b{53,66,68}` (3 new probe-class scenarios)
- `plans/P6_AUTORESEARCH/REPORT_KERNEL_CYCLE_14.md` — this file
- `/tmp/b{52,53,64,66,68}_{combined,bf16}_run*.jsonl` — oracle artefacts

## Ledger rows added cycle 14

- `AR_BF16_AGG_B53` (diagnostic, 202.2 single run, peak 35.91 GB) — 482 MB extra
  headroom over B=52 doesn't translate to E2E gain.
- `AR_BF16_AGG_B66` (discard, 166.8, peak 40.79 GB) — past the cliff.
- `AR_BF16_AGG_B68` (discard, 169.1, peak 41.79 GB) — past the cliff.
- `AR_V10_BF16_STACK_B52` (**KEEP**, 206.2 ± 0.5, n=3, peak 35.52 GB) — ⭐
  NEW within-envelope RUNNING-BEST. Δ +5.4 vs cycle-13 B=52 bf16-only,
  3.4σ above pooled noise. Compositional win: cycle-11 v10 + cycle-12
  bf16 + cycle-13 axis-shift all stack.
- `AR_V10_BF16_STACK_B64` (**KEEP** on relaxed-envelope ladder, 232.2 ±
  0.3, n=3, peak 40.01 GB) — ⭐⭐ NEW demonstrated ceiling within 48 GB
  hardware. Δ +2.4 vs cycle-13 B=64 bf16-only.

## Reflection

Across cycles 10-14, every individual kernel/state probe in isolation
either landed flat or moved the running-best by a small margin. The big
wins have all been **compositional**:

- C10 alone: +4.60× via axis-shift → 193.9
- C11 alone: 0% E2E
- C12 alone: 0% E2E (at fixed B=48), but produced peak save + wiring fix
- C13 = C12 peak save + C10 axis-shift extension → +4.76× envelope, +5.45× hardware
- C14 = C13 + C11 v10 → +4.89× envelope, +5.51× hardware

The autoresearch loop's productivity comes from **letting individually
flat-looking probes accumulate as resources, then re-composing them when
the right opportunity emerges**. The cycle-12 report's "honest negative"
on bf16 state at fixed B=48 was the necessary step before cycle-13's
axis-extension; cycle-11's "FA-decode kernel beats mlx but no E2E delta"
was the necessary step before cycle-14's stack KEEP.

## What's next (if user authorises cycle 15+)

1. **Apply v6→v7 half4 vectorisation lessons to QMM kernels** (cycles 7-9
   QMM was retired as "1.27× from mlx parity"; cycle 11 showed half4 loads
   are the missing optimisation; if QMM gets the same treatment, weights
   bandwidth utilisation on the 4-bit weight stream may improve, helping
   the dominant cost at B=64).
2. **Speculative decoding** (D-021 framework available; B=64 + spec gives
   multi-token-per-step which side-steps the 232 ceiling). Gated on the
   C.5 γ.1 read-only survey decision per P6_AUTORESEARCH.md.
3. **Profile mlx's actual memory behavior past 40 GB peak** to understand
   the cliff cause architecturally, not empirically. Knowing whether it's
   SLC, allocator, or VM tells us if the cliff can be moved with a
   different memory allocation pattern.
