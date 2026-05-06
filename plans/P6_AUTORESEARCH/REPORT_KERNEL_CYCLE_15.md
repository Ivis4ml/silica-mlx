# P-6 Autoresearch — fifteenth cycle (2026-05-04) — local-optimum confirmation round

| Field | Value |
| --- | --- |
| Date | 2026-05-04 |
| Branch | `opus` |
| Status | Diagnostic — 5 probes, no new KEEP. Cycle-14 v10+bf16 stack at B=52 = 206.2 ± 0.5 (envelope) and B=64 = 232.2 ± 0.3 (hardware) confirmed as local optimum. |
| User authorisation | "Great, and let us continue optimizing" |
| Companion docs | `REPORT_KERNEL_CYCLE_{10..14}.md` |

## TL;DR

Five probes against the cycle-14 KEEPs, all flat or regression. The
v10+bf16 stack at B=52 / B=64 with `SPLIT_K=128` is a local optimum
along the levers tried in cycles 10-14.

| Probe | tok/s | vs cycle-14 KEEP | verdict |
| --- | ---: | --- | --- |
| C14 envelope KEEP (B=52 v10+bf16) baseline | 206.2 ± 0.5 | — | reference |
| B=53 v10+bf16 | 205.2 (n=1) | within noise | diagnostic |
| `SILICA_DECODE_CHUNK=2` at B=52 v10+bf16 | OOG: warmup-stability | gate fail | discard |
| B=52 v10+bf16+silu_mul (fused SwiGLU) | 204.6 (n=1) | -1.6, hurts | discard |
| B=52 v10+bf16, SPLIT_K=256 (fewer splits) | 202.7 (n=1) | -3.5, hurts | discard |
| B=52 v10+bf16, SPLIT_K=64 (more splits) | 203.3 (n=1) | -2.9, hurts | discard |

## Findings

**1. SPLIT_K=128 is the sweet spot.** Tried 64 and 256; both hurt.
Smaller splits cost merge launches; larger splits leave parallelism
on the table. The default chosen at cycle 11 lands cleanly.

**2. fused_silu_mul does not stack net-positively.** SwiGLU activation
in DeltaNet's MLP is small enough that the fused-kernel launch
overhead exceeds the saved HBM passes at B=52. Same finding as cycle 1
(silu_mul at fixed B=4 was 1.006× — honest negative).

**3. Chunked decode (chunk=2) still fails the warm-decode oracle's
stability gate** even with the smallest possible chunk. Confirms
cycle-4/5/12 finding that chunked-lazy-decode is incompatible with
warm-decode oracle's per-step rate-stability check.

**4. B=53 doesn't lift the within-envelope KEEP.** The 482 MB headroom
over B=52 to the 36 GB envelope absorbs one batch row worth of growth
(345 MB) but the per-step gain is below the noise floor of B=52.

## Why cycle 15 found no further lever

Cycle 14 already composed the major levers cycles 10-13 produced:

- C10 axis-shift (B≤48) — saturated at cycle 14
- C11 v10 FA-decode kernel — already in stack
- C12 bf16 DeltaNet state — already in stack
- C13 B-axis extension to 52..64 — already at the cliff (40 GB)

Remaining un-pulled levers per `REPORT_KERNEL_CYCLE_14.md` "What's next":

- **Apply v6→v7 half4 vectorisation lessons to QMM kernels** —
  substantial rewrite, plus must beat mlx's tuned QMM by enough to
  show E2E (mlx already at 32% bandwidth peak for QMM at production shape)
- **Speculative decoding** — gated on C.5 γ.1 read-only survey decision;
  cycle-1 αprobes showed accept rate ~0.09 with current drafters
- **`mx.compile` on the transformer block** — graph trace might amortise
  dispatch overhead at B=64 where launch overhead is significant
- **Profile mlx memory behaviour past 40 GB peak** — diagnostic; would
  inform whether the cliff is movable

None are quick probes; each needs the kind of sustained design effort
cycle 11/12 took. Cycle 15 confirms there are **no easy local lifts**
left along the cycles 10-14 lever set.

## Files added in cycle 15

- `plans/P6_AUTORESEARCH/REPORT_KERNEL_CYCLE_15.md` — this file
- 5 oracle JSONL artefacts in `/tmp/b52_*_run1.jsonl`

## Ledger rows added cycle 15

- `AR_C15_B53_STACK` (diagnostic, 205.2 single run, peak 35.91 GB) —
  one extra batch row vs B=52; in noise band of B=52 v10+bf16 KEEP.
- `AR_C15_CHUNK2_FAIL` (discard, OOG warmup-stability) — re-confirms
  cycle 4/5/12 chunked-decode incompatibility with warm-decode oracle.
- `AR_C15_SILU_MUL_STACK` (discard, 204.6, -1.6 vs C14 KEEP) — fused
  SwiGLU does not stack net-positively at B=52.
- `AR_C15_SPLITK_256` (discard, 202.7, -3.5) — fewer splits cost gain
  from K-axis parallelism.
- `AR_C15_SPLITK_64` (discard, 203.3, -2.9) — more splits cost merge launches.

## Cycle 15 takeaway

**Diminishing returns warning.** The cycle-1 → cycle-14 progression has
extracted a 4.89× / 5.51× lift over cycle-1 baseline. Further lifts on
the same B/kernel/state lever set are below 3σ noise. To break 207
tok/s within strict envelope or 233 within hardware ceiling, a different
class of lever is needed — not more local kernel/state tuning.

Per P6_AUTORESEARCH.md "Stop conditions" §, both cycle-1 stop conditions remain
cleared by margin. The autoresearch loop is in a healthy stable state
with two compositional KEEPs landing in the last 2 cycles. No
deliverable urgency to push further unless authorised lever (spec
decoding, mx.compile, etc.) is opened.
