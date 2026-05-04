# P-6 Autoresearch — twenty-first cycle (2026-05-04) — drafter survey closes; tree-spec is the path

| Field | Value |
| --- | --- |
| Date | 2026-05-04 |
| Branch | `opus` |
| Status | **Drafter survey shows scaling/architecture changes don't lift the accept rate.** Three candidates (0.8B / 4B / 27B-3bit) produce essentially identical coverage curves; @1 ranges 6.3-8.0%, @64 all converge to 39-41%. The 4-bit-target's distribution is the structural bottleneck. **Tree-spec at b=64 is the only path to >40% accept rate.** |
| User authorisation | continued research-loop authorization |
| Companion docs | cycles 19 + 20 reports |

## TL;DR

Drafter survey (3 off-the-shelf candidates):

| Drafter | @1 | @4 | @8 | @16 | @32 | @64 | @128 | @256 | @512 | @1000 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Qwen3.5-0.8B (β.2 baseline) | 0.063 | 0.143 | 0.198 | 0.258 | 0.341 | **0.405** | 0.499 | 0.605 | 0.703 | 0.767 |
| Qwen3.5-4B | 0.074 | 0.139 | 0.166 | 0.231 | 0.286 | 0.391 | 0.497 | 0.603 | 0.706 | 0.765 |
| Qwen3.5-27B-3bit (same arch) | **0.080** | 0.145 | 0.194 | 0.252 | 0.298 | 0.393 | **0.528** | **0.638** | **0.730** | **0.787** |

**Best @1 = 8.0%** (27B-3bit, +27% over 0.8B baseline). The 27B-3bit drafter
shares the target's hybrid DeltaNet+Attn architecture — same family, same
data, just one bit-width quantisation step away. Even THAT only delivers
8.0% top-1 match. Drafter capacity / family alignment / quantization tier
all together produce a 1.7% absolute lift; nowhere near the >40% goal.

**The structural ceiling is the 4-bit target's distribution, not the
drafter.** Scaling the drafter costs more per propose forward but doesn't
buy meaningful coverage@1 lift.

## Reading

The cycle-1 reorientation memo had it right:
> "drafter capacity is not the ceiling, the 4-bit-target distribution
> divergence is."

Three independent drafters confirm. The 27B-3bit case is particularly
informative: same model family, same training data, same hybrid
architecture as the target, only differing in a single quantization
step (4-bit affine vs 3-bit). Even with that maximum architectural
alignment, @1 stays at 8.0%. There is something specific to the 4-bit
quantization process that shifts the argmax in a way no drafter
captures without explicit distillation against the 4-bit target's
own outputs.

## What this means for the >40% goal

**Drafter improvements are a dead end** without quantization-aware
distillation training (multi-day work, requires authorising the GPU
budget). The path is:

| Path | Expected ceiling | Cost | Status |
| --- | --- | --- | --- |
| Off-the-shelf drafter swap | @1 ≈ 8% (linear spec gives ~9-10% accept) | trivial | **closed — cycle 21** |
| Tree-spec at b=64 (cycle 19+20 confirm feasibility) | @64 = 40.5% — matches user threshold | 6-10 hours integration | **open — cycle 22+** |
| Distillation drafter (KD on 4-bit target) | @1 = ? (likely 0.20-0.40 if successful) | multi-day, GPU train budget | not authorized |
| MTP head trained on 4-bit target | similar to KD drafter | similar | not authorized |

Cycle 22+ pursues the open path — tree-spec at b=64.

## Drafter cost considerations (cost ≠ coverage)

Even if drafter coverage@1 stayed at ~8%, choosing the cheapest drafter
matters for net throughput. Per-step drafter forward cost (warm cache):

| Drafter | weights | est. forward ms | cost ratio vs target's 60 ms |
| --- | --- | --- | --- |
| Qwen3.5-0.8B (bf16) | ~1.6 GB | 28 ms (β.1 measured) | 0.47 |
| Qwen3.5-4B (bf16) | ~8 GB | ~80 ms est. | 1.33 |
| Qwen3.5-27B-3bit | ~14 GB | ~50 ms est. | 0.83 |

The 27B-3bit drafter is structurally appealing (slightly higher coverage,
same arch) but per-step cost is 3× the 0.8B drafter. Given coverage@1
gains are tiny (8.0% vs 6.3%), the 0.8B drafter still has best throughput
math at fixed verify-k. Confirmed: **0.8B drafter stays as the canonical
choice for any further linear-spec or tree-spec experiments.**

## Files added in cycle 21

- `plans/P6_C5_DDTREE/coverage_qwen3_5_4b.jsonl` — 4B drafter probe
- `plans/P6_C5_DDTREE/coverage_qwen3_5_27b_3bit.jsonl` — 27B-3bit probe
- `plans/P6_AUTORESEARCH/REPORT_KERNEL_CYCLE_21.md` — this file

## Ledger row added cycle 21

- `AR_C21_DRAFTER_SURVEY_CLOSED` (diagnostic, KEEP — closes drafter
  arm) — three drafter candidates, all flat coverage curves. Best @1 =
  8.0% (27B-3bit), best @64 = 40.5% (0.8B). Confirms cycle-1 reorientation
  finding: drafter is not the bottleneck. Pivots research to tree-spec at
  b=64 implementation (cycle 22+).

## Cycle 22 plan

Build minimal tree-spec at b=64 to actually realize the 40% accept rate
that cycle 19's β.2 probe established as feasible and cycle 20's verify-
cost probe established as computationally cheap.

The minimal implementation needs:

1. **Tree-aware drafter API**: at each draft position, return top-b
   candidates (not just argmax). Existing `silica.speculative.engine`
   has `DraftEngine` protocol — add a tree-mode variant or extend
   existing API.
2. **Tree-aware verify**: pass b candidates as a flat input batch with a
   tree-shaped causal mask so each candidate sees only its prefix
   ancestors. Target forward at k=b returns target's argmax for each
   tree-position.
3. **Tree-walk acceptance**: walk down the tree picking matches; track
   accepted prefix length per cycle.

For first iteration, target a depth-1 tree (b=64 candidates at one
position only). Expected accept = coverage@64 = 0.405. Per-cycle
throughput ≈ 0.405 tokens / (drafter_cost + verify_cost).

If the depth-1 tree doesn't translate to net throughput improvement
over the v10+bf16 KEEP, escalate to depth-2 tree (b=8 wide, 8 candidates
deep at first hit) or beyond.
