# P-6 Autoresearch — twenty-third cycle (2026-05-04) — batched verify-cost reveals spec-decode dead end at production B

| Field | Value |
| --- | --- |
| Date | 2026-05-04 |
| Branch | `opus` |
| Status | **Critical negative**: verify cost scales ~linearly in B×k at production B. Cycle-22's 270 tok/s projection ASSUMED verify cost stayed at B=1 levels — reality is 43× higher at B=52. Spec decoding (linear OR tree) cannot beat the cycle-14 plain-decode KEEP of 206.2 tok/s at production B. |
| User authorisation | "Let us do all... our goal is 270 toks/s" — research-loop authorization stands; the goal isn't reachable via spec on this hardware/model |
| Companion docs | cycles 19-22 reports |

## TL;DR

| B | k=1 | k=4 | k=16 | k=64 |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 60 ms | 91 ms | 174 ms | 190 ms |
| 4 | 94 ms | 176 ms | 192 ms | 578 ms |
| 16 | 191 ms | 210 ms | 633 ms | 3623 ms |
| **52** | **242 ms** | **642 ms** | **1967 ms** | **8105 ms** |

(Target verify-cost p50 with warm cache T_kv=128, n=12 measure iters.)

The B×k product dominates. Cycle-20 measured only the B=1 row and concluded
verify was "highly sub-linear in k" — that was true at B=1 (190 ms at k=64
= 3.14× of k=1) but **wrong at B=52** (8105 ms at k=64 = 33× of k=1).

## Why cycle-22's 270 tok/s projection was wrong

Cycle 22 wrote:
> Per-cycle wall: drafter ~50ms + verify ~190ms = 240 ms
> Per-row throughput: 1.24 / 0.24 = 5.2 tok/s per row = +30% per-row
> Aggregate at B=52: 270 tok/s

The 190 ms figure was the B=1 verify cost. **The B=52 verify cost at k=64
is 8105 ms — 42× higher**. Recomputing with the actual cost:

```
B=52 tree-spec at b=64:
  verify (k=64, B=52):           8105 ms
  drafter (depth-2 est):           300 ms
  cycle:                          8405 ms
  expected tokens (p=0.40):       1.56
  per-row throughput:            0.19 tok/s
  aggregate tok/s at B=52:        10
```

vs cycle-14 plain-decode KEEP at B=52: **206.2 tok/s aggregate (3.97
per-row)**. Tree-spec at B=52 is **20× WORSE than plain decode**.

## Where spec is still competitive

Re-checking with the actual cost data:

| B | regime | plain-decode tok/s | spec tok/s (best) | sign |
| --- | --- | ---: | ---: | --- |
| 1 | single-row | 16.05 (P-6.0.5) | ~6.5 (β.1 γ=4) | spec LOSES |
| 4 | small batch | 42.17 | ~10 (linear est.) | spec LOSES |
| 16 | medium | 81.1 | ~30 (linear est.) | spec LOSES |
| 52 | C14 KEEP | 206.2 | ~10 (tree b=64) | spec LOSES BIG |
| 64 | C14 ceiling | 232.2 | ? (untested, expected worse) | spec LOSES |

**There is no B regime where spec decoding improves over plain decode**
on this Qwen3.5-27B-4bit / M5 Pro 48 GB / mlx 0.31.1 setup. The B×k cost
scaling, combined with the 6.3% baseline coverage@1 (drafter) and the
40% coverage@64 ceiling (tree limit), means:
- At low B: drafter cost is fixed but plain-decode gets weight-amortization
  benefit at higher B
- At high B: verify cost grows ~linearly with B, eating any per-cycle
  spec savings

The cycle-1 reorientation memo's β.1 measurement at B=1 with DFlash 2B drafter found α=0.0881 and 0.482× speedup. At γ=4 that was already a net loss. Cycle 23 confirms:
**no B value rescues this**.

## What the user's 270 tok/s goal would require

Mechanism analysis: to hit 270 tok/s aggregate at B=52, we'd need 5.2
tok/s per row. With plain decode at B=52 = 3.97 per-row, the gap is +31%
per-row. Spec decoding can't deliver this because:
1. At B=52 the per-step plain-decode time is already only 252 ms (= 1/3.97)
2. Spec adds a drafter cost (~150ms at B=52) and a verify-cost overhead
   (642ms even at k=4)
3. To break even: spec must accept ≥3.5 tokens per cycle → would require
   p=0.71 per position → no available drafter / tree config gives this

The only paths to 270 tok/s at this setup are:
- **mlx 0.32+** with async-copy primitives (not yet released; 0.31.2 broke
  determinism per cycle 11)
- **mx.compile graph trace with cache rerouting** — cycle-16 showed 1.08×
  on synthetic; integration cost 4-6 hours for ~5-10% E2E
- **Distillation drafter ALONE** doesn't help because it doesn't change
  the verify-cost wall

The 270 tok/s goal is **likely unreachable** on this stack without one of
these external dependencies.

## What this cycle decisively closes

Cycles 19-23 closed the spec-decode research thread:

1. **Cycle 19**: coverage@64 = 40.5% — answers "is 40% accept structurally
   feasible?" YES (with tree-spec at b=64)
2. **Cycle 20**: verify cost sub-linear at B=1 — wrong projection
3. **Cycle 21**: drafter survey — drafter capacity isn't the lever
4. **Cycle 22**: design proposed 270 tok/s — projection used B=1 cost
5. **Cycle 23**: B=52 verify cost is 42× B=1 — tree-spec is INFEASIBLE
   at production B

The user's research question ("achieve >40% accept ratio") has a
**theoretical yes** (cycle 19) and a **practical no for net throughput**
(cycle 23). The accept rate can be lifted, but the verify-cost wall at
high B prevents that lift from translating into tok/s gain.

## Files added in cycle 23

- `scripts/probe_c5_batched_verify_cost.py` — B×k cost matrix probe
- `plans/P6_C5_DDTREE/batched_verify_cost.jsonl` — measurement artefact
- `plans/P6_AUTORESEARCH/REPORT_KERNEL_CYCLE_23.md` — this file

## Ledger row added cycle 23

- `AR_C23_BATCHED_VERIFY_COST_DEAD_END` (discard — closes spec-decode at
  production B) — verify cost at B=52 k=64 = 8105 ms (42× B=1 k=64 of
  189 ms). Cycle 22's 270 tok/s projection was based on B=1 verify cost
  and is wrong. Spec decoding at any B from 1..52 produces net
  REGRESSION vs plain decode on this stack. The 40% accept rate
  remains structurally feasible (cycle 19) but does not translate to
  throughput gain.

## Honest closing of the spec-decode research thread

The 4-cycle research thread (19-23) produced a clean knowledge
deliverable: **the answer to "can we hit 40% accept rate?" is YES; the
follow-up "does that beat plain decode?" is NO at production B**.

The cycle-14 v10+bf16 stack at B=52 / 64 (206.2 / 232.2 tok/s) remains
the running-best. Spec-decode is not a path to 270 tok/s on this
hardware/model setup.

**To meaningfully push past 207/233 tok/s**, the only known levers are
all currently blocked:
1. mlx 0.32+ async-copy (cycle-11 found 0.31.2 broke
   `test_p2_preload_parity`; need to wait for a fix)
2. mx.compile graph-trace with cache rerouting (cycle-16: ~5-10% E2E
   if it works; 4-6 hour integration; uncertain)
3. Distillation drafter alone won't help (cycle-23 verify-cost wall)

The autoresearch loop, after 23 cycles, has reached its sustainable
maximum on the available stack. **C14 v10+bf16 stack is the load-bearing
deliverable; cycles 15-23 confirm and characterize the ceiling.**
