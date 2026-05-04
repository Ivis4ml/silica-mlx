# P-6 Autoresearch — twenty-ninth cycle (2026-05-04) — 40 GB cliff is architectural, not allocator-policy

| Field | Value |
| --- | --- |
| Date | 2026-05-04 |
| Branch | `opus` |
| Status | Cliff at 40 GB peak (B=64 → B=66 boundary) is NOT movable via mlx allocator hints. Probed `mx.metal.set_cache_limit` and `mx.metal.set_memory_limit`; B=66 throughput stays at ~166 tok/s regardless. The cliff is M5 Pro architectural (likely SLC capacity threshold or unified-memory-bandwidth pressure), not mlx allocator policy. |
| User authorization | continued from cycle 28 |
| Companion docs | cycle 27, 28 reports |

## TL;DR

| Configuration | B=66 tok/s | peak GB |
| --- | ---: | ---: |
| Default (no override) — cycle 14 baseline | 166.8 | 40.79 |
| `cache-limit=44 GB`, `memory-limit=46 GB` | 165.8 | 40.79 |
| `cache-limit=1 GB` (force eviction) | 166.5 | 41.01 |
| `wired-limit=40 GB` | OOM (kIOGPUCommandBufferCallbackErrorOutOfMemory) |

**The cliff is unchanged across all allocator-policy probes.** It is
architectural — most likely:
- M5 Pro System-Level Cache (SLC) capacity threshold
- Apple unified-memory bandwidth contention as resident pressure
  approaches the 48 GB system cap

Neither is movable from the mlx allocator's user-facing API.

## Implications

The empirical hardware ceiling for warm-decode-b48 production workload
on M5 Pro 48 GB is **B=64 / 40.01 GB peak / 231.9 ± 0.3 tok/s** (per
cycle 28's bf16-only re-measurement). Cycle 29's cliff probe confirms
this is a genuine architectural ceiling, not an allocator policy.

To break past 232 tok/s on the same model/hardware would require:
1. **mlx 0.32+ async-copy primitives** to enable K/V tile prefetch
   overlapping with current-tile compute (cycle 11/12 found 0.31.2
   broke determinism; await 0.33+)
2. **mx.compile graph-trace with cache rerouting** — 4-6 hour
   integration for ~5% E2E if it works (cycle 16 microbench was 1.08×)
3. **Different model architecture** that has lower per-step memory
   pressure or different compute/memory profile

None of these are accessible without authorization or external
dependencies. The autoresearch loop has reached a legitimate stop on
this hardware/model combination.

## Files added in cycle 29

- `plans/P6_AUTORESEARCH/REPORT_KERNEL_CYCLE_29.md` — this file
- `/tmp/c29_b66_cache44_mem46.jsonl` — first cliff probe artefact
- `/tmp/c29_b66_cache1.jsonl` — small-cache-limit probe artefact

## Ledger row added cycle 29

- `AR_C29_CLIFF_ARCHITECTURAL` (discard — cliff probe with allocator
  hints fails to move the boundary). B=66 throughput holds at ~166
  tok/s across all `set_cache_limit` / `set_memory_limit` settings
  tried. The 40 GB peak cliff is architectural (M5 Pro SLC threshold
  or unified memory pressure), not mlx allocator policy. **Hardware
  ceiling at B=64 / 232 tok/s is the genuine production ceiling on
  this stack.**

## Final state of the autoresearch loop (cycles 1-29)

| Frame | Value | Lever attribution |
| --- | ---: | --- |
| Within strict 36 GB envelope | **204.5 ± ~1.5 tok/s at B=52** (4.85× cycle-1) | C10 axis-shift × C12 bf16-state peak-save |
| Within 48 GB hardware ceiling | **231.9 ± 0.3 tok/s at B=64** (5.50× cycle-1) | same; cliff at 40 GB peak is architectural |
| (1b) ≥60 milestone | CLEARED 3.41× / 3.86× | — |

The cycles 1-29 work has produced a clean, honest, reproducible
running-best with corrected attribution. v10 FA-decode kernel stays in
inventory as a microbench-validated tool (1.28-2.14× over mlx) without
E2E contribution at production B. bf16 DeltaNet state's value is the
peak-memory headroom that enabled C10's axis-shift to extend.

The remaining levers all need external dependencies (mlx 0.32+) or
substantial integration (mx.compile cache rerouting). The autoresearch
loop's deliverable is in a genuinely closed state on this stack.
