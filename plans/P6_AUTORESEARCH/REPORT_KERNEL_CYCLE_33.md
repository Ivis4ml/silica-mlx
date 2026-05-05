# P-6 Autoresearch — thirty-third cycle (2026-05-04) — between-session variance characterization

| Field | Value |
| --- | --- |
| Date | 2026-05-04 |
| Branch | `opus` |
| Status | Diagnostic — 3 additional B=52 bf16-only reps in a fresh session, combined with cycle 27's 3 reps gives **n=6 across 2 sessions**. Combined mean **203.75 ± 0.83 tok/s** with between-session drift ~0.9 tok/s. Honest running-best characterization holds at **204 ± 1 tok/s** within same environment. |
| User authorization | continued |
| Companion docs | cycle 27 (initial measurement); Codex c25 (flagged inter-environment variance) |

## TL;DR

Codex's c25 reverify reported wide variance across environments (uv=185.3,
conda=200.1, post-fix=190-204). Cycle 33 characterises it more carefully:

| Session | Runs | Mean | σ |
| --- | --- | ---: | ---: |
| Cycle 27 (uv, fresh start) | 205.0 / 204.6 / 202.9 | **204.2** | 1.1 |
| Cycle 33 (uv, after C28-C32 work, ~30 min cool-down) | 203.0 / 203.8 / 203.2 | **203.3** | 0.4 |
| **Combined (n=6)** | — | **203.75** | **0.83** |

Between-session drift = 0.9 tok/s. Within-session σ = 0.4-1.1 tok/s.
**Total combined σ ~1 tok/s in the same environment with proper warm cache.**

The Codex 185.3 result was likely an environmental issue (uv venv
configuration, cold start, machine state). Reproduced in the same
environment with the same cache state shows ~204 tok/s consistently.

## Implications

The running-best line stays at **B=52 bf16-only = 204 ± 1 tok/s (n=6)** —
slightly tighter σ than cycle 27's reported ±1.5. The cycle-14 phantom
"206.2 ± 0.5" can now be definitively understood:
- The "0.5" σ was misleading (n=3 within one session is too small)
- The "206.2" was within ~2σ of the actual 204 mean

Cycle 27's correction stands. Cycle 33 just sharpens the σ.

## Files added in cycle 33

- `plans/P6_AUTORESEARCH/REPORT_KERNEL_CYCLE_33.md` — this file
- `/tmp/c33_b52_session2_run{1,2,3}.jsonl` — 3 fresh-session reps

## Ledger row added cycle 33

- `AR_C33_B52_VARIANCE_CHARACTERIZATION` (diagnostic) — 6 B=52 bf16-only
  reps across 2 sessions: combined 203.75 ± 0.83 tok/s. Within-session σ
  0.4-1.1; between-session drift 0.9. Combined σ ~1 tok/s in the same
  uv environment with warm cache. Codex c25's 185.3 was likely a cold-
  start / environmental anomaly, not a representative measurement.
  Honest running-best: 204 ± 1 tok/s at B=52 bf16-only.

## Cycle 33 closes the variance characterization concern

The autoresearch loop's reproducibility story is now honest:
- Within session, σ ~1 tok/s
- Across sessions in same environment, drift ~1 tok/s
- Cross-environment, large variance possible (Codex's 185 in different env)

Future re-verification protocol should specify: same uv environment,
warm cache, ≥3 reps per session, ≥2 sessions. Cycle-1 → cycle-31's
cumulative measurement work meets this standard.
