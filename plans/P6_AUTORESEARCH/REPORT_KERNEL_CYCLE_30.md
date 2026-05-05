# P-6 Autoresearch — thirtieth cycle (2026-05-04) — per-step decomposition at B=64 reveals DeltaNet now 88% of step time

| Field | Value |
| --- | --- |
| Date | 2026-05-04 |
| Branch | `opus` |
| Status | Diagnostic — instrumented per-step decomposition at B=64 with cycle-14 (post-correction) v10+bf16 stack. **DeltaNet share grew from 74.2% (cycle 1, B=4) to 87.9% (B=64).** Full-attention dropped from 21.9% to 12.5%. Identifies DeltaNet as the only remaining E2E lever at production B. |
| User authorization | continued |
| Companion docs | cycles 27, 28, 29 reports (corrected attribution), cycle-1 baseline decomposition |

## TL;DR

| Layer kind | Cycle 1 (B=4) | **Cycle 30 (B=64)** |
| --- | ---: | ---: |
| DeltaNet (linear-attention) | 74.2% | **87.9%** |
| Full-attention | 21.9% | 12.5% |
| Overhead | 4.0% | 0.3% (instrumented) |

(Numbers from per-layer barrier-timed decompose; instrumented step total
inflated due to barriers but proportions are informative.)

**Why DeltaNet share grew**: state R/W scales linearly with B. At B=4
state is 12 MB/layer; at B=64 it's 144 MB/layer at fp32 (72 MB at bf16).
Compute scales sub-linearly because matmul amortises across B.

**Why full-attention share shrank**: attention's KV-cache R/W also
scales with B but the per-token per-row fraction drops as B grows.

**Implication**: at B=64, kernel-level wins on full-attention (cycle
11/14 v10) cannot move E2E by much (12.5% × X% kernel speedup ≤ a few
percent). DeltaNet at 87.9% is the only path to a meaningful E2E lift.

## Files added in cycle 30

- `plans/P6_AUTORESEARCH/REPORT_KERNEL_CYCLE_30.md` — this file
- `/tmp/c30_decode_attr_b64.jsonl` — 661-row decomposition artefact
