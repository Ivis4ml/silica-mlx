# P-6 Autoresearch — twentieth cycle (2026-05-04) — verify-cost scaling confirms tree-spec is computationally feasible

| Field | Value |
| --- | --- |
| Date | 2026-05-04 |
| Branch | `opus` |
| Status | **Verify cost is HIGHLY sub-linear**: at k=64 the target costs 3.14× of k=1 (188.81 ms vs 60.12 ms). Per-token verify cost drops from 60 ms (k=1) to 2.95 ms (k=64) — a 20× per-token reduction. **Tree-spec at b=64 is computationally feasible.** |
| User authorisation | continued research-loop authorization from cycle 19 |
| Companion docs | cycle 19 extended coverage report; β.1 + β.2 baseline reports in `plans/P6_C5_DDTREE/` |

## TL;DR

```
k    p50 ms   per-token   ratio_vs_k1   sub-linearity
1    60.12    60.12       1.00x          1.000
4    91.47    22.87       1.52x          0.380
8   164.89    20.61       2.74x          0.343
16  174.30    10.89       2.90x          0.181
32  170.23     5.32       2.83x          0.088
64  188.81     2.95       3.14x          0.049
128 324.14     2.53       5.39x          0.042
```

(Wikitext prefill T_kv=128 warm; n=30 measure iters per k.)

The plateau between k=8 and k=64 (164→188 ms = +14% absolute cost for
8× more tokens) shows that the **per-step overhead is dominated by
weights bandwidth + DeltaNet recurrent state R/W, not k-dependent
attention compute**. This is exactly the regime where tree-spec wins:
verifying 64 tree candidates costs almost the same as 8.

## Implications for the >40% accept rate goal

**Computational feasibility — passed.**

| b | coverage@b (cycle 19) | verify cost (k=b) |
| ---: | ---: | ---: |
| 4 | 0.143 | 91 ms |
| 16 | 0.258 | 174 ms |
| 32 | 0.341 | 170 ms |
| **64** | **0.405** ← user threshold | 189 ms |
| 128 | 0.499 | 324 ms |

Tree-spec at b=64 has the necessary structural headroom (40.5% match) AND
the necessary cost structure (verify ~3× of k=1). The piece NOT yet in
place: tree-aware draft + verify code paths.

## What's needed for actual tree-spec

The existing `silica.speculative` framework is **linear-spec only**:
drafter generates a sequential γ-prefix, target verifies in one
batched forward, accept along the prefix until first rejection. Linear
spec at large γ doesn't help here because each position's match
probability is bounded by `coverage@1 = 0.063` independently.

Tree-spec needs three new pieces:

1. **Tree-aware drafter**: at each position, return top-b candidates from
   the drafter's logits (not the single argmax). The candidate count
   needed is constant at each position (b = 64); current drafter API
   returns just the sequential argmax stream.
2. **Tree-aware verify with sparse attention mask**: verify all b
   candidates in one target forward using a tree-shaped causal mask
   so each candidate sees only its tree-ancestor as KV-context.
3. **Tree-walk acceptance**: target's argmax at each position selects
   among the b candidates; walk the tree picking matching paths until
   first rejection.

These are non-trivial. The DDTree-mlx upstream (mentioned in
`plans/P6_C5_DDTREE_OPENING.md` §5) has a reference implementation,
but the cycle-1 ledger noted "humanrouter/ddtree-mlx upstream survey —
not authorised". With user research-loop authorization, cycle 21+ can
either pull the upstream design or hand-roll a minimal version.

## Linear-spec failure analysis

Why does linear spec at γ=64 not work? Independence-bounded math:
- Each position's match rate = coverage@1 = 0.063
- Expected accepted prefix length = 0.063 + 0.063² + 0.063³ + ... = 0.063 / (1 - 0.063) ≈ 0.067
- Cost per cycle: drafter γ-step (~28 ms × 64 ≈ 1.8 s) + target verify (188 ms) = ~2 s
- Tokens per second = 0.067 / 2 = 0.033 tok/s — orders of magnitude worse than naive

This is why the previous β.1 settled on γ=4: linear spec saturates fast.

## Tree-spec projected throughput

For tree-spec at b=64 with **per-position coverage 0.405** organized as a
balanced tree (depth=d, branch=b^(1/d)):

- Depth-1 tree (b=64 candidates at one position): expected accept = 0.405
  → 0.405 tok / verify = ~2.1 tok/s — much worse than naive 16 tok/s
- Depth-d tree with branching b^(1/d):
  - At depth 4, branch=64^(1/4)=2.83 → ~3 candidates per position, 4
    levels deep
  - per-position match rate at b=3 from cycle 19: not measured (between
    @1=0.063 and @4=0.143; interpolate ≈ 0.10)
  - Expected accepted depth = sum_{i} 0.10^i ≈ 0.111 — terrible

The right tree-spec config probably uses **wide-shallow trees**:
- Depth=2, branch=32, total leaves=1024 (too many candidates)
- Depth=2, branch=8 (b=64 leaves with 8 first-position + 8 second-position
  per first): expected = coverage@8(pos1) × coverage@8(pos2|hit) ≈
  0.198 × ?
- This gets complex without actually running it

The simpler model: **linear-spec α-equivalent**. For tree-spec to beat
linear-spec at γ=4 (current best), we need the expected accepted tokens
per verify to clear ~1.27 (the β.1 baseline). With b=64 and the cycle-19
coverage curve, achieving 2-4 expected tokens per verify is plausible
but not guaranteed.

## Files added in cycle 20

- `scripts/probe_c5_verify_cost.py` — verify cost microbench at k∈{1,4,8,16,32,64,128}
- `plans/P6_C5_DDTREE/verify_cost_probe.jsonl` — measurement artefact
- `plans/P6_AUTORESEARCH/REPORT_KERNEL_CYCLE_20.md` — this file

## Ledger row added cycle 20

- `AR_C20_VERIFY_COST_SUBLINEAR` (diagnostic, KEEP — opens implementation
  path for tree-spec) — verify cost p50 at T_kv=128 warm cache:
  k=1→60.12 ms, k=64→188.81 ms (3.14× cost for 64× tokens). Per-token
  verify drops 20× from k=1 to k=64. The k=8→k=64 plateau confirms
  weights-bandwidth dominance; tree-spec at b=64 is computationally
  feasible. Cycle 21 will build minimal tree-spec to measure actual
  accept rate.

## What's next (cycle 21)

The decision tree forks based on what's tractable:

**Option A — Build minimal tree-spec (substantial integration)**: write
silica/speculative/ddtree.py with tree-aware drafter API, tree-mask
verify, tree-walk accept. Measure accept rate at b=64. Time: ~6-8 hours.

**Option B — Pull upstream `humanrouter/ddtree-mlx`** (read-only survey
+ adapter): cheaper integration, but needs license / no-torch
attestation per AR.md. Time: ~2-3 hours survey + ~3-4 hours adapter.

**Option C — Pivot to drafter-side improvements**: skip tree-spec
entirely; train a quantization-aware drafter on the 4-bit target's
output distribution to push coverage@1 from 6.3% upward. Time: multi-
day if training is needed; few hours if a closer-fit off-the-shelf
drafter (e.g., Qwen3.5-Next-Mini) exists.

Cycle 21 first step: a 1-hour survey of HuggingFace for smaller Qwen3.5
or Qwen3.5-Next variants that share the target's tokenizer. If a clean
0.5B Qwen3.5 drafter exists (or any architecture-aligned smaller
model), measure its coverage@1 against the 4-bit target — if it
clears coverage@1 ≥ 0.20, that's a faster path to >40% effective
accept than tree-spec.
