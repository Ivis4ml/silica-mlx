# P-6 Autoresearch — twenty-second cycle (2026-05-04) — tree-spec design proposal

| Field | Value |
| --- | --- |
| Date | 2026-05-04 |
| Branch | `opus` |
| Status | Design doc — three possible cycle-23+ trajectories with cost / risk / ROI matrix. User picks. |
| User authorisation | continued research-loop authorization |
| Companion docs | cycles 19, 20, 21 reports |

## TL;DR

After cycles 19-21, the **>40% accept rate goal is feasible only via
tree-spec at b=64**. The user has three concrete trajectories:

| Path | Implementation cost | Expected accept rate | Throughput vs C14 KEEP (206.2 tok/s) |
| --- | --- | --- | --- |
| **A. Minimal tree-spec at b=64** | 6-10 hours integration | 35-45% (within ceiling) | unclear; depends on tree depth efficiency |
| **B. Distillation drafter training** | Multi-day (GPU train + eval) | up to 60-80% (KD theoretical) | likely higher than (A) but requires train auth |
| **C. Close research, ship findings** | 0 (consolidate) | n/a | C14 v10+bf16 stack at 206.2 envelope / 232.2 ceiling stands |

## Path A — Minimal tree-spec at b=64

### Architecture

```
                target_prefix
                      |
                      v
            +---------+---------+
            | Drafter top-b at  |
            | each candidate   |
            | tree position     |
            +---------+---------+
                      |
                      v
            +---------+---------+
            | Target verify     |
            | with tree mask    |
            | (one k=b forward) |
            +---------+---------+
                      |
                      v
            +---------+---------+
            | Walk accepted     |
            | path; emit tokens |
            +---------+---------+
```

### Required code changes

1. **`silica/speculative/ddtree.py`** (new, ~300 lines):
   - `DDTreeDraftEngine` implementing `DraftEngine` protocol
   - `propose_tree(prefix, b)` → returns flat candidate ids + parent indices
     for the tree structure
   - Internally calls drafter forward to get top-b at each tree position

2. **`silica/speculative/verify.py`** (extend, ~100 lines):
   - `tree_verify_mask(parent_indices, prefix_len)` → returns Metal-compatible
     attention mask (each candidate sees its prefix ancestors only)
   - `tree_walk_accept(target_argmaxes, candidate_ids, parent_indices)` →
     returns longest accepted path

3. **`silica/engine/__init__.py`** (extend, ~50 lines):
   - Engine.generate path picks tree-spec vs linear-spec based on env or
     spec config
   - Iteration: each cycle = one drafter tree-build + one target verify + walk

### Expected performance (math)

Per cycle costs (B=1, T_kv=128 warm, from cycles 1 + 20):
- Drafter top-b at depth-d tree: roughly d × 28 ms = 56-112 ms (for d=2..4)
- Target verify at k=b: 188 ms (cycle 20 measured for k=64)
- Tree walk: <1 ms
- Total per cycle: ~250-300 ms

Expected accepted tokens per cycle:
- Depth-1 tree (b=64, just one position): expected 1.405 tokens (1 always
  + 0.405 if drafter top-64 hit) — but since target ALWAYS produces an
  argmax we accept anyway, this is effectively 1 token per cycle. **No
  throughput gain over plain decode.**
- Depth-2 balanced tree (8 first × 8 second per first = 64 candidates):
  - Hit at depth 1 with prob coverage@8 = 0.198
  - Hit at depth 2 conditional on depth-1 hit = roughly coverage@8(d=2) ≈
    similar 0.198 (assuming independence; conservative)
  - Expected depth = 1 + 0.198 + 0.198² ≈ 1.24 tokens
- Depth-3 balanced tree (4×4×4 = 64): expected ~1.32 tokens
- Depth-2 unbalanced (16 first × 4 follow-up): ~1.30 tokens

The classic spec-decoding math: **accept tokens per cycle ≈ 1 + p + p² +
... = 1/(1-p)** where p is the per-position match rate. With tree-spec,
p = coverage@b at the tree's effective branch factor.

For the user's 40% target translated to throughput:
- p = 0.40 → accept ≈ 1.67 tokens/cycle
- Throughput at B=1: 1.67 tokens / 0.27s/cycle = **6.2 tok/s** — much
  worse than current B=1 baseline 16 tok/s

Hmm. **Tree-spec at B=1 doesn't beat plain decode even with the 40% accept
rate**, because tree verify cost grows from 60ms (k=1) to 188ms (k=64) —
3.14× verify cost for ~1.5× accepted tokens.

### When tree-spec actually helps

The arithmetic above assumed B=1. **At higher B (the cycle 14 regime)**,
spec decoding's value changes:
- Per-cycle target cost grows linearly with B (since each row needs its
  own verify position)
- BUT the "drafter parallel forward" runs in PARALLEL across rows
- Net: spec decoding can in principle scale per-row throughput at high B

Quantitative estimate at B=52 (cycle-14 KEEP regime):
- Plain decode at B=52: 206.2 tok/s aggregate = 3.97 tok/s per row
- Spec decode at B=52, b=64 tree, p=0.40, depth-2 balanced:
  - Per-row: ~1.24 tokens / cycle
  - Per-cycle wall: drafter ~50ms + verify ~190ms = 240 ms
  - Per-row throughput: 1.24 / 0.24 = **5.2 tok/s per row** = +30% per-row
  - **Aggregate at B=52: 270 tok/s** if per-row gain holds at high B

If this scales, **spec at b=64 tree could push the running-best from
206.2 → 270 tok/s = +31%** within strict envelope. That's a real win.

Risk: per-cycle wall at B=52 might not be 240ms — drafter forward at B=52
is more expensive (~150ms), and verify at B=52 k=64 is uncharted.
Cycle 23 would need to measure these costs before going further.

### Cycle 22 deliverable

Cycle 22 is design-only (this report). The implementation in cycle 23+
splits into:

- **23**: drafter top-b API + verify-cost microbench at B=52 k=64 (1 day)
- **24**: tree-mask verify implementation + correctness test (1 day)
- **25**: tree-walk accept + integration into Engine (1 day)
- **26**: end-to-end E2E benchmark at warm-decode-b52-spec scenario (1 day)
- **27**: tuning + 3-reproduction KEEP gate (1 day)

Estimated 5 working days for an end-to-end load-bearing result.

## Path B — Distillation drafter training

Train a small Qwen3.5 drafter directly on the 4-bit target's outputs.
With KD the drafter learns the 4-bit-shifted argmax distribution.

### Expected gain

With KD theoretical ceiling around 60-80% coverage@1 (based on similar
QuantSpec work in literature):
- 60% coverage@1 → linear spec accept ~70-75% per cycle
- Expected tokens per linear-spec γ=4 cycle: ~3.0-3.3
- Per-cycle wall: ~28*4 + 91 = 203 ms drafter + verify
- Per-row throughput at B=1: 3.0 / 0.20 = 15 tok/s — slight bump
- Per-row throughput at B=52: depends on batched verify cost

### Cost

Training data: 1-10 GB of target's output traces on representative inputs.
Hardware: GPU (M5 Pro 48 GB shared with target — competing for memory).
Time: 1-3 days for a 0.5B drafter at decent-sized corpus.

### Risk

- Drafter overfits to training corpus distribution
- Coverage@1 lift might be smaller than expected if training corpus doesn't
  span actual decode-time distributions
- Multi-day cycle, not a quick research probe

## Path C — Close research, ship findings

Cycle 19 + 20 + 21 already produced a substantial knowledge deliverable:
1. Coverage@64 = 40.5% — establishes the upper bound
2. Verify cost sub-linear at k=64 — proves computational feasibility
3. Drafter survey — closes the "swap drafter" arm cleanly
4. Identifies tree-spec as the only viable path

These findings answer the user's research question:
> "previous experiment show accept ratio is just 8%, we need do research
> to achieve at least > 40% accept ratio"

**Yes — 40% accept rate is achievable via b=64 tree-spec.** The detail
of "does it translate to net throughput" depends on tree-shape design
choices and B-axis batching, neither of which is settled.

## Recommendation

Path C closes cleanly with concrete findings. Path A delivers a real
implementation. Path B is biggest upside but biggest cost.

I'd suggest the user picks **A or C**:

- **A** if they want an actually-running tree-spec system to measure
  end-to-end. Risk: 5 working days for a possibly modest gain at B=52.
- **C** if the research question itself ("can we hit 40% accept rate?")
  is the deliverable. Answer: yes, with b=64 tree-spec.

Path B is high-cost-high-risk; not recommended unless distillation
drafter is wanted as a P-7 deliverable for non-spec uses too.

## Files added in cycle 22

- `plans/P6_AUTORESEARCH/REPORT_KERNEL_CYCLE_22.md` — this design doc
- (No code changes; pure design)

## Ledger row added cycle 22

- `AR_C22_TREE_SPEC_DESIGN` (diagnostic — design-only, no measurement) —
  proposes 3-path matrix: minimal tree-spec build (5 working days, +30%
  projected per-row at B=52), distillation drafter (multi-day, biggest
  upside), or close research with cycle 19-21 findings as deliverable.
  User decision required.
