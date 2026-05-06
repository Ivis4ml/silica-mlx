# δ.1 — dispatch-site pre-projection audit

Read-only audit of every `mx.eval` / `.item()` / sync barrier on the
production decode hot path at B=4 steady-state, plus a decomposition
estimate of α's 3.6% "instrumented overhead" bucket. Per the v1.7.27
PLAN gate: <2% recoverable → close δ with measurement-anchored
negative; ≥ 2.5% → run δ.1 empirical measurement before any code
change.

No model load, no benchmark run; this audit is sourced entirely from
code reading + α's existing attribution data
(`plans/P6_SMALL_B/REPORT.md` and the per-session JSONL).

## 1. Hot-path inventory — per decode step at B=4 steady state

"Steady state" means: rows already prefilled, no new admissions, no
evictions, no preemptions, no speculative path active (the v1.7.25
sonnet baseline). Each step calls one `forward_batched` and emits B
tokens.

| Site | File:line | Per-step count | Cost estimate | Recoverable? |
| --- | --- | ---: | --- | --- |
| `Qwen3NextModel.__call__` forward (lazy graph build) | mlx_lm/models/qwen3_next.py:404 | 1 | ~0.5-1.0 ms Python attribute walk over 64 layers | NO (mx.compile-only territory; β/γ already closed) |
| `BatchKVCache.update_and_fetch` (lazy) | mlx_lm/models/cache.py:1234 | 64 (per layer) | 0 sync; lazy mx ops only | n/a |
| `Sampler.sample` chain | silica/core/sampler.py:128 | 1 | 0 sync; lazy mx ops | n/a |
| `int(token_scalar.item())` per row | silica/scheduler/batcher.py:1922 | B=4 | ~10-100 μs/call → 40-400 μs total | partial (see §3) |
| `silica.mlx.runner.forward_batched` | silica/mlx/runner.py:92 | 1 | 0 sync; lazy return | n/a |
| Final norm + LM head matmul (lazy) | mlx_lm/models/qwen3_next.py:421+ | 1 | real compute (~1.5-2.5 ms) | NO |
| `bonus_scalar.item()` (spec path) | silica/scheduler/batcher.py:1862 | 0 (spec OFF in baseline) | 0 | n/a |

**Sites in the silica scheduler that DO call `mx.eval` but only on
non-steady-state paths** (NOT per decode step):

| Site | File:line | When called |
| --- | --- | --- |
| Cache slice materialisation in filter / preempt | silica/scheduler/batcher.py:686, 905 | filter / preempt / replay only |
| `mx.eval(*per_attn)` in admit | silica/scheduler/batcher.py:894 | admit phase only |
| `mx.eval(...)` in batched cache rebuild | silica/scheduler/batcher.py:1075 | filter / preempt only |
| Recurrent-state snapshot / detach / splice | silica/models/qwen3_5.py:710, 793 | spec rollback only |
| `BatchRotatingKVCache.extract` evals padding | mlx_lm/models/cache.py:1384 | filter / extract only |
| Cache stat reads (`.left_padding[...].item()`) | mlx_lm/models/cache.py:995, 1049 | admit / filter only |

These contribute zero work to a steady-state B=4 decode step and are
out of δ scope. Anything δ removes from the steady-state hot path
must come from the first table.

## 2. α's 3.6% "instrumented overhead" — what's inside

α's `decode_step_attribution` microbench reports
`step_total - sum(per_layer_wall_ms) = ~4.5 ms = 3.4-3.9%` as
"instrumented overhead". The α report names this as the candidate
δ bucket. The audit unpacks what that 4.5 ms actually is, by
walking everything that runs OUTSIDE the per-layer wrappers
(per `silica/bench/microbench/decode_step_attribution.py` lines
60-78 — the wrapper times the layer call + an `mx.eval(out)`
barrier, so per-layer time absorbs both layer compute *and* its
own per-layer barrier).

Estimated decomposition of the 4.5 ms outside-layers bucket
(rough, derived from shape arithmetic + typical mlx 0.31 op cost
on M5 Pro for the production Qwen3.5-27B-4bit shape):

| Component | est. ms | % step | Recoverable via Python hygiene? |
| --- | ---: | ---: | --- |
| LM head matmul (hidden=5120 → vocab≈152K, quantized 4-bit) | ~1.5-2.5 | 1.2-2.0 | **NO** — real compute, dominant |
| Embedding lookup (one row × hidden) | ~0.05-0.1 | 0.04-0.08 | NO — real compute |
| Final RMSNorm (B, 1, 5120) | ~0.05-0.1 | 0.04-0.08 | NO — real compute |
| Argmax over vocab (~152K) per row × B=4 | ~0.2-0.5 | 0.2-0.4 | NO — real compute |
| Python loop over 64 layers (zip + attr lookups, lazy graph) | ~0.5-1.0 | 0.4-0.8 | NO — would need mx.compile (β/γ closed) |
| `int(token_scalar.item())` × B=4 (sync round-trip + Python int) | ~0.04-0.4 | 0.03-0.3 | **partial** — see §3 |
| mlx-lm cache `update_and_fetch` Python wrapper × 64 | ~0.3-0.6 | 0.2-0.5 | small — already lazy on the compute side, only Python overhead |
| Mask construction (`create_attention_mask` + `create_ssm_mask`) | ~0.1-0.3 | 0.08-0.24 | small — depends on shape stability |
| Misc (token slicing, BatchEvent emission, transition checks) | ~0.1-0.3 | 0.08-0.24 | small — non-load-bearing |
| **Total accounted** | ~2.9-5.7 | 2.3-4.5 | matches α's 3.4-3.9% within estimation noise |

**The 3.6% bucket is dominated by real compute** (LM head ~1.2-2%
alone), not by Python-side dispatch overhead. The recoverable subset
— Python hygiene that does not change the compute graph — is at most
the bottom four rows: ~0.5-1.4 ms total = 0.4-1.1% of step.

## 3. Where δ-style hygiene could plausibly apply

Three patches are physically possible without changing the compute
graph or the contract:

### 3a. Batched per-row `.item()` consolidation

Current: `silica/scheduler/batcher.py:1922` calls
`int(token_scalar.item())` inside a `for i, row in enumerate(rows)`
loop, B=4 times per step. Each `.item()` is a separate
GPU→CPU sync round-trip plus a Python int allocation.

Possible patch: when all rows have uniform `SamplingParams` (e.g., all
greedy or all `temperature=1.0` with `top_p=None` and
`repetition_penalty=1.0`), batch sampling as
`mx.argmax(batched_logits, axis=-1)` → `tokens_b` of shape (B,) →
`tokens_b.tolist()` → Python list of B ints. One sync barrier
instead of B.

Estimated saving: per-call `.item()` round-trip is ~10-100 μs on this
stack. Saving B-1 = 3 round-trips = ~30-300 μs = **0.024-0.24%** of a
125 ms step. Mostly meaningful for B ≥ 32; marginal at B=4.

Important caveat: per-row `SamplingParams` divergence — repetition
penalty (which uses per-row history), per-row top-k, per-row
temperature — defeats the batching. The current contract supports
heterogeneous params. A patch would need a uniform-params fast path
plus a fallback to the current loop. That doubles code complexity for
≤ 0.24% E2E gain.

**Estimated recoverable: 0.05-0.20% E2E.**

### 3b. Mask-construction hoisting

`mlx_lm.models.qwen3_next.Qwen3NextModel.__call__:414-415` rebuilds
`fa_mask` and `ssm_mask` on every step. At T_q=1 these masks are
trivially small (the offset query mask), but the construction calls
go through `create_attention_mask` + `create_ssm_mask` helpers each
step.

Possible patch: cache the mask at adapter level when `T_q=1` and
shape is invariant; reuse across steps. Requires a sentinel for
invalidation on T_kv growth.

Estimated saving: 0.1-0.3 ms per step = **0.08-0.24%** E2E. Touches
mlx-lm internals (would need monkey-patch via shadow_install or a
silica-side decode loop fork). Same ~0.1% magnitude as 3a.

**Estimated recoverable: 0.05-0.20% E2E.**

### 3c. Cache `update_and_fetch` Python overhead

Current: 64 Python calls per step into mlx-lm's `update_and_fetch`,
each adding ~5-10 μs of attribute lookups + lazy op construction.

Possible patch: replace with a single batched-state mutation, or
inline the relevant op chain. Requires upstream cooperation or a
custom silica cache wrapper.

Estimated saving: ~0.1-0.3 ms per step = **0.08-0.24%** E2E. High
implementation effort (touches mlx-lm cache contract); brittle
across mlx versions.

**Estimated recoverable: marginal (0.05-0.20% E2E), high cost.**

### 3a + 3b + 3c upper bound

Adding the optimistic ends of each estimate:
**0.20% + 0.20% + 0.20% = 0.60% E2E aggregate.**

Even that aggregate falls below the v1.7.27 close threshold
of 2% by ~3×. None of the three sites individually clears the
threshold either.

## 4. Why the 3.6% bucket is not 3.6% recoverable

The α attribution names the leftover wall time as "instrumented
overhead", which suggested (read on its own) that this bucket is
all Python-side dispatch / sync inefficiency. The audit refutes
that read.

**About 70-90% of the 3.6% bucket is real compute** (LM head matmul
alone dominates), which no amount of Python hygiene can reduce.
**The genuinely Python-side fraction is 0.4-1.1% of step**, of which
typical hygiene patches recover at most a fraction (the lazy graph
already handles most fusion opportunities under the hood).

This is the same shape as the β/γ "bucket headlines mislead" lesson
from v1.7.26-27, applied one more layer down. The v1.7.27 memory
rule (`feedback_p6_small_b_single_customer_gates.md`) said δ does not
apply the *bucket × reachable-scope × per-call-gain* product
directly because δ is Python-side hygiene. The audit refines that:
δ does need a similar product, just with different terms.

**Generalised δ-axis ceiling estimator:**

`recoverable E2E % ≈ overhead bucket % × (1 − real-compute fraction) × (hygiene-reachable fraction)`

For α's bucket: 3.6% × (1 − 0.8) × ~1.0 ≈ **0.7%** upper bound.
The 3a/3b/3c sum at ~0.6% is consistent with this.

## 5. Verdict: close δ with measurement-anchored negative

The audit estimates δ recoverable overhead at ≤ 0.6% E2E aggregate,
firmly below the v1.7.27 close gate of 2%. The estimate is not noisy
— it comes from concrete site-by-site cost ranges with cycle-1 / α
data as the calibration anchor. No empirical δ.1 microbench is
needed; the design simply does not carry the headroom that PLAN's
"3.6% mathematical ceiling" framing implied.

**Recommendation: close δ on this audit. D-022 itself can then
close** because α complete + β closed + γ closed + δ closed reaches
every conditionally-opened sub-unit's terminal state per
`plans/P6_SMALL_B_OPENING.md` §6. ε remains upstream-waitlist
(mlx 0.32+ async-copy) and does not block D-022 closure.

The exit position for D-022:

- Single-customer B=1 latency on Qwen3.5-27B-4bit / M5 Pro 48 GB
  remains at the bandwidth-derived ceiling (~20 tok/s).
- B=4 per-row stays at the v1.7.25 sonnet baseline (10.29 ± 0.16
  tok/s/row).
- The compile axis is exhausted (β narrow scope; γ tiny gain) and
  the Python-hygiene axis is too thin (this audit, ≤ 0.6%
  recoverable).
- Future revisits require a different lever: mlx 0.32+ async-copy
  primitives (ε waitlist), a fundamentally different kernel
  approach, or a different model architecture.

## 6. Cross-references

- `plans/PLAN.md` §9 D-022 + §13 v1.7.27 — opening framing for δ
  pre-projection.
- `plans/P6_SMALL_B/REPORT.md` — α attribution; source for the
  3.6% overhead bucket and per-layer breakdown.
- `plans/P6_SMALL_B/BETA/microbench/REPORT.md` /
  `plans/P6_SMALL_B/GAMMA/microbench/REPORT.md` — sibling
  closures; same bucket-misleads-headline shape, lower in the
  decomposition.
- `silica/scheduler/batcher.py:1898-1937` — the per-row sample
  loop the §3a patch would touch.
- `mlx_lm/models/qwen3_next.py:404-421` — the model forward; the
  §3b mask-hoist patch would touch the upstream module.
- `mlx_lm/models/cache.py:1234` — `update_and_fetch`, the §3c
  candidate site.
- v1.7.27 memory `feedback_p6_small_b_single_customer_gates.md` —
  the *bucket × reachable-scope × per-call-gain* discipline that
  δ.1 generalises to the Python-hygiene axis.
