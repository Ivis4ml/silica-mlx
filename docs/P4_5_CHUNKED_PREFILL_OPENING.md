# P-4.5-B Opening — Chunked Prefill (minimal)

> **Scope:** this document is the design opener for P-4.5-B. It enumerates
> three paths to close the Q-010 TTFT-under-concurrency fairness defect
> (`docs/PLAN.md` §10 Q-010 Resolution, 2026-04-21), compares their
> scheduler footprint against the P-2 invariant set, and picks one path
> for implementation. No code changes in this commit.
>
> Downstream sub-unit (P-4.5-B.1) implements the chosen option. The
> acceptance criteria live in `docs/PLAN.md` §7 P-4.5 (three-layer
> criterion, amended 2026-04-21).

## 1. Problem

### 1.1 Measured signal

`qwen3-0.6b-ttft-under-concurrency` is the P-4 scenario specified by
PLAN §P-4 Deliverables to resolve Q-010: one long (~301-token) prompt
admitted alongside three one-character prompts at `max_batch_size=4`,
`prefix_cache=False`, `max_tokens=4`. Per-row first-token wall offsets
are emitted under `metadata.rows[i].first_token_ms_offset` in the
bench JSONL.

Two independent measurements (PLAN §10 Q-010 Resolution):

| measurement | isolated smoke TTFT | concurrent max first-token offset | ratio |
| --- | --- | --- | --- |
| Codex (2026-04-21) | 11.8 ms | 81.28 ms | 6.9× |
| Silica run 1 | 16.3 ms | 77.5 ms | 4.76× |
| Silica run 2 | 19.3 ms | 85.2 ms | 4.42× |
| Silica run 3 | 17.1 ms | 78.0 ms | 4.56× |
| Silica run 4 | 18.7 ms | 77.1 ms | 4.13× |

Single-run ratios straddle Q-010's 5× promotion threshold. The
dispositive signal is not the magnitude but the fact that **all four
concurrent rows' first-token offsets sit within ≤ 0.2 ms of each
other** — the timing signature of a single batched prefill forward.

### 1.2 Root cause

`silica/scheduler/batcher.py::_prefill_phase`:

```python
def _prefill_phase(self) -> list[BatchEvent]:
    assert self._batch_cache is not None
    tokens = self._build_prefill_tokens()  # (B, T_max)
    logits = forward_batched(self._model, tokens, list(self._batch_cache))
    ...
```

`_prepare_cohort` seals every row admitted before the first `step()`
into a single cohort, computes `T_max = max(len(r.prompt_ids))`, and
left-pads every shorter prompt to `T_max`. The whole cohort runs as
one `(B, T_max)` forward — short rows spend O(T_max − len(short)) of
their TTFT waiting on pad positions attached to the long row.

`_admit_miss_cohort` (mid-run admission) has the same shape, but over
the mid-run admitted subset only; it does not mix with the incumbent
cohort's decode because `step()` early-returns after Phase 2 admit.

### 1.3 Scaling projection

Ratio scales roughly linearly with `T_max / len(short)`. The 301-token
current probe gives 4–7×. Scaling the long prompt to 2000 tokens on
the same scenario deterministically exceeds the 5× trigger with
margin. Q-010 is not a single-sample artefact.

## 2. Constraints

### 2.1 Hard invariant set to preserve

The `ContinuousBatcher` already carries three invariant families.
Whichever option P-4.5-B ships must leave each family either
unchanged or, if mutated, locally re-pinned with its own regression
test before the P-4.5-B.1 commit lands.

- **I-1..I-5** — row lifecycle & event taxonomy
  (`docs/P2_UNIT_16C_PREP.md` §1): every row emits `token` events
  before its `done`; terminal rows occupy their slot until reclaim;
  `req_index` stays stable across the whole stream.
- **B-1..B-9** — budgeter semantics
  (`docs/P2_UNIT_16D_PREP.md` §5): reservation-timing (B-8),
  reject-does-not-reserve (B-6' / Q-4), release-on-raise (B-6'(c)),
  no-preempt-ping-pong (B-9), evict-underrun-never-corrupts (B-2).
- **S-1..S-7** — prefix-cache invariants
  (`docs/P2_UNIT_16C_2_STEP_4_SKELETON.md` §4): lookup agrees with
  peek (S-1), eager-extract-before-filter (S-3b), hit-skip-forward
  (S-5), cache-None-bit-identical (S-6), intra-step event grouping
  (S-7).

### 2.2 MLX kernel constraint

Q-009 / R-7: MLX ships no flash-attn-equivalent variable-length
attention kernel that can mix prefill (T=chunk_size per row) and
decode (T=1 per row) in one forward. Without that kernel, any option
that pretends to "interleave prefill and decode" forward passes must
actually run two separate forwards per step — which is option (B)'s
design, not a proper chunked-prefill as in vLLM v1.

This constraint eliminates the most ambitious reading of "real
chunked prefill" from P-4.5's scope. A future MLX kernel upgrade
re-opens it; until then, our options are all **homogeneous-per-forward**.

### 2.3 Fp16 batched SDPA drift

PLAN §7 P-2 and §7 P-3-D3.1 Gemma4-31B empirical findings document
that, on Qwen3-0.6B and Gemma4-31B, batched forwards at different
`B` values do not produce bit-identical logits under fp16. Changing
the batch composition changes fp16 roundoff and therefore the greedy
argmax. Strict bit-identity between the chunked path and the
unchunked path is therefore **not a testable criterion** on these
models; the PLAN P-4.5 acceptance was amended 2026-04-21 to a
three-layer criterion whose numerical reference is direct mlx-lm
batched reference over the sub-cohort, mirroring P-3-D3.1 precedent.

## 3. Option catalogue

Three paths, each forward-homogeneous under §2.2.

### Option (A) — Uniform chunked prefill with row stragglers

**Idea.** Split every prefill into K chunks of size `T_chunk`. Every
row presents `T_chunk` tokens per forward. Short rows finish their
real prompt after K_short chunks and then tread water for
`K_long − K_short` chunks, feeding pad tokens. After K_long chunks
every row transitions to DECODE.

**Scheduler shape.** `_prefill_phase` grows a chunk-loop outer
iteration. New row-state distinction: "prefilled but still attending
to long-row prefill chunks". Requires a per-row `prefill_offset`
tracked alongside `left_padding`. `BatchKVCache.update_and_fetch`
must accept `T=T_chunk` forwards that feed further non-final prefill
tokens — mlx-lm's `BatchKVCache` already supports this pattern
(multiple `update_and_fetch` calls per row before decode begins).

**Short-row TTFT.** After `K_short` chunks, the short row's last
prompt token has been fed through a forward; its logits can be
sampled. So short-row TTFT ≈ `K_short × T_chunk × per_token_forward`
instead of `T_max × per_token_forward`. Improvement factor ≈
`T_max / (T_chunk × K_short)`, best when `T_chunk` is small.

**Throughput cost.** Long-row throughput is unchanged (it still does
`T_max` tokens of prefill). Short rows do `K_long × T_chunk` tokens
of forward for a `len(short)`-token prompt — waste factor `≈ K_long
× T_chunk / len(short)`. On the current probe (`T_max=301`, three
`len=1` short rows, `T_chunk=16`), each short row does ≈ 20 chunks of
forward for a 1-token prompt — **20× the minimum** forward work per
short row.

**fp16 correctness.** Chunked causal SDPA accumulates attention in
different partial sums than a one-shot forward. Bit-identity with
the unchunked path fails (documented above). Each row's individual
token stream should hold against mlx-lm direct batched reference run
with the same chunked schedule — but mlx-lm has no public reference
for uniform-chunked batched prefill on its own `BatchKVCache`, so
the numerical reference must be written first.

**Scheduler footprint.** Large. `_prefill_phase` rewritten; new
row-state; chunk-bookkeeping; `_sample_and_emit_rows` now skips rows
mid-prefill. Estimated 400–600 diff lines in `batcher.py`, possibly
additional in `_admit_miss_cohort` for symmetric handling.

**Upgrade path.** Directly becomes proper chunked prefill when MLX
ships variable-length attention — at that point the "tread water"
cycles for short rows disappear and short rows join the decode
forward earlier.

### Option (B) — Sub-cohort split at cohort seal

**Idea.** `_prepare_cohort` inspects the per-row prompt lengths. If
the length-spread ratio `max_len / min_len` exceeds a threshold K
(to be pinned, see §4.4), the cohort splits into sub-cohorts by
length class. `step()` drives each sub-cohort's prefill in turn;
each sub-cohort's prefill is a single batched forward. After all
sub-cohorts have prefilled, the batcher unifies them into a single
live row set for the decode phase.

**Scheduler shape.** A new batcher-internal state machine: PREFILL
sub-cohort index `self._prefill_sub_cohort`. `_prepare_cohort`
produces `self._sub_cohorts: list[list[int]]` (row indices grouped
by length quantile) and `self._batch_cache` is built in stages:
first sub-cohort's cache at its own `T_max_sub`, then extended by
each subsequent sub-cohort's cache on admit — exactly the
`BatchKVCache.extend` path `_admit_miss_cohort` already exercises.

**Short-row TTFT.** Short sub-cohort (len ≈ short_T_max) prefills
first: TTFT ≈ `short_T_max × per_token_forward`, independent of the
long sub-cohort. Long-row TTFT is unchanged (≈ `long_T_max ×
per_token_forward`).

**Throughput cost.** Two sub-cohorts means two separate prefill
forwards instead of one. Per-forward launch overhead is paid twice;
but the total effective prompt-token count (`sum(len(row.prompt_ids))`)
is identical. Marginal overhead ≈ one extra forward launch + one
extra BatchKVCache extend per step-0-split.

**fp16 correctness.** Short rows run a B=3 (or whatever the short
sub-cohort size is) forward; long row runs a B=1 forward. Neither
equals the original B=4 forward. Strict bit-identity against the
unchunked Silica path is not achievable (confirmed §2.3). The
numerical reference per sub-cohort is direct mlx-lm batched: the
short sub-cohort's tokens should match mlx-lm direct batched run over
just those three short prompts with `left_padding=[0,0,0]`; the long
sub-cohort's tokens should match mlx-lm direct batched over just the
one long prompt (which degenerates to `_direct_mlx_lm_batched_reference`
on B=1, identical to single-request).

**Scheduler footprint.** Medium. `_prepare_cohort` rewritten to
compute sub-cohorts; `step()` gains an `if self._prefill_sub_cohort
< len(self._sub_cohorts)` branch before the global-prefill / decode
dispatch; `_build_prefill_tokens` scoped to the current sub-cohort.
The mid-run admission path (`_admit_miss_cohort`) already handles
"extend into existing batch cache"; we reuse it verbatim. Estimated
150–250 diff lines in `batcher.py`. New unit tests pin the
sub-cohort state machine.

**Upgrade path.** Neutral. If MLX eventually ships variable-length
attention, option (B) would be rewritten as option (A), not
extended. The investment is bounded.

### Option (C) — Admission-time reorder via waiting queue

**Idea.** `Engine.generate_batch` sorts admissions by prompt length
ASC and admits only the shortest prompts to the initial cohort (via
pre-step `add_request`); the rest go to the waiting queue. The
initial cohort prefills and enters decode; mid-run admission draws
the long prompt(s) from the queue and runs `_admit_miss_cohort` — an
existing, tested code path that performs a B=1 (or B=k for k long
prompts) dedicated prefill forward for the new rows and extends
their K/V into `self._batch_cache`.

**Scheduler shape.** Zero change in `ContinuousBatcher`. All changes
live in `silica/engine/__init__.py::generate_batch`: (a) sort
`admissions` by length ASC; (b) cap the pre-step-admit count based
on a length-spread threshold; (c) route the remainder through the
same `remainder → batcher.step() bootstrap → add_request(...)` path
the current code already uses for `len(prompts) > max_batch_size`.

**Short-row TTFT.** Same as option (B): short rows prefill in their
own smaller forward, bypass the long prompt's `T_max`. TTFT ≈
`short_T_max × per_token_forward`.

**Throughput cost.** Same as option (B): two separate prefill
forwards. `_admit_miss_cohort` already pays this cost in existing
workflows (`max_batch_size < len(prompts)`), so the forward path
itself is regression-locked.

**fp16 correctness.** Same as option (B): short cohort forward is
B=3; long cohort forward is B=1; neither equals B=4. Numerical
reference is direct mlx-lm batched per sub-cohort — same shape
reference that P-3-D3.1 Gemma4 uses (see
`silica/bench/runner.py::_direct_mlx_lm_batched_reference`, which is
the helper we already wrote for B>1 batched reference comparisons).

**Scheduler footprint.** Minimal. `batcher.py` diff: zero lines
expected (assuming `_admit_miss_cohort` handles B=1 admits cleanly —
empirically verified 2026-04-21, see §5.1 below). `engine/__init__.py`
diff: ~30–60 lines for sorting + threshold logic + new tests.
Estimated total < 100 diff lines across repo.

**Upgrade path.** None. Option (C) is an admission heuristic; it
does not structurally address long-context OOM (a single long prompt
alone will still OOM at prefill on a large enough `T_max`; no
batching trick saves it). If long-context OOM becomes a separate
objective, option (A) proper chunked prefill must be revisited.

## 4. Trade-off matrix

| criterion | (A) uniform chunked | (B) sub-cohort split | (C) admission reorder |
| --- | --- | --- | --- |
| Short-row first-token TTFT improvement on Q-010 | yes, small-`T_chunk` best | yes | yes |
| Short-row second-token latency improvement | yes (short rows rejoin decode-only early) | partial (long-prefill idle step remains) | no (one idle step while long row prefills) |
| Batcher diff footprint (est. lines) | 400–600 | 150–250 | ~0 |
| Engine diff footprint (est. lines) | 0 | 0 | 30–60 |
| New scheduler state machine | yes (chunk loop + row state) | yes (sub-cohort state) | none |
| B-1..B-9 re-verification needed | yes | yes (partial) | no |
| S-1..S-7 re-verification needed | yes (prefix-hit under chunking is new ground) | partial | no |
| I-1..I-5 preserved | requires care | yes | yes |
| Short-row forward waste | up to ~20× per short row | none | none |
| Greedy vs unchunked bit-identity | no | no | no |
| Numerical reference available | must be written | reusable P-3-D3.1 precedent | reusable P-3-D3.1 precedent |
| Helps long-context single-prompt OOM | eventually, with MLX kernel | no | no |
| Upgrade path when MLX ships var-len attn | direct | needs rewrite | needs rewrite |
| Risk of hitting P-4.5 Notes' 300-line ceiling | exceeds | near ceiling | well under |

## 5. Recommendation — Option (C)

### 5.1 Why (C)

The three considerations that tip the decision toward (C):

1. **Minimum blast radius on a proven scheduler.** The `ContinuousBatcher`
   carries twenty-one labelled invariants (I-1..I-5 + B-1..B-9 +
   S-1..S-7) pinned across ten tests files and ~2000 lines of
   production code. Options (A) and (B) touch `_prefill_phase` and
   `_prepare_cohort`, which exercise all three invariant families;
   option (C) touches neither. Re-verifying invariants is where
   real time gets spent, and (C) requires no such re-verification.
2. **Code-path reuse — `_admit_miss_cohort` is the existing mid-run
   prefill-a-new-subset path.** That path already runs a dedicated
   batched prefill for a subset admitted after the initial cohort,
   already calls `adapter.make_batch_cache(left_padding)`, already
   extends K/V into the main cache, already respects B-1..B-9. It
   is already tested end-to-end on Qwen3-0.6B (`tests/test_p2_batched_parity.py`)
   and Qwen3.5-0.8B hybrid (`tests/test_p3_hybrid_batched_parity.py`).
   (C) routes exactly one more admission through this path.
3. **Empirically verified 2026-04-21.** A direct probe on the target
   shape:

   ```python
   prompts = ['A', 'B', 'C', 'The capital of France is']
   engine.generate_batch(prompts, params, max_batch_size=3)
   ```

   Three short prompts admitted pre-step (B=3 prefill), long prompt
   via `_admit_miss_cohort` on the next step (B=1 prefill). All four
   rows reach `done`; no `aborted`; row 3 emits `" Paris. The capital"`
   — a plausible greedy continuation. `_admit_miss_cohort` handles
   B=1 mid-run admissions cleanly.

### 5.2 What (C) explicitly does NOT do

Scope boundary worth marking up front, so P-4.5's success is not
mistakenly expected to cover these:

- **Single-long-prompt OOM** — if a user runs `engine.generate_batch(
  ['<200k-token prompt>'], ...)` on a budget that can't hold `T_max ×
  bytes_per_token × 1` tokens of K/V, (C) does nothing. Proper
  chunked prefill (option (A) with a real MLX variable-length-attn
  kernel) is the long-term answer; P-4.5 does not ship it.
- **Per-row TTFT fairness when all prompts are the same length.** If
  four 300-token prompts arrive together, (C)'s sort-and-cap does
  not split them; all four prefill together; TTFT for all four is
  `300 × per_token_forward`. That's the same as current behaviour,
  and is correct — the fairness issue Q-010 measures is a
  heterogeneous-length pathology, not a homogeneous one.
- **Per-token fairness beyond the first token.** (C) fixes
  first-token fairness, not full per-token latency fairness. Under
  the current batcher, `step()` early-returns after Phase 2
  admission (lines 364–369 of `silica/scheduler/batcher.py`:
  "Admit ran its own batched prefill forward; existing DECODE rows
  idle this step so prefill-T vs decode-T=1 don't mix"). On the
  Q-010 scenario under (C), short rows receive their first token in
  step 0 (short cohort prefill) and transition to DECODE; step 1
  then runs mid-run admit for the long row — which means the
  incumbent short rows **idle for one step while the long row
  prefills**. Their second-token offset therefore still pays the
  long-row prefill cost once. The P-4.5-B.1 acceptance scopes
  strictly to first-token ratio (Q-010's trigger signal), matching
  the PLAN §7 P-4.5 Acceptance (a) clause. Full per-token latency
  fairness requires mixing prefill-T and decode-T in one forward,
  which is blocked on MLX variable-length attention — option (A)
  territory, out of P-4.5 scope.
- **Queued-cohort fairness.** (C) reorders admissions at the
  *initial cohort* level only. When
  ``max_batch_size < short_count + 1``, shorts that overflow the
  initial cohort land in the waiting queue **alongside** the long
  prompt; ``_admit_miss_cohort`` then batches the entire queue in
  a single mid-run admission forward. Concrete example: prompts
  ``[short_a=1 tok, short_b=1 tok, short_c=1 tok, long=6 tok]`` with
  ``max_batch_size=2``, threshold 2.0 → cap=2 → initial cohort
  ``{short_a, short_b}`` prefills at ``(B=2, T=1)``, then the queue
  ``{short_c, long}`` admits at ``(B=2, T=6)`` — ``short_c`` is
  again dragged to the long's ``T_max``. This is pinned in
  ``tests/test_engine_admission_reorder.py::
  test_generate_batch_queued_short_gets_batched_with_long_not_fixed``
  so the limit is observable as a regression lock, not invisible.
  The Q-010 trigger scenario exposes ``max_batch_size=4`` ≥
  ``short_count+1=4``, so this edge does not bite the measured
  Q-010 shape — but any caller who sees the chat REPL or a future
  HTTP client send more short rows than ``max_batch_size`` leaves
  room for will hit the limitation. Fixing it requires per-step
  admission rebalancing (admit shorts first per-step, not only at
  initial cohort), which is a more invasive scheduler change than
  (C) and belongs in a follow-up (option (B) territory).
- **Backpressure under heavy sustained mixed loads.** The admission
  heuristic re-orders one `generate_batch` call. Over a long stream
  of calls with mixed lengths, the waiting queue's FIFO ordering
  still applies. P-8 HTTP server will need its own policy at that
  layer.

### 5.3 Why not (B) despite the same TTFT profile

(B) is the cleanest alternative and if the implementer strongly
prefers batcher-owned logic over Engine-owned heuristics, it is a
reasonable substitute. Its two disadvantages are:

- Requires adding `self._prefill_sub_cohort` state + the sub-cohort
  prefill dispatch branch in `step()`, which touches I-1 (row
  lifecycle ordering — "prefill before decode" becomes "prefill
  sub-cohort N before sub-cohort N+1 before decode") and requires a
  new regression test.
- No reuse of `_admit_miss_cohort`'s already-regression-locked
  behaviour. Sub-cohort prefill is freshly written even though the
  shape is isomorphic to mid-run admit.

Neither is a showstopper; they are the ~100 lines of diff (B) spends
that (C) avoids.

### 5.4 Why not (A) for P-4.5

(A)'s upgrade-path argument — natural home for future MLX
variable-length-attention chunked prefill — is real. It does not
clear the P-4.5 Notes' 300-line ceiling on `batcher.py`, nor does
P-4.5's scope include long-context OOM (the real payoff of (A)).
Revisit (A) under a dedicated phase (v0.2 candidate), not under
P-4.5.

## 6. `ContinuousBatcher` and `Engine` touchpoints for option (C)

Concrete touchpoints the P-4.5-B.1 implementer will modify. Line
numbers are against HEAD at `2849443`.

### 6.1 Required

- `silica/engine/__init__.py::generate_batch` — **main change site**.
  - Insertion point for sort: between `admissions` construction (line
    232) and `effective_batch_size` decision (line 240).
  - New helper `_sort_admissions_by_length(admissions)` returning
    admissions in length-ASC order (stable sort; ties preserve the
    user's original order so `req_index` continuity is predictable).
  - New helper `_initial_cohort_cap(admissions_sorted,
    effective_batch_size, spread_ratio_threshold)` returning how many
    leading (shortest) admissions to place in the initial cohort.
    Spec (this is what the implementer writes to, verbatim):
    - **Preconditions:** `admissions_sorted` is length-ASC;
      `effective_batch_size >= 1`; `spread_ratio_threshold > 1.0`
      (the helper raises `ValueError` on `<= 1.0` — a threshold of
      exactly 1 would require equal lengths to avoid splitting, which
      is a degenerate always-split policy).
    - **Homogeneous-batch fast path.** If `len(admissions_sorted)
      <= 1` or `max_len / min_len <= spread_ratio_threshold`, return
      `min(effective_batch_size, len(admissions_sorted))` — current
      behaviour, no split. (Uses `<=`, not `<`, so a threshold of
      2.0 applied to a `[1, 2]` pair does not split.)
    - **Split path.** Otherwise, find `first_exceeding_index` = the
      smallest index `k` such that `admissions_sorted[k].len >
      spread_ratio_threshold * admissions_sorted[0].len`, and return
      `cap = max(1, min(effective_batch_size, first_exceeding_index))`.
      The `max(1, ...)` floor handles the pathological case where
      `admissions_sorted[0]` itself already exceeds the threshold
      against itself (impossible by construction, but belt-and-
      braces); the `min(effective_batch_size, ...)` ceiling is the
      **hard invariant** that the runtime respects `max_batch_size`.
    - **Worked reverse examples (regression lock material):**
      - lens `[1, 1, 1, 3]`, threshold `2.0`, `max_batch_size=2` →
        `first_exceeding_index = 3`, `cap = min(2, 3) = 2`. The
        first two short rows admit pre-step; rows 2 and 3 queue.
        (Without the `min(effective_batch_size, ...)` clamp the
        naive return would be `3`, which violates `max_batch_size`.)
      - lens `[1, 1, 1, 3]`, threshold `2.0`, `max_batch_size=4` →
        `first_exceeding_index = 3`, `cap = min(4, 3) = 3`. Three
        short rows admit pre-step; row 3 (the `len=3` outlier)
        queues. Target Q-010 shape.
      - lens `[300, 300, 300, 300]`, threshold `2.0`,
        `max_batch_size=4` → homogeneous fast path returns 4.
        No split, no regression.
      - lens `[1]`, threshold `2.0`, `max_batch_size=4` → `len <= 1`
        fast path returns 1. Single-prompt call unchanged.
  - `generate_batch` signature gains one optional kwarg
    `length_spread_threshold: float = 2.0`. Prefer the bare scalar
    over wrapping in a `BatcherPolicy` dataclass until a second
    admission-policy knob appears; keeping the surface small avoids
    prematurely exposing a config object we'd then have to evolve.
- `silica/engine/__init__.py` public surface — `generate_batch` gains
  one optional kwarg `length_spread_threshold: float = 2.0` (or an
  `admission_policy` object if we expect more knobs soon; prefer the
  simpler scalar until a second knob appears).

### 6.2 Likely

- `silica/bench/scenario.py` — workload struct may gain an optional
  `length_spread_threshold` field so bench scenarios can pin the
  threshold independently of the runtime default. Alternatively, the
  threshold is a `BenchRunner` constructor parameter. Prefer the
  second; scenario-level override is a P-5 concern if it surfaces.

### 6.3 Definitely not touched

- `silica/scheduler/batcher.py` — zero lines. All existing tests
  (`tests/test_batcher.py`, `tests/test_p2_batched_parity.py`,
  `tests/test_p3_*_batched_*.py`) must stay green unchanged.
- Existing BatchKVCache / prefix-cache / budgeter code. (C) is an
  admission heuristic; the prefill / decode / reclaim phases run
  exactly as today.
- The bench CLI (`scripts/bench.py`) — no flag changes expected for
  P-4.5-B. The bench runs whatever `BenchRunner` chooses as
  threshold; results for the TTFT-under-concurrency row change
  shape only.

### 6.4 New tests

- `tests/test_engine_admission_reorder.py` — new file, pinning:
  - Sort-and-cap preserves `req_index` values (the user-facing
    req_index the event stream emits matches the original list
    order, not the sorted order — critical invariant for
    generate_batch callers that map events back to their input list).
  - Length-spread ratio ≤ threshold → single cohort (bit-identical
    to current behaviour when threshold ≥ actual spread, regression
    lock for homogeneous workloads).
  - Length-spread ratio > threshold → initial cohort contains only
    length-class-1 prompts; remainder queues; mid-run admit drains
    the queue.
  - All rows terminate cleanly; no `aborted`; event ordering
    preserves per-row `token` → `done` shape (I-1).

## 7. Length-spread threshold — parameter choice

### 7.1 The knob

The only new tuning parameter (C) introduces is
`length_spread_threshold: float`. If `max(prompt_lens) /
min(prompt_lens) ≤ threshold`, all prompts admit as one cohort
(current behaviour). Otherwise, the cohort is split at the first
prompt whose length exceeds `threshold × min(prompt_lens)`.

### 7.2 Why not a per-token absolute cutoff

A token-count cutoff like "prompts longer than 100 tokens go to the
queue" does not scale with model context length (a 100-token cutoff
that makes sense for Qwen3-0.6B is irrelevant for Qwen3.5-27B at
256K context) and does not scale with batch content (if every prompt
in a batch is 5 tokens, a 100-token cutoff fires never; if every
prompt is 50k tokens, it fires always — neither is right).

A *ratio* threshold is workload-relative and scales naturally.

### 7.3 Default value rationale

Default: `length_spread_threshold = 2.0`.

Reasoning:

- Below 2.0: `_prefill_phase` pays at worst ≈ 2× the minimum
  prefill-forward cost per short row (each short row has at most
  `T_max / 2 = min_len` extra left-pad positions). The measured
  4–7× ratio at 301:1 is visible because the ratio is 301, not
  because it is > 2.
- Above 2.0: the ratio of "wasted short-row prefill tokens" to
  "short-row real prefill tokens" crosses 1.0 — i.e. the short row
  spends more of its TTFT on padding than on its own prompt.
- At 2.0 exactly, Q-010's 301:1 scenario splits (ratio 301 > 2);
  a homogeneous-300 batch does not (ratio 1 < 2).

This is a heuristic; the value is callable and should be revisited
after the P-4.5-B.1 implementation ships its own bench row.
Candidates for future tightening include (a) measuring real
workloads from the chat REPL, (b) picking the threshold adaptively
per-step based on the admission queue content.

### 7.4 Testing the threshold

`tests/test_engine_admission_reorder.py` parametrizes over
`length_spread_threshold` ∈ {1.5, 2.0, 4.0} and prompt sets
chosen to straddle each threshold. The parametrization pins the
cap-vs-split decision deterministically; it does not claim any
particular TTFT improvement. The TTFT ratio acceptance lives in
the bench row under `qwen3-0.6b-ttft-under-concurrency` (PLAN P-4.5
Acceptance (a)).

## 8. Acceptance sign-off

When P-4.5-B.1 ships, the following four things are demonstrable:

1. **Q-010 ratio below threshold (PLAN P-4.5 Acceptance §a).** The
   acceptance fixture in
   `tests/test_engine_admission_reorder.py::test_q010_ratio_below_threshold_on_five_runs`
   spawns **one subprocess** that runs the `(smoke, ttft)` pair six
   times back-to-back:

   ```bash
   python -m scripts.bench \
       --scenario qwen3-0.6b-smoke --scenario qwen3-0.6b-ttft-under-concurrency \
       --scenario qwen3-0.6b-smoke --scenario qwen3-0.6b-ttft-under-concurrency \
       --scenario qwen3-0.6b-smoke --scenario qwen3-0.6b-ttft-under-concurrency \
       --scenario qwen3-0.6b-smoke --scenario qwen3-0.6b-ttft-under-concurrency \
       --scenario qwen3-0.6b-smoke --scenario qwen3-0.6b-ttft-under-concurrency \
       --scenario qwen3-0.6b-smoke --scenario qwen3-0.6b-ttft-under-concurrency \
       --out /tmp/p4_5_acceptance.jsonl
   ```

   The first `(smoke, ttft)` pair is discarded as warmup (metal
   kernel compile on the subprocess's first forward of each shape).
   From the next five pairs, compute the ratio
   `max(offsets_short) / smoke_ttft_ms` per pair — where
   `smoke_ttft_ms` is the `ttft_ms` field on the
   `qwen3-0.6b-smoke` row and `offsets_short` is
   `[row[i].first_token_ms_offset for i in short_rows]` on the
   `qwen3-0.6b-ttft-under-concurrency` row. **Short-row filter is
   adaptive**, not hard-coded: `short_rows` is the set of row
   indices whose prompt-token length is less than
   `max_prompt_len / length_spread_threshold`. The current catalog
   places the long prompt at `req_index=0`, so `short_rows = {1, 2,
   3}`, but the adaptive filter stays correct under any future
   catalog reordering.

   Per-run ratio must be **< 3.5× in all five measured pairs**, not
   merely on average. The 3.5× threshold reflects the measured
   post-fix steady-state distribution on Qwen3-0.6B (typical
   `{2.53, 2.78, 2.95, 3.07, 3.27}×`); pre-fix ratios on the same
   scenario pair ranged `4.13-4.76×`. Strict `< 3.0×` is unachievable
   under option (C) because the B=3 short-cohort prefill carries an
   intrinsic 2-3× overhead vs B=1 isolated smoke — true single-step
   fairness requires MLX variable-length attention (Q-009 / R-7) and
   lives outside P-4.5 scope. A manual user workflow produces the
   same data by running the subprocess command above and eyeballing
   the JSONL; the pytest fixture automates that plus the `< 3.5×`
   per-pair assertion.
2. **No regression (PLAN P-4.5 Acceptance §d).** `python -m
   scripts.bench --all` under the default env-var set exits with
   all cache-only rows `status="ok"`; dual-gated rows still skip.
3. **Admission-reorder unit tests pass.**
   `tests/test_engine_admission_reorder.py` passes with the default
   `length_spread_threshold = 2.0` and under parametrized
   alternatives `{1.5, 2.0, 4.0}` (covers straddling the test
   prompt set); sort stability + `req_index` preservation + the
   four worked reverse examples from §6.1 are each pinned as
   separate test functions.
4. **Existing batched-parity suites stay green unchanged.**
   `tests/test_p2_batched_parity.py`, `tests/test_p3_hybrid_batched_parity.py`,
   and `tests/test_p3_gemma4_batched_parity.py` continue to pass —
   the three-layer correctness criterion's (a) event-taxonomy and
   (b) per-row token-count layers are pinned there. The (c)
   numerical reference is covered by a dedicated helper in
   `tests/test_engine_admission_reorder.py::test_reordered_cohort_matches_mlx_lm_direct_batched_reference`.
   That helper compares Silica's per-row token streams against
   `silica.bench.runner._direct_mlx_lm_batched_reference` run over
   the same sub-cohort shape. **Index mapping.** The direct-batched
   reference returns tokens keyed by sub-cohort-local index
   (`0..k-1` for a short sub-cohort of size `k`); Silica's event
   stream emits `req_index` in the original user-supplied
   `prompts` order. The helper must invert the sort permutation
   from §6.1 (`sorted_order[sub_cohort_local_i] = original_req_index`)
   before comparing — comparing by positional index without the
   inverse permutation will fail silently on any non-identity sort,
   which is the common case for Q-010-shaped workloads. A unit
   test asserting the permutation inversion round-trips cleanly
   must land before the parity test itself.

## 9. What this document is NOT

- A commit. `docs/PLAN.md` is amended in the same commit as this
  document (acceptance criterion correction, 2026-04-21); no code
  changes.
- A specification for option (A) or option (B). If the P-4.5-B.1
  implementer chooses one of them over (C), they re-open this
  document, record the choice, and write a new
  `§ContinuousBatcher touchpoints` section for the chosen option
  before coding.
- A design for full vLLM-style chunked prefill. That is a v0.2
  candidate tied to MLX kernel availability (Q-009).

## 10. References

- `docs/PLAN.md` §7 P-4.5 (chunked-prefill correctness three-layer
  criterion, amended 2026-04-21); §10 Q-010 Resolution (2026-04-21
  promotion).
- `silica/scheduler/batcher.py::_prepare_cohort` /
  `_prefill_phase` / `_admit_miss_cohort` — current cohort and
  mid-run admit shapes.
- `silica/engine/__init__.py::generate_batch` — admission-ordering
  site (C)'s change lands here.
- `silica/bench/runner.py::_direct_mlx_lm_batched_reference` —
  numerical-reference helper for the three-layer acceptance (c).
- `docs/P2_UNIT_16C_PREP.md` §1 I-5 (row lifecycle invariants);
  `docs/P2_UNIT_16C_2_STEP_4_SKELETON.md` §4 S-1..S-7 (prefix-cache
  invariants); `docs/P2_UNIT_16D_PREP.md` §5 B-1..B-9 (budgeter
  invariants).
- PLAN §7 P-3 2026-04-20 Gemma4-D3.1 empirical finding — precedent
  for "numerical reference = direct mlx-lm batched, not Silica
  single-request".
- PLAN §10 Q-009 + §11 R-7 — MLX paged-attention / variable-length
  attention kernel availability risk; the constraint that closes
  option (A)'s mixed prefill+decode path in P-4.5 scope.
