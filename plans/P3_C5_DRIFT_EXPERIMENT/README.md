# P-3-C5.2 numeric-drift experiment

Pre-implementation probe answering: under Qwen3.5 + bf16 + greedy
sampling, does today's replay-by-re-prefill path produce a
**byte-equal** recurrent state to the original single-step decode
path? The answer decides what shape C5.2's acceptance takes.

The experiment is run BEFORE C5.2 implementation lands. Result
shapes the test gate; see "Implication for C5.2 acceptance" below.

**Probe:** `plans/P3_C5_DRIFT_EXPERIMENT/probe.py`. Loads one
Qwen3.5 repo via `adapter_for_repo`, runs two paths (single-step
decode vs batched re-prefill of the same input sequence), and
compares per-layer recurrent state plus the M-step continuation
token stream.

## Methodology

Both paths process the same input sequence — the difference is
the per-step boundary at which `forward_batched` is invoked.

**Path A (single-step, simulates the original no-preempt path):**

1. `forward_batched(model, prefix, cache_A)` — prefill `T` tokens
   in one batched call. Cache state has consumed positions
   `0..T-1`.
2. Decode loop: `forward_batched(model, [[last_tok]], cache_A)`
   `× (k - 1)` times. Each call consumes one new input position.
   Cache state ends having consumed `T + (k - 1)` tokens.
3. Snapshot `state_A` — every DeltaNet layer's `(conv_state,
   recurrent_state)` slot pair via
   `adapter.snapshot_recurrent_state(cache_A, row_idx=0)`.

**Path B (batched re-prefill, simulates today's replay path):**

1. `forward_batched(model, prefix + decoded_a[:k-1], cache_B)`
   — prefill `T + (k - 1)` tokens in one batched call. The
   composite is exactly the input sequence path A processed.
2. Snapshot `state_B`.

**Comparisons:**

- Per-layer `mx.array_equal(state_A, state_B)` for both `conv` and
  `recurrent` slots — the strict byte-equality gate.
- Per-layer relative Frobenius error on fp32-cast tensors when
  not bit-equal, to quantify the drift magnitude.
- Continuation: from each path's final state, run `M` more
  greedy decode-steps and compare token streams. First divergence
  (if any) is recorded.

## Results

### Qwen3.5-0.8B (`drift_qwen3_5_0_8b*.json`)

`T = 61` prompt tokens (prompt fixed in the probe).

| `k` | `M` | layers bit-exact (conv) | layers bit-exact (recurrent) | worst rel-frob (conv) | worst rel-frob (recurrent) | tokens match |
| ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 8 | 8 | 0 / 18 | 0 / 18 | 9.28e-03 | 1.15e-02 | yes through 8 |
| 8 | 64 | 0 / 18 | 0 / 18 | 9.28e-03 | 1.15e-02 | yes through 64 |
| 32 | 128 | 0 / 18 | 0 / 18 | 8.94e-03 | 9.86e-03 | yes through 128 |

### Qwen3.5-4B (`drift_qwen3_5_4b.json`)

| `k` | `M` | layers bit-exact (conv) | layers bit-exact (recurrent) | worst rel-frob (conv) | worst rel-frob (recurrent) | tokens match |
| ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 8 | 32 | 0 / 24 | 0 / 24 | 2.23e-02 | 1.62e-02 | yes through 32 |

## Interpretation

This is the **middle case** the experiment was designed to
distinguish:

- Recurrent state is **NOT byte-equal** between single-step decode
  and batched re-prefill on either target. Per-layer
  fp32-cast relative Frobenius drift is ~1% on Qwen3.5-0.8B and
  ~2% on Qwen3.5-4B. The drift is reproducible — same input
  sequence, same seeds, same hardware; difference comes from
  reduction-order / per-step kernel layout differences between
  the two code paths.
- **Greedy argmax tokens DO match** through every continuation
  horizon tested (`M ∈ {8, 32, 64, 128}` on 0.8B; `M = 32` on
  4B). The drift sits below the argmax boundary.

## Implication for C5.2 acceptance

C5.2 should NOT bind its acceptance to "fixes observable token
drift" — that drift is not observable under greedy bf16 sampling
on the C5 targets at the horizons tested.

A first draft of the C5.2 implementation tried to bind a
**bit-exact restore vs no-preempt oracle** gate on the
full-replay path (snapshot taken at preempt, restore applied
after the replay's batched re-prefill, before
`_batch_cache[layer].extend(...)`). Wiring that and running it
against Qwen3.5-0.8B revealed a fundamental boundary mismatch
that the experiment's own math admits but did not call out
explicitly:

```text
preempt time live cache:
  cache consumed = T + len(generated) - 1
  (T prompt tokens fed; the first len(generated) - 1 decoded
   tokens fed back via decode-step; the latest sample is held
   in the row's `generated` list but NOT yet consumed by the
   cache.)

replay's _admit_miss_cohort:
  composite_prompt = prompt + generated  (T + len(generated) tokens)
  forward_batched(composite, k_batch_cache) consumes all of it
  cache consumed = T + len(generated)
```

The snapshot describes a `T + len(generated) - 1` consumed
state; the replay's post-prefill cache is at `T + len(generated)`
consumed — exactly one token (the last generated one) ahead.
Restoring the snapshot at "after replay's prefill, before
extend" rewinds the cache by one position and erases the last
generated token's consumption from the recurrent state. The
next decode-step then feeds the replay's freshly emitted token
into a cache that's missing one position from the row's
trajectory, breaking the sequence.

There is no boundary-aligned restore site on the **full-replay**
path without restructuring `_admit_miss_cohort` to skip the last
token of the composite during prefill and run an extra single-
step decode (a non-trivial change to the P-2 main path). The
2026-04-24 design call landed on **C5.2 = capture / stash; C5.3
= restore**, deferring restore enablement to admission paths
whose post-prefill boundary naturally matches the snapshot
boundary (the prefix-hit replay, where the prefill ends at the
prefix-block edge the snapshot was captured against).

C5.2 acceptance gates (the post-pivot list):

1. **Snapshot-before-filter** — `snapshot_recurrent_state` is
   invoked inside `_preempt_active_row` before
   `layer_cache.filter(kept)` mutates the batched cache.
2. **Snapshot stashed on `_PendingAdmit.recurrent_snapshot`** —
   the captured snapshot travels on the replay record; field
   defaults to `None` for non-recurrent adapters or for
   pendings created via `add_request`.
3. **Restore is NOT called on the full-replay path** — the
   `_admit_miss_cohort` path's full-prefill of
   `composite_prompt = prompt + generated` makes the post-
   prefill cache one token ahead of the snapshot. Restore here
   is contractually disabled at C5.2 (verified by a direct
   spy-adapter test).
4. **Pending-lifetime alias defense** — the snapshot stashed on
   a replay pending stays byte-identical across intervening
   `step()` calls that mutate the live `_batch_cache`. Re-
   exercises the R-C5-2 invariant C5.1's snapshot constructor
   already enforced, now across the longer pending lifetime.
5. **Boundary metadata** — the snapshot's boundary
   (`T + len(generated) - 1` consumed) is one token earlier than
   the replay record's `len(prompt_ids)` (= `T + len(generated)`
   composite). The off-by-one is intentional and documented.

C5.2 explicitly does **NOT** prove "bit-exact restore vs
no-preempt oracle" because the full-replay path's natural
boundary makes this impossible without restructuring; C5.3
(prefix-hit cooperation) is where the boundary-aligned restore
actually lands.

**Why C5.2 is still a correctness foundation, not a noise
optimization:** the snapshot capture / stash plumbing is what
C5.3 / P-7 build on. C5.3's prefix-hit path snapshots at block
boundaries during prefill (a different capture site, but using
the same `RecurrentStateAdapter` API surface C5.1 / C5.2
ratified). P-7's draft-snapshot lifecycle reuses the same
`RecurrentSnapshot` value object. By landing the capture path
end-to-end at C5.2, the API contract is fixed before either
follow-on unit needs it.

## Artefacts

| file | contents |
| --- | --- |
| `probe.py` | self-contained probe script; reproducible per repo and per (k, M) |
| `drift_qwen3_5_0_8b.json` | k=8 M=8 result |
| `drift_qwen3_5_0_8b_M64.json` | k=8 M=64 result |
| `drift_qwen3_5_0_8b_k32_M128.json` | k=32 M=128 result |
| `drift_qwen3_5_4b.json` | Qwen3.5-4B k=8 M=32 result |
| `README.md` | this document |

## Reproduction

The four committed JSON files were produced by the four invocations
below — running them on a clean checkout reproduces the numbers in
this README modulo machine-level reduction-order differences (the
drift magnitude is hardware/compiler-stack-sensitive at the
~1e-2 rel-frobenius scale; the bit-exact / not-bit-exact
classification is robust).

```bash
# 0.8B k=8 M=8 (default).
PYTHONPATH=. uv run python plans/P3_C5_DRIFT_EXPERIMENT/probe.py \
    --repo Qwen/Qwen3.5-0.8B --decode-k 8 --continue-m 8 \
    --out plans/P3_C5_DRIFT_EXPERIMENT/drift_qwen3_5_0_8b.json

# 0.8B k=8 M=64 — longer continuation horizon.
PYTHONPATH=. uv run python plans/P3_C5_DRIFT_EXPERIMENT/probe.py \
    --repo Qwen/Qwen3.5-0.8B --decode-k 8 --continue-m 64 \
    --out plans/P3_C5_DRIFT_EXPERIMENT/drift_qwen3_5_0_8b_M64.json

# 0.8B k=32 M=128 — longer split-point AND longer horizon.
PYTHONPATH=. uv run python plans/P3_C5_DRIFT_EXPERIMENT/probe.py \
    --repo Qwen/Qwen3.5-0.8B --decode-k 32 --continue-m 128 \
    --out plans/P3_C5_DRIFT_EXPERIMENT/drift_qwen3_5_0_8b_k32_M128.json

# 4B k=8 M=32 — confirm scale-independence of the picture.
PYTHONPATH=. uv run python plans/P3_C5_DRIFT_EXPERIMENT/probe.py \
    --repo Qwen/Qwen3.5-4B --decode-k 8 --continue-m 32 \
    --out plans/P3_C5_DRIFT_EXPERIMENT/drift_qwen3_5_4b.json
```

35B-A3B is not in the committed evidence set; the same `--repo
mlx-community/Qwen3.5-35B-A3B-4bit` invocation runs the probe
against it if the HF cache is present, and the experiment's
conclusion (drift exists, tokens match under greedy) is expected
to hold by the same numeric mechanism, but C5.2's acceptance
shape is decided here on the 0.8B / 4B data above.

Skip-gated on the requested repo's HF cache; probe exits with
status 2 when the repo is absent.

## C5.2 implementation pause point

This experiment closes. C5.2 implementation lands the
**capture/stash plumbing only** per the post-pivot acceptance
shape above:

1. `_PendingAdmit` schema gains
   `recurrent_snapshot: RecurrentSnapshot | None = None` (frozen
   dataclass field; lifecycle bound to the replay record per the
   user's 2026-04-24 ruling).
2. `Qwen3_5Adapter` snapshot/restore surface (already landed at
   C5.1) is wired into `_preempt_active_row` (snapshot before
   `filter`). `_admit_miss_cohort` does NOT call
   `restore_recurrent_state` on the full-replay path — the
   boundary mismatch documented above prevents a correct restore
   at that site. The restore call site is reserved for C5.3
   (prefix-hit admission), where the boundary aligns.
3. Tests prove call-order (snapshot-before-filter), pending-stash
   (`_PendingAdmit.recurrent_snapshot` populated), restore-NOT-
   called (full-replay path explicitly disabled), and R-C5-2
   aliasing across the longer pending lifetime. Bit-exact-restore
   is omitted (impossible at C5.2's natural boundaries); a
   greedy-token sanity row is also omitted because today's
   replay path already produces matching tokens (the experiment
   above documents this), so a "C5.2 doesn't break tokens" test
   would just re-verify pre-C5.2 behavior.
4. The `has_recurrent_state + prefix_cache` guard at
   `silica/scheduler/batcher.py:152-166` stays in place — guard
   removal is C5.4 by design.
