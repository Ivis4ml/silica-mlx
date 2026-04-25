# P-3-C5.2 numeric-drift experiment

Pre-implementation probe answering: under Qwen3.5 + bf16 + greedy
sampling, does today's replay-by-re-prefill path produce a
**byte-equal** recurrent state to the original single-step decode
path? The answer decides what shape C5.2's acceptance takes.

The experiment is run BEFORE C5.2 implementation lands. Result
shapes the test gate; see "Implication for C5.2 acceptance" below.

**Probe:** `docs/P3_C5_DRIFT_EXPERIMENT/probe.py`. Loads one
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
on the C5 targets at the horizons tested. Following the user's
2026-04-24 framing for this exact case, C5.2 instead locks:

1. **Call-order invariants** — `snapshot_recurrent_state` is
   invoked **before** `layer_cache.filter(kept)` inside
   `_preempt_active_row`; `restore_recurrent_state` is invoked
   **after** the replay's prefill `forward_batched` completes
   and **before** `self._batch_cache[layer].extend(row_cache[layer])`
   stitches it into the active batch.
2. **Snapshot boundary** — the snapshot stored on the replay's
   `_PendingAdmit.recurrent_snapshot` field describes the
   recurrent state at the exact token boundary `n_prompt +
   len(generated)` for the victim row, byte-aligned with the
   `composite_prompt` length the same record carries.
3. **Bit-exact restore** — post-restore `mx.array_equal` between
   the live cache's per-layer slots and a parallel no-preempt
   oracle's snapshot at the same token boundary. This is the
   strict gate that proves C5.2 actually changes the recurrent
   bytes the live cache holds, regardless of whether the change
   is observable through tokens.
4. **R-C5-2 aliasing defense** — the snapshot stashed on a
   replay `_PendingAdmit` survives multiple subsequent `step()`
   calls (more forwards on the original `_batch_cache` happen
   between preempt and replay-admit). The snapshot tensors must
   remain unchanged across those forwards. Same R-C5-2 invariant
   C5.1 already enforced inside the snapshot constructor; C5.2
   re-exercises it across the longer pending lifetime.

C5.2 explicitly does **NOT** prove "token bit-exactness vs
no-preempt oracle" because today's replay path already
satisfies that under greedy. It IS still useful to record the
token-stream observation as a sanity check (assert no
divergence post-implementation), but it is not the gate.

**Why C5.2 is still a correctness foundation, not a noise
optimization:** the drift exists at fp32 precision, sits ~1-2%
on the relative-Frobenius scale on tested targets, and is below
greedy-argmax under bf16 today. Several follow-on contexts can
push it across the boundary: (a) higher-precision sampling
(temperature > 0, top-p, beam) where small logit shifts move
sampled tokens; (b) longer horizons under repeated preempt/replay
cycles where drift may compound; (c) different hardware /
compiler stacks. C5.2 closes that drift before it becomes visible
rather than after, and creates the snapshot pathway C5.3 / P-7
will lean on for compute reuse and draft rollback.

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
PYTHONPATH=. uv run python docs/P3_C5_DRIFT_EXPERIMENT/probe.py \
    --repo Qwen/Qwen3.5-0.8B --decode-k 8 --continue-m 8 \
    --out docs/P3_C5_DRIFT_EXPERIMENT/drift_qwen3_5_0_8b.json

# 0.8B k=8 M=64 — longer continuation horizon.
PYTHONPATH=. uv run python docs/P3_C5_DRIFT_EXPERIMENT/probe.py \
    --repo Qwen/Qwen3.5-0.8B --decode-k 8 --continue-m 64 \
    --out docs/P3_C5_DRIFT_EXPERIMENT/drift_qwen3_5_0_8b_M64.json

# 0.8B k=32 M=128 — longer split-point AND longer horizon.
PYTHONPATH=. uv run python docs/P3_C5_DRIFT_EXPERIMENT/probe.py \
    --repo Qwen/Qwen3.5-0.8B --decode-k 32 --continue-m 128 \
    --out docs/P3_C5_DRIFT_EXPERIMENT/drift_qwen3_5_0_8b_k32_M128.json

# 4B k=8 M=32 — confirm scale-independence of the picture.
PYTHONPATH=. uv run python docs/P3_C5_DRIFT_EXPERIMENT/probe.py \
    --repo Qwen/Qwen3.5-4B --decode-k 8 --continue-m 32 \
    --out docs/P3_C5_DRIFT_EXPERIMENT/drift_qwen3_5_4b.json
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

This experiment closes. C5.2 implementation now proceeds with the
acceptance shape locked above:

1. `_PendingAdmit` schema gains
   `recurrent_snapshot: RecurrentSnapshot | None = None` (frozen
   dataclass field; lifecycle bound to the replay record per the
   user's 2026-04-24 ruling).
2. `Qwen3_5Adapter` snapshot/restore surface (already landed at
   C5.1) is wired into `_preempt_active_row` (snapshot before
   `filter`) and `_admit_miss_cohort` (restore after
   `forward_batched`, before `extend`).
3. Tests prove call-order, snapshot-boundary, bit-exact-restore,
   R-C5-2 aliasing across the longer pending lifetime, plus a
   greedy-token sanity row.
4. The `has_recurrent_state + prefix_cache` guard at
   `silica/scheduler/batcher.py:152-166` stays in place — guard
   removal is C5.4 by design.
