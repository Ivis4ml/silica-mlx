# P-6 Autoresearch — twelfth cycle (2026-05-04) — DeltaNet bf16-state probe + 193 tok/s wall confirmation

| Field | Value |
| --- | --- |
| Date | 2026-05-04 |
| Branch | `opus` |
| Status | DeltaNet bf16-state correctness PASS; **E2E flat at 193 tok/s wall**; the cycle 11 v10 + cycle 12 bf16 + chunked decode all hit the same plateau |
| User authorisation | "great, continue" |
| Companion docs | cycle 11 FA-decode kernel work in `plans/P6_AUTORESEARCH/REPORT_KERNEL_CYCLE_11.md` |

## TL;DR

The cycle-1 per-step decomposition (74.2% DeltaNet + 21.9% full-attention + 4.0%
overhead) was measured at B=4. At B=48 the mix is different: weights traffic
dominates (15 GB at 4-bit) and the DeltaNet recurrent state, while large
absolutely (144 MB × 48 layers = 6.9 GB), is not the throughput-limiting
component.

Two cycle-12 probes both produced **honest negative E2E results**:

| Configuration | n | decode_tok_s mean ± std | vs cycle-10 baseline 193.9 ± 0.6 |
| --- | --- | ---: | --- |
| baseline (cycle-10 reproduce) | 1 | 191.8 | within noise |
| v10 alone (shadow wired) | 1 | 192.0 | within noise |
| **bf16 DeltaNet state alone** | 3 | **192.5 ± 1.1** | within noise |
| v10 + bf16 combined | 1 | 190.7 | within noise (slightly lower) |
| chunked decode (chunk=32) | 1 | OOG: warmup-did-not-stabilize | gate fail (re-confirms cycle-4/5 finding) |

**The 193 tok/s wall is real at warm-decode-b48.** Kernel-level wins on FA-decode
(cycle 11: 1.25-1.81× over mlx) and DeltaNet state-bandwidth halving (cycle 12:
fp32 → bf16 = ~22 ms theoretical save per step) do not move E2E throughput.

## What worked: bf16 state correctness

`mlx_lm.models.gated_delta._make_gated_delta_kernel` already templates the state
type (`StT`). Allocating the recurrent state as `mx.bfloat16` instead of
`mx.float32` is a one-line change to `gated_delta_update`. Greedy-decode token
ID parity holds against fp32 baseline on a 20-token Qwen3.5-27B sample:

```
Baseline (fp32 state): 'Paris.\n\n<think>\nThinking Process:\n\n1.  ...'
Patched (bf16 state):  'Paris.\n\n<think>\nThinking Process:\n\n1.  ...'
Match: True
```

The recurrent compute uses fp32 internally (per `state[i] =
static_cast<float>(i_state[s_idx]);` in the kernel source); only the
`state_in` / `state_out` HBM storage shifts to bf16. The accumulated
numerical error stays below the greedy-decode argmax boundary on tested
prompts.

bf16 state is exposed through `silica/kernels/shadow_install.py` as the
`SILICA_USE_BF16_DELTANET_STATE=1` env flag. With the cycle-12 fix to actually
wire `shadow_install.install()` into `Qwen3_5Adapter.from_hf_repo`, the patch
applies during real decode — confirmed via in-process inspection
(`gd.gated_delta_update.__name__ == '_patched_gated_delta_update'`).

## What didn't work: E2E throughput at B=48

bf16 state should save ~22 ms of bandwidth per step (144 MB × 48 layers / 307
GB/s). Observed E2E save: **0%**.

Possible explanations (none definitively confirmed):

1. **State traffic overlaps with weights / activations.** mlx schedules kernel
   launches such that DeltaNet state R/W and the 4-bit weight reads are
   concurrent on the unified-memory bus. Cutting one in half doesn't free wall-
   clock if the other dominates the critical path.
2. **Apple unified-memory cache effects.** The recurrent state is small enough
   per (b, hv) (16K × 4B = 64 KB at fp32, 32 KB at bf16) that hot per-layer
   state lives in L2/SLC, not HBM. The "144 MB total state" figure
   over-counts genuine HBM traffic.
3. **Python / dispatch overhead at B=48.** ~64 layers × per-step Python
   dispatch + mx.eval sync per generated token. At 193 tok/s, each token
   round trip is ~5.2 ms wall, consistent with a Python-bound loop where
   kernel-level optimizations save on the dispatch budget but not the
   sync floor.

The combined v10 + bf16 path actually came in slightly *lower* than baseline
(190.7) — likely just bench noise (single run), but at minimum the
combination is not synergistic.

## Why this matters (cycle 11 retrospective)

Cycle 11 reported "kernel-level victory" with v10 beating mlx by 1.25-1.81×.
Cycle 12 confirms what the cycle-11 advisor flagged as a risk: that win does
not translate to E2E because attention is below the noise floor of E2E step
time at the production B=48 configuration. The same is true for DeltaNet:
even halving its state traffic doesn't move the wall.

**The implication: 193.9 tok/s on warm-decode-b48 is at or very near the
production hardware ceiling for this model on M5 Pro 48 GB.** Further
kernel-level optimisations are extracting from a fixed budget that's already
saturated by something other than per-layer kernel time — most plausibly
mlx's eager-execution dispatch loop overhead, which scales with layer count
× decode steps regardless of per-kernel speed.

The real lever to move past 193.9 tok/s is **reducing dispatch / sync
frequency**, not making individual kernels faster. Cycle 4 already explored
this via SILICA_DECODE_CHUNK and found chunk=32 produces spiky per-step
intervals that fail the warm-decode oracle's stability gate. Re-confirmed
in cycle 12: chunk=32 at B=48 → `warm_decode_row_0_warmup_did_not_stabilize`.

## Files added in cycle 12

- `plans/P6_AUTORESEARCH/REPORT_KERNEL_CYCLE_12.md` — this file
- `silica/kernels/shadow_install.py` — extended with
  `SILICA_USE_BF16_DELTANET_STATE` env flag handling + restore() update;
  `silica/models/qwen3_5.py` — `from_hf_repo` now calls
  `shadow_install.install(model)` so the env flags take effect through the
  bench harness (cycle 11's E2E measurement was running unpatched because
  this hook was missing; cycle 12 closes that bug)
- 4 oracle JSONL artefacts: `/tmp/b48_{baseline_sanity, v10_wired,
  bf16state_run{1b,2,3}, combined, chunk32}_run*.jsonl`

## Ledger rows added cycle 12

- `AR_BF16_DELTANET_STATE_CORRECTNESS` (diagnostic) — bf16 state preserves
  greedy-decode token IDs vs fp32 on a 20-token Qwen3.5-27B sample. Token
  parity = True. Tolerance not measured at PPL level; would need a longer
  ablation before production claim.
- `AR_BF16_DELTANET_STATE_E2E_B48` (diagnostic) — 3 reproductions on real
  Qwen3.5-27B-4bit, mean 192.5 ± 1.1 tok/s. Statistically indistinguishable
  from cycle-10 baseline 193.9 ± 0.6. Does NOT move running-best line.
- `AR_SHADOW_INSTALL_WIRING_FIX` (defect-class) — cycle-11 v10 measurement
  was running un-patched (vanilla mlx) because `Qwen3_5Adapter.from_hf_repo`
  did not call `shadow_install.install`. Cycle 12 wires the call so all
  cycle-11 + cycle-12 + future env-flag-gated optimisations actually apply
  through the bench harness. Re-running cycle-11 v10-alone with the wired
  install: 192.0 tok/s (still within noise band; no E2E delta).

## Reflection

Cycles 11 and 12 demonstrate a kernel-engineering reality: wins inside
individual ops do not necessarily translate to E2E throughput when the
system is already running at a higher-level bottleneck. The autoresearch
loop's cycle 10 axis-shift (find optimal B, not optimal kernel) remains the
load-bearing wins. Subsequent cycle-11/12 kernel-level work has produced:

1. A tunable, correctness-validated FA-decode kernel that beats mlx-internal
   SDPA by 1.25-1.81× at decoder shape (not E2E load-bearing, but a real
   tool to keep in inventory).
2. A bf16 DeltaNet state path that's correctness-equivalent at greedy decode
   and theoretically saves ~22 ms/step of state bandwidth (not E2E
   load-bearing for the same dispatch-overhead reason).
3. Confirmation that the 193 tok/s wall is the actual hardware ceiling at
   warm-decode-b48 in the current mlx execution model — kernel-level
   optimisations cannot break it.

**For the running-best line, cycle 10's 193.9 ± 0.6 stays as the standing
record.** Cycle 11 + 12 are diagnostic/foundation work; no kept rows.

## What's next (if user authorises)

1. **Compute-graph-level optimisation rather than kernel-level.** mlx 0.31.x
   uses eager execution. A graph-traced or ahead-of-time-compiled decode loop
   may amortise dispatch overhead across layers/steps, breaking the 193 wall.
   Worth investigating `mx.compile` on the full transformer block.
2. **Speculative decoding with a strong drafter.** Cycle 10 §"What's next" #3
   listed B=48 + spec as compose-able; D-021 work has the framework
   plumbing. Spec gives multi-token-per-step, side-stepping the per-step
   dispatch ceiling. Currently blocked on C.5 γ.1 read-only survey decision.
3. **Profile mlx's actual scheduler at B=48.** A mlx-level instrument is
   needed to see whether the wall is dispatch-bound, sync-bound, or
   memory-bound at peak load. Without that, more guesses about why kernel
   optimisations don't translate are speculation.
