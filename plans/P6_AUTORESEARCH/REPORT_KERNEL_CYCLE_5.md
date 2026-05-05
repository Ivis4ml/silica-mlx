# P-6 Autoresearch — fifth cycle (2026-05-03)

| Field | Value |
| --- | --- |
| Date | 2026-05-03 |
| Branch | `opus` |
| Status | **Honest negative — the cycle-4 chunked-decode "win" was a measurement artifact** |
| User authorisation | "continue" (2026-05-03) |
| Companion docs | cycle 1-4 REPORTs in `plans/P6_AUTORESEARCH/` |

## TL;DR

This cycle pursued the cycle-4 chunked-decode lever further, integrating it into ContinuousBatcher's `_decode_phase` so it would apply to the production warm-decode-b4 oracle path. The integration works, all 2673 tests pass, and one early run showed a tantalizing 47.92 tok/s through the oracle. **But careful reproducibility (10 runs at chunk=64) exposed the 47.92 reading as a measurement artifact**: when the oracle's stability check happened to pass, its post-warmup measurement window fell entirely within a chunk's lazy-emit phase (where intervals are nanoseconds because logits are already materialised). The window measured the *speed of `.item()` extraction* on cached logits, not real throughput.

End-to-end raw-wall-clock comparison across the same 13 runs:

| Configuration | Raw aggregate tok/s | Mean ± std |
| --- | --- | --- |
| chunk=1 (baseline, 3 runs) | 38.76, 38.17, 38.89 | **38.61 ± 0.38** |
| chunk=64 (10 runs) | 37.90, 35.92, 36.59, 37.64, 37.51, 37.63, 37.11, 37.59, 37.43, 37.73 | **37.30 ± 0.55** |
| Difference | -1.31 tok/s | within 3σ of combined noise (1.31 < ~1.8) |

**The chunked-decode path is at best null on the production B=4 path; possibly a slight regression within noise.** The +7% I measured in the cycle-4 raw `forward_batched` microbench did not translate through ContinuousBatcher's production loop.

## Why the cycle-4 microbench result didn't translate

Three independent reasons consistent with the data:

1. **mlx's lazy graph already does most of the pipelining I tried to add manually.** Calling `forward_batched` K times in a row without intermediate `mx.eval` builds a K-step graph that mlx evaluates at the next sync point. Per-call Python overhead is small relative to per-step GPU compute (~95 ms/step), so reducing it doesn't surface in wall time. The microbench's +7% was likely measurement noise or specific to the synthetic-input pattern (constant zeros input vs real argmax-chained input).

2. **ContinuousBatcher's per-step bookkeeping** (per-row admission state, prefix-cache integration, BatchEvent emission) is not the bottleneck. Reducing it to per-chunk doesn't free compute time because compute, not Python, dominates per step on B=4 dense 27B-4bit at ~95 ms/step.

3. **The chunked path adds overhead the unchunked path doesn't have**: per-step `mx.argmax` on the (B=4, V=248320) logits to lazily produce the next-step input, plus larger lazy graphs that mlx must schedule. These small costs nibble at the savings from reduced per-step .item() syncs.

## What was attempted (cycle 5)

### 1. Add `ContinuousBatcher._decode_phase_chunked(chunk)` method

`silica/scheduler/batcher.py` — runs K consecutive `forward_batched` calls with lazy `mx.argmax` chaining, single `mx.eval` at chunk end, then per-row `_sample_and_emit_batched` over each materialised logits slice. Skipped when spec is active or slice-prefill is active. Default `chunk=1` preserves byte-identical behavior; opt-in via `SILICA_DECODE_CHUNK` env var.

```python
def _decode_phase_chunked(self, chunk: int) -> list[BatchEvent]:
    if self._spec_active():
        return self._decode_phase_spec()
    pending_logits: list[mx.array] = []
    next_tokens: mx.array | None = None
    for k in range(chunk):
        tokens = self._build_decode_tokens() if k == 0 else next_tokens
        logits = forward_batched(self._model, tokens, list(self._batch_cache))
        pending_logits.append(logits)
        next_tokens = mx.argmax(logits, axis=-1, keepdims=True).astype(mx.int32)
    mx.eval(pending_logits[-1])  # single sync for the chunk
    all_events = []
    for logits_k in pending_logits:
        all_events.extend(self._sample_and_emit_batched(logits_k, is_prefill=False))
    return all_events
```

`step()` dispatches to `_decode_phase_chunked` when env > 1 and not in spec/slice-prefill mode. **All 2673 tests pass with default chunk=1.**

### 2. End-to-end measurement: warm-decode-b4 with chunk={1,8,16,32,64,128}

Oracle pass/fail across chunk sizes (single run each):

| chunk | Status | Reported decode_tok_s | Raw aggregate |
| ---: | --- | ---: | ---: |
| 1 | ok | 41.1 | 38.8 |
| 8 | failed | n/a | (similar) |
| 16 | failed | n/a | (similar) |
| 32 | failed | n/a | (similar) |
| 64 | ok (1 run) / failed (others) | 47.9 (1 run) | (37-38 across all) |
| 128 | failed | n/a | (similar) |

The oracle reports `decode_tok_s_warm_aggregate` only when it passes. When chunked path passes by alignment-luck, the post-warmup window measures within-chunk lazy intervals — gives misleadingly fast reading.

### 3. Reproducibility sweep at chunk=64 (the only chunk that ever passes)

10 runs:

```
37.90, 35.92, 36.59, 37.64, 37.51, 37.63, 37.11, 37.59, 37.43, 37.73
```

Mean **37.30 ± 0.55 tok/s**. Compared to chunk=1 baseline (3 runs) **38.61 ± 0.38 tok/s** — the chunked path is **slightly slower or flat**, not faster. 0/10 chunk=64 runs in this sweep passed the oracle (the earlier 1/3 was alignment luck).

### 4. Identification of the artifact

When chunk=64 passes the oracle:
- Warmup advances to boundary=368 (out of 384 tokens × 4 rows = 1536 intervals; per-row 384 intervals, warmup at 368 leaves 16 measurement intervals)
- Measurement window: intervals 368-383, all within chunk 6 (tokens 320-384)
- All 16 measurement intervals are tiny (~ms `.item()` extractions on already-materialised logits, since the chunk's `mx.eval` happened at token 320)
- `decode_tok_s = 1000 / mean_interval_ms` = misleadingly fast reading

The `decode_tok_s_warm_aggregate` of 47.92 was actually measuring "rate of token extraction from cached logits", not "rate of token generation by the model". The model's actual generation rate (raw aggregate = total_tokens / total_decode_wall_time) is ~37.5 tok/s — slightly slower than chunk=1.

## Aggregated finding across cycles 1-5

The autoresearch loop has now empirically tested **all single-session-tractable lever families** on dense Qwen3.5-27B-4bit B=4. None has produced a kept improvement on the running-best frame.

| Cycle | Lever family | Result |
| --- | --- | --- |
| 1-2 | Pointwise-fusion custom kernels (`fused_gated_output`, `fused_silu_mul`, `fused_qk_norm`) | All flat or slower than mx.compile-fused references. mx.compile already fuses pointwise ops. |
| 3 | Custom 4-bit QMM kernel (naive + SIMD-cooperative) | 0.694× and 0.241× slower than `mx.quantized_matmul`. Beating mlx's QMM requires `simdgroup_matrix` MMA primitives (multi-day). |
| 3 | mx.compile MLP wrap | Flat — cannot fuse across `mx.quantized_matmul`. |
| 3 | Layer-skip self-spec on off-the-shelf checkpoint | Max realistic 1.13× = 48 tok/s; checkpoint lacks early-exit training. |
| 4 | Engine.generate chunked-decode | +1.1% at B=1 within noise; ALL chunks fail warm-decode oracle stability gate. |
| 4 | Raw forward_batched lazy-chain microbench | +7.3% measured in synthetic microbench. |
| 5 | ContinuousBatcher chunked-decode integration | **Did not translate**: synthetic +7% became flat-to-slight-regression on production warm-decode-b4. The earlier 47.92 reading was a measurement artifact. |

**Running best on `decode_tok_s` remains 42.17 tok/s after 5 cycles.**

## What this empirically rules out

Cycles 1-5 collectively rule out the following levers as session-tractable for closing the 42 → 67+ gap:

1. **Naive Metal kernels via `mx.fast.metal_kernel`** for any pointwise fusion or QMM — mlx's tuned internals are the floor.
2. **mx.compile graph fusion** beyond what mlx-lm already applies — cannot fuse across non-pointwise ops.
3. **Layer-skip self-spec** on the production checkpoint — agreement floor too low.
4. **Lazy-graph decode chunking** on the production batcher path — doesn't materialise as actual speedup once integrated.

The composed envelope of 67-100 tok/s remains valid (per `plans/P6_AUTORESEARCH_NOT_LIMIT_PROOF.md`), but every empirically-tested session-tractable lever has been exhausted.

## What's left — multi-day or external-authorization paths only

1. **Custom simdgroup_matrix MMA QMM kernel** (3-7 days). The cycle-3 attempt confirmed naive Metal kernels lose; tuning with `simdgroup_matrix` primitives is the path. Estimated +5-10 tok/s lift if it matches mlx.
2. **Fused FA-2 with output gate baked in** (5-14 days). 2026-Q2 survey confirmed no public Apple-Silicon implementation. Targets 6.7% of step time.
3. **Fused conv1d + gated_delta_update** (3-7 days). Targets 24.9% of step time on DeltaNet layers.
4. **Speculative composition** with a viable drafter checkpoint. Currently blocked: Qwen3.5-0.8B (retired in C.4), DFlash drafter (retired), MTP head (absent in production safetensors), layer-skip self-spec (insufficient agreement).
5. **C.5 γ.1 read-only kernel survey** (1 day, pending user decision since 2026-05-02). Could re-open the spec lever if the upstream `humanrouter/ddtree-mlx` ships a no-torch tree-attention kernel.
6. **A new checkpoint** — early-exit-trained, MTP-preserving, smaller-group-size 4-bit, or activation-aware 3-bit. Multi-week + ~50 GB downloads + conversion + user authorization.

## Files added in cycle 5

- `silica/scheduler/batcher.py` — `_decode_phase_chunked` method + dispatch in `step()`; `_decode_chunk` field in `__init__`. Default chunk=1 = byte-identical to pre-cycle behavior. All 2673 tests pass.
- `/tmp/b4_c{1,8,16,32,64,128}.jsonl` — single-run sweep across chunk sizes
- `/tmp/b4_c1_{1,2,3}.jsonl` — chunk=1 baseline reproducibility
- `/tmp/b4_c64_run{1..10}.jsonl` — chunk=64 reproducibility sweep
- `plans/P6_AUTORESEARCH/REPORT_KERNEL_CYCLE_5.md` (this file)

## Ledger rows appended this cycle

- `AR_BATCHER_CHUNKED_B4` — discard; chunk=64 raw_agg 37.30 ± 0.55 vs chunk=1 raw_agg 38.61 ± 0.38; not a speedup; the earlier 47.92 reading was an oracle measurement artifact.
- `AR_BATCHER_CHUNKED_INTEGRATION` — diagnostic; ContinuousBatcher integration landed; default-OFF safe; opt-in path exists but doesn't produce measurable end-to-end gain.

## Running best on `decode_tok_s`

Unchanged at **42.17 tok/s**. After 5 cycles of substantive engineering, the running best has not moved.

## Honest stop — for real this time

Cycles 4 and 5 collectively prove that **the lazy-decode-chunking lever does not produce a measurable speedup on the production warm-decode-b4 path**, despite the synthetic microbench suggesting otherwise. This was the last single-session-tractable lever; cycles 1-3 had already exhausted simple kernel writing and self-spec.

**The empirical, measurement-anchored conclusion:**

> Closing the gap from 42.17 to the demonstrated 67-100 tok/s envelope requires multi-day kernel engineering (custom simdgroup_matrix MMA QMM, fused FA-2 with gate, fused gated_delta_update + conv1d) OR external authorization (new checkpoint, C.5 γ.1 survey decision). Continuing single-session experimentation will empirically reproduce the same null results.

This is the third time I've reached this conclusion across cycles 3, 4, and 5. The pattern is consistent: every single-session lever returns either correctness-pass-but-perf-flat, or a measurement artifact that doesn't survive careful reproducibility. The +7% raw-forward microbench in cycle 4 was the most promising signal in 5 cycles, and even it didn't translate through the production path.

**The next genuine progress requires a multi-day commitment**, not another single-session attempt.

## Updates to existing artifacts

- `plans/P6_AUTORESEARCH_LOG.tsv` — cycle 5 rows added; cycle 4 chunked-bench row updated to reflect the corrected understanding (the cycle-4 +17% claim was the artifact, not real).
- `plans/P6_AUTORESEARCH_PROGRESS_DECODE_TOK_S.png` — regenerated with cycle 5 data.
