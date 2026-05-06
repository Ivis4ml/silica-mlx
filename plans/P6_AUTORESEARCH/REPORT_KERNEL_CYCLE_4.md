# P-6 Autoresearch — fourth cycle (2026-05-03)

| Field | Value |
| --- | --- |
| Date | 2026-05-03 |
| Branch | `opus` |
| Status | **Lever exists but doesn't clear running-best 3σ + breaks oracle stability gate — diagnostic, not kept** |
| User authorisation | "plz continue" (2026-05-03) — open authorization to keep pursuing |
| Companion docs | cycle 1/2/3 REPORTs in `plans/P6_AUTORESEARCH/` |

## TL;DR

This cycle pursued **execution-pattern levers** (the only remaining single-session-tractable category after cycles 2-3 confirmed simple kernels can't beat MLX internals). Found a real signal: a "lazy argmax chain" decode pattern that defers per-step `.item()` materialisation across N decode steps gives **+7.3% on raw `forward_batched` at B=4** (42.38 vs 39.49 tok/s within same run). Wired this into `Engine._drive` as `SILICA_DECODE_CHUNK` env flag (default 1 = byte-identical to pre-cycle behavior, all 2673 tests pass).

**Two caveats prevent this from clearing the kept-improvement gate:**

1. **The +7% applies to raw forward_batched, not the production warm-decode-b4 oracle path.** The B=4 production bench uses `Engine.generate_batch` → `ContinuousBatcher`, which has its own per-step `.item()` syncs in `_decode_phase` (e.g., `silica/scheduler/batcher.py:1922`). My change in `Engine._drive` only affects single-request `Engine.generate` (B=1 path).
2. **At B=1 (where the change applies), gain is +1.1% (16.55 vs 16.37 within same run, well under 3σ=0.63).** AND chunk=2..32 ALL fail the warm-decode oracle's per-step stability check (chunked path has spiky per-step timing — bursts of near-zero ms during lazy chunks, then big sync spike — which the oracle reads as decode-rate instability and rejects with `warm_decode_row_0_warmup_did_not_stabilize`).

The lever is **real but unconverted into a running-best gain in this session**. Translating it requires either (a) ContinuousBatcher chunked-decode integration (multi-day, risky given the batcher's admission/stop-token/streaming logic), or (b) a warm-decode oracle adaptation that accepts chunk-aggregated stability rather than per-step (oracle-policy change, requires user authorization per P6_AUTORESEARCH.md "changing gates or acceptance criteria").

## What was attempted

### 1. Decode-pattern microbench at B=4 raw forward_batched

`scripts/microbench_kernel_e2e.py` extended with three patterns: `per_step_eval` (current production-equivalent), `lazy_chain` (constant-zeros input), `lazy_argmax_chain` (next input = argmax of prev logits, true dependency chain).

| Pattern | chunk_size | tok/s p50 | vs per_step_eval (39.49) |
| --- | ---: | ---: | ---: |
| per_step_eval | — | 39.49 ± 0.75 | baseline |
| lazy_argmax_chain | 64 (single sync) | 41.83 ± 0.58 | +5.9% |
| lazy_argmax_chain | 8 | 42.02 ± 0.36 | +6.4% |
| lazy_argmax_chain | 16 | 42.28 ± 0.11 | +7.1% |
| **lazy_argmax_chain** | **32** | **42.38 ± 0.19** | **+7.3%** |

**Reading:** the lever is real and reproducible. Chunk=32 maximises throughput; chunk=64 (single sync) is slightly slower than chunk=32 (longer dependency graphs may have eval overhead). The `lazy_argmax_chain` keeps a real dependency (next input depends on previous logits via lazy `mx.argmax`); it is NOT the synthetic constant-input pattern. **This pattern is production-applicable** in principle — modulo the integration challenges below.

### 2. Wire SILICA_DECODE_CHUNK into Engine._drive

`silica/engine/__init__.py` — added env var read at `__init__`, plus a chunked-decode branch in the spec-off path:

```python
if self._decode_chunk > 1:
    chunk = min(self._decode_chunk, params.max_tokens - n)
    pending: list[mx.array] = []
    prev_arr = mx.array([tok_int], dtype=mx.int32)
    for _ in range(chunk):
        logits_step, _ = self._adapter.decode_step(prev_arr, handle)
        prev_arr = mx.argmax(logits_step, axis=-1, keepdims=True).astype(mx.int32)
        pending.append(prev_arr)
    mx.eval(pending[-1])
    for tok_arr in pending:
        tok_int = int(tok_arr.item())
        yield tok_int
        n += 1
        decode_count += 1
        history.append(tok_int)
        ctx.output_token_ids.append(tok_int)
        if tok_int in params.stop_token_ids:
            return
    continue
```

Default `SILICA_DECODE_CHUNK=1` preserves byte-identical pre-cycle behavior. **All 2673 tests pass with default.** Stop-token detection happens at chunk boundaries (overshoot of up to chunk-1 tokens before exit).

### 3. End-to-end via Engine.generate at B=1

| chunk | tok/s p50 (3 iters) | vs chunk=1 (16.37) |
| ---: | ---: | ---: |
| 1 | 16.37 | baseline |
| 32 | 16.55 | +1.1% |

**Reading:** the lift at B=1 single-request is real but small (+0.18 tok/s, ~1σ from baseline noise σ≈0.04 within-run). The much-smaller magnitude vs the +7% raw-forward result is consistent with: at B=1 the per-step compute is already dominant (one weight stream produces 1 token), so saving Python<->Metal sync overhead is a smaller fraction of total step time than at B=4 where the compute amortises across rows.

### 4. End-to-end via warm-decode-b1 oracle

| SILICA_DECODE_CHUNK | tok/s | status | reason |
| ---: | ---: | --- | --- |
| 1 | 16.9 | **ok** | clean warm-decode oracle pass |
| 2 | 17.1 | failed | warmup_did_not_stabilize:rel_std exceeds 5% |
| 4 | 17.0 | failed | warmup_did_not_stabilize |
| 8 | 16.9 | failed | warmup_did_not_stabilize |
| 16 | 16.9 | failed | warmup_did_not_stabilize |
| 32 | 17.1 | failed | warmup_did_not_stabilize |

**Reading:** the throughput improvement (≤+1.2%) is within run-to-run noise (σ≈0.21 tok/s, 3σ floor for keep = 0.63 tok/s). AND every chunked variant **fails the per-step rate-stability check** the warm-decode oracle imposes (rel_std > 5% by design — chunked path produces 0-ms intervals during the lazy phase, then a big spike at the chunk-boundary `mx.eval`). The oracle reads this as "decode rate did not stabilise after warmup" and flips status to `failed`.

This is by design on both sides: the oracle's stability check is correct for catching real regressions where decode slows mid-run; the chunked path is correct in batching syncs to amortise overhead. The two are simply incompatible without an oracle adaptation that accepts chunk-aggregated stability.

## Aggregated finding across cycles 1-4

The autoresearch loop has now empirically tested every single-session-tractable lever family on dense Qwen3.5-27B-4bit B=4:

1. **Pointwise-fusion custom kernels** (cycle 1-2): `fused_gated_output` (1.006× = flat), `fused_silu_mul` (0.755×, slower), `fused_qk_norm` (0.751×, slower). Honest negative — `mx.compile` already fuses pointwise ops.
2. **Quantised matmul custom kernel** (cycle 3): naive 1-thread/output 0.694× (1.44× slower), SIMD-cooperative 0.241× (4× slower). Honest negative — beating `mx.quantized_matmul` requires `simdgroup_matrix` MMA primitives (multi-day work).
3. **mx.compile-wrap larger ops** (cycles 2-3): full SwiGLU MLP shows no speedup; cannot fuse across `mx.quantized_matmul`.
4. **Layer-skip self-spec on off-the-shelf checkpoint** (cycle 3): max realistic speedup 1.13× = 47.7 tok/s. Off-the-shelf models lack early-exit training; agreement at cheap-skip levels too low.
5. **Execution-pattern lazy-chain decode** (cycle 4): real +7% on raw forward_batched B=4, but doesn't translate through production warm-decode oracle (chunk-stability incompatibility) AND requires ContinuousBatcher integration for B=4 production path.

**No experiment in cycles 1-4 has produced a kept improvement on the running-best `qwen3.5-27b-warm-decode-b4` frame.** Running best stays at 42.17 tok/s.

The composed envelope of 67-100 tok/s remains valid (per `plans/P6_AUTORESEARCH_NOT_LIMIT_PROOF.md`) but reaching it needs one of:

- **Multi-day kernel engineering**: simdgroup_matrix-tuned QMM (3-7d), fused FA-2 with output gate (5-14d), or fused gated_delta_update + conv1d (3-7d).
- **ContinuousBatcher integration of cycle-4's lazy-chain pattern** (3-5d, given ContinuousBatcher's admission / stop-token / streaming complexity) PLUS a warm-decode oracle adaptation accepting chunk-aggregated stability.
- **A new checkpoint** with early-exit training, MTP weights preserved, smaller group_size, or activation-aware 3-bit quantisation. Multi-week + downloads + conversion.
- **C.5 γ.1 read-only kernel survey** (1d) — still pending user decision since 2026-05-02.

## Files added in cycle 4

- `silica/engine/__init__.py` — `SILICA_DECODE_CHUNK` env var wired into `Engine.__init__` and `_drive` spec-off branch; default 1 preserves existing behavior; all 2673 tests pass.
- `scripts/microbench_kernel_e2e.py` — extended with `--decode-pattern` and `--chunk-size` arguments.
- `plans/P6_AUTORESEARCH/e2e_pattern_baseline.jsonl`
- `plans/P6_AUTORESEARCH/e2e_pattern_lazy_full.jsonl`
- `plans/P6_AUTORESEARCH/e2e_pattern_lazy_chunk{8,16,32}.jsonl`
- `plans/P6_AUTORESEARCH/REPORT_KERNEL_CYCLE_4.md` (this file)

## Ledger rows appended this cycle

- `AR_LAZY_DECODE_RAW_B4` — diagnostic; +7.3% at B=4 raw forward_batched; production-applicable via ContinuousBatcher integration (multi-day work).
- `AR_ENGINE_DECODE_CHUNK` — diagnostic; engine integration landed; all tests pass with default chunk=1.
- `AR_E2E_BENCH_B1_CHUNKED` — discard; warm-decode-b1 oracle bench with chunk=2..32 returns +0-0.2 tok/s (within run-to-run noise) AND fails per-step stability check.

## Running best on `decode_tok_s`

Unchanged at **42.17 tok/s**. Cycle 4 produced the first signal of a real lever (lazy-chain pattern, +7% raw) but no kept improvement on the running-best frame.

## Honest recommendation for next cycle

The session-tractable lever set is empirically exhausted on the dense 27B B=4 path. The next iteration legitimately needs **one of three multi-day or external-authorization paths**:

1. **ContinuousBatcher chunked-decode integration** + warm-decode oracle adaptation (3-5d). Lifts the cycle-4 lever to the production B=4 path; would unlock +5-7 tok/s = ~47-49 tok/s. Requires user approval for the oracle change ("changing gates or acceptance criteria" per P6_AUTORESEARCH.md).
2. **Custom simdgroup_matrix QMM kernel** (3-7d). Would unlock the bandwidth-utilisation lever from 52% toward the demonstrated 82.7%; estimated +5-10 tok/s lift = ~47-52 tok/s.
3. **C.5 γ.1 read-only kernel survey** (1d). The C.5 escalate state's only remaining branch — still pending since 2026-05-02. Could re-open the spec lever if the upstream `humanrouter/ddtree-mlx` ships a no-torch tree-attention kernel with sub-linear T=32 verify cost.

Continuing in-session will empirically reproduce the same null results — every realistic single-session lever has been tested.
