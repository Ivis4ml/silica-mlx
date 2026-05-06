# P-6 small-B sub-unit α — RUNBOOK

Sonnet-side baseline refresh per `plans/P6_SMALL_B_OPENING.md` §4 α.
Diagnostic only. No speed-up expected. Goal is a sonnet-timestamped
B=4 / B=8 / B=12 warm-decode baseline plus per-step bucket
decomposition that informs which sub-unit (β / γ / δ) opens next.

## Pre-flight (run once, not per session)

```
# 1. Toolchain pin
uv run python -c "import mlx, mlx.core; print(mlx.__version__)"
uv run python -c "import mlx_lm; print(mlx_lm.__version__)"
# Expected: mlx 0.31.1, mlx-lm 0.31.2 (per cycle-24 pin in pyproject.toml).

# 2. Determinism gate
uv run --extra bench pytest tests/test_p2_preload_parity.py -q
# Expected: 3/3 passed.

# 3. HF cache for the production target
uv run python -c "from huggingface_hub import snapshot_download; \
    snapshot_download(repo_id='mlx-community/Qwen3.5-27B-4bit')"
# Expected: cache-hit (~15 GB).

# 4. Shadow-flag verification (both default OFF for α)
echo "SILICA_USE_FA_DECODE_V10=${SILICA_USE_FA_DECODE_V10:-unset}"
echo "SILICA_USE_BF16_DELTANET_STATE=${SILICA_USE_BF16_DELTANET_STATE:-unset}"
# Expected: both unset (or "0"). Enabling either is a different baseline.
```

## Per-session run (~25 min wall clock)

Each session runs the same five commands in this order. Repeat the
entire session at least once (≥2 total sessions). Sessions should be
separated in time (different login session, different terminal,
ideally different day) so environmental drift surfaces in σ rather
than hiding within a single hot run.

Session label convention: `SESSION=$(date +%Y%m%d_%H%M%S)` — appended
to artifact paths.

```
SESSION=$(date +%Y%m%d_%H%M%S)
mkdir -p plans/P6_SMALL_B/$SESSION

# Three warm-decode rows. --seeds 0,1,2 = three within-session reps per row.
SILICA_REAL_QWEN3_5_27B=1 \
    uv run --extra bench python -m scripts.bench \
        --scenario qwen3.5-27b-warm-decode-b4 \
        --seeds 0,1,2 \
        --out plans/P6_SMALL_B/$SESSION/warm_decode_b4.jsonl

SILICA_REAL_QWEN3_5_27B=1 \
    uv run --extra bench python -m scripts.bench \
        --scenario qwen3.5-27b-warm-decode-b8 \
        --seeds 0,1,2 \
        --out plans/P6_SMALL_B/$SESSION/warm_decode_b8.jsonl

SILICA_REAL_QWEN3_5_27B=1 \
    uv run --extra bench python -m scripts.bench \
        --scenario qwen3.5-27b-warm-decode-b12 \
        --seeds 0,1,2 \
        --out plans/P6_SMALL_B/$SESSION/warm_decode_b12.jsonl

# Two attribution microbenches. iters=20 / 10 give per-iter median;
# the microbench has its own warmup loop independent of the bench harness.
SILICA_REAL_QWEN3_5_27B=1 \
    uv run python -m silica.bench.microbench.decode_step_attribution \
        --b 4 --warmup 3 --iters 20 \
        --out plans/P6_SMALL_B/$SESSION/decode_step_attr_b4.jsonl

SILICA_REAL_QWEN3_5_27B=1 \
    uv run python -m silica.bench.microbench.layer_internal_attribution \
        --b 4 --warmup 3 --iters 10 \
        --out plans/P6_SMALL_B/$SESSION/layer_internal_attr_b4.jsonl
```

## Optional: B=8 attribution probe

Per `P6_SMALL_B_OPENING.md` §4 α, B=8 attribution is optional. If you
want to compare B=4 vs B=8 step-share to confirm the small-B regime
behaves consistently, add:

```
SILICA_REAL_QWEN3_5_27B=1 \
    uv run python -m silica.bench.microbench.decode_step_attribution \
        --b 8 --warmup 3 --iters 20 \
        --out plans/P6_SMALL_B/$SESSION/decode_step_attr_b8.jsonl
```

## After ≥2 sessions: aggregate and gate-check

```
uv run python plans/P6_SMALL_B/aggregate_variance.py \
    --root plans/P6_SMALL_B \
    --out plans/P6_SMALL_B/REPORT.md
```

The aggregator reads every per-session JSONL, groups rows by
scenario, computes combined mean / σ / n across all sessions, and
emits a Markdown summary plus pass / fail status against the α gate
(combined σ ≤ 1.5 tok/s on each of B=4 / B=8 / B=12 aggregate
throughput).

## Stop conditions (from §4 α)

Declare α complete only if all three hold:

1. ≥2 sessions captured for each of B=4 / B=8 / B=12.
2. Combined σ ≤ 1.5 tok/s on each row's aggregate throughput.
   If σ > 1.5, do not proceed to β / γ / δ — investigate environmental
   drift (wall-clock gap between sessions, background load, thermal
   state, mlx version drift) and document the root cause before
   accepting the baseline.
3. Bucket distribution from `decode_step_attribution_b4.jsonl` is
   readable: per-layer-kind median wall time available for both
   `is_linear=True` (DeltaNet) and `is_linear=False` (full-attention)
   rows, plus instrumented-overhead row.

## What α decides

The bucket distribution from the attribution microbenches gates
β / γ / δ:

| Bucket           | Threshold              | Sub-unit unblocked |
| ---              | ---                    | ---                |
| Full-attention   | ≥ 15% of step time     | β (`mx.compile` graph-trace with cache reroute) |
| Dispatch overhead| ≥ 3% of step time      | δ (`mx.eval` cadence / per-layer loop sync) |
| MLP-attributable | ≥ 5% of step time      | γ (`mx.compile` on `Qwen3NextMLP`) |
| DeltaNet         | ≥ 95% of step time     | **close P-6 small-B line** (no reachable lever; cycle-31 confirmed mlx `gated_delta` at HBM-bandwidth limit, vectorisation = 1.001×) |

If neither full-attn nor overhead clears the threshold and DeltaNet
sits at 75-90%, declare α complete with the conclusion that small-B
on this stack is bandwidth-saturated by recurrent state R/W and
document accordingly.

## Cross-references

- `plans/P6_SMALL_B_OPENING.md` — opening doc with goal, framing,
  non-goals, sub-unit definitions, load-bearing references.
- `plans/P6_AUTORESEARCH.md` — the 35-cycle ledger this line follows
  on from. Cycle-1 B=4 step-share decomposition (DeltaNet 74% /
  full-attn 22% / overhead 4%) is the load-bearing anchor that
  α refreshes on the sonnet branch.
- `plans/P6_AUTORESEARCH_NOTES.md` cycle-30 (B=64) decomposition is
  **not** the small-B anchor; cycle-1 B=4 is.
- `plans/P6_0_5_BASELINE/` — the v1.7.17 baseline this α refreshes;
  same artifact layout convention (per-row JSONL, per-row Markdown
  summary, cross-row `REPORT.md`).
