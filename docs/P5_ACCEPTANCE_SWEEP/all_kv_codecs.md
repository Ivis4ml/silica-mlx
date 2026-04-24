# P-5 Acceptance (2) — one-command fp16-vs-codec report

**Gate (PLAN.md §7 P-5 Acceptance item 2):** "For the same scenario
set, fp16 vs codec quality delta and memory savings are available
from the bench in one command." Per `docs/P5_OPENING.md` §7(e)
(narrowed at v1.7.4): the command must produce a coherent
per-(scenario, codec, seed) report covering `status`, quality
signal (`delta_ppl` / `delta_ppl_pct` on PPL rows, `admit_ratio` /
`n_delta` on admission-headroom rows, paired decode tok/s on
prefix-hit-decode rows), memory signal (`resident_bytes` /
`resident_bytes_per_block` on compression-row metadata, `peak_memory_mb`
top-level on every ok row), decode signal (`decode_tok_s` top-level
on prefix-hit-decode rows and `row1_decode_tok_s` metadata for the
paired 0.85× gate), and the `vqbench_gap` xcheck column. The xcheck
column is **structurally present on every row** (report schema is
uniform) but **populated only when two conditions co-occur**:
(i) the scenario declares a `VqbenchXcheckSpec` — currently the
`qwen3-0.6b-wikitext-ppl-block-tq-b64-b4-vqbench-aligned` row only,
per the P-5-C.6 declarative-spec contract — and (ii)
`--vqbench-xcheck` is passed to the runner so
`BenchRunner.vqbench_xcheck_enabled=True`. Populated xcheck values
belong to Acceptance (4-b) and live in
`docs/P5_D2_INVESTIGATION/d2a_verification_3seeds.jsonl`, not in
the `--all --all-kv-codecs` sweep that (2) verifies. This sweep
deliberately did **not** pass `--vqbench-xcheck`, so `vqbench_gap`
is empty across all 924 rows and (2) binds to the report-schema
coverage, not to numerical agreement.

**Run date:** 2026-04-24 (commit `ec8ed27` base, v1.7.3).

## Command

```bash
uv run python scripts/bench.py \
    --all --all-kv-codecs --seeds 42,43,44 \
    --out docs/P5_ACCEPTANCE_SWEEP/all_kv_codecs.jsonl \
    --report-md docs/P5_ACCEPTANCE_SWEEP/all_kv_codecs_report.md \
    2>&1 | tee docs/P5_ACCEPTANCE_SWEEP/all_kv_codecs.log
```

Exit code: `0`. No uncaught exceptions in the runner; all failed
rows reached the bench reporter with a structured `status=failed`
+ `reason=<classifier>:<detail>` row.

## Output files

- **`all_kv_codecs.jsonl`** — 924 rows (one per scenario × codec ×
  seed). Each row has full `metadata`: `codec_id`, `seed`, plus
  oracle-specific fields — `delta_ppl` / `delta_ppl_pct` on PPL
  rows, `resident_bytes` / `resident_bytes_per_block` on
  compression rows, `row0_decode_tok_s` / `row1_decode_tok_s` on
  prefix-hit-decode rows. `vqbench_delta_ppl` /
  `vqbench_delta_ppl_gap` / `vqbench_divergence_warning` are
  **absent** from every row in this sweep — `--vqbench-xcheck`
  was deliberately not passed, so the vqbench subprocess path was
  skipped for all rows including the `-vqbench-aligned` one.
  Populated xcheck numbers live in
  `docs/P5_D2_INVESTIGATION/d2a_verification_3seeds.jsonl` under
  Acceptance (4-b).
- **`all_kv_codecs_report.md`** — GFM Markdown report with
  per-(scenario, codec) aggregation header (`runs / ok / skipped /
  failed` counts + median timings) and per-seed detail rows. Columns:
  `{id, codec, runs, ok, skipped, failed, ttft_ms, decode_tok_s,
  resident_mb, peak_mb, wall_s, tokens, vqbench_gap}`.
- **`all_kv_codecs.log`** — the runner's own per-row table
  progression stream (tee'd during the sweep). Useful for auditing
  reason-code classification without replaying the JSONL.

## Coverage

- **Scenarios swept:** 28 distinct ids
  (22 Qwen3-0.6B + 1 Qwen3.5-0.8B + 2 Gemma4-31B + 1 Gemma4-MoE +
  1 Qwen3.5-27B + 1 Qwen3.5-MoE, per `--all`).
- **Codecs swept:** 11 — `fp16`, `tq_mse_b3`, `tq_mse_b4`,
  `block_tq_b32_b3`, `block_tq_b32_b4`, `block_tq_b64_b3`,
  `block_tq_b64_b4`, `rabitq_b1`, `ext_rabitq_b2`, `ext_rabitq_b3`,
  `ext_rabitq_b4` (the full `CODEC_REGISTRY`).
- **Seeds swept:** 3 — `{42, 43, 44}`.
- **Total rows:** `28 × 11 × 3 = 924`. All 924 rows present in
  JSONL and report.

## Status summary

| status    | count |
| ---       | ---:  |
| `ok`      |   360 |
| `skipped` |     0 |
| `failed`  |   564 |
| **total** |   924 |

## Failed-row classification — all 564 failures are expected compatibility misses

Three disjoint failure classes, each explainable from the scenario
/ codec / workload contract without any runner or report bug:

### Class A — `codec_override_invalid` (528 rows)

**Cause.** `--all-kv-codecs` overrides `workload.kv_codec` with
each registry codec, but the target workload has
`prefix_cache=False`; codecs install on the prefix cache's
store, so there is nothing to install on. The runner surfaces this
via the Workload guard in `silica/core/request.py`.

Affected scenario classes (workload-level `prefix_cache=False`):

- **PPL fp16 baseline** (`qwen3-0.6b-wikitext-ppl-fp16`) — the PPL
  oracle bypasses `engine.generate_batch` entirely and drives the
  adapter through `silica.bench.ppl_oracle`, so its workload does
  not need `prefix_cache=True`. The codec-pinned PPL rows
  (`-tq-mse-b4`, `-block-tq-b64-b4`, `-block-tq-b64-b4-vqbench-aligned`,
  `-ext-rabitq-b4`) **do** declare `prefix_cache=True` and DO run
  under the sweep — they carry the actual ΔPPL signal.
- **Admission-headroom** (`qwen3-0.6b-admission-headroom-prefix-heavy`)
  — the oracle constructs the comparison codecs internally via
  `oracle_config["fp16_codec"]` / `["compressed_codec"]`, so
  workload-level `kv_codec` is intentionally `None` and
  `prefix_cache=False`. Coverage for this row is in the
  dedicated (3) evidence run
  (`docs/P5_ACCEPTANCE_SWEEP/admission_headroom.{jsonl,md}`),
  not this sweep.
- **Smoke / parity / routing rows** (`qwen3-0.6b-{smoke,
  b1-parity, bgt1-parity, short-in-long-out, long-in-short-out,
  teacher-forced-argmax, ttft-under-concurrency}`) — these rows
  verify engine / scheduler / adapter contracts with
  `prefix_cache=False`, so they have no codec slot by design.
- **All big-model smoke / parity rows** (`gemma4-31b-*`,
  `gemma4-moe-smoke`, `qwen3.5-27b-smoke`, `qwen3.5-moe-smoke`,
  `qwen3.5-0.8b-b1-parity`) — all `prefix_cache=False`; several
  additionally sit behind `gate_env_var` and would not have run
  on a fresh HF cache anyway. The `codec_override_invalid` fires
  before the `gate_env_var` check because the Workload guard is
  parse-time.

Row count: `16 scenarios × 11 codecs × 3 seeds = 528`.

### Class B — `ValueError: kv_codec='rabitq_b1' is not symmetric` (33 rows)

**Cause.** `rabitq_b1` is K-only (`k_supported=True,
v_supported=False` — the estimator-native attention path the
`ip_coeff` field feeds lives on K by construction). The symmetric
`kv_codec=` shorthand installs one codec on both sides, so the
runner rejects the combination at workload resolution.

Affected rows: every Qwen3-0.6B `prefix_cache=True` scenario
crossed with `codec=rabitq_b1` × 3 seeds.

Row count: `11 scenarios × 1 codec × 3 seeds = 33`.

### Class C — `RuntimeError: codec_quality_path='vqbench_aligned' requires a symmetric codec` (3 rows)

**Cause.** The D.3-landed guard on the `-vqbench-aligned` row's
pre-RoPE projection-patch oracle (`silica/bench/runner.py::_run_ppl`
dispatcher) rejects the asymmetric `rabitq_b1` codec — the
projection patch needs a codec that can serve both K and V. This
is a second-layer guard on top of Class B's workload guard,
triggered specifically because `vqbench_aligned` has its own
codec-installation code path.

Affected rows: `qwen3-0.6b-wikitext-ppl-block-tq-b64-b4-vqbench-aligned`
× `rabitq_b1` × 3 seeds.

Row count: `1 scenario × 1 codec × 3 seeds = 3`.

### Cross-check

`528 + 33 + 3 = 564 = total failed`. Every failed row is
accounted for by one of the three compatibility classes above.
**No runner bug, no report bug, no unclassified exceptions.**

## ok-row coverage — 12 scenarios × 10 symmetric codecs × 3 seeds = 360

| scenario | ok rows (all 10 symmetric codecs × 3 seeds) |
| --- | ---: |
| `qwen3-0.6b-compression-fp16` | 30 |
| `qwen3-0.6b-compression-tq-mse-b4` | 30 |
| `qwen3-0.6b-compression-block-tq-b64-b4` | 30 |
| `qwen3-0.6b-compression-ext-rabitq-b4` | 30 |
| `qwen3-0.6b-concurrent-shared-prefix` | 30 |
| `qwen3-0.6b-prefix-hit-decode-fp16` | 30 |
| `qwen3-0.6b-prefix-hit-decode-block-tq-b64-b4` | 30 |
| `qwen3-0.6b-prefix-hit-decode-ext-rabitq-b4` | 30 |
| `qwen3-0.6b-wikitext-ppl-tq-mse-b4` | 30 |
| `qwen3-0.6b-wikitext-ppl-block-tq-b64-b4` | 30 |
| `qwen3-0.6b-wikitext-ppl-block-tq-b64-b4-vqbench-aligned` | 30 |
| `qwen3-0.6b-wikitext-ppl-ext-rabitq-b4` | 30 |

This is the P-5 bench surface the (2) gate actually binds to —
quality (PPL), memory (compression), and decode (prefix-hit
decode) per-codec coverage across the three seeds.

## Column and metadata coverage — schema + oracle-emitted values

(2) is scoped to "the command produces a coherent report";
populated xcheck values are owned by (4-b) / P-5-D.3 on the
dedicated `-vqbench-aligned` 3-seed run, not by this sweep. The
sweep's role is to verify the **report schema** is coherent and
that **every oracle emits the metrics its class is responsible
for**. Top-level runner columns and oracle-specific metadata
together cover the `{quality, memory, decode, xcheck}` gate.

Top-level runner columns on the 360 ok rows:

| column | populated / 360 | scope |
| --- | ---: | --- |
| `wall_s` | 360 | every ok row |
| `peak_memory_mb` | 360 | every ok row |
| `total_tokens` | 360 | every ok row |
| `decode_tok_s` | 90 | prefix-hit-decode rows only (by oracle design) |
| `ttft_ms` | 90 | prefix-hit-decode rows only (by oracle design) |
| `resident_mb` | 0 | not emitted by any P-5 oracle in v1.7.3; memory signal carried in oracle-specific metadata |
| `prefill_tok_s` | 0 | not emitted by any oracle exercised here |

The top-level `decode_tok_s` / `ttft_ms` / `resident_mb` / `prefill_tok_s`
are populated only when the oracle elects to report them. The
sweep does not require each ok row to fill every top-level column;
it requires each oracle's **chosen** metrics to be emitted
coherently. Oracle-specific metadata (JSONL `metadata` field) is
where most of the gate signal lives:

| oracle / scenario class | gate-relevant metadata keys |
| --- | --- |
| PPL (`wikitext-ppl-*`) | `ppl`, `delta_ppl`, `delta_ppl_pct`, `nll_sum`, `n_tokens`, `chunk_size`, `codec_id`, `codec_quality_path` (quality) |
| STORAGE (`compression-*`) | `resident_bytes`, `resident_bytes_per_block`, `block_size`, `live_blocks`, `prefix_cache_hits`, `codec_id` (memory) |
| DECODE_TOK_S_WITH_PREFIX_HIT (`prefix-hit-decode-*`) | `row0_decode_tok_s`, `row1_decode_tok_s`, `row0_first_token_ms`, `row1_first_token_ms`, `prefix_cache_hits`, `codec_id` (decode + per-row pairing for the 0.85× gate) |
| SMOKE variants (`concurrent-shared-prefix`) | `rows`, `total_tokens`, `max_token_id`, `codec_id` (scheduler-side smoke) |

This is the substantive column coverage: memory lives on
`compression-*` rows via `resident_bytes` /
`resident_bytes_per_block`, decode lives on `prefix-hit-decode-*`
rows via `row1_decode_tok_s` against the fp16 arm, and quality
lives on `wikitext-ppl-*` rows via `delta_ppl` / `delta_ppl_pct`.
All three columns are populated on their respective scenario classes
× 10 symmetric codecs × 3 seeds, as the ok-row-coverage table
above shows (30 ok rows per scenario).

## xcheck scope — schema only in this sweep; populated values owned by (4-b)

- **`--vqbench-xcheck` was NOT passed** to this sweep. Without the
  flag, `BenchRunner` skips the vqbench subprocess path entirely:
  `vqbench_*` metadata is null on all 924 rows and the
  `vqbench_gap` column in the report is structurally present but
  empty. Declaring a `VqbenchXcheckSpec` on a scenario is a
  necessary but not sufficient condition for populated xcheck
  values; the runner flag is also required. This matches
  `BenchRunner`'s `vqbench_xcheck_enabled=False` contract
  (`silica/bench/runner.py::BenchRunner.__init__`).
- **Populated xcheck values are owned by Acceptance (4-b).** The
  3-seed verification in `docs/P5_D2_INVESTIGATION/d2a_verification_3seeds.jsonl`
  (produced at P-5-D.2a with `--vqbench-xcheck` on, bound to the
  `-vqbench-aligned` scenario) is what closed (4-b) at v1.7.3.
  That run is the primary source for `vqbench_delta_ppl_gap` /
  `vqbench_divergence_warning` numbers and for the mean-over-seeds
  aggregated gate.
- **Why (2) doesn't rerun with `--vqbench-xcheck` globally.** Per
  §7(e) (as rewritten at v1.7.4), only the `-vqbench-aligned` row
  carries a `VqbenchXcheckSpec`; running `--vqbench-xcheck` under
  `--all-kv-codecs` would either (a) leave `vqbench_*` null on all
  other codec arms (no spec to drive them) or (b) require the
  runner to infer unvalidated method / bits mappings. Neither
  helps (2). (2) therefore proves the report-schema coverage here
  and leaves the populated xcheck numbers to (4-b)'s dedicated
  run.

## Conclusion

**Acceptance (2) is empirically satisfied on the narrowed §7(e)
scope.** `scripts/bench.py --all --all-kv-codecs --seeds 42,43,44
--out <jsonl> --report-md <md>` produces a coherent 924-row report
covering every declared scenario × every codec × 3 seeds in a
single invocation. The report schema is coherent; each oracle
emits the `{quality, memory, decode}` signal its class is
responsible for on its 30 ok rows (PPL rows → `delta_ppl` /
`delta_ppl_pct`; compression rows → `resident_bytes` /
`resident_bytes_per_block`; prefix-hit-decode rows →
`row0_decode_tok_s` / `row1_decode_tok_s`). All 564 failed rows
are classified into three expected compatibility classes (528
`codec_override_invalid`, 33 K-only `rabitq_b1`, 3
vqbench-aligned symmetric-codec guard) that together account for
`528 + 33 + 3 = 564` rows; no runner or report bugs surfaced.
The `vqbench_gap` column is structurally present but empty in
this sweep (`--vqbench-xcheck` deliberately not passed); populated
xcheck numbers are owned by Acceptance (4-b) and were landed at
v1.7.3 / P-5-D.3 via
`docs/P5_D2_INVESTIGATION/d2a_verification_3seeds.jsonl`.

Ready to flip PLAN.md §7 P-5 Acceptance item (2) `[ ]` → `[x]` at
v1.7.4.
