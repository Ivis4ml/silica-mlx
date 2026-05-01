# Benchmark harness

`scripts/bench.py` is the single entry point that drives every
P-4 / P-5 acceptance scenario, an oracle stack (smoke, B=1 parity,
B>1 direct-reference, teacher-forced argmax, perplexity, storage,
admission headroom), and emits JSONL + Markdown reports plus an
optional vqbench subprocess cross-check column.

The CLI is a thin wrapper around {doc}`api/silica.bench`. Anything
the CLI does is reachable programmatically — see the API page for the
runner / scenario / oracle classes.

## Run the catalog

```bash
# full built-in catalogue, write JSONL + Markdown report
python -m scripts.bench --all \
    --out bench-results.jsonl \
    --report-md bench-results.md

# single scenario by id
python -m scripts.bench --scenario qwen3-0.6b-bgt1-parity

# list registered scenarios
python -m scripts.bench --list
```

Typical on-device output for `--all` without any dual-gate env vars
set (cache-only rows run, dual-gated rows skip):

```
| id | status | reason | ttft_ms | decode_tok_s | resident_mb | peak_mb | wall_s | tokens |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| qwen3-0.6b-smoke | ok |  | 11.6 | 151.8 | 29.4 | 1222.8 | 0.43 | 4 |
| qwen3-0.6b-b1-parity | ok |  | 16.3 | 151.1 | 29.4 | 1252.0 | 0.56 | 4 |
| qwen3-0.6b-bgt1-parity | ok |  |  |  |  | 1279.3 | 0.53 | 16 |
| qwen3-0.6b-short-in-long-out | ok |  | 13.3 | 167.8 | 29.4 | 1222.9 | 0.90 | 64 |
| qwen3-0.6b-long-in-short-out | ok |  | 46.1 | 165.1 | 58.7 | 1815.2 | 0.45 | 4 |
| ... |
| gemma4-31b-smoke | skipped | env_var_not_set:SILICA_REAL_GEMMA4_31B | | ... |
| qwen3.5-moe-smoke | skipped | env_var_not_set:SILICA_REAL_QWEN3_5_MOE | | ... |
```

## Scenario catalog

Run `python -m scripts.bench --list` for the live roster. The catalog
is registered in `silica.bench.scenarios.BUILTIN_SCENARIOS`; the table
below summarises what's currently shipped.

### Cache-only rows

Run on any dev box that has pulled `Qwen/Qwen3-0.6B`.

| id | oracle | shape |
| --- | --- | --- |
| `qwen3-0.6b-smoke` | SMOKE | 1 prompt, max_tokens=4 |
| `qwen3-0.6b-b1-parity` | B1_PARITY_VS_SINGLE | 1 prompt, B=1 |
| `qwen3-0.6b-bgt1-parity` | BGT1_DIRECT_BATCHED_REFERENCE | 2 prompts, B=2, different tokenized lengths |
| `qwen3-0.6b-short-in-long-out` | SMOKE | 1 prompt ("Hello"), max_tokens=64 |
| `qwen3-0.6b-long-in-short-out` | SMOKE | 301-token prompt, max_tokens=4 |
| `qwen3-0.6b-concurrent-shared-prefix` | SMOKE | 4 prompts w/ shared "The capital of" prefix, prefix_cache=True |
| `qwen3-0.6b-ttft-under-concurrency` | SMOKE | 1 long + 3 short prompts, B=4 (Q-010 signal) |
| `qwen3-0.6b-teacher-forced-argmax` | TEACHER_FORCED_ARGMAX | silica vs direct mlx-lm positional argmax, ≥0.98 agreement |
| `qwen3.5-0.8b-b1-parity` | B1_PARITY_VS_SINGLE | Qwen3.5 hybrid DeltaNet, 1 prompt |
| `qwen3-0.6b-warm-decode-b1` | WARM_DECODE | sustained decode_tok_s, B=1, 256-token gen (P-6.0 oracle validation) |
| `qwen3-0.6b-warm-decode-b2` | WARM_DECODE | same, B=2 (validates batched timestamp collector) |
| `qwen3.5-0.8b-warm-decode-b1` | WARM_DECODE | hybrid-DeltaNet warm-decode validation |

### Dual-gated rows

Cache + `SILICA_REAL_<family>=1`.

| id | gate env var | shape |
| --- | --- | --- |
| `qwen3.5-27b-smoke` | `SILICA_REAL_QWEN3_5_27B` | SMOKE, ~16 GB checkpoint |
| `qwen3.5-27b-warm-decode-b1` | `SILICA_REAL_QWEN3_5_27B` | **P-6 dense primary baseline**, 384-token gen |
| `qwen3.5-27b-warm-decode-b1-4k` | `SILICA_REAL_QWEN3_5_27B` | sustained 4K-context probe (P5.9 step 2(d)); ~3500-token prompt + 600 max |
| `qwen3.5-27b-warm-decode-b1-8k` | `SILICA_REAL_QWEN3_5_27B` | sustained 8K-context probe (P5.9 step 2(d)); ~7500-token prompt + 600 max |
| `qwen3.5-moe-smoke` | `SILICA_REAL_QWEN3_5_MOE` | MoE SMOKE, ~20 GB checkpoint, ~30 GB peak |
| `qwen3.5-moe-35b-a3b-warm-decode-b1` | `SILICA_REAL_QWEN3_5_MOE` | MoE B=1 warm-decode |
| `qwen3.5-moe-35b-a3b-warm-decode-b4` | `SILICA_REAL_QWEN3_5_MOE` | **P-6 MoE stretch validator** (≥100 tok/s aggregate gate) |
| `gemma4-31b-smoke` | `SILICA_REAL_GEMMA4_31B` | SMOKE, ~18 GB checkpoint |
| `gemma4-31b-b1-parity` | `SILICA_REAL_GEMMA4_31B` | B=1 parity on dense 31B |
| `gemma4-31b-bgt1-parity` | `SILICA_REAL_GEMMA4_31B` | B=2 parity vs direct mlx-lm |
| `gemma4-31b-warm-decode-b1` | `SILICA_REAL_GEMMA4_31B` | dense Gemma4-31B warm-decode baseline |
| `gemma4-31b-warm-decode-b1-4k` | `SILICA_REAL_GEMMA4_31B` | sustained 4K-context probe (P5.9 step 2(d)); sliding/full hybrid layout |
| `gemma4-31b-warm-decode-b1-8k` | `SILICA_REAL_GEMMA4_31B` | sustained 8K-context probe (P5.9 step 2(d)); sliding-window-capped KV growth |
| `gemma4-moe-smoke` | `SILICA_REAL_GEMMA4_MOE` | MoE SMOKE, ~16 GB checkpoint |
| `gemma4-moe-26b-a4b-warm-decode-b1` | `SILICA_REAL_GEMMA4_MOE` | second-MoE-family warm-decode baseline |

### Speculative-decoding rows (D-021 step 5 sub-unit (h))

Foundation closed at v1.7.19. These rows mirror their non-spec
warm-decode cousins and route through the spec engine when invoked
under `--speculative draft_target`; under the default
`--speculative none` they degrade to plain warm-decode against the
target alone (drafter checks skipped, byte-identical to the b1
baseline).

| id | gates (all four required for spec on) | shape |
| --- | --- | --- |
| `qwen3.5-27b-warm-decode-spec-on` | `SILICA_REAL_QWEN3_5_27B` (target) + `SILICA_REAL_QWEN3_5_0_8B_DRAFT` (drafter) + HF cache hits on both | dense 27B B=1, 384-token gen; drafter `Qwen/Qwen3.5-0.8B`, `verify_k=4` |
| `qwen3.5-moe-35b-a3b-warm-decode-spec-on` | `SILICA_REAL_QWEN3_5_MOE` (target) + `SILICA_REAL_QWEN3_5_0_8B_DRAFT` (drafter) + HF cache hits on both | MoE 35B-A3B B=1, 384-token gen; same drafter / `verify_k` as the dense row |

The two new rows are quad-gated. `_check_gates` skips loud with
`env_var_not_set:` / `draft_env_var_not_set:` / `draft_cache_missing:`
on any missing gate. Sharing the target row's existing strong gate
(`SILICA_REAL_QWEN3_5_27B` / `SILICA_REAL_QWEN3_5_MOE`) preserves
the existing target opt-in; the drafter gate
`SILICA_REAL_QWEN3_5_0_8B_DRAFT` is a separate per-checkpoint
toggle so users who only have the target weights cached do not
trip a drafter download.

`ScenarioResult.metadata` carries the seven
`silica.bench.spec_metrics` fields when the row runs under
`--speculative draft_target`:

| field | meaning |
| --- | --- |
| `accept_rate` | fraction of proposed drafts the target argmax accepted |
| `verify_cost_ms` | mean ms per target-side `decode_step_multi` verify forward |
| `draft_cost_ms` | mean ms per drafter `propose` call (`0.0` for same-model self-spec) |
| `tokens_per_target_forward` | `(yielded_drafts + bonus_tokens) / target_forward_count` — headline speedup observable |
| `rollback_count` | cycles that fired a target-side KV rollback (`un_committed > 0`) |
| `tree_node_visits` | always `0` for trajectory drafters; non-zero only on tree variants (C.5) |
| `quality_parity_status` | `parity` / `diverged` / `not_tested` (set by the harness, default `not_tested`) |

`validate_speculative_metrics` runs at row exit; missing or
invalid fields flip the row to `status="failed"` with the
violation tags appended to `reason`.

#### How to read first spec results

1. Run the baseline first (no spec):

   ```bash
   SILICA_REAL_QWEN3_5_27B=1 python -m scripts.bench \
       --scenario qwen3.5-27b-warm-decode-b1 \
       --out spec-off.jsonl
   ```

2. Run the spec-on row with both gates set:

   ```bash
   SILICA_REAL_QWEN3_5_27B=1 SILICA_REAL_QWEN3_5_0_8B_DRAFT=1 \
       python -m scripts.bench \
       --speculative draft_target \
       --scenario qwen3.5-27b-warm-decode-spec-on \
       --out spec-on.jsonl
   ```

3. Compare `decode_tok_s` between the two JSONL rows; read
   `metadata.accept_rate` and `metadata.tokens_per_target_forward`
   on the spec-on row to attribute any speedup to drafter
   acceptance vs verify-amortisation.

The OPENING-time goal of ≥1.2× decode-throughput vs spec-off is
**tracked, not blocking** at foundation closure (per
`plans/P6_SPEC_FOUNDATION_OPENING.md` §6.2 and Decision Gate 1
v1.7.18); v0.1 spec foundation passes when correctness + metric
schema land, and Track C.4 / C.5 in P-6 carry the throughput
bullet. The (c) slice 3 multi-request hybrid batched-spec path is
deferred — both spec-on rows are B=1 and route through
`Engine.generate`, not `ContinuousBatcher`.

### Warm-decode oracle (P-6.0 measurement gate)

The `WARM_DECODE` oracle measures sustained warm-start
`decode_tok_s` on a long-running generation, with first-forward
kernel-compile latency excluded by a two-stage warm-up rule:

1. Discard at least `warmup_min_steps` decode steps (default 32 —
   covers MLX kernel-compile cost on a 64-layer 27B first forward).
2. Continue discarding until the rolling-window of
   `warmup_rolling_window` (default 16) inter-token interval
   std/mean falls below `warmup_rel_std_threshold` (default 5%).
3. Whichever finishes later defines the warm-up boundary; the
   remaining decodes form the measurement window.

Reported metrics in `ScenarioResult.metadata`:

- `decode_tok_s_warm_aggregate` — total measurement decodes across
  all rows / aggregate measurement-window wall (also promoted to
  `ScenarioResult.decode_tok_s`); directly comparable to
  vllm-mlx-style headline numbers.
- `decode_tok_s_warm_per_row_mean` — average of per-row steady-state
  rates.
- `rows[].decode_tok_s_warm` / `warmup_steps_used` /
  `measurement_steps` / `decode_interval_ms_{mean,std,rel_std}` /
  `cold_ttft_ms` — per-row diagnostic detail.

The oracle reports the measurement; it does not enforce a target.
Phase-level acceptance (`plans/P6_OPENING.md` §6) compares the
reported numbers against the dense-60 / MoE-100 gates.

### KV codec sweep

`--all-kv-codecs` expands every PPL / storage / admission-headroom
row across the codec catalogue (BlockTQ B={32,64} × b={3,4},
RaBitQ-1, ExtRaBitQ B∈{2,3,4}, plus the fp16 IdentityCodec
baseline). Used to produce the P-5 acceptance sweep:

```bash
python -m scripts.bench --all --all-kv-codecs \
    --seeds 42,43,44 \
    --out plans/P5_ACCEPTANCE_SWEEP/all_kv_codecs.jsonl \
    --report-md plans/P5_ACCEPTANCE_SWEEP/all_kv_codecs.md
```

## vqbench cross-check

For the numeric cross-check P-5 Acceptance (4) requires, silica
wraps the existing `vqbench/` reproduce script in a subprocess and
parses the PPL headline row.

```bash
python scripts/vqbench_baseline.py \
    --python-executable /path/to/vqbench/venv/bin/python \
    --out vqbench-baseline.jsonl
```

`--python-executable` is mandatory for a real run — the silica venv
does **not** depend on torch / transformers / datasets (D-009 hot-
path constraint), so the vqbench script must run under its own
Python. Output is one
{class}`silica.bench.VqbenchBaselineResult` per row with `model`,
`method`, `bits`, `ppl_fp16`, `ppl_quant`, `delta_ppl`, `delta_pct`.

The `--vqbench-xcheck` flag on `scripts/bench.py` integrates the
same path inline: every PPL row gains a `vqbench_gap` column and a
diagnostic `vqbench_divergence_warning` boolean.

## Programmatic usage

For tests and ad-hoc reports, drive the runner directly:

```python
from silica.bench import BenchRunner, get_scenario, render_markdown_report

runner = BenchRunner()
results = runner.run(
    [get_scenario("qwen3-0.6b-bgt1-parity")],
    output_path="results.jsonl",
)
print(render_markdown_report(results))
```

See {doc}`api/silica.bench` for the full surface — `BenchRunner`,
`Scenario`, `ScenarioResult`, `OracleKind`, `Workload`, the codec
registry, and the per-oracle helpers under `silica.bench.oracles` /
`silica.bench.ppl_oracle`.
