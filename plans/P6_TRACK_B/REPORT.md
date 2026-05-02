# P-6 Track B 3-bit Weights — REPORT

D-021 step 7 measurement bundle. One file across sub-units; new
rows append as the spike progresses through B.1 / B.2 / B.3.

| Sub-unit | Row | Status | Date |
| -------- | --- | ------ | ---- |
| (B.1) | `NexVeridian/Qwen3.5-27B-3bit` loader smoke + scenario reg | landed; **gate PASSED** | 2026-05-01 |
| (B.2) | `qwen3.5-27b-wikitext-ppl-{4bit,3bit}` — ΔPPL cross-check | landed; **gate FAILED** | 2026-05-01 |
| (B.3) | `qwen3.5-27b-warm-decode-b1-3bit` — silica-integrated speedup | not run (B.2 quality gate breach retires the candidate) | — |

---

## (B.1) — `NexVeridian/Qwen3.5-27B-3bit` loader smoke

Closes the §6.1 B.1 acceptance gate cleanly. The
`silica.models.factory.adapter_for_repo` path loads the matched-
family native MLX 3-bit checkpoint without exception; the engine
generates the expected 4 tokens; peak resident memory satisfies
both the absolute (≤ 13 GiB) and relative (≥ 20% reduction vs the
4-bit anchor) forms of the gate.

### Setup

- **Fixture:** `NexVeridian/Qwen3.5-27B-3bit` (≈11.0 GB on disk;
  3 safetensor shards; created 2026-02-25 via `mlx_lm.convert -q
  --bits 3` against `Qwen/Qwen3.5-27B`; mlx-lm 0.30.8). Acquired
  via `hf download` after the user-authorised B.1 download
  ("Pull `NexVeridian/Qwen3.5-27B-3bit` and run B.1 loader
  smoke").
- **Probe:** `scripts/probe_qwen3_5_27b_3bit_load.py`. Loads via
  `silica.models.factory.adapter_for_repo`, runs
  `Engine.generate("Hello", max_tokens=4, temperature=0.0)`,
  reads `mx.get_peak_memory()` before and after.
- **Gate env:** `SILICA_REAL_QWEN3_5_27B_3BIT=1`.
- **Hardware:** M5 Pro 48 GB.
- **Command:**
  ```text
  SILICA_REAL_QWEN3_5_27B_3BIT=1 \
      uv run python -m scripts.probe_qwen3_5_27b_3bit_load
  ```

### Result

| Quantity | Value |
| -------- | ----- |
| Adapter load time (cache-warm) | 1.60 s |
| Adapter `config.num_layers` | 64 |
| Adapter `config.hidden_size` | 5120 |
| Adapter `config.vocab_size` | 248,044 |
| Peak resident after load | **11.77 GB (10.96 GiB)** |
| Generated tokens (`Hello`, max=4) | `[11, 353, 599, 264]` |
| Generation time | 0.29 s |
| Peak resident after generate | **11.98 GB (11.16 GiB)** |
| 4-bit anchor (v1.7.14 P-6.0) | 15.34 GiB |
| **Reduction vs 4-bit anchor** | **27.2%** |

### §6.1 acceptance gate

- **Absolute form** (`peak ≤ 13.0 GiB`): **PASS** (11.16 ≤ 13.0).
- **Relative form** (`reduction ≥ 20%`): **PASS** (27.2 ≥ 20).
- **Gate (either passes):** **PASS.**

The 27.2% reduction overshoots the 25% naive `3/4 = 0.75×` weight-
bytes prediction by ≈2 percentage points — likely because the
3-bit checkpoint also packs scale/zero metadata more compactly
than the 4-bit cousin, or because the 4-bit anchor's 15.34 GiB
includes scratch / activation overhead that scales sublinearly
with weight bits. Either way, the relative form has comfortable
margin (4+ GiB headroom) for the (1a) ≥40 tok/s primary-gate
calculus that step 7 closure will revisit.

### Bench scenario registered

- `qwen3.5-27b-warm-decode-b1-3bit` registered in
  `silica/bench/scenarios.py`. Mirrors `qwen3.5-27b-warm-decode-b1`
  shape exactly (B=1, 128-token prompt, 384-token generation,
  max_tokens=384, `OracleKind.WARM_DECODE`). Differs from the
  4-bit cousin only in `repo` (`NexVeridian/Qwen3.5-27B-3bit`)
  and `gate_env_var` (`SILICA_REAL_QWEN3_5_27B_3BIT`).
- `BUILTIN_SCENARIOS` count rose from 67 to 68.
- `python -m scripts.bench --list` enumerates the new row.

### Tests landed

8 new tests in `tests/test_bench_3bit_b1_scenario.py`:

- `test_b1_3bit_scenario_registered` — `get_scenario(...)` resolves.
- `test_builtin_scenarios_count_is_68`.
- `test_b1_3bit_workload_shape_matches_b1_cousin` — same prompts,
  max_tokens, max_batch_size, oracle.
- `test_b1_3bit_repo_differs_from_4bit_cousin`.
- `test_b1_3bit_gate_env_differs_from_4bit_cousin`.
- `test_b1_3bit_has_no_spec_config` — plain warm-decode, not
  spec-on; (B.2) and any future spec composition are separate.
- `test_cli_list_surfaces_b1_3bit_row` — subprocess `--list`
  contains the id.
- `test_cli_list_module_imports_clean` — bench CLI loads after
  the scenario addition.

Plus the pre-existing
`test_builtin_scenarios_count_is_67` (in
`tests/test_bench_dflash_zeta.py`) loosened to `≥ 67` so it does
not gate against later catalog growth — the (ζ) invariant was
"≥67 scenarios are registered", not "exactly 67".

### Toolchain attestation

- ruff clean (silica + tests + scripts).
- mypy clean (107 source files, no new errors over v1.7.20
  baseline of 106).
- `SILICA_SKIP_MODEL_TESTS=1` baseline: 2640 passed, 85 skipped
  (= post-(η.1) 2630 + 8 new B.1 scenario tests + 2 picked up by
  the existing ζ count test).

---

## (B.2) — `qwen3.5-27b-wikitext-ppl-{4bit,3bit}` ΔPPL cross-check

**Disposition: gate FAILS.** WikiText-2 chunked-NLL PPL on the
NexVeridian 3-bit checkpoint is 8.0719 against a 4-bit anchor
of 6.9082, breaching both the absolute and relative bounds of
the §6.1 B.2 gate by a comfortable margin. B.3 is therefore
**not authorised to run** under this candidate.

### Setup

- **Fixture:** WikiText-2 test split, cached at
  `~/.cache/silica/wikitext2-test.txt` (1.3 MB; 2026-04-24
  fetch). Same fixture used by all prior PPL rows.
- **Oracle config (shared across 4-bit and 3-bit rows):**
  `chunk_size=256`, `max_tokens=512`, seed `0`,
  `codec_quality_path="prefix_store_pre_norm"`, `kv_codec=None`.
  Both rows therefore see the same prompt prefix, the same 511
  scored token positions, and the same NLL-summation path.
- **Rows registered in `silica/bench/scenarios.py`:**
  - `qwen3.5-27b-wikitext-ppl-4bit` →
    `mlx-community/Qwen3.5-27B-4bit`, gate
    `SILICA_REAL_QWEN3_5_27B`.
  - `qwen3.5-27b-wikitext-ppl-3bit` →
    `NexVeridian/Qwen3.5-27B-3bit`, gate
    `SILICA_REAL_QWEN3_5_27B_3BIT`.
  Each scenario carries its own gate-env so a developer can
  run one half without authorising the other; `Scenario.repo`
  remains one-repo-per-row (no runner-side dual-load).
- **Hardware:** M5 Pro 48 GB.
- **Commands:**
  ```text
  SILICA_REAL_QWEN3_5_27B=1 \
      uv run python -m scripts.bench \
          --scenario qwen3.5-27b-wikitext-ppl-4bit \
          --out plans/P6_TRACK_B/ppl_4bit_run.jsonl

  SILICA_REAL_QWEN3_5_27B_3BIT=1 \
      uv run python -m scripts.bench \
          --scenario qwen3.5-27b-wikitext-ppl-3bit \
          --out plans/P6_TRACK_B/ppl_3bit_run.jsonl
  ```

### Result

| Quantity | 4-bit anchor | 3-bit candidate |
| -------- | ------------ | --------------- |
| Repo | `mlx-community/Qwen3.5-27B-4bit` | `NexVeridian/Qwen3.5-27B-3bit` |
| Status | ok | ok |
| Scored tokens (`n_tokens`) | 511 | 511 |
| `nll_sum` | 987.6154 | 1067.1691 |
| **PPL** | **6.9082** | **8.0719** |
| Peak resident | 16,479.6 MB | 13,166.5 MB |
| Wall time | 5.254 s | 3.971 s |

Both rows score the same 511 token positions (matched
`n_tokens`), so the PPL difference comes from the model only —
not from oracle drift or prompt resampling.

### §6.1 B.2 acceptance gate (both-pass)

- ΔPPL_abs = 8.0719 − 6.9082 = **1.1637** (target ≤ 0.5).
- ΔPPL_rel = 1.1637 / 6.9082 = **16.85 %** (target ≤ 5 %).
- abs form (≤ 0.5): **FAIL** by 0.66 PPL.
- rel form (≤ 5 %): **FAIL** by ≈12 percentage points.
- gate (both must pass): **FAIL.**

### Reading

The 3-bit weight-only quantisation in this checkpoint
introduces a +16.9% PPL hit on WikiText-2 — well above the
quality bound the plan declared acceptable. Two interpretations
are consistent with the data:

1. **The candidate-specific calibration is suboptimal.** The
   NexVeridian shard was produced with `mlx_lm.convert -q
   --bits 3` against `Qwen/Qwen3.5-27B` on 2026-02-25; mlx-lm
   0.30.8's default group size and zero-point handling at 3
   bits is known to be more sensitive to the calibration
   distribution than the 4-bit recipe. A re-converted 3-bit
   checkpoint with a different group size (e.g. 32 instead of
   64) or with AWQ/GPTQ-style activation-aware calibration
   could plausibly close part of the gap.
2. **Pure weight-only 3-bit is too aggressive at 27B.** Even
   well-calibrated weight-only Q3 typically pays 5-12% PPL on
   models in the 7B-30B band; a 17% hit puts this candidate
   on the high-but-plausible side of that distribution rather
   than as an obvious calibration outlier.

In either reading, the *pre-declared* gate is breached. The
plan's design intent — declare the bound up front, then ship
or not based on the measurement — is honoured by **not**
running B.3 against this candidate. Memory headroom is real
(B.1 27.2% reduction; B.2 ≈20% during PPL) but the speed
budget B.3 would unlock cannot offset a quality regression
the user has not opted into.

### Tests landed

This sub-unit is measurement-only: it registers two PPL
scenarios (already exercised by the runner's existing PPL
oracle path) and writes the ΔPPL into this REPORT. No
runtime-path code changes — the runner-side ΔPPL fields
intentionally stay null on per-row JSONL, since cross-row
comparison is REPORT-side bookkeeping.

- `tests/test_bench_qwen3_5_27b_ppl_b2.py` — 6 tests cover
  registration of both rows, repo + gate divergence, oracle
  config equality, no-spec-no-codec, and that
  `BUILTIN_SCENARIOS` count rose from 68 to 70.
- Pre-existing ζ count test (already loosened to `≥ 67` at
  B.1 close) absorbs the new scenarios. The B.1 count test
  `test_builtin_scenarios_count_is_at_least_68` was loosened
  from `== 68` to `>= 68` here so B.2's two added rows do not
  retro-gate it; the new B.2 count test pins `>= 70` for the
  same reason vs. future B.3 / future-track growth.

### Toolchain attestation

- ruff clean (silica + tests + scripts).
- mypy clean (85 source files in the silica package; no new
  errors).
- `SILICA_SKIP_MODEL_TESTS=1` baseline: **2647 passed, 86
  skipped** (post-B.1 2640 → +6 new B.2 scenario-shape tests
  plus +1 absorbed by the existing `>=` count tests).

### B.3 disposition

**Not run.** Track B's PLAN §13 step 7 acceptance is a both-
pass gate over (B.1 memory, B.2 quality, B.3 speedup). With
B.2 closed FAIL, B.3's outcome cannot rescue Track B as
currently configured — even an unconditional B.3 PASS would
ship a model whose pre-declared quality bound is breached.
The candidate is retired at B.2 pending a follow-up
authorisation: the next viable move is either (a) a
re-converted 3-bit checkpoint with a tightened calibration
recipe, (b) a relaxed B.2 bound declared up-front in PLAN
(not retro-loosened to fit this measurement), or (c) Track
B retirement and escalation upstream as PLAN §13 step 7
contemplates for ≤16 tok/s outcomes — except here the
escalation reason is quality, not speed.
