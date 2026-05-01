# P-6 Track B 3-bit Weights — REPORT

D-021 step 7 measurement bundle. One file across sub-units; new
rows append as the spike progresses through B.1 / B.2 / B.3.

| Sub-unit | Row | Status | Date |
| -------- | --- | ------ | ---- |
| (B.1) | `NexVeridian/Qwen3.5-27B-3bit` loader smoke + scenario reg | landed; **gate PASSED** | 2026-05-01 |
| (B.2) | `qwen3.5-27b-wikitext-ppl-{4bit,3bit}` — ΔPPL cross-check | pending | — |
| (B.3) | `qwen3.5-27b-warm-decode-b1-3bit` — silica-integrated speedup | pending | — |

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

### B.2 / B.3 deferred

- (B.2) WikiText-2 PPL cross-check requires running the existing
  PPL chunked-NLL oracle against both 4-bit and 3-bit
  checkpoints. Not run at B.1 close — env-affecting, requires
  separate user authorisation.
- (B.3) 27B 3-bit warm-decode attestation runs the new
  `qwen3.5-27b-warm-decode-b1-3bit` scenario under both gate
  envs and reports `decode_tok_s` against the 16.05 b1 anchor.
  Performance gate per PLAN §13 step 7: ≥21 tok/s = 1.31×
  passes; 16 < tok/s < 21 ships opt-in; ≤16 escalates upstream.
