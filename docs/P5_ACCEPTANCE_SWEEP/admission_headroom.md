# P-5 Acceptance (3) — Admission headroom evidence

**Gate (PLAN.md §7 P-5 Acceptance item 3):** "With BlockTQ on, the
same memory budget admits more requests (quantitatively verifies
Principle 8)." Specific rule (`docs/P5_OPENING.md` §7(c)): under a
shared-prefix warmup to `store.resident_bytes() >=
cap_bytes * warmup_ratio`, count the consecutive `AdmitDecision`s
against each codec's residency; strict inequality `n_block > n_fp16`
is the close criterion.

**Run date:** 2026-04-24 (commit `ec8ed27` base, v1.7.3).
**Scenario:** `qwen3-0.6b-admission-headroom-prefix-heavy`
(`silica/bench/scenarios.py`, `OracleKind.ADMISSION_HEADROOM`).
**Raw JSONL:** `docs/P5_ACCEPTANCE_SWEEP/admission_headroom.jsonl`
(3 rows, one per seed).

## Command

```bash
uv run python scripts/bench.py \
    --scenario qwen3-0.6b-admission-headroom-prefix-heavy \
    --seeds 42,43,44 \
    --out docs/P5_ACCEPTANCE_SWEEP/admission_headroom.jsonl
```

Report emitted by the runner (GFM table; row 4 of the runner's own
summary table):

```text
| id                                         | codec | seed | status | wall_s | tokens |
|--------------------------------------------|-------|------|--------|--------|--------|
| qwen3-0.6b-admission-headroom-prefix-heavy |       |   42 |  ok    |  0.632 |     11 |
| qwen3-0.6b-admission-headroom-prefix-heavy |       |   43 |  ok    |  0.576 |     11 |
| qwen3-0.6b-admission-headroom-prefix-heavy |       |   44 |  ok    |  0.485 |     11 |
```

`codec` is empty because the workload declares `kv_codec=None`; the
oracle consumes `fp16_codec` / `compressed_codec` directly from
`oracle_config` (`fp16_codec="fp16"`, `compressed_codec="block_tq_b64_b4"`).

## Per-seed numbers

All three seeds produce byte-identical admission counts and
residency numbers. This is structural, not a seed artefact: the
warmup phase writes a deterministic `n_prompt=128` shared-prefix
block recipe up to `cap_bytes * warmup_ratio`, then the oracle
replays the **same** block recipe under each codec. Haar-rotation
seed variance inside `BlockTurboQuantMSE` affects reconstructed
values but not residency bytes per block or per-request
`worst_case_bytes`, so the admission-count comparison is
codec-residency-driven and seed-independent by construction.

| field | value |
| --- | --- |
| `cap_bytes` | `134_217_728` (128 MB) |
| `weights_bytes` | `0` (isolates the signal on pure prefix residency) |
| `warmup_ratio` | `0.5` |
| `warmup_blocks` | `37` (blocks inserted to reach `resident_bytes_fp16 >= 64 MB`) |
| `n_prompt` | `128` |
| `max_tokens` | `16` |
| `fp16_codec` | `fp16` (IdentityCodec baseline, §4.7 mode B) |
| `compressed_codec` | `block_tq_b64_b4` (BlockTurboQuantMSE B=64 4-bit, §4.7 mode C) |
| `resident_bytes_fp16` | `67_895_296` (~67.895 MB) |
| `resident_bytes_block` | `18_034_688` (~18.035 MB) |
| `residency_ratio` | `0.266` (`resident_bytes_block / resident_bytes_fp16`) |
| `n_fp16` | `4` (consecutive `AdmitDecision`s under IdentityCodec) |
| `n_block` | `7` (consecutive `AdmitDecision`s under BlockTQ) |
| `n_delta` | `+3` |
| `admit_ratio` | `1.75` (`n_block / n_fp16`) |

**Seeds 42 / 43 / 44 all report the above values bit-identically**
(raw JSONL confirms). 3-seed mean / std is therefore degenerate
(std = 0); a seed sweep is still recorded to match the other P-5
acceptance rows' seed surface and so a future non-determinism
regression shows up as a row-level divergence.

## Gate check

- **Hard inequality.** `n_block > n_fp16` → `7 > 4` — **pass** on
  every seed.
- **Strict-greater-by-at-least-one margin.** `n_delta >= 1` →
  `n_delta = 3` — pass with margin.
- **Residency compression sanity.** `residency_ratio ≈ 0.266`
  (`≈ 1 / 3.76`) matches vqbench REPORT §3.1's headline BlockTQ
  B=64 4-bit K+V total-KV compression, so the admission gain is
  being earned by honest compressed residency rather than by a
  byte-accounting artefact.
- **Theoretical upper bound reference** (§7(c)):
  `N_block ≈ N_fp16 × (1 + compression_factor × prefix_fraction)`
  with `compression_factor ≈ 3.76`, `prefix_fraction ≈ 0.5`
  (warmup\_ratio), gives a theoretical ceiling of `N_fp16 × 2.88`.
  Observed `admit_ratio = 1.75` sits below the ceiling —
  consistent with the §7(c) rationale ("the `reserved_bytes` still
  charges fp16 worst-case for every admitted request, so the gain
  is bounded by how much of `cap_bytes` is prefix-cache vs
  active-reservation"). The gate is the strict-inequality floor,
  not the ceiling.

## Scope delimiters

- **Mode (B) vs mode (C) only.** The bench row runs with
  `account_prefix_residency=True` on both arms; §4.7 mode (A)
  byte-for-byte regression is a separate regression lock
  (`tests/test_kvcodec_integration.py` / `test_memory_budgeter.py`),
  not part of this row.
- **Weights-bytes isolation.** `weights_bytes = 0` keeps the
  comparison on pure prefix-cache residency. A real deployment
  also pays for weight residency; §7(c) notes this and the
  admission gain is smaller when `weights_bytes` consumes most of
  `cap_bytes`. Running the same row with a realistic weight
  budget is a tuning exercise, not a close-gate requirement.
- **Synthetic prefix recipe.** The shared-prefix cohort is
  deterministic synthetic K/V, not real Qwen3-0.6B activations.
  Real-activation variants live in the PPL rows, not this row —
  the admission gate is a byte-budget structural invariant and is
  content-independent by design.

## Conclusion

**Acceptance (3) is empirically satisfied.** Under a `cap_bytes =
128 MB` budget with `warmup_ratio = 0.5`, BlockTurboQuantMSE
`B=64` 4-bit K+V frees ~3.76× of prefix-cache residency compared
to IdentityCodec fp16, and the memory budgeter admits `+3` more
trial requests (`n_fp16 = 4` → `n_block = 7`,
`admit_ratio = 1.75`). The result is reproducible and seed-
independent by construction.

Ready to flip PLAN.md §7 P-5 Acceptance item (3) `[ ]` → `[x]` at
v1.7.4.
