# P-5 quality regression gate — operator's guide

| Field | Value |
| --- | --- |
| Phase | P-6 (D-021 step 2(g)) |
| Anchor | `qwen3-0.6b-wikitext-ppl-block-tq-b64-b4-vqbench-aligned` |
| First close | v1.7.3 (commit `ed57be1`, 2026-04-24) |
| Decision math | `silica.bench.p5_regression_gate` |
| Pinned snapshot | `silica.bench.p5_regression_gate.SILICA_V1_7_3_SNAPSHOT` |
| Pre-merge gate | `evaluate_silica_regression` (no vqbench dependency) |
| Phase-exit attestation | `evaluate_4b_gate` (full silica + vqbench cross-check) |

This document is the operator's how-to for the P-5 quality
regression gate that every Track A / B / C PR must pass before
merging into the main branch. The decision math + the v1.7.3
pinned reference values live in
`silica.bench.p5_regression_gate`; this file documents *how* to
run the gate and *what* the canonical numbers are.

## 1. Why this gate exists

The (4-b) two-part aggregated gate closed P-5 Acceptance item (4)
at v1.7.3. PLAN.md §7 P-5 records the gate text:

> `|mean_gap| <= 2 * SEM_diff` AND `|mean_gap| < 1.0` PPL,
> where `mean_gap = mean_seeds(silica.ΔPPL_seed − vqbench.ΔPPL_seed)`
> and `SEM_diff = sqrt(std(silica.ΔPPL_seeds)^2 / n + std(vqbench.ΔPPL_seeds)^2 / n)`
> with `n = 3`.

D-021 step 2(g) (v1.7.14) lifts that one-off acceptance event into
an operational pre-merge contract. The gate becomes a routine
check that fires on every Track A / B / C PR, so a regression
that breaks the v1.7.3 PPL footprint (wrong rotation, codec shape
mismatch, missed projection-output capture, etc.) is caught
before it lands rather than during a phase-exit re-attestation.

## 2. Two evaluation modes

### Mode 1 — silica-only regression check (cheap, pre-merge default)

Compares silica's mean ΔPPL across the canonical 3 seeds against
the v1.7.3 pinned snapshot. No vqbench dependency; runs in
~10 seconds on a dev box with the Qwen3-0.6B HF cache pulled.

**Use this for every Track A / B / C PR.**

```bash
python -m scripts.bench \
    --scenario qwen3-0.6b-wikitext-ppl-block-tq-b64-b4-vqbench-aligned \
    --seeds 42,43,44 \
    --out /tmp/p5_regression_check.jsonl
```

Then evaluate the gate:

```python
import json

from silica.bench.p5_regression_gate import (
    SILICA_V1_7_3_SNAPSHOT,
    evaluate_silica_regression,
)

rows = [
    json.loads(line)
    for line in open("/tmp/p5_regression_check.jsonl")
]
silica_seed_delta_ppls = [
    row["metadata"]["delta_ppl"]
    for row in rows
    if row["status"] == "ok"
]
result = evaluate_silica_regression(silica_seed_delta_ppls)
assert result.passes, result.reason
```

The default tolerance is `0.5` PPL — twice the v1.7.3 silica SEM,
chosen to absorb legitimate seed-to-seed variance while still
catching structural regression. Track C variants whose own
convergence story warrants tighter / looser bands can pass an
explicit `tolerance_ppl=` argument.

### Mode 2 — full (4-b) aggregated gate (phase-exit attestation)

Runs both silica's path and vqbench's reproduce-script subprocess
via the existing `--vqbench-xcheck` plumbing. Requires a separate
vqbench venv (silica's runtime carries no torch / transformers /
datasets per D-009). Produces the truthful (4-b) gate result the
phase-exit attestation needs.

```bash
python -m scripts.bench \
    --scenario qwen3-0.6b-wikitext-ppl-block-tq-b64-b4-vqbench-aligned \
    --seeds 42,43,44 \
    --vqbench-xcheck \
    --python-executable /path/to/vqbench/venv/bin/python \
    --vqbench-script /path/to/vqbench/scripts/reproduce_qwen35_0_6b_anchor.py \
    --out /tmp/p5_full_gate.jsonl \
    --report-md /tmp/p5_full_gate.md
```

Then:

```python
import json

from silica.bench.p5_regression_gate import evaluate_4b_gate

rows = [
    json.loads(line) for line in open("/tmp/p5_full_gate.jsonl")
]
silica = [r["metadata"]["delta_ppl"] for r in rows]
vqbench = [r["metadata"]["vqbench_delta_ppl"] for r in rows]
result = evaluate_4b_gate(silica, vqbench)
assert result.passes, result.reason
```

## 3. Pinned reference values

All values from the v1.7.3 close attestation
(`plans/P5_D2_INVESTIGATION/d2a_verification_3seeds.jsonl`):

| Side | mean ΔPPL | std ΔPPL (n-1) | SEM = std/√n |
| --- | --- | --- | --- |
| silica | +0.511 | 0.354 | ~0.204 |
| vqbench | +0.661 | 0.347 | ~0.200 |
| `mean_gap` | -0.150 | — | — |
| `2 * SEM_diff` | — | — | ~0.572 |

Two-part gate at v1.7.3:
- `|mean_gap| = 0.150` ≤ `2 * SEM_diff = 0.572` ✓ (~3.8× headroom)
- `|mean_gap| = 0.150` < `absolute_threshold = 1.0` ✓ (~6.7× headroom)

These values are pinned in
`silica.bench.p5_regression_gate.SILICA_V1_7_3_SNAPSHOT` and
`VQBENCH_V1_7_3_SNAPSHOT`. The
`tests/test_p5_regression_gate.py::test_full_gate_v1_7_3_evidence_reproduces`
test reproduces the recorded `mean_gap = -0.150` and
`aggregate_band ≈ 0.572` from these snapshots.

## 4. When to run which mode

| Situation | Mode | Frequency |
| --- | --- | --- |
| Pre-merge gate on any PR touching `silica.kvcache` / `silica.vq` / `silica.scheduler` / `silica.models.qwen3*` / `silica.bench` | Mode 1 (silica-only) | Every PR |
| Pre-merge gate on Track A / B / C PRs | Mode 1 (silica-only) | Every PR |
| Phase-exit P-6 attestation | Mode 2 (full 4-b) | Once per phase exit |
| New C.x variant landing | Mode 2 (full 4-b) | At C.x close gate |
| Routine dependency bump (mlx-lm version, etc.) | Mode 1 (silica-only) | When the bump lands |

Mode 1 is the routine gate. Mode 2 is the formal attestation —
needed once per phase exit and once per C.x variant close, not
per PR.

## 5. What to do when the gate fails

A `GateResult.reason` of the form `silica_regression_drift:...`
or `full_4b_fail:...` means silica's path has shifted away from
the v1.7.3 footprint. Investigation playbook:

1. **Confirm the drift is structural, not flaky.** Re-run with
   different seeds and confirm the drift is reproducible. If
   only one seed misbehaves, the issue is likely sampling /
   environment variance and not a real regression.
2. **Bisect the offending change.** `git bisect` between the
   landing PR and `bbdb7f7` (the v1.7.14 P-6 contract sync) on
   `qwen3-0.6b-wikitext-ppl-block-tq-b64-b4-vqbench-aligned`. The
   commit that introduces the drift is almost always the one
   that touched `silica.vq` / `silica.kvcache.codec` /
   `silica.kvcache.store` or the prefix-store capture path
   (P-5-F (3b)).
3. **Decide: revert or update the snapshot.** A genuine
   regression in silica's path should be reverted. A *deliberate*
   change that re-anchors the mean — for instance, flipping
   per-head Haar rotation to default-on, which v1.7.10 / v1.7.11
   evidence supports — should be accompanied by a PLAN.md
   Decisions Log entry plus an update to
   `silica.bench.p5_regression_gate.SILICA_V1_7_3_SNAPSHOT` (or
   a new snapshot, e.g. `SILICA_V1_8_0_SNAPSHOT`, with the old
   one preserved for traceability).

## 6. Cross-references

- PLAN.md §7 P-5 Acceptance (4-b): closes via this gate.
- PLAN.md §9 D-021 step 2(g): records this gate as P-6's
  per-track regression contract.
- PLAN.md §13 v1.7.3: original close attestation.
- `silica/bench/p5_regression_gate.py`: decision math + pinned
  snapshots.
- `tests/test_p5_regression_gate.py`: 16 tests pinning the math
  + the v1.7.3 reproduction.
- `plans/P5_D2_INVESTIGATION/d2a_verification_3seeds.jsonl`: raw
  per-seed evidence behind the snapshot.
