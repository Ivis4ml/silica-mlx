# P-5-C.2 Step 3 — PPL OracleKind + Qwen3-0.6B WikiText Bench Rows

| Field        | Value                                                 |
| ------------ | ----------------------------------------------------- |
| Phase        | P-5-C.2 (codec-backed PPL oracle + bench rows)        |
| Status       | Step 3a landed; step 3b pending                       |
| Depends on   | C.2 step 1 (oracle, commit `e25893a`), step 2 (loader, commit `72c686f`) |
| Maintainer   | Xin Zhou                                              |
| Opened       | 2026-04-23                                            |

---

## 1. Motivation

C.2 step 1 landed the two PPL oracles
(`teacher_forced_chunked_nll` for the fp16 baseline path;
`teacher_forced_chunked_nll_with_codec` for the codec-backed path).
C.2 step 2 landed the offline WikiText text loader
(`silica.bench.wikitext.load_wikitext_text`,
`tokenize_for_ppl`). Neither piece is reachable from the bench
harness today — `scripts/bench.py --scenario qwen3-0.6b-wikitext-ppl-*`
does not know about any of it.

Step 3 turns the C.1 / C.2 oracles + WikiText loader into four
bench rows on the Qwen3-0.6B adapter:

| Row id                                      | kv_codec          | Oracle path                                 |
| ------------------------------------------- | ----------------- | ------------------------------------------- |
| `qwen3-0.6b-wikitext-ppl-fp16`              | (None)            | `teacher_forced_chunked_nll` (fp16 baseline) |
| `qwen3-0.6b-wikitext-ppl-tq-mse-b4`         | `tq_mse_b4`       | `teacher_forced_chunked_nll_with_codec`     |
| `qwen3-0.6b-wikitext-ppl-block-tq-b64-b4`   | `block_tq_b64_b4` | `teacher_forced_chunked_nll_with_codec`     |
| `qwen3-0.6b-wikitext-ppl-ext-rabitq-b4`     | `ext_rabitq_b4`   | `teacher_forced_chunked_nll_with_codec`     |

The fp16 row establishes the baseline PPL; the three codec rows
report ΔPPL against the baseline. Model pinned to Qwen3-0.6B for
consistency with the existing `qwen3-0.6b-prefix-hit-decode-*`
rows (`has_recurrent_state=False`, accepts `RadixPrefixCache`).

## 2. Sub-step split

Step 3 is split into two commits:

### Step 3a — PPL plumbing + scenarios (this commit)

Lands everything below the HF-integration-test boundary:

- `OracleKind.PPL` enum value
- `ppl_oracle_fn` in `silica/bench/oracles.py`
- `_run_ppl` runner helper in `silica/bench/runner.py`
- `OracleKind.PPL` dispatch branch in `BenchRunner.run`
- Four `Scenario` definitions in `silica/bench/scenarios.py`
- Unit tests: oracle-fn validation; `_run_ppl` on the
  BatchKVCache-compat fake adapter + `tests/fixtures/wikitext2_tiny.txt`

Expected ~300-500 LOC total.

### Step 3b — Production integration test + cache script (next commit)

- `scripts/prepare_wikitext2_cache.py` — one-shot script that
  extracts WikiText-2 raw test split via HuggingFace datasets and
  writes `~/.cache/silica/wikitext2-test.txt`.
- Cache-gated integration test that loads real Qwen3-0.6B and runs
  the four rows end-to-end (or at least the fp16 row; the codec
  rows already have decode-path coverage from A.3c / B.3).

Separated because the HF-datasets dependency is heavy and the test
fixtures exist only on opted-in developer machines (same cache-
gating pattern as the decode-speed tests).

## 3. Design — step 3a

### 3.1 `OracleKind.PPL`

Add to `silica/bench/scenario.py`:

```python
class OracleKind(str, Enum):
    ...
    PPL = "ppl"
```

Docstring update: one bullet describing the kind (teacher-forced
streaming PPL with optional codec-backed arm; collected payload is
`{nll_sum, n_tokens, ppl}`; oracle_config carries `wikitext_path`,
`chunk_size`, `max_tokens`, `min_scored_tokens`).

### 3.2 `ppl_oracle_fn`

Signature:

```python
def ppl_oracle_fn(
    scenario: Scenario,
    collected: Any,
    context: Any,
) -> tuple[bool, str | None, dict[str, Any]]: ...
```

- `collected` is a mapping-shaped result:
  `{"nll_sum": float, "n_tokens": int, "ppl": float}`.
- `context` is a dict carrying chunk / path / codec metadata for
  downstream consumers; step 3a never populates `ppl_fp16` inside
  `_run_ppl`, so `delta_ppl` always lands as `None` on in-tree
  rows. The `ppl_fp16` handling stays wired so a downstream
  bench-report pass (or C.6 vqbench cross-check) can supply it
  when comparing codec rows to their baseline without the oracle
  function moving.

Validation:

- Shape check on `collected`: required keys present; `n_tokens` is
  an int ≥ 0; `nll_sum` is a float; `ppl` is finite (≥ 0).
- `n_tokens >= scenario.oracle_config["min_scored_tokens"]`
  (defaults to 1 if not set).
- If `context["ppl_fp16"]` is present (future / report layer):
  compute `delta_ppl = ppl - ppl_fp16` and surface in metadata.

Metadata returned:

```python
{
    "nll_sum": float,
    "n_tokens": int,
    "ppl": float,
    "ppl_fp16": float | None,
    "delta_ppl": float | None,
    "delta_ppl_pct": float | None,
    "chunk_size": int,
    "max_tokens": int,
    "wikitext_path": str,
    "kv_codec": str | None,
}
```

No pass/fail gate on the `delta_ppl` magnitude here — the vqbench
cross-check in C.6 is the authoritative gate. This oracle's only
pass/fail decision is structural validation.

### 3.3 `_run_ppl` runner helper

Landed shape (not engine-dependent; the PPL oracle bypasses
`engine.generate_batch` entirely and only drives the adapter):

```python
def _run_ppl(
    scenario: Scenario,
    adapter: ModelAdapter,
) -> tuple[dict[str, float | int], dict[str, Any]]:
    """..."""
    cfg = scenario.oracle_config
    wikitext_path = Path(cfg["wikitext_path"])
    chunk_size = int(cfg.get("chunk_size", 256))
    max_tokens = int(cfg.get("max_tokens", 512))

    # Tokenize offline from local file. ModelAdapter.tokenizer is a
    # Protocol method (not a property) — call it to get the bound
    # Tokenizer instance before handing it to tokenize_for_ppl.
    text = load_wikitext_text(wikitext_path)
    tokenizer = adapter.tokenizer()
    tokens = tokenize_for_ppl(
        tokenizer, text,
        max_tokens=max_tokens, min_tokens=chunk_size,
    )

    # Reuses the existing _maybe_build_prefix_cache helper — returns
    # None for kv_codec=None (fp16 baseline) and a populated
    # RadixPrefixCache wrapping a SyntheticPrefixBlockStore with the
    # registered codec pair otherwise. No new helper needed.
    prefix_cache = _maybe_build_prefix_cache(scenario.workload, adapter)

    if prefix_cache is None:
        # fp16 baseline — straight through the adapter's fp16 cache.
        nll_sum, n_tokens = teacher_forced_chunked_nll(
            adapter, tokens, chunk_size=chunk_size,
        )
    else:
        # Codec-backed — drive the C.2 oracle through the RadixPrefixCache.
        nll_sum, n_tokens = teacher_forced_chunked_nll_with_codec(
            adapter, prefix_cache, tokens, chunk_size=chunk_size,
        )

    ppl = perplexity_from_nll(nll_sum, n_tokens)

    # Step 3a does NOT add ``ppl_fp16`` here — no runner inter-row
    # baseline cache exists, so every in-tree row records
    # ``delta_ppl = None``. Downstream bench-report consumers (or
    # C.6 vqbench cross-check) recompute deltas by pairing rows'
    # recorded ``ppl`` values.
    collected = {
        "nll_sum": float(nll_sum),
        "n_tokens": int(n_tokens),
        "ppl": float(ppl),
    }
    context = {
        "chunk_size": chunk_size,
        "max_tokens": max_tokens,
        "wikitext_path": str(wikitext_path),
        "kv_codec": scenario.workload.kv_codec,
    }
    return collected, context
```

No new prefix-cache builder is needed — the existing
`_maybe_build_prefix_cache(workload, adapter)` helper already
routes `codec_registry` lookup + `adapter.kv_layout()` into a
`SyntheticPrefixBlockStore` + `RadixPrefixCache` with hard-coded
`block_size=16` (matching every other runner path).

### 3.4 Runner dispatch branch

In `BenchRunner.run` (around line 306 in `runner.py`):

```python
elif scenario.oracle == OracleKind.PPL:
    ppl_collected, oracle_context = _run_ppl(scenario, adapter)
    oracle_input = ppl_collected
    total_tokens = int(ppl_collected["n_tokens"])
```

The oracle is dispatched through the standard `ORACLES[...]` lookup
as with other kinds.

### 3.5 Scenario entries

`silica/bench/scenarios.py`:

```python
_DEFAULT_WIKITEXT_PATH = str(
    Path.home() / ".cache" / "silica" / "wikitext2-test.txt"
)

_QWEN3_0_6B_WIKITEXT_PPL_FP16 = Scenario(
    id="qwen3-0.6b-wikitext-ppl-fp16",
    repo="Qwen/Qwen3-0.6B",
    workload=Workload(
        name="wikitext-ppl-fp16",
        prompts=(),  # PPL is a teacher-forced oracle, no prompts
        max_tokens=0,
        max_batch_size=1,
        prefix_cache=False,
        kv_codec=None,
    ),
    oracle=OracleKind.PPL,
    oracle_config={
        "wikitext_path": _DEFAULT_WIKITEXT_PATH,
        "chunk_size": 256,
        "max_tokens": 512,
        "min_scored_tokens": 256,
    },
    gate_env_var=None,
    description=(
        "P-5-C.2 baseline PPL row. Drives Qwen3-0.6B on "
        "WikiText-2 raw test split via "
        "teacher_forced_chunked_nll; chunk_size=256, max_tokens=512 "
        "(two chunks, matches vqbench REPORT headline). Cache-only "
        "gate plus presence of the WikiText-2 text cache file at "
        "~/.cache/silica/wikitext2-test.txt (populate with "
        "scripts/prepare_wikitext2_cache.py)."
    ),
)
```

Three more rows identical shape, differing in `workload.kv_codec`
and `workload.prefix_cache=True`. Registered in
`BUILTIN_SCENARIOS`.

Note: the bench Workload currently requires a non-empty prompt
tuple for B=1 smoke rows. The PPL oracle bypasses `engine.generate*`
entirely, so `prompts=()` and `max_tokens=0` are legal for this
oracle. Revisit the Workload validation if the existing
`__post_init__` rejects these values on codec rows.

### 3.6 Workload-field adjustment (if required)

Current `Workload.__post_init__` (scenario.py:108-131) enforces
`prefix_cache=True` when `kv_codec` is set. That is still correct
for the codec PPL rows, which DO need a prefix cache. The rows
will set `prefix_cache=True, kv_codec="<id>"`.

For the `fp16` baseline row, `kv_codec=None` and
`prefix_cache=False` — the oracle drives the adapter's own fp16
`BatchKVCache` without involving the prefix cache at all.

### 3.7 Tests (step 3a)

Two test files:

#### `tests/test_ppl_oracle_fn.py` (new)

Unit tests on `ppl_oracle_fn`:

- `test_accepts_valid_collected` — returns `(True, None, ...)` on a
  well-shaped `collected`.
- `test_rejects_missing_key_nll_sum`
- `test_rejects_missing_key_n_tokens`
- `test_rejects_missing_key_ppl`
- `test_rejects_non_finite_ppl` (inf / nan)
- `test_rejects_negative_n_tokens`
- `test_rejects_n_tokens_below_min_scored_tokens`
- `test_computes_delta_ppl_when_ppl_fp16_in_context`
- `test_metadata_includes_all_declared_keys`

No adapter / engine dependency; oracle_fn is pure validation.

#### `tests/test_run_ppl.py` (new)

Tests on `_run_ppl` using the BatchKVCache-compat fake adapter
from `tests/test_ppl_oracle_codec.py` plus the tiny WikiText
fixture:

- `test_run_ppl_fp16_baseline_path` — `kv_codec=None` scenario,
  asserts `collected["n_tokens"] > 0`, `ppl` finite.
- `test_run_ppl_codec_path_with_identity_matches_baseline` — same
  token stream, identity codec, asserts (fp16) == (identity codec)
  to rel/abs 1e-5.
- `test_run_ppl_raises_when_wikitext_path_missing`.
- `test_run_ppl_honours_oracle_config_overrides` — non-default
  `chunk_size`, `max_tokens` flow through.

Integration with a real Qwen3-0.6B lives in step 3b.

### 3.8 Acceptance (step 3a)

- [x] `OracleKind.PPL` in `silica.bench.scenario`.
- [x] `ppl_oracle` in `silica.bench.oracles`, registered in
      `ORACLES`; 20 unit tests green in
      `tests/test_ppl_oracle_fn.py`.
- [x] `_run_ppl` in `silica.bench.runner`; 10 unit tests green
      in `tests/test_run_ppl.py` (including 3 gate-check tests
      for the PPL wikitext-path skip path).
- [x] Runner dispatches `OracleKind.PPL` through `_run_ppl`.
- [x] `_check_gates` skips `OracleKind.PPL` rows when
      ``oracle_config['wikitext_path']`` is absent from disk,
      before engine-factory load. Prevents wasting a ~600 MB
      Qwen3-0.6B weight load on a missing tokenizer input.
- [x] Four `Scenario` entries registered in `BUILTIN_SCENARIOS`
      (`qwen3-0.6b-wikitext-ppl-{fp16, tq-mse-b4, block-tq-b64-b4,
      ext-rabitq-b4}`). Scenario descriptions state plainly that
      step 3a emits raw PPL numbers only; ΔPPL vs the fp16
      baseline is a downstream derivation (bench report / C.6
      vqbench cross-check), not wired into the runner in 3a.
- [ ] End-to-end bench CLI smoke (`python -m scripts.bench --scenario
      qwen3-0.6b-wikitext-ppl-fp16`) — deferred to step 3b
      alongside the `prepare_wikitext2_cache.py` script.
- [x] Adjacent regression sweep green: 204-test sweep across
      `test_ppl_oracle_fn`, `test_run_ppl`, `test_ppl_oracle`,
      `test_ppl_oracle_codec`, `test_wikitext_loader`,
      `test_bench_runner`, `test_bench_cli`, `test_codec_registry`.
- [x] ruff + mypy clean on the modified `silica/bench/*` modules
      and the new test files (via a ``_call_run_ppl`` cast wrapper
      that keeps `_run_ppl`'s `ModelAdapter` type annotation honest
      at test sites using a subset-Protocol fake adapter).

## 4. Design — step 3b (forward reference)

### 4.1 `scripts/prepare_wikitext2_cache.py`

Standalone script (no `silica.*` imports beyond stdlib + HF
datasets) that extracts `wikitext-2-raw-v1` test split, joins the
`text` field with double newlines, and writes UTF-8 to
`~/.cache/silica/wikitext2-test.txt`.

```python
# scripts/prepare_wikitext2_cache.py
from __future__ import annotations
from pathlib import Path

def main() -> None:
    from datasets import load_dataset
    ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    text = "\n\n".join(row["text"] for row in ds)
    out_path = Path.home() / ".cache" / "silica" / "wikitext2-test.txt"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(text, encoding="utf-8")
    print(f"Wrote {len(text):,} chars to {out_path}")

if __name__ == "__main__":
    main()
```

### 4.2 Integration test

`tests/test_bench_ppl_scenarios_integration.py` (new, cache-gated):

- Dual gate: HF cache present for `Qwen/Qwen3-0.6B` AND wikitext
  cache file present at `~/.cache/silica/wikitext2-test.txt`.
- Runs `BenchRunner.run(_QWEN3_0_6B_WIKITEXT_PPL_FP16)` end-to-end.
- Asserts `status=="ok"`, `metadata["n_tokens"] > 0`,
  `metadata["ppl"]` finite and positive.

Codec rows deliberately not run in this integration test — their
codec path is already covered by `test_prefix_hit_decode_speed_gate.py`
(A.3c + B.3), and a real-model PPL run for three codecs per CI
sweep is expensive for marginal coverage.

## 5. Open questions (resolved during step 3a implementation)

### 5.1 Adapter tokenizer access — resolved

`ModelAdapter.tokenizer` is a **Protocol method**
(`silica/models/adapter.py:170`), not a property. Every existing
runner path (`runner.py:431`, `:607`, `:741`) calls it as
`adapter.tokenizer()`; `_run_ppl` matches that convention.

### 5.2 Engine `block_size` — resolved

Not consulted. `_maybe_build_prefix_cache` already hard-codes
`block_size=16` — the same constant every pytest-side prefix-cache
test has relied on since P-2. PPL rows reuse that constant; the
oracle's `chunk_size % block_size == 0` contract is satisfied by
setting `chunk_size=256` (16 × 16).

### 5.3 Existing prefix-cache builder — resolved

Reused verbatim. `_maybe_build_prefix_cache(workload, adapter)`
already routes `codec_registry` lookup + `adapter.kv_layout()` into
a `SyntheticPrefixBlockStore` + `RadixPrefixCache`. No
`_build_prefix_cache_for_codec` helper needed; `_run_ppl` calls
`_maybe_build_prefix_cache` directly and branches on the `None`
return (fp16 baseline) vs populated cache (codec path).

## 6. Risks

- **Context / scope growth**: step 3a touches `scenario.py`,
  `oracles.py`, `runner.py`, `scenarios.py`, plus two new test
  files. Keep the runner dispatch branch tight; defer anything
  that naturally fits in C.3 (STORAGE oracle) or C.4 (multi-seed)
  to those sub-units.
- **Workload schema stretch**: PPL has no prompts / max_tokens in
  the `Workload` sense. Using `prompts=()` and `max_tokens=0` is
  pragmatic but documents via comment that this oracle bypasses
  `engine.generate_batch`. Resist adding a PPL-specific field to
  `Workload` — oracle-specific knobs belong in `oracle_config`.
- **`adapter.tokenizer` availability**: if the adapter does not
  expose the tokenizer at the oracle-runner boundary, step 3a may
  need a small adapter-surface addition. Prefer reading whatever
  the scheduler already uses.

## 7. Changelog

- **2026-04-23**: doc opened with the 3a / 3b split.
- **2026-04-23**: step 3a landed —
  `OracleKind.PPL`, `ppl_oracle`, `_run_ppl`, four
  `qwen3-0.6b-wikitext-ppl-*` scenarios, 30 new unit tests
  (`test_ppl_oracle_fn.py`, `test_run_ppl.py`). Production
  bench CLI integration + real-model test remain for step 3b.
- **2026-04-23**: post-review fixes applied before commit —
  (1) `_run_ppl` calls `adapter.tokenizer()` as a method
  (matches the real Protocol); (2) `_ByteTokenizer` caps byte
  values at `_VOCAB` so test CE is numerically stable under
  arbitrary test-order permutations; (3) `_check_gates`
  extended with a PPL wikitext-path gate so missing cache
  file skips pre-load; (4) scenario descriptions no longer
  claim runner-side ΔPPL propagation (not wired in 3a).
