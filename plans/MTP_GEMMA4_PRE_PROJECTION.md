# D-023 — Gemma 4 MTP drafter pre-projection (half-day external spike)

| Field        | Value                                                              |
| ------------ | ------------------------------------------------------------------ |
| Decision ID  | D-023                                                              |
| Status       | Opening — gate (i) verified 2026-05-06 (outcome A\*); gate (ii) downloads done 2026-05-06 (target `dcb78c3` 17 GB + drafter `28e9227` 926 MB); gate (iii) reframed 2026-05-06 — install via isolated venv, NOT project dev-deps (mlx-vlm 0.5.0 requires `mlx>=0.31.2 / mlx-lm>=0.31.3` which would force-bump the v1.7.21 determinism anchor); gate (iv) license reconciliation deferred — spike does not bundle weights |
| Created      | 2026-05-06                                                         |
| Origin       | External evidence: Google's Gemma 4 multi-token-prediction release |
| Frame        | Track C reopen probe; not D-022 reopen, not C.3 continuation        |
| Phase order  | Runs before P-8 OpenAI HTTP server; P-6 already done at v1.7.28    |

---

## TL;DR

Google released Gemma 4 multi-token-prediction (MTP) drafters with a claimed
~2.2× Apple Silicon speedup at B=4-8. Silica-MLX's Track C speculative line
retired at v1.7.20 (DFlash) and v1.7.22 (DDTree); MTP is **external evidence
on a closed line**, so a half-day external spike against the public weights
through `mlx_vlm` is appropriate before multi-day P-8 work begins. The spike
runs entirely outside `silica.*` runtime — no integration code lands until
the measured B=1 per-row gate clears.

The gate is **measured** off vs on tok/s. Any speedup formula in this
document exists for intuition only and never enters the verdict.

**Feasibility caveat at the top.** The advertised MTP pair on the HF /
mlx-vlm side is target `mlx-community/gemma-4-31B-it-bf16` (~62.5 GB) +
drafter `mlx-community/gemma-4-31B-it-assistant-bf16` (~939 MB). The
BF16 target exceeds the M5 Pro 48 GB unified-memory ceiling, so the
spike is not runnable as-advertised on this hardware. Gate (i) resolved
to outcome A\*: use the 4-bit IT target
`mlx-community/gemma-4-31b-it-4bit` plus the BF16 assistant drafter.
That pairing is hardware-feasible but mixed-precision / undocumented;
accept-rate and parity remain empirical, and the verdict must carry the
precision-mismatch caveat.

---

## §1 Provenance — D-023 vs C.3 reopen

The closed Track C entries in PLAN.md §9 D-021 step 4 / step 6 / step 8
were Qwen3.5-target-framed:

- v1.7.20 retired **C.4 DFlash** at 0.482× silica-integrated speedup on
  dense Qwen3.5-27B-4bit (η.1 measurement: `accept_rate = 0.0881`,
  `draft_cost_ms = 35.70` ≫ `verify_cost_ms = 2.45`).
- v1.7.22 retired **C.5 DDTree** at the production-B verify-cost wall
  (cycle-23: B=52 k=64 verify cost 8105 ms vs B=52 plain decode ~252 ms).
- C.1 / C.2 / C.3 / C.6 deprioritised at v1.7.23 because (1b) ≥60 tok/s
  cleared via the C10 × C12 composition; spec-decode no longer required
  to clear the dense gate.

C.3 was never instantiated because the current Silica production target,
`mlx-community/Qwen3.5-27B-4bit`, ships no MTP weights. (The broader
Qwen3.5 family / training story may include MTP heads in some configurations;
the load-bearing fact is that the specific 4-bit checkpoint silica targets
in production does not.) Gemma 4 is a new family with a publicly-released
MTP drafter, so opening as **D-023** (not C.3 continuation) keeps the
v1.7.20-22 Track C closure record intact and frames the spike around new
external evidence rather than re-litigating an old verdict. P-6 stays done;
D-023 runs parallel to P-8 in the §9 register.

---

## §2 License-verify — 待核查 (verify before reuse)

Two lines of license metadata must be reconciled before any model file
is reused or redistributed:

- **Official source:** Google's blog post and `ai.google.dev/gemma/docs/mtp/overview`
  describe Gemma 4 MTP drafters as **Apache-2.0** licensed.
- **mlx-community conversion metadata:** the HuggingFace card for
  `mlx-community/gemma-4-31B-it-assistant-bf16` shows
  `License: gemma`. **Verify before reuse** — the conversion may inherit
  the upstream Gemma terms-of-use which restrict redistribution.

Status: **`official: apache-2.0; mlx-community conversion metadata:
License: gemma — verify before reuse.`** Reuse decision (whether silica may
bundle, fine-tune, or redistribute the drafter) is deferred until both
lines reconcile. No code or weights are pulled into `silica.*` during the
spike — `mlx_vlm` runs the drafter externally — so the spike itself does
not require the reuse decision.

---

## §3 Supported-pairing + hardware-feasibility verification

This is the load-bearing stop-gate before any download or install runs.
The cache check in §4 is bookkeeping; this section is the decision.

### §3.1 Advertised pairing on the HF / mlx-vlm card

Per `https://huggingface.co/mlx-community/gemma-4-31B-it-assistant-bf16`
and the `mlx-vlm` README, the supported MTP pair is:

| Role     | Repo                                              | Format | Approx size |
| -------- | ------------------------------------------------- | ------ | ----------- |
| target   | `mlx-community/gemma-4-31B-it-bf16`               | BF16   | ~62.5 GB    |
| drafter  | `mlx-community/gemma-4-31B-it-assistant-bf16`     | BF16   | ~939 MB     |

The `mlx-community/gemma-4-31b-4bit` already in the local cache is
**not the supported MTP target**: it is the non-IT 4-bit variant and
the assistant drafter has not been validated against it. Wiring the
cached 4-bit base target to the BF16 IT drafter is unsupported and
the accept-rate / parity behaviour is unknown.

### §3.2 Hardware-feasibility against M5 Pro 48 GB

The BF16 target alone is ~62.5 GB. M5 Pro unified memory is 48 GB. The
target plus the ~939 MB drafter does not fit in unified memory; even
weight-streaming would require the working set to spill, and `mlx_vlm`
does not run the dense forward off SSD in this configuration. **The
spike is not runnable as-advertised on this hardware.**

### §3.3 What the supported-pairing verification must answer

The verification is read-only: HF model search via
`https://huggingface.co/api/models?search=gemma-4-31B-it` (or the
website model-list filter) plus the drafter card's "Recommended target"
field plus the `mlx-vlm` README for documented mixed-precision support.
No download, no install — the verification gates whether either
follows.

Before any download or install runs, three questions need primary-source
answers (HF cards, mlx-vlm docs, dflash-mlx style follow-up):

1. **Is there an MTP-supported 4-bit IT target variant for Gemma 4-31B?**
   E.g., `mlx-community/gemma-4-31B-it-4bit` paired with a 4-bit
   assistant. If yes, this is the pairing to use; the working set drops
   to ~17-19 GB and the spike is feasible.
2. **If only BF16 IT exists, does the assistant drafter run against the
   cached non-IT 4-bit base target?** Likely no — the assistant is
   trained against the IT target's hidden states / layer geometry, so
   accept-rate is the empirical question. This path requires explicit
   user authorization because an unsupported pairing risks a wrong
   measurement.
3. **Does mlx-vlm support a mixed-precision target + drafter (4-bit
   target + bf16 drafter or vice versa)?** If yes, mixed precision could
   make the spike feasible without downloading 62.5 GB.

If all three questions resolve negative (no 4-bit IT pairing, drafter
incompatible with non-IT base, no mixed-precision support), this
M5-Pro external spike closes with a hardware-feasibility negative; the
broader Gemma 4 MTP question stays open and is monitored for a future
4-bit IT target conversion or documented mlx-vlm mixed-precision
support. Recorded as a fourth hard block in the gate matrix below.

### §3.4 Outcome of the verification

The verification is a primary-source read (HF model search, mlx-vlm
README, drafter card) plus a brief check that any downloads stay in
range. The outcome is one of:

- **A** — 4-bit IT target exists with a precision-matched (4-bit) drafter,
  the documented pair is genuinely 4-bit on both sides → pull both,
  ~18 GB total, run spike with no caveat.
- **A\*** (hybrid mixed precision, no precision-matched drafter exists) —
  4-bit IT target exists but only bf16 drafter exists, and the bf16
  drafter is the documented partner of the bf16 IT target → pull the
  4-bit IT target + bf16 drafter, accept the precision-mismatch as a
  documented unsupported pairing in the verdict, run spike with the
  caveat that accept-rate is empirical (drafter was trained against
  bf16 IT hidden states; loading 4-bit IT instead changes the layer
  geometry by precision only, not by training data, so accept-rate
  is plausibly close to the documented baseline but not vendor-warranted).
- **B** — only BF16 pair exists, mixed-precision unsupported → close
  this M5-Pro external spike with hardware-feasibility negative
  (gate row PAIR-INFEASIBLE below). The broader Gemma 4 MTP question
  remains open and gets monitored for a future 4-bit IT target
  conversion or documented mlx-vlm mixed-precision support;
  re-run the verification step when either appears.
- **C** — only BF16 pair exists, mixed-precision documented as
  supported → ask user authorization to download the 939 MB BF16
  drafter, run spike with cached 4-bit base target as a documented
  unsupported pairing (with a caveat in the verdict).
- **D** — BF16 pair only and explicit user authorization to use a
  remote box / cloud GPU for the spike → out of scope for "half-day
  external probe on M5 Pro" framing; defer.

### §3.5 Verification result (2026-05-06)

Performed via `https://huggingface.co/api/models?search=gemma-4-31B-it&author=mlx-community`,
the drafter card at `https://huggingface.co/mlx-community/gemma-4-31B-it-assistant-bf16`,
and the `mlx-vlm` README at `https://github.com/Blaizzy/mlx-vlm`.

Findings:

- `mlx-community/gemma-4-31b-it-4bit` exists on HF (most-downloaded
  variant of the family at 56,661 downloads as of the verification
  date). 4-bit IT target solves the M5 Pro 48 GB ceiling.
- No precision-matched (4-bit) drafter exists. The only drafter on HF
  is `mlx-community/gemma-4-31B-it-assistant-bf16` (939 MB bf16). The
  drafter card recommends `mlx-community/gemma-4-31B-it-bf16` as the
  documented target (precision-matched bf16 pair).
- mlx-vlm README does not document mixed-precision pairings. Examples
  show matching precision only.

Resolution: **outcome A\*** (hybrid mixed precision). User authorized
2026-05-06: pull `mlx-community/gemma-4-31b-it-4bit` (~17-19 GB est.) +
`mlx-community/gemma-4-31B-it-assistant-bf16` (~939 MB), run spike with
the precision-mismatch caveat documented in the verdict. The
hardware-feasibility hard block (PAIR-INFEASIBLE) is therefore not
fired; the spike proceeds to gate (ii) download authorization, gate
(iii) isolated-venv setup for `mlx-vlm 0.5.0`, and gate (iv) license
reconciliation.

---

## §4 Runtime dependencies — isolated venv fork

The spike runs through Google's published `mlx_vlm` runtime. The
canonical CLI command form per the HF / mlx-vlm README is:

```
python -m mlx_vlm.generate \
    --model <target_repo> \
    --draft-model <drafter_repo> \
    --draft-block-size <N> \
    --temp 0 \
    --prompt "..." \
    --max-tokens <M>
```

A Python `batch_generate` API exists for B > 1 measurement; the
exact entry-point is `mlx_vlm.utils.batch_generate` (verify exact
import path against the installed version before scripting the spike).

### §4.1 Why an isolated venv, not project dev-deps

`mlx-vlm 0.5.0` (the version with Gemma 4 MTP CLI) requires
`mlx>=0.31.2 / mlx-lm>=0.31.3 / transformers>=5.5.0`. Silica's project
pin is `mlx==0.31.1 / mlx-lm==0.31.2 / mlx-metal==0.31.1` per the
v1.7.21 anchor in `tests/test_p2_preload_parity.py` (the cycle-11
2026-05-04 upgrade attempt produced a deterministic argmax flip at
greedy-decode index 5; bisect across the three packages was deferred).

`uv add --dev mlx-vlm` would therefore either fail at the resolver
step or force-bump the pin and break the determinism anchor.

Decision recorded 2026-05-06: **install `mlx-vlm 0.5.0` into a
project-external isolated venv; do NOT touch `pyproject.toml` or
`uv.lock`.** This decouples the spike from the pin-bump decision
entirely. The spike's verdict will record the runtime-stack divergence
explicitly (see §10).

### §4.2 Isolated venv install plan

| Item                  | Path / version                                                    |
| --------------------- | ----------------------------------------------------------------- |
| venv root             | `~/.cache/silica-d023-mtp/.venv`                                  |
| Python                | match silica's `>=3.12`                                           |
| `mlx-vlm`             | `==0.5.0` (Gemma 4 MTP CLI; 2026-05-06 release)                   |
| transitively pinned   | `mlx>=0.31.2`, `mlx-lm>=0.31.3`, `transformers>=5.5.0`, `Pillow`, etc. |
| silica `pyproject.toml` | **untouched** at `mlx==0.31.1 / mlx-lm==0.31.2 / mlx-metal==0.31.1` |

After install, the spike scripts run via
`~/.cache/silica-d023-mtp/.venv/bin/python -m mlx_vlm.generate ...`,
keeping the command path explicit to avoid accidental shadowing by
the project venv on `$PATH`.

### §4.3 Pin-divergence ledger (records what changed during the spike)

The verdict template (§10) records the exact installed versions of
`mlx`, `mlx-lm`, and `mlx-vlm` inside the isolated venv at spike
runtime. Future-self consults that block to know precisely what
runtime stack produced the measurements; the silica project pin is
separately anchored by the still-untouched
`tests/test_p2_preload_parity.py` determinism gate.

---

## §5 HuggingFace cache check (bookkeeping)

The pairing decision in §3 governs which repos matter. The cache state
as of 2026-05-06:

| Repo                                                     | Cached? | Local size | Role for D-023                                            |
| -------------------------------------------------------- | ------- | ---------- | --------------------------------------------------------- |
| `mlx-community/gemma-4-31b-4bit` (non-IT)                | ✅      | 17 GB      | **Not the supported MTP target**; not used under outcome A\* |
| `mlx-community/gemma-4-26b-a4b-4bit` (MoE non-IT)        | ✅      | 15 GB      | Out of scope for D-023 (MTP drafter is dense-paired)       |
| `mlx-community/gemma-4-31b-it-4bit` (4-bit IT target)    | ✅      | 17 GB      | **Outcome A\* target.** Downloaded 2026-05-06 (snapshot `dcb78c3`)        |
| `mlx-community/gemma-4-31B-it-bf16` (advertised target)  | ❌      | (~62.5 GB) | Cannot fit on 48 GB unified memory; not used under outcome A\* |
| `mlx-community/gemma-4-31B-it-assistant-bf16` (drafter)  | ✅      | 926 MB     | **Outcome A\* drafter.** Downloaded 2026-05-06 (snapshot `28e9227`)        |

Both outcome A* models cached locally. The non-IT 4-bit target snapshot
has 4 safetensors shards plus tokenizer + `processor_config.json`
(multimodal-capable processor; the spike runs text-only, mirroring
D-014).

---

## §6 Measurement plan

Applies only when §3 resolves to outcome A, outcome A\*, or outcome C.

All measurements run through `mlx_vlm` external runtime; no `silica.*`
imports.

### §6.1 Per-call decomposition

For each `(B, draft_block_size)` pair, capture from a single
representative warm decode trace:

- `drafter_ms` — wall time to produce the draft block.
- `verify_k_ms` — target-side wall time to score the candidate tokens.
- `accept_rate` — fraction of candidate tokens accepted at
  `temperature=0`.
- `bonus_token_rate` — fraction of cycles that emit the target-side
  bonus token.
- `rollback_count` — number of cycles that triggered the
  no-accepts-rollback path.

### §6.2 `k` vs `draft_block_size` semantics

`mlx-vlm`'s `--draft-block-size` flag controls one cycle's draft length
**including the already-accepted / bonus token slot**, so the number of
fresh candidate tokens proposed per cycle is `k_candidates = block_size - 1`.
The HF card recommends `block_size = 6` for single-request decoding
(5 fresh candidates) and `block_size = 3` for batched (2 fresh
candidates), reflecting accept-rate distribution over draft position.

This document uses `draft_block_size` (CLI form) as the parameter and
records `k_candidates` alongside it; speedup is reported per
`(B, draft_block_size)`.

### §6.3 Sweep

- `draft_block_size ∈ {2, 3, 6, 9}` →
  `k_candidates ∈ {1, 2, 5, 8}`.
  - `block_size = 2` (`k_candidates = 1`) is the verify-cost floor —
    diagnostic only, not a decision row (see §7 below).
  - `block_size = 3` (`k_candidates = 2`) is the card's batched
    recommendation; **B=4 decision row**.
  - `block_size = 6` (`k_candidates = 5`) is the card's
    single-request recommendation; **B=1 decision row**.
  - `block_size = 9` (`k_candidates = 8`) hits the v1.7.18 verify-k
    zero-drafter ceiling 2.93×; stress row to see how MTP scales when
    accept-rate distribution thins.
- `B ∈ {1, 4}` — B=1 is the single-customer chat gate; B=4 is the
  v1.7.25 D-022 sonnet-baseline reference (10.29 ± 0.16 tok/s/row at
  Qwen3.5-27B-4bit; the Gemma 4 baseline will differ and is established
  by the off-spec leg of each measurement).
- Greedy mode: `temperature=0`, deterministic.
- Greedy parity: same prompt, same seed, off-spec vs on-spec output
  must match byte-for-byte at **`max_tokens=1` (cycle-1)**. Long-run
  (`max_tokens=N`) divergence is recorded but does not by itself fire
  the §7 row 2 hard block (see §7 row 2 amendment for the rationale
  vs the v1.7.19 D-021 step 5 precedent). Output quality (degeneracy
  / repetition / format collapse) is handled separately by §7 row 2.5.

### §6.4 Headline metric

Two numbers per `(B, draft_block_size)` pair drive the gate:

- **`off_tok_per_sec`** — drafter disabled, plain decode through
  `mlx_vlm` on the same target.
- **`on_tok_per_sec`** — drafter enabled, MTP path active.
- **`speedup = on_tok_per_sec / off_tok_per_sec`** — measured ratio.

Per-row speedup at B=4 means `(on_aggregate / 4) / (off_aggregate / 4)`,
which equals the aggregate ratio when row count is held constant.

### §6.5 Decision row vs diagnostic row

For each B, the **decision row** is the `draft_block_size` whose
measured `on_tok_per_sec` is highest (i.e., the best single-customer
speedup achievable in this config). Hard-block triggers (§7 rows 1, 2,
4) evaluate against the decision row. Diagnostic rows (`block_size = 2`,
floor) inform the per-call decomposition but do not, by themselves,
close D-023 — falling on a non-decision shape is not a verdict.

### §6.6 Variance discipline

Two sessions per the cycle-27 protocol, n=3 reps per session, combined
σ check on `on_tok_per_sec`. If combined σ > 1.5 tok/s on either side
of the gate threshold, the measurement defers until the drift source
is named.

> **Noise-floor caveat at B=1.** σ ≤ 1.5 tok/s is an absolute bound
> calibrated against the v1.7.25 D-022 sonnet baseline (B=4 aggregate
> 41.11 ± 0.64, B=8 45.03 ± 0.36, B=12 63.89 ± 0.05). At B=1 the
> bandwidth ceiling is ~20 tok/s, so 1.5 tok/s is ~7.5% relative on
> each side and the speedup ratio inherits ~10% combined uncertainty.
> Distinguishing 1.3× from 1.0× at that band is well-resolved;
> distinguishing 1.3× from 1.2× is not. If the measured speedup at
> B=1 lands in [1.2×, 1.4×], the absolute-σ rule does not cleanly
> resolve the gate and a relative supplement (e.g., `σ_ratio ≤ 0.05`)
> must be added at measurement time. Recorded as a footnote here so
> the discipline does not need to be improvised mid-measurement.

### §6.7 Cross-session protocol

Sessions must be cross-day (or at minimum ≥ 30 minutes apart with the
M5 Pro at thermal idle), since cycle-27 thermal drift can hide a real
1.05-1.10× win at the noise floor of the warm-decode gate.

---

## §7 Gate matrix — measured

Evaluated **top-down**; the first row whose trigger fires drives the
verdict. Hard blocks live at the top so they cannot be shadowed by a
pass condition fired against an unstable measurement.

| order | row name           | outcome / verdict                                                                | trigger                                                                              |
| ----- | ------------------ | -------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------ |
| 1     | PAIR-INFEASIBLE    | hard block — close this M5-Pro external spike with hardware-feasibility negative; no measurement runs; monitor for a future 4-bit IT target conversion or documented mlx-vlm mixed-precision support and re-run verification when either appears | §3 verification resolves to outcome B (only BF16 pair exists, mixed-precision unsupported, BF16 target > 48 GB) |
| 2     | GREEDY-PARITY-FAIL | hard block — drafter or runtime compatibility issue; investigate before any verdict | measured, `temperature=0`, **`max_tokens=1` cycle-1 byte parity** off-spec output ≠ on-spec output (sha256 mismatch). Long-run `max_tokens=N` byte divergence is **not** by itself a fail — it is a diagnostic / native-integration caveat (see row 2.5). This matches the v1.7.19 D-021 step 5 closure precedent: silica's own fp16 path produces `max_tokens=N` divergence from sequential reference under `BatchKVCache`, validated through (h) bench scenarios rather than against a sequential reference. |
| 2.5   | OUTPUT-QUALITY-FAIL | hard block — drafter introduces quality regression; investigate before any verdict | on-spec output exhibits **degeneracy** (looped repetition, format collapse, partial-token corruption, gibberish). Detected by: (a) repeated n-gram fraction > 25% over the generated tail, OR (b) pre-/post-pairing sentence-level fluency comparison flags clear regression on visual inspection. Distinguishes "different but coherent paraphrase" (acceptable, not a fail) from "broken" (hard block). Triggered separately from row 2 — long-run paraphrase divergence does not fire row 2.5 if outputs remain non-degenerate, well-formed, and topical. |
| 3     | DRAFT-VERIFY-WALL  | hard block — replicates DFlash η.1 physics; close D-023                          | measured `draft_cost_ms / verify_cost_ms ≥ 0.5` at the **best decision row** for B=1 AND B=4 (i.e., the ratio is ≥ 0.5 even at the most favourable `draft_block_size`); diagnostic-only `block_size = 2` does not fire this row. Note: `mlx_vlm.GenerationResult` does not separately expose `draft_cost_ms` / `verify_cost_ms`, so the external spike cannot directly evaluate this row. The external spike can only record whether a DFlash-like **net regression** is observed in tok/s. Direct ratio measurement is deferred to native integration. |
| 4     | B=1-PASS           | open D-023 native integration ladder (covers Gemma4 hidden-capture, see §8)      | measured B=1 per-row speedup ≥ 1.3× at the decision row, σ check PASS                 |
| 5     | B=4-ONLY-PASS      | record "serving / concurrency reopen value" only in §9; **no auto-integration** | measured B=4 per-row speedup ≥ 1.3× at the decision row AND row 4 did not fire        |
| 6     | NEGATIVE           | close D-023 with measurement-anchored negative; Track C remains closed           | measured B=1 < 1.3× AND B=4 < 1.3× at the respective decision rows                   |

**Diagnostic-only rows do not close D-023.** Decision rows are the
`draft_block_size` whose `on_tok_per_sec` is the highest at each B
(see §6.5). Failing on `block_size = 2` (the verify-cost floor) is not
a verdict.

**Row 2 / 2.5 amendment (2026-05-06, applied during D-023 measurement):**
The original row 2 wording specified "off-spec output ≠ on-spec output
bytewise" without bounding the comparison to cycle-1. As written, that
language is stricter than the silica-internal precedent set by v1.7.19
D-021 step 5 closure, where `DraftTargetEngine` was admitted into the
spec foundation with documented `max_tokens=N` divergence from the
sequential fp16 reference under `BatchKVCache` (validated via (h) bench
scenarios rather than long-run byte equality). Since D-023's purpose
is to decide whether to open the native-integration ladder, holding an
external spec drafter to a stricter parity standard than silica
already accepts internally is logically inconsistent. The amendment
narrows row 2 to cycle-1 byte parity (the discriminator that proves
the drafter introduces no logits-level bias) and splits output-quality
detection into a new row 2.5 (so paraphrase-level long-run divergence
does not fire when outputs remain non-degenerate). The native
integration ladder still requires its own cycle-1 parity gate plus
scenario-level output sanity plus rollback correctness; PASS at this
spike does **not** transfer that gate to silica-native code.

---

## §8 Native-integration gap — what the ladder must cover if D-023 PASS

The pre-projection runs externally through `mlx_vlm`. Native silica
integration on a PASS requires closing the following gap:

- `silica/models/hidden_capture.py:158` defines `HiddenCaptureAdapter`
  as a `@runtime_checkable` Protocol with two methods —
  `decode_step_multi_with_capture(tokens, kv_handle, capture_layer_ids)`
  and `prefill_with_capture(tokens, kv_handle, capture_layer_ids)`.
- The Protocol docstring (lines 161-167) explicitly states the current
  scope: "D-021 step 6 (αβ.1) lands this Protocol on `Qwen3_5Adapter`
  only; (αβ.2) extends it to `Qwen3_5MoeAdapter`. Other families
  (`Qwen3Adapter`, `Gemma4Adapter`, `Gemma4MoeAdapter`) do not ship the
  capture surface."
- `Gemma4Adapter` (`silica/models/gemma4.py:90`) and `Gemma4MoeAdapter`
  (`silica/models/gemma4_moe.py:77`) therefore **do not implement** the
  Protocol. Any target-conditioned drafter (DFlash, MTP head with
  hidden-state input, future C.6 self-spec) running against Gemma 4
  would fail the runtime gate at `silica/bench/runner.py:537`:

```python
from silica.models.hidden_capture import HiddenCaptureAdapter
...
if not isinstance(adapter, HiddenCaptureAdapter):
    raise NotImplementedError(
        "C.4 DFlash bench scenario requires a "
        "HiddenCaptureAdapter target ..."
    )
```

- `silica.speculative.engine.TargetHiddenConsumer` (the side-channel
  Protocol the drafter implements to consume captured hidden states)
  is independent of the adapter side and stays unchanged.
- `silica/speculative/draft_target.py` ships `DraftTargetEngine` for
  the (h) bench rows; this is the integration entry-point that a
  native MTP drafter would compose against.

**Native integration ladder (only triggered on B=1 PASS):**

1. Add `decode_step_multi_with_capture` + `prefill_with_capture` to
   `Gemma4Adapter` mirroring `Qwen3_5Adapter`'s implementation; verify
   `isinstance(Gemma4Adapter, HiddenCaptureAdapter)` returns `True`.
2. Author MTP-drafter wrapper alongside `DFlashDrafter` that consumes
   the capture surface (or routes through `DraftTargetEngine` if the
   MTP path doesn't need target hidden states — TBD by external spike
   findings).
3. Wire into `silica/bench/runner.py` and `silica/bench/scenarios.py`
   alongside the existing dense / MoE warm-decode-spec-on rows.
4. Re-run cycle-1 greedy-parity gate, three-rollback synthetic, and
   real-model parity tests.

**The pre-projection spike avoids items 1-4 entirely** — `mlx_vlm`
runs the drafter through its own runtime. Going native is a 1-2 week
follow-up that only opens on the B=1 ≥ 1.3× trigger.

---

## §9 Rough projection — intuition only

> **This subsection does not contribute to the gate decision.** The
> formula below is for sanity-check intuition before measurement. Per
> the cycle-22 lesson — the projected 270 tok/s headroom collapsed at
> the cycle-23 8105 ms verify wall — projection-only conclusions are
> structurally unreliable. Do not let a near-1.3× projection bias the
> reading of an ambiguous measurement.

A rough first-order model:

```
expected_speedup ≈ (1 + accept_rate × k_candidates) / (1 + draft_cost_ms / verify_cost_ms)
```

This ignores:

1. The bonus-token rule (target's own next-token sample after the last
   accepted draft slot — note that `draft_block_size = k_candidates + 1`
   in mlx-vlm reflects exactly this slot).
2. Verify-batch efficiency at B > 1 (cycle-23 production-B verify-cost
   wall: at B = 52, k = 64, verify cost = 8105 ms — not linear in B).
3. Draft block-size failure distribution (DFlash η.1 had `accept_rate
   = 0.0881` at k = 8 because most rejection happens early in the draft
   block; the formula treats `accept_rate` as a flat per-token property).
4. `mlx_vlm` Python wrapper overhead (every cycle pays a Python-side
   coordination cost not present in a silica-native integration).

Even with the optimistic Google figure (`accept_rate ≈ 0.6`,
`k_candidates = 5` at the card's `draft_block_size = 6` recommendation,
`draft_cost / verify_cost ≈ 0.3`) the formula yields ≈ 3.1× — which
sits at the v1.7.18 verify-k zero-drafter linear ceiling 2.93× at
B = 4 k = 8, so the optimistic claim is at the edge of physics.
**Plausible at the edge of physics is not a gate-pass.**

---

## §10 Verdict template

> Populated **after** the spike runs. Numbers below are placeholders;
> do not reuse them as projections. If §3 resolves to outcome B, the
> template fills in only the toolchain / pairing block and disposition
> ("PAIR-INFEASIBLE — this M5-Pro spike closes; broader Gemma 4 MTP
> question monitored for 4-bit IT target / mlx-vlm mixed-precision
> support").

```
=== D-023 verdict (date: YYYY-MM-DD) ===

Toolchain — **MUST record runtime-stack divergence** (§4.3)
  isolated venv path  : <e.g., ~/.cache/silica-d023-mtp/.venv>
  mlx (in venv)       : <e.g., 0.31.2>
  mlx-lm (in venv)    : <e.g., 0.31.3>
  mlx-vlm (in venv)   : <e.g., 0.5.0>
  silica project pin  : mlx==0.31.1 / mlx-lm==0.31.2 / mlx-metal==0.31.1 (untouched)
  divergence note     : "External spike stack != silica pinned stack;
                         spike result informs D-023 decision only and
                         does NOT constitute a silica runtime
                         attestation. tests/test_p2_preload_parity.py
                         remains anchored on the silica project pin."
  mx.metal device     : <device name + memory>

Pairing
  outcome (§3.4)        : <A | A* | B | C | D>     (A* hybrid mixed precision per §3.5)
  target  : <repo>      (snapshot <hash>)
  drafter : <repo>      (snapshot <hash>)
  precision mismatch?   : <NO | YES — record caveat in disposition>
  license reuse decision : <PENDING | RECONCILED-APACHE-2 | RESTRICTED-NO-REUSE>

Greedy parity (temperature=0)
  prompt token count             : <N>
  off-spec output (cycle-1)      : <sha256>
  on-spec  output (cycle-1)      : <sha256>
  cycle-1 bytewise match         : <PASS | FAIL>   ← drives §7 row 2
  off-spec output (max_tokens=N) : <sha256>
  on-spec  output (max_tokens=N) : <sha256>
  long-run bytewise match        : <PASS | FAIL>   ← caveat only, not row 2
  output quality (visual)        : <NO-DEGENERACY | DEGENERATE>   ← drives §7 row 2.5

Per-call decomposition (B=1)
                                  block=2  block=3  block=6  block=9
                                  k=1      k=2      k=5      k=8
  drafter_ms                  :   …        …        …        …
  verify_k_ms                 :   …        …        …        …
  draft/verify ratio          :   …        …        …        …
  accept_rate                 :   …        …        …        …
  bonus_token_rate            :   …        …        …        …
  rollback_count              :   …        …        …        …
  off_tok_per_sec  (n=6)      :   …±…      …±…      …±…      …±…
  on_tok_per_sec   (n=6)      :   …±…      …±…      …±…      …±…
  speedup                     :   …        …        …        …
  decision row?               :   diag     …        …        …

Per-call decomposition (B=4)
                                  block=2  block=3  block=6  block=9
                                  k=1      k=2      k=5      k=8
  drafter_ms                  :   …        …        …        …
  verify_k_ms                 :   …        …        …        …
  draft/verify ratio          :   …        …        …        …
  accept_rate                 :   …        …        …        …
  bonus_token_rate            :   …        …        …        …
  rollback_count              :   …        …        …        …
  off_tok_per_sec  (n=6)      :   …±…      …±…      …±…      …±…
  on_tok_per_sec   (n=6)      :   …±…      …±…      …±…      …±…
  speedup                     :   …        …        …        …
  decision row?               :   diag     …        …        …

Decision rows (per §6.5: highest on_tok_per_sec at each B)
  B=1 decision row : block_size = <…>, speedup = <…>
  B=4 decision row : block_size = <…>, speedup = <…>

Variance discipline
  combined σ B=1   : <PASS | FAIL>
  combined σ B=4   : <PASS | FAIL>
  B=1 in [1.2, 1.4] grey band? : <NO | YES — relative σ_ratio ≤ 0.05? PASS|FAIL>

Gate matrix evaluation (top-down, first FIRE wins)
  Row 1   PAIR-INFEASIBLE        : <FIRE | NO>
  Row 2   GREEDY-PARITY-FAIL     : <FIRE | NO>   (cycle-1 max_tokens=1 byte parity)
  Row 2.5 OUTPUT-QUALITY-FAIL    : <FIRE | NO>   (degenerate / format-collapse / repetition)
  Row 3   DRAFT-VERIFY-WALL      : <FIRE | NOT-EVALUATED>   (external spike lacks direct ratio; native integration must measure)
  Row 4   B=1-PASS               : <FIRE | NO>
  Row 5   B=4-ONLY-PASS          : <FIRE | NO>
  Row 6   NEGATIVE               : <FIRE | NO>

Disposition
  <one paragraph: which row fired, what next, link to PLAN.md §9 D-023 update>
```

---

## §11 References

### External

- Google blog: <https://blog.google/innovation-and-ai/technology/developers-tools/multi-token-prediction-gemma-4/>
- Google docs: <https://ai.google.dev/gemma/docs/mtp/overview>
- HuggingFace drafter card: <https://huggingface.co/mlx-community/gemma-4-31B-it-assistant-bf16>
- HuggingFace target card (advertised pair): <https://huggingface.co/mlx-community/gemma-4-31B-it-bf16>

### Cross-references inside silica-mlx

- v1.7.20 — C.4 DFlash retirement (η.1 = 0.482×, draft_cost_ms = 35.70).
  `plans/P6_C4_DFLASH/REPORT.md`. Failure mode to compare MTP against.
- v1.7.18 — verify-k zero-drafter ceiling 2.93× at B=4 k=8 linear.
  Constrains plausible MTP upper-band claims.
- v1.7.19 — D-021 step 5 spec foundation (`DraftTargetEngine` +
  bonus-token rule + three rollback paths). Reusable for native MTP
  integration if D-023 PASS.
- v1.7.22 — C.5 DDTree retirement (cycle-23 production-B verify-cost
  wall). Different physics from MTP (tree-shape vs linear k), but
  same family of "spec-decode at production B is fragile" lesson.
- v1.7.27 / v1.7.28 — D-022 closure framing: "compile axis exhausted;
  future revisits require a different lever". MTP qualifies as
  "different lever" but is properly Track C reopen, not D-022 reopen.
- `silica/models/hidden_capture.py:158` — `HiddenCaptureAdapter`
  Protocol definition.
- `silica/bench/runner.py:537` — runtime gate raising
  `NotImplementedError` for non-capture adapters.
- `silica/speculative/draft_target.py` — `DraftTargetEngine` integration
  surface.
- `silica/speculative/engine.py` — `TargetHiddenConsumer` side-channel
  Protocol.
