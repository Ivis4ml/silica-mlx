# P-5-(a-real) — real-activation Frobenius cross-check on Qwen3.5-0.8B

| Field        | Value                                                              |
| ------------ | ------------------------------------------------------------------ |
| Unit         | post-P-5 follow-up (v1.7.2 Notes → v1.7.4 Status)                  |
| Version      | v0.1 (design gate)                                                 |
| Last updated | 2026-04-24                                                         |
| Status       | closed at v1.7.5 — evidence in `docs/P5_ACCEPTANCE_SWEEP/real_activation_xcheck.{md,jsonl}` |
| Parent doc   | `docs/P5_OPENING.md` §7(a-real); `docs/PLAN.md` §7 P-5 Notes       |
| Scope        | Land the real-activation half of P-5 §7(a). Silica MLX BlockTQ vs |
|              | inlined NumPy reference on real Qwen3.5-0.8B K/V tensors extracted|
|              | from a forward pass, gated on the HF cache.                       |

## 1. What this unit closes

v1.7.2 narrowed P-5 §7 Acceptance (4-a) to synthetic Gaussian
inputs (closed at P-5-A.1c, `tests/test_block_tq_vqbench_xcheck.py`)
and moved the real-activation half to Notes as
"(a-real) — post-P-5 required follow-up". This design doc scopes
the landing of (a-real) as a standalone test that:

- Loads `Qwen/Qwen3.5-0.8B` via the existing `Qwen3_5Adapter`.
- Runs one deterministic forward pass on a short fixed prompt.
- Extracts the **pre-RoPE K / V projection outputs** from every
  GLOBAL (full-attention) layer (i.e. every layer with
  `layer.is_linear == False`).
- Feeds the extracted tensors through two pipelines:
  1. **silica MLX hot path** — `BlockTurboQuantMSE.encode_tensor` →
     `decode_tensor` at the production configuration
     `B=64, num_bits=4` (and the companion set
     `{B=32, num_bits=3}`, `{B=32, num_bits=4}`, `{B=64, num_bits=3}`
     for parity with (a-algo)).
  2. **NumPy reference** — `_numpy_block_tq_round_trip` from
     `tests/test_block_tq_vqbench_xcheck.py` (the verbatim
     vqbench-transcribed reference locked at P-5-A.1c at tolerance
     `5e-3` / `1e-3` on synthetic Gaussian).
- Compares per-block relative Frobenius error between the two
  pipelines and asserts the silica-vs-reference parity holds at
  the same tolerances as (a-algo).
- Compares silica's per-block relative Frobenius to a baseline
  measured by running `IdentityCodec` through the same encode /
  decode surface.

## 2. Design decisions (locked before implementation)

### 2.1 Layer subset — GLOBAL layers only

Qwen3.5-0.8B is hybrid-DeltaNet: some layers are linear-attention
(`layer.is_linear == True`) and do not carry a per-token K / V
tensor. (a-real) extracts from **every layer where
`layer.is_linear == False`** — the GLOBAL / full-attention subset.
This matches P-5's codec scope (BlockTQ only applies to
`(B, n_kv_heads, T, head_dim)` K / V tensors on
`AttentionKind.GLOBAL` layers; `AttentionKind.HYBRID_DELTANET`
recurrent state is outside the codec surface).

### 2.2 Extraction space — pre-RoPE K, vqbench-aligned

K is extracted **after `self_attn.k_proj` but before RoPE is
applied**. This matches the injection space of the D.2a
vqbench-aligned oracle landed at v1.7.3 (`codec_quality_path="vqbench_aligned"`)
and the vqbench `_QuantizedProj` harness. V has no RoPE, so there
is no ambiguity there.

**Why pre-RoPE, explicitly.** The D.2 investigation
(`docs/P5_D2_INVESTIGATION/README.md` §Root cause) established
that noise injection space matters for downstream quality, and
that silica's production prefix-cache store currently sits in
post-RoPE space. The future P-5-F pre-RoPE production routing
unit will consume (a-real)'s evidence; (a-real) must therefore
verify silica BlockTQ on the K space P-5-F is trying to adopt, not
on the post-RoPE space the production store currently uses.
Post-RoPE K is a valid BlockTQ input too, but the (a-real) gate is
an algorithmic-parity check — what matters is that silica and
vqbench both see the same K — and the vqbench harness injects
pre-RoPE.

### 2.3 Reference side — inline NumPy, not vqbench subprocess

The v1.5.1-era P5_OPENING §7(a-real) text specified a vqbench venv
subprocess + recon-specific driver script. This design gate
**supersedes** that: the reference side reuses the
already-landed `_numpy_block_tq_round_trip` from
`tests/test_block_tq_vqbench_xcheck.py` (the verbatim vqbench
algorithmic transcription, faithfulness-locked at tolerance `5e-3`
/ `1e-3` on synthetic Gaussian at P-5-A.1c). Rationale:

- **Pattern continuity.** The repo's established xcheck idiom
  transcribes vqbench's NumPy reference inline (memory:
  "vqbench is reference-only — never import in silica runtime OR
  tests"). A vqbench subprocess for (a-real) would introduce a
  second reference-access idiom for no additional fidelity gain —
  the transcription's faithfulness is already pinned.
- **No scipy dependency.** The transcribed reference consumes only
  `numpy` + `silica.vq._calibration` helpers (Haar + Lloyd-Max,
  both verbatim NumPy ports of vqbench's own calibration). It
  runs in silica's venv directly.
- **Simpler gate surface.** One gate (HF cache has Qwen3.5-0.8B),
  not two (HF cache + `VQBENCH_PYTHON_EXECUTABLE`).
- **Transcription risk already paid for.** The `5e-3` / `1e-3`
  bounds at (a-algo) already catch any regression in the
  transcription. (a-real) stacks on top of that guarantee.

`P5_OPENING.md` §7(a-real) body and `PLAN.md` §7 P-5 Notes
(a-real) bullet will be updated in the same commit as the
implementation to reflect the inline-NumPy decision; the
"subprocess driver + new recon script" wording is stale.

### 2.4 Baseline — `IdentityCodec` round-trip

The baseline noise floor is established by running `IdentityCodec`
through the same `encode_tensor` → `decode_tensor` surface the
BlockTQ codec uses. Three candidate forms were considered:

1. **bf16 → fp16 → bf16 cast.** Measures dtype-coercion noise in
   isolation.
2. **`IdentityCodec.encode_tensor` → `decode_tensor` on the same
   tensor.** Measures the actual production fp16 floor.
3. **NumPy reference's own identity pass** (no codec, no cast).

**Decision: (2).** `IdentityCodec` is the production baseline
codec and is the right point of reference for "what noise the
store path introduces with no compression". Its `encode_tensor`
returns a `RawFp16Payload` that stores the tensor directly, so the
round-trip is likely dtype-preserving and the observed baseline
Frobenius may be `≈ 0`. **If that is the case, (a-real) records
the measurement and falls back to absolute silica-vs-reference
Frobenius as the primary gate.** The `silica_frob ≤ 2 × baseline`
gate becomes degenerate when `baseline ≈ 0`; the evidence doc
names this outcome explicitly rather than discovering it at
implementation time and patching silently.

### 2.5 Tolerance — reuse (a-algo)'s pinned bounds

**Silica-vs-NumPy-reference per-block relative Frobenius:**

- `B=64, num_bits=4` (production recommendation): `< 1e-3`
  (matches (a-algo)'s production-recommended lock).
- `(B, bits) ∈ {32, 64} × {3, 4}`: `< 5e-3`
  (matches (a-algo)'s general bound).

**Silica vs `IdentityCodec` baseline:** `silica_frob ≤ 2 × baseline_frob`
per the v1.7.2 (a-real) wording. If `baseline ≈ 0` (see §2.4),
this bound is degenerate; fall back to the (a-algo) bounds above
as the primary gate and record the baseline value for
documentation.

**Why not invent a tighter "real activations are less adversarial"
number.** The transcription is the same whether the input is
synthetic or real; the tolerance envelope the synthetic check
validates is the same envelope real K must respect. Tightening
requires evidence, not intuition. If one run measures silica-vs-NumPy
parity well below `1e-3`, a subsequent tightening revision can
pin the new bound.

### 2.6 Skip gate — HF cache only, single-gate

**Gate:** `_hf_cache_has_repo("Qwen/Qwen3.5-0.8B")` per the
existing pattern at
`tests/test_engine_admission_reorder.py:607-626`. No
`VQBENCH_PYTHON_EXECUTABLE` gate (inline NumPy path; see §2.3).
If the cache is missing, the test skips with a clear message
pointing at how to populate it (any Qwen3.5-0.8B cache-only bench
scenario — `qwen3.5-0.8b-b1-parity` — will do).

v1.7.2 Notes' "dual-gate pattern" wording needs updating in the
same commit to reflect single-gate.

### 2.7 Prompt — checked-in deterministic string constant

The extracted K / V only needs to reflect realistic Qwen3.5-0.8B
activations; (a-real) does not depend on WikiText content the way
the PPL oracles do. Pinning the prompt to a WikiText-2 cache file
(e.g. `~/.cache/silica/wikitext2-test.txt`) would add an implicit
second skip dependency beyond §2.6's HF-cache gate — if the
WikiText file is missing the test would fail rather than skip.

**Decision:** embed a ~128-token deterministic English prompt as
a module-level string constant inside
`tests/test_block_tq_real_activation_xcheck.py`. No external data
dependency, no second gate, no WikiText cache plumbing. A
multi-paragraph generic English passage (e.g. a reshaped fixed
public-domain excerpt, or a handwritten encyclopedia-style
paragraph) gives realistic token-sequence statistics — what
matters is that the tokenizer produces ~128 tokens across
several sentences, not that the content match PPL scenarios.

The specific string is committed alongside the test file so the
prompt is reproducible across machines without any network or
cache state.

### 2.8 Seeds — n = 3 (codec RNG)

`BlockTurboQuantMSE`'s Haar rotation is seeded from the codec
ctor's `seed` parameter (P-5-D.1). (a-real) runs three seeds
`{42, 43, 44}` per (4-b) convention. Each seed produces a
different rotation, so per-seed Frobenius numbers will differ;
gate applies per seed. No mean-over-seeds aggregation here —
(a-real) is an algorithmic-parity gate, not a distributional one.

## 3. Evidence file

`docs/P5_ACCEPTANCE_SWEEP/real_activation_xcheck.md` (+ raw
`real_activation_xcheck.jsonl`). Landed schema (1 metadata row +
144 data rows):

- **Metadata row** (`_record: "metadata"`): `repo`,
  `n_global_layers`, `global_layer_indices`, `n_tokens`,
  `n_kv_heads`, `head_dim`, `dtype`, `cells` (list of active
  `(vq_block_size, num_bits)` pairs), `skipped_cells` (cells
  incompatible with `head_dim`), `seeds`, `sides`,
  `tolerance_production`, `tolerance_regular`,
  `baseline_degenerate_eps`.
- **Per (layer_idx, side, vq_block_size, num_bits, seed) data
  row:** `layer_idx`, `side ∈ {"K", "V"}`,
  `vq_block_size`, `num_bits`, `seed`, `silica_frob`,
  `numpy_frob`, `baseline_frob`, `silica_vs_numpy_abs_gap`
  (`|silica - numpy|`), `gate_pass_silica_vs_numpy` (bool),
  `gate_tolerance` (`1e-3` for the production `(B=64, b=4)` cell,
  `5e-3` otherwise), `is_production_cell` (bool),
  `baseline_degenerate` (bool — true when
  `baseline_frob < baseline_degenerate_eps`).
- **Summary in the MD file:** per-`(B, bits) × side` distribution
  (min / p50 / p95 / max), per-layer worst `|gap|` with K / V
  breakdown, absolute BlockTQ Frobenius mean / min / max per cell
  for D.2 cross-reference, baseline-degeneracy confirmation
  (expected: 144 / 144), and pointers to the test file and design
  doc.

The `side` axis is an explicit column per the K/V schema
guardrail — without it, (a-real) would be recorded as a K-only
gate even though V is also run; the schema makes K and V
independent observables.

Place under `P5_ACCEPTANCE_SWEEP/` rather than
`P5_D2_INVESTIGATION/` because (a-real) is an Acceptance
close-evidence artifact (it closes a v1.7.2-deferred half of
Acceptance item (4-a)'s real-activation half), not an
investigation record.

## 4. Test file layout

`tests/test_block_tq_real_activation_xcheck.py`:

1. Module-level imports + `_SKIP_A_REAL` gate via `_hf_cache_has_repo`.
2. Fixture: load Qwen3.5-0.8B adapter once, run forward, capture
   pre-RoPE K / V per GLOBAL layer into a dict keyed by layer
   index. Capture mechanism: temporarily hook `self_attn.k_proj` /
   `self_attn.v_proj` on each GLOBAL layer for the duration of the
   fixture to record the output, then restore — same patching
   idiom `silica/bench/ppl_oracle.py::_install_vqbench_wrappers`
   uses for the D.2a vqbench-aligned oracle (but read-only
   capture, not mutation).
3. Parametrized test over `(B, bits) ∈ {(32, 3), (32, 4), (64, 3),
   (64, 4)}` × `seed ∈ {42, 43, 44}`.
4. Per-layer loop within each parameter cell: compute silica
   Frobenius, NumPy-reference Frobenius, baseline Frobenius;
   assert the two gates in §2.5; record a row into the JSONL
   evidence file.
5. Worst-case summary assertion across layers (sanity check that
   no layer blows the bound).

Test-runtime cost estimate: Qwen3.5-0.8B load ~30-60 s on M5 Pro,
one forward pass on 128 tokens ~1-2 s, **6 GLOBAL layers**
(measured 2026-04-24 at indices `[3, 7, 11, 15, 19, 23]` on the
24-hidden-layer Qwen3.5-0.8B with `full_attention_interval=4`) ×
2 sides (K, V) × 4 (B, bits) × 3 seeds = **144 codec-round pairs**
(silica + NumPy reference), each round ~10 ms on a small
`(1, n_kv_heads=2, T=138, head_dim=256)` tensor. Total ~1-2
minutes per invocation. Gateable via the HF-cache gate; skipped
on CI.

## 5. Docs update in the same commit

Per advisor feedback, the §7(a-real) wording must flip at the
same time as the implementation lands, or docs will describe a
subprocess path the code doesn't take. Three surfaces to update:

- **`docs/P5_OPENING.md` §7(a-real) body** — rewrite the
  "vqbench venv subprocess + new recon-specific driver script"
  paragraph to describe the inline-NumPy pipeline + the
  `IdentityCodec` baseline; update the "dual-gate" → "single-gate"
  sentence.
- **`docs/PLAN.md` §7 P-5 Notes (a-real) bullet** — mirror the
  subprocess → inline-NumPy switch; the `silica.bench.vqbench_baseline`
  subprocess driver is no longer the infrastructure (a-real)
  stands on.
- **`docs/PLAN.md` §13 Changelog** — add v1.7.5 entry covering
  the (a-real) close, the evidence file, the §7(a-real) wording
  correction, and the §7 Status update from
  "post-P-5 follow-up: Qwen3.5 real-target xcheck (a-real) /
  (b-static)" to "(a-real) closed at v1.7.5; (b-static) pending
  P-3-C5".

Version bump: **v1.7.4 → v1.7.5** (minor, docs + new test, no
runtime code change).

## 6. Explicitly out of scope for this unit

- **(b-static) Qwen3.5-4B end-to-end PPL vs vqbench REPORT.md
  static baseline.** Stays in backlog per PLAN §7 Notes; blocked
  on P-3-C5 or on the monkey-patch route. (a-real) does not
  unlock (b-static).
- **Post-RoPE K xcheck.** The production prefix-cache store's
  post-RoPE K space is a different quality question owned by
  P-5-F. (a-real) deliberately extracts pre-RoPE K so it feeds
  the P-5-F verification pipeline; a symmetric post-RoPE
  (a-real-postrope) test could be added later if P-5-F needs it.
- **Qwen3.5-4B / Qwen3.5-35B-A3B extensions.** Design uses 0.8B
  as the minimal target per v1.7.2 wording "Qwen3.5-0.8B (or
  larger)". Extending to 4B / 35B-A3B is natural but not required
  for this close; adds HF-cache weight download (4B ≈ 8 GB, 35B
  ≈ 20 GB) which changes the gate calculus. Leave as
  parametrizable `--repo` escape hatch but default to 0.8B only.

## 7. Implementation pause point

Ship this design doc. User reviews §2 decisions (especially §2.2
pre-RoPE extraction space, §2.4 baseline choice, §2.5 tolerance
reuse, §2.7 checked-in prompt). On approval, implement in this
order:

1. `tests/test_block_tq_real_activation_xcheck.py` skeleton —
   fixture + first parametrize cell (`B=64, b=4, seed=42`), no
   layer loop yet. Includes the checked-in prompt constant from
   §2.7.
2. Run the skeleton, observe the baseline measurement and first
   layer's silica-vs-numpy gap; adjust if §2.4 degeneracy
   prediction is wrong or if tolerance reuse is off.
3. Expand to full layer loop × (B, bits) parameter cells × 3 seeds.
4. Land evidence file `docs/P5_ACCEPTANCE_SWEEP/real_activation_xcheck.md`.
5. Update P5_OPENING §7(a-real) + PLAN §7 Notes + v1.7.5
   Changelog in the same commit.

One pause point between step 2 and step 3 — confirm the
measurement shape before committing to the full evidence sweep.

**Implementation checklist (minor fixes piggy-backed on the landing commit):**

- **`tests/test_block_tq_vqbench_xcheck.py` module docstring
  update.** Current top-of-file wording says "(a-real): real
  Qwen3.5-0.8B activations vs vqbench subprocess — defers to P-5-C
  where the bench harness already owns the subprocess + HF-cache
  + model-load plumbing." This is stale on two counts: (1)
  (a-real) does not defer to P-5-C (P-5 closed at v1.7.4); (2)
  (a-real) does not use a vqbench subprocess (§2.3 supersedes
  that). Rewrite to point at the new
  `tests/test_block_tq_real_activation_xcheck.py` and name the
  inline-NumPy reference reuse.
- **Layer count — measured and corrected.** §4 previously
  estimated "~12 full-attention layers" and the earlier
  guardrail suggested "~8". Actual: **6 GLOBAL layers** at
  indices `[3, 7, 11, 15, 19, 23]` on the
  24-hidden-layer Qwen3.5-0.8B (`full_attention_interval=4`).
  §4 updated in the same commit. 6 layers × 2 sides × 4 (B,
  bits) × 3 seeds = 144 evidence rows.
