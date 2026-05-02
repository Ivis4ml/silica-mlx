# P-6 Track B 3-bit Follow-up — Read-only Candidate Survey

**Date:** 2026-05-01
**Scope:** D-021 step 7 Track B follow-up after the v1.7.21
candidate retirement of `NexVeridian/Qwen3.5-27B-3bit`.
**Method:** Read-only HF Hub queries (`hf models ls`,
`hf models info`, model-card fetches) — no downloads, no
conversions.
**Timebox:** ≤ 1 hour HF-query effort.
**Deliverable:** This doc + closure decision.

---

## 1. Question

After B.2 retired `NexVeridian/Qwen3.5-27B-3bit` on a +16.85% PPL
drift, the user-confirmed framing for follow-up is: only run a
small read-only HF survey for Qwen3.5-27B 3-bit MLX-native
checkpoints that have **reason to believe lower drift** than the
NexVeridian recipe — activation-aware (AWQ / GPTQ / AutoRound /
DWQ), smaller group size (e.g. 32 instead of 64), or model-card
PPL/quality evidence. **No auto re-conversion** of the ~52 GB
full-precision Qwen3.5-27B weights from this branch.

**Disposition path:**

- **No viable candidate** → Track B fully retired pending future
  checkpoint; this doc is the audit trail.
- **Viable candidate exists** → Return for explicit download
  authorisation; no auto-pull.

## 2. Acceptance criteria for a viable candidate

A repo qualifies as a Track B B.2 re-attempt candidate iff all
five hold:

1. **Matched-family base model.** Base must be
   `Qwen/Qwen3.5-27B` (or its `*-Instruct` 1:1 weight
   equivalent). Distilled, abliterated, fine-tuned, or
   continued-pretrained variants are out — comparing those
   against the 4-bit anchor `mlx-community/Qwen3.5-27B-4bit`
   does not isolate the weight-bits delta the gate is testing.
2. **MLX-native loadable.** `library_name: mlx` (or a layout
   `silica.models.factory.adapter_for_repo` can read via
   mlx-lm). HF-only `transformers` repos require conversion,
   which is out of survey scope. GGUF is out.
3. **3-bit weight quantisation.** B.2 measures the 4-bit → 3-bit
   PPL drift; 4-bit and below-3-bit candidates do not test the
   step-7 hypothesis.
4. **Reason to expect lower drift than NexVeridian.** At least
   one of:
   - **Activation-aware** quantisation method (AWQ, GPTQ,
     AutoRound, DWQ) — uses calibration data to bias
     quantisation error toward less-active dimensions.
   - **Smaller group size** than mlx-lm default
     (`group_size=64`) — finer-granularity scales reduce
     intra-group reconstruction error.
   - **Model-card PPL / quality evidence** the candidate is
     measurably better than NexVeridian's recipe.
5. **Pre-declared gate unchanged.** The §6.1 B.2 both-pass gate
   (ΔPPL_abs ≤ 0.5 AND ΔPPL_rel ≤ 5%) is **not** to be relaxed
   to fit a new candidate; the candidate must look likely to
   pass the existing gate, not the other way around.

## 3. Survey method

```text
hf models ls --search "Qwen3.5-27B" --limit 50 --sort downloads
hf models ls --search "Qwen3.5-27B 3bit"
hf models ls --search "Qwen3.5-27B 3-bit"
hf models ls --author mlx-community --search "Qwen3.5-27B"
hf models ls --search "Qwen3.5-27B" --filter mlx --limit 50
hf models ls --search "Qwen3.5-27B AWQ 3"
hf models ls --search "Qwen3.5-27B GPTQ"
hf models ls --search "Qwen3.5-27B DWQ"
hf models ls --search "Qwen3.5-27B OptiQ"
hf models ls --author kaitchup --search "Qwen3.5-27B"
hf models ls --author Intel --search "Qwen3.5-27B"
hf models info <repo>  # for any candidate that surfaced 3-bit
```

Plus model-card fetches for the two MLX-native 3-bit candidates
identified at B.1 lookup (NexVeridian, RepublicOfKorokke) and the
single non-MLX 3-bit AWQ/GPTQ candidate (telvenes).

## 4. Findings table

| # | Repo | Bits | Library | Base | Method | Verdict |
|---|------|------|---------|------|--------|---------|
| 1 | `NexVeridian/Qwen3.5-27B-3bit` | 3 | mlx | matched | `mlx_lm.convert -q --bits 3` (gs=64) | **Retired at B.2 (PPL 8.07).** |
| 2 | `RepublicOfKorokke/Qwen3.5-27B-mlx-lm-3bit` | 3 | mlx | matched | `mlx_lm.convert -q --q-bits 3` (gs=64) | **Same recipe as #1; expected same drift.** No PPL evidence; no activation-aware step. |
| 3 | `ianleelamb/qwen3.5-27b-turboquant-3bit` | 3 | safetensors (NOT mlx) | matched | TurboQuant (`turboquant_config.json` only) | **Out:** Not MLX-loadable. mlx-lm has no TurboQuant loader; vqbench reference is NumPy-only and not in the silica runtime path (D-009). |
| 4 | `telvenes/Qwen3.5-27B-abliterated-GPTQ-3bit` | 3 | transformers (NOT mlx) | **abliterated** | GPTQ | **Out:** (a) abliterated base — not matched-family; (b) GPTQ-3bit Hugging Face safetensors are not directly mlx-lm loadable. |
| 5 | `enet45/Qwen3.5-27B-Claude-4.6-OS-...-mlx-3Bit` | 3 | mlx | **distilled** (Claude-Opus reasoning distill) | mlx-lm | **Out:** Not matched-family — comparing against `mlx-community/Qwen3.5-27B-4bit` does not isolate the weight-bits delta. |

**Adjacent findings (note-only — outside Track B scope):**

| Repo | Bits | Note |
|------|------|------|
| `mlx-community/Qwen3.5-27B-4bit-DWQ` | 4 | DWQ exists in MLX at 4 bits only; no public 3-bit DWQ Qwen3.5-27B. |
| `mlx-community/Qwen3.5-27B-OptiQ-4bit` | 4 | OptiQ is 4-bit only in this listing. |
| `mlx-community/Qwen3.5-27B-GPTQ-Int4` | 4 | MLX-community GPTQ exists at 4 bits, not 3. |
| `Intel/Qwen3.5-27B-int4-AutoRound` | 4 | AutoRound matched-family but 4-bit + transformers (not MLX). |
| `kaitchup/Qwen3.5-27B-NVFP4` / `MXFP4A16` | 4 (fp4) | Microscaling FP4 schemes — 4-bit-equivalent, not 3-bit. |
| `McG-221/Qwen3.5-27B-abliterated-mlx-gs32` | 8 | smaller `group_size=32` exists in the wild, but at 8-bit on an abliterated base. Confirms the gs32 knob is reachable; does not produce a 3-bit candidate. |

## 5. Conclusion

**No viable better-calibrated MLX-native 3-bit matched-family
Qwen3.5-27B candidate currently exists on HF Hub.** The two known
matched-family MLX 3-bit checkpoints both use the
`mlx_lm.convert -q --bits 3` recipe at default `group_size=64`
without any activation-aware calibration step. Activation-aware
methods (AWQ, GPTQ, AutoRound, DWQ, OptiQ) are well represented
for Qwen3.5-27B at **4 bits** in the MLX ecosystem — at 3 bits
they exist only in transformers/GPTQ form on abliterated bases
or as research-format artefacts (TurboQuant) without an
MLX-loadable path.

This rules out a B.2 re-attempt under the survey's acceptance
criteria. **Track B is therefore fully retired at v1.7.21 pending
a future checkpoint.**

## 6. Re-look triggers

A future survey is justified iff one of these surfaces:

1. **Any of the four well-represented 4-bit activation-aware MLX
   checkpoints** (`*-DWQ`, `*-OptiQ-4bit`, `*-GPTQ-Int4`,
   `*-AutoRound`) ships a **3-bit** sibling for Qwen3.5-27B.
   Trigger: an `hf models ls --filter mlx --search "Qwen3.5-27B
   3"` query returns a matched-family activation-aware result.
2. **mlx-lm's quantisation tooling adds an activation-aware
   path at 3 bits.** Currently `mlx_lm.convert -q --bits 3` is
   weight-only RTN at the default `group_size`. If MLX-LM
   upstream lands a calibration step (analogous to AWQ scale
   estimation), that motivates a re-conversion attempt with
   user authorisation for the ~52 GB base download.
3. **A Qwen3.5-27B model card publishes WikiText / MMLU PPL
   numbers for a 3-bit MLX-native checkpoint** that would
   plausibly clear the §6.1 B.2 gate. Trigger: explicit
   model-card evidence rather than inferring from method tags.

None of these is a deliverable for this survey — they are
watch-list items for ad-hoc future polling.

## 7. Closure

Track B native 3-bit weight-streaming lever is closed in the
same disposition as v1.7.21: candidate retired, gate not
relaxed, no auto-conversion. The `qwen3.5-27b-warm-decode-b1-3bit`
scenario stays in the catalog as the load-bearing artefact for
any future re-attempt — the row's `repo` field is the swappable
knob the moment a viable candidate appears.

Mainline next move stays the C.5 tree-shape spike (D-021 step 8)
for the (1b) ≥60 tok/s survival path, per the user's stated
ordering ("先 #2 read-only survey ... 然后开 #1 C.5 orientation").
This survey's outcome closes the #2 leg.
