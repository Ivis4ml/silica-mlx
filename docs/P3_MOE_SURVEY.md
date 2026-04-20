# P-3-E0 MoE Survey

Scope: mlx-lm's MoE plumbing for the two P-3 targets (Qwen3.5-35B-A3B
and Gemma4-26B-A4B), combined with Phase 1 metadata from the real
quantized 4-bit checkpoints. Purpose: record the data needed to
design E1 (`Qwen3_5MoeAdapter` / `Gemma4MoeAdapter`), E2 (D-011
per-expert call-path unit tests), and E3 (real-model forward smoke)
without having to re-derive anything on the fly, and to surface the
architectural questions that E1 must answer before code lands.

Counterpart of `docs/P3_DELTANET_SURVEY.md` (C-track) and
`docs/P3_GEMMA4_SURVEY.md` (D-track dense). Empirical evidence comes
from two `--run-forward`-disabled probes run on 2026-04-20
(`scripts/probe_qwen3_5_moe_load.py` and
`scripts/probe_gemma4_moe_load.py`). The probes use
`silica.models.factory.adapter_from_loaded_model` so each checkpoint
is loaded exactly once.

## 1. Summary

| Family | Repo probed | model_type | num_layers | num_experts | top_k | moe_mid | extra FFN branch alongside experts |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Qwen3.5-MoE | `mlx-community/Qwen3.5-35B-A3B-4bit` (20.4 GB) | `qwen3_5_moe` | 40 | 256 | 8 | 512 | shared-expert MLP, sigmoid-gated (`shared_expert_intermediate_size=512`) |
| Gemma4-MoE | `mlx-community/gemma-4-26b-a4b-4bit` (15.6 GB) | `gemma4` | 30 | 128 | 8 | 704 | always-on dense MLP, ungated sum |

Both families route through `mlx_lm.models.switch_layers.SwitchGLU`
with `gather_mm` — **all experts' weights are stacked into one
tensor per layer and dispatched internally**. This is the load-bearing
observation for the D-011 call-path tension (§4.1).

Factory dispatch:

- Qwen3.5-MoE reports `model_type="qwen3_5_moe"` — distinct from the
  dense Qwen3.5 key, so a new `_ADAPTERS` entry is enough.
- **Gemma4-MoE reports `model_type="gemma4"` — identical to
  Gemma4-dense.** Factory keying on `model_type` alone is ambiguous;
  the existing `Gemma4Adapter.__init__` guard catches the MoE
  checkpoint via `enable_moe_block` but E1 needs a real dispatch
  split. §4.2.

## 2. mlx-lm MoE plumbing

### 2.1 Qwen3.5-MoE module tree

```
mlx_lm/models/qwen3_5_moe.py   (53 lines)
  class Model(Qwen3_5Model):
      def sanitize(weights):  # remap experts.{gate_up,down}_proj ->
                              # switch_mlp.{gate,up,down}_proj for SwitchGLU

mlx_lm/models/qwen3_5.py       (reused by qwen3_5_moe)
  class DecoderLayer:
      if args.num_experts > 0:
          self.mlp = SparseMoeBlock(args)     # <- from qwen3_next
      else:
          self.mlp = MLP(args.hidden_size, args.intermediate_size)

mlx_lm/models/qwen3_next.py
  class Qwen3NextSparseMoeBlock(nn.Module):
      self.gate = nn.Linear(dim, num_experts)
      self.switch_mlp = SwitchGLU(dim, moe_intermediate, num_experts)
      self.shared_expert = Qwen3NextMLP(dim, shared_expert_intermediate)
      self.shared_expert_gate = nn.Linear(dim, 1)

      def __call__(self, x):
          gates = softmax(self.gate(x))
          k = self.top_k
          inds = argpartition(gates, -k)[..., -k:]
          scores = take_along_axis(gates, inds)
          scores = scores / scores.sum() if norm_topk_prob else scores
          y = self.switch_mlp(x, inds)
          y = (y * scores[..., None]).sum(-2)
          shared_y = self.shared_expert(x)
          shared_y = sigmoid(self.shared_expert_gate(x)) * shared_y
          return y + shared_y
```

Key properties:

- `qwen3_5_moe.Model` inherits `qwen3_5.Model` whole-cloth. Hybrid
  DeltaNet + GQA attention handling (C-track) is reused as-is. The
  MoE addition is purely on the FFN side.
- `DecoderLayer.is_linear = (layer_idx + 1) % full_attention_interval != 0`
  — on the 35B-A3B config (`full_attention_interval=4`) this yields
  the familiar `[D, D, D, G, D, D, D, G, ...]` 3:1 pattern: 30
  DeltaNet + 10 full-attention over 40 layers.
- The shared-expert branch is Qwen3-Next-style and non-standard for
  MoE transformers: every token passes through a small dense MLP
  gated by a scalar sigmoid, added to the sparse expert output.

### 2.2 Gemma4-MoE module tree

```
mlx_lm/models/gemma4_text.py   (sole module; no separate gemma4_moe)
  class DecoderLayer:
      self.mlp = MLP(config, layer_idx)            # ALWAYS constructed
      self.pre_feedforward_layernorm  = RMSNorm(...)
      self.post_feedforward_layernorm = RMSNorm(...)

      self.enable_moe = config.enable_moe_block    # model-wide bool
      if self.enable_moe:
          self.router   = Router(config)
          self.experts  = Experts(config)
          self.post_feedforward_layernorm_1 = RMSNorm(...)  # for MLP branch
          self.post_feedforward_layernorm_2 = RMSNorm(...)  # for experts branch
          self.pre_feedforward_layernorm_2  = RMSNorm(...)  # for experts branch

      def __call__(self, h):
          if self.enable_moe:
              # BOTH branches run; dense MLP is NOT replaced.
              h1 = self.post_feedforward_layernorm_1(self.mlp(
                       self.pre_feedforward_layernorm(h)))
              top_k_indices, top_k_weights = self.router(h)
              h2 = self.post_feedforward_layernorm_2(self.experts(
                       self.pre_feedforward_layernorm_2(h),
                       top_k_indices, top_k_weights))
              h = h1 + h2                         # ungated sum
          else:
              h = self.mlp(self.pre_feedforward_layernorm(h))
          h = self.post_feedforward_layernorm(h)
          # (residual addition happens outside this block)

  class Router:
      self.proj = nn.Linear(hidden_size, num_experts)
      self.scale = mx.ones((hidden_size,))         # RMSNorm-style scale
      self.per_expert_scale = mx.ones((num_experts,))

      def __call__(self, x):
          x = fast.rms_norm(x, scale * root_size)
          expert_scores = self.proj(x)
          top_k_indices = argpartition(expert_scores, -top_k_experts)[..., -top_k_experts:]
          top_k_weights = softmax(take_along_axis(expert_scores, top_k_indices))
          top_k_weights = top_k_weights * per_expert_scale[top_k_indices]
          return top_k_indices, top_k_weights

  class Experts:
      self.switch_glu = SwitchGLU(hidden_size, moe_intermediate_size,
                                  num_experts, activation=GeGLU())

      def __call__(self, x, top_k_indices, top_k_weights):
          w = expand_dims(top_k_weights, -1)
          y = self.switch_glu(x, top_k_indices)
          return (w * y).sum(-2)
```

Key properties vs Qwen3.5-MoE:

- **Always-on dense MLP branch ADDED to the MoE experts branch,
  ungated.** Gemma4-MoE does not replace the dense MLP with
  `SparseMoeBlock` (as Qwen3.5-MoE does). Instead every token
  runs through BOTH the dense MLP *and* the top-k expert branch,
  with the two outputs summed (no sigmoid gate, no learned
  mixing weight). The `self.mlp = MLP(...)` line is constructed
  unconditionally (`gemma4_text.py:277`); the MoE branch adds
  three extra layernorms and the Router/Experts pair. Contrast
  with Qwen3-Next's shared-expert branch (Qwen3.5-MoE inherits
  this), which is a smaller dense MLP gated by
  `sigmoid(shared_expert_gate(x))`.
- Router has a pre-projection RMSNorm-style scale and a
  `per_expert_scale` bias after softmax — slightly different
  scoring arithmetic than Qwen3.5's plain softmax-then-topk.
- Activation for the experts branch is `GeGLU` (GELU-gated)
  rather than SwiGLU — mirrors the dense Gemma4 MLP's
  `geglu(gate_proj, up_proj)` convention.
- MoE is layer-local: every decoder layer checks `enable_moe_block`.
  Since the flag is a single bool on the config (not per-layer),
  every layer gets MoE when the flag is set — including sliding-
  window layers. This is important for E1: MoE co-exists with
  both SLIDING and GLOBAL attention kinds on Gemma4-26B-A4B.

### 2.3 Shared primitive: `SwitchGLU` + `gather_mm`

`mlx_lm/models/switch_layers.py` contains the shared sparse-MLP
machinery used by both families:

```
class SwitchLinear:
    # One 3D tensor of shape (num_experts, output_dims, input_dims)
    # holds ALL experts' weights stacked along dim 0.

    def __call__(self, x, indices, sorted_indices=False):
        # _gather_sort(x, indices) to cluster tokens per expert,
        # gather_mm against self.weight, _scatter_unsort to restore order.
        ...

class SwitchGLU(nn.Module):
    # Wraps three SwitchLinear (gate_proj, up_proj, down_proj) +
    # an activation function. Forward: sparse two-step GLU per token.

    def __call__(self, x, indices):
        x_gate = self.gate_proj(x, indices)
        x_up = self.up_proj(x, indices)
        x = self.activation(x_gate, x_up)         # SwiGLU or GeGLU
        x = self.down_proj(x, indices)
        return x
```

Two load-bearing facts for §4.1:

1. Expert weights are a **single stacked 3D tensor** per layer, not
   `num_experts` independent tensors. Loading / indexing is by
   `expert_id` as an axis, not as a dict key.
2. Dispatch happens **inside SwitchGLU via `gather_mm`** on the
   activated `top_k` indices. The caller does not enumerate experts;
   it passes the indices and lets the kernel fuse the per-expert
   matmul into one call.

`QuantizedSwitchLinear` (switch_layers.py:27) is the quantized
counterpart used when the checkpoint is 4-bit. Same stacking
semantics; the 4-bit weights are held as one quantized tensor with
an expert-indexed dispatch path.

## 3. Real-checkpoint findings

### 3.1 Qwen3.5-35B-A3B-4bit (probe 2026-04-20)

Config highlights (via `args.text_config`):

| Field | Value | Note |
| --- | --- | --- |
| `model_type` | `qwen3_5_moe` | distinct from dense `qwen3_5` |
| `num_hidden_layers` | 40 | |
| `hidden_size` | 2048 | smaller than 27B's 5120 |
| `num_attention_heads` | 16 | |
| `num_key_value_heads` | 2 | GQA 8:1 |
| `head_dim` | 256 | |
| `full_attention_interval` | 4 | 3 DeltaNet + 1 full per block → 30 D + 10 G |
| `num_experts` | 256 | |
| `num_experts_per_tok` | 8 | top-k = 8 |
| `moe_intermediate_size` | 512 | per-expert hidden |
| `shared_expert_intermediate_size` | 512 | shared-expert hidden, equal to MoE |
| `norm_topk_prob` | key absent in probe; `qwen3_5.TextModelArgs` dataclass default is `True` | runtime value is `True` → Qwen3NextSparseMoeBlock applies `scores / scores.sum()` on the top-k scores after `take_along_axis`. `qwen3_next.ModelArgs` has the opposite default `False`, so a probe that only reads config keys can look ambiguous — check the dataclass the wrapper actually extends (`qwen3_5.TextModelArgs` in this case). |
| `vocab_size` | 248044 | same tokenizer as dense Qwen3.5 |

Extra config keys surfaced by the probe, worth flagging for E1:

- `mlp_only_layers` — see §4.4; the config carries this list but
  `qwen3_5.DecoderLayer` does **not** consult it (only
  `qwen3_next.Qwen3NextDecoderLayer` does). If the list is
  non-empty on a Qwen3.5-MoE checkpoint, layers in that list
  should be dense MLPs; mlx-lm's qwen3_5 decoder would silently
  wire MoE onto them.
- `mtp_num_hidden_layers`, `mtp_use_dedicated_embeddings` —
  multi-token-prediction heads that Qwen3-Next checkpoints carry
  as a post-base-model extension. The `mtp.` weight prefix is
  recognised by the qwen3_next sanitize path but is orthogonal to
  MoE. Flag for E-open-*.
- `attn_output_gate` — Qwen3-Next attention gate. The qwen3_5
  `Attention` class may or may not honour it; needs a source check
  before E1.
- `router_aux_loss_coef` — training-only; inference ignores it.
- `linear_*` (`linear_conv_kernel_dim`, `linear_key_head_dim`,
  `linear_num_key_heads`, `linear_num_value_heads`,
  `linear_value_head_dim`) — DeltaNet hyperparameters. These are
  `Qwen3_5Adapter.config.extra`-worthy for observability but do
  not change the E1 MoE work.
- `mamba_ssm_dtype` — DeltaNet internal dtype (fp32 expected, per
  C-track finding).

Per-layer breakdown (derived from `full_attention_interval=4` and
`num_hidden_layers=40`):

- 30 × `HYBRID_DELTANET` layers
- 10 × `GLOBAL` (full attention) layers
- Every layer with `num_experts > 0` is MoE (per qwen3_5
  `DecoderLayer.__init__`), regardless of attention kind — unless
  `mlp_only_layers` overrides it (§4.4, unverified).

### 3.2 Gemma4-26B-A4B-4bit (probe 2026-04-20)

Config highlights:

| Field | Value | Note |
| --- | --- | --- |
| `model_type` | `gemma4` | **same as Gemma4-31B-dense** (§4.2) |
| `num_hidden_layers` | 30 | |
| `hidden_size` | 2816 | |
| `num_attention_heads` | 16 | |
| `num_key_value_heads` | 8 | sliding |
| `num_global_key_value_heads` | 2 | full (dense 31B had 4) |
| `head_dim` | 256 | sliding |
| `global_head_dim` | 512 | full |
| `sliding_window` | 1024 | same as 31B |
| `sliding_window_pattern` | absent / `None` | 31B set 5; here missing, but `layer_types` is authoritative |
| `attention_k_eq_v` | True | full-attention layers share K/V weights (but not caches at runtime) |
| `num_kv_shared_layers` | 0 | no cross-layer KV sharing |
| `hidden_size_per_layer_input` | 0 | not a 2B/4B per-layer-input variant |
| `enable_moe_block` | True | MoE on |
| `num_experts` | 128 | |
| `top_k_experts` | 8 | |
| `moe_intermediate_size` | 704 | per-expert hidden |
| `vocab_size` | 262144 | same tokenizer as 31B-dense |

Per-layer breakdown (from `layer_types` counts):

- 25 × `SLIDING` layers
- 5 × `GLOBAL` (full attention) layers
- 5:1 sliding-to-full ratio preserved; pattern within `layer_types`
  mostly repeats `[S, S, S, S, S, F]` (first 8 observed:
  `S, S, S, S, S, F, S, S`).
- **Every layer is MoE** because `enable_moe_block=True` is a
  single bool on the config, not per-layer — MoE co-exists with
  both SLIDING and GLOBAL attention kinds.

D4 `KVLayout.bytes_per_token_total` for 26B-A4B (bfloat16):

```
  25 sliding × 2 (K+V) × 8 kv_heads × 256 head_dim × 2 bytes = 204,800
  + 5 full   × 2 (K+V) × 2 kv_heads × 512 head_dim × 2 bytes =  20,480
  -------------------------------------------------------------------
                                                              225,280 bytes/token
```

Less than a quarter of Gemma4-31B's 901,120 bytes/token — consistent
with the smaller model + narrower full-attention shape. The D4 field
applies uniformly; no new layout work is needed for 26B-A4B beyond
populating it in `Gemma4MoeAdapter._build_kv_layout`.

## 4. Silica integration concerns

### 4.1 D-011 per-expert call path vs fused `SwitchGLU`

D-011 (PLAN.md) requires the MoE adapter's forward pass to invoke
`WeightProvider.get_expert(layer_idx, expert_id)` for each activated
top-k expert, and **not** use `get_layer(layer_idx)` to load whole
layers' expert weights. This makes P-6 per-expert weight streaming
observable via a mock-`WeightProvider` unit test.

mlx-lm's MoE path does the opposite: `SwitchGLU` holds all experts
as one stacked tensor and dispatches internally via `gather_mm`.
Silica has three options, each with tradeoffs:

**(a) Fused wrapper, D-011 literal compliance deferred.** Treat
`SwitchGLU` as an opaque component; the adapter calls `get_layer`
once per layer and the whole switch-mlp weight tensor goes through.
Fast, matches mlx-lm's quantized fast path (`QuantizedSwitchLinear`
+ `gather_mm`). Violates the letter of D-011; P-6 per-expert
streaming would require unfurling the stacked tensor per-access
which loses the `gather_mm` benefit.

**(b) Per-expert dispatch, D-011 literal compliance.** Replace
`SwitchGLU` with a custom routing layer that walks the activated
top-k indices and calls `get_expert(layer_idx, expert_id)` once per
activation. Slower (no `gather_mm` fusion; loses quantized fast
path for 4-bit). Matches D-011 verbatim; P-6 per-expert streaming
drops in naturally.

**(c) Hybrid: fused at inference, per-expert at the unit-test
seam.** The adapter's forward uses `SwitchGLU` for speed; the I-4
`WeightProvider` interface retains `get_expert` for P-6, and the
adapter threads top-k indices through a thin shim that still lets
the D-011 unit test assert "`get_expert` was called for every
activated expert" even though the underlying weight fetch goes
through a single `get_layer`-equivalent. This is a redefinition of
D-011 — "the **dispatch** uses per-expert ids, regardless of
whether the **fetch** is fused."

Recommended for E1 / E2: **(c)**. Reasoning:

- Option (a) loses D-011 as a testable invariant, which is
  PLAN.md's explicit ask for MoE structural correctness.
- Option (b) sacrifices the quantized fast path that matters most
  on the 35B-A3B / 26B-A4B checkpoints (both are 4-bit, both
  depend on `QuantizedSwitchLinear + gather_mm` for real
  throughput).
- Option (c) preserves inference throughput AND the unit-testable
  per-expert-index invariant. The interpretation of D-011 becomes
  "per-expert at the dispatch level," which matches how P-6
  streaming would decide what to pull in — streaming needs
  `(layer_idx, expert_id)` observability, not one-fetch-per-expert
  granularity.

Option (c) requires a **WeightProvider API clarification** —
specifically, whether `get_expert(layer_idx, expert_id)` returns a
view into a larger stacked tensor (allowed under option (c)) or a
freshly allocated per-expert tensor (required under option (b)).
Flagged as E-open-1.

### 4.2 Factory dispatch: Gemma4-MoE shares `model_type` with dense

`adapter_for_repo` keys on `model.model_type`. The Qwen3.5-MoE
checkpoint reports `qwen3_5_moe` (clean); the Gemma4-MoE checkpoint
reports `gemma4` (collides with Gemma4-31B-dense). Today this is
"handled" only by `Gemma4Adapter.__init__` raising
`NotImplementedError` when `enable_moe_block=True`.

Three options for E1:

**(i) Secondary dispatch inside `_build_gemma4`.** Inspect
`enable_moe_block` / `num_experts` on the loaded model and return
`Gemma4Adapter` or `Gemma4MoeAdapter` accordingly. Keeps the
`_ADAPTERS` map simple; adds a family-local decision.

**(ii) Two-key factory.** Widen `_ADAPTERS` from
`dict[str, Builder]` to `dict[tuple[str, ...], Builder]` where the
tuple carries `(model_type, extra_flag)`. More principled; changes
the public dispatch API.

**(iii) Register builder that returns the right adapter itself.**
`_build_gemma4` already receives `model` / `tokenizer` — it can
branch on `getattr(model.args.text_config, "enable_moe_block", False)`
before constructing. This is option (i) phrased more carefully.

Recommended for E1: **(i) / (iii)**. Option (ii) is a larger
factory refactor that no other family needs today; Gemma4's
single-outer-model_type split is the only known case. A local
branch inside `_build_gemma4` keeps the blast radius contained.

### 4.3 `KVLayout.bytes_per_token_total` for MoE checkpoints

D4 already set up an optional `bytes_per_token_total: int | None`
field on `KVLayout` and has `MemoryBudgeter.for_adapter` prefer it
when set (commit `f9664cf`). The field is orthogonal to MoE
routing — it describes the KV cache alone.

- **Qwen3.5-MoE (35B-A3B)**: GQA, 2 KV heads × 256 head_dim × 40
  layers × fp16 = `2 * 40 * 2 * 256 * 2 = 81,920` bytes/token. The
  dense `Qwen3_5Adapter` formula gives the same number for an
  equivalent GQA shape; E1 need NOT populate `bytes_per_token_total`
  — the fallback formula is already correct. (Recurrent state on
  DeltaNet layers lives outside KV; the budgeter accounts for it
  via `StateDelta.recurrent_bytes()`, not `KVLayout`.)
- **Gemma4-MoE (26B-A4B)**: heterogeneous KV shape (sliding 8×256
  + full 2×512). E1 **MUST** populate `bytes_per_token_total`
  using the per-kind sum (§3.2 gives 225,280 bytes/token) because
  the `Gemma4Adapter` D4 logic is the reference. `Gemma4MoeAdapter`
  should reuse or refactor the existing `_build_kv_layout`
  helper — not re-derive.

### 4.4 Qwen3.5-MoE `mlp_only_layers` is unhandled by qwen3_5 decoder

The 35B-A3B config exposes `mlp_only_layers` in `text_config_keys`,
but `mlx_lm/models/qwen3_5.py::DecoderLayer.__init__` does **not**
consult it — only `qwen3_next.py::Qwen3NextDecoderLayer` does. The
qwen3_5_moe wrapper extends qwen3_5's decoder, so:

- If the 35B-A3B checkpoint's `mlp_only_layers` is **empty**, this
  is a benign inherited config field.
- If the 35B-A3B checkpoint's `mlp_only_layers` is **non-empty**,
  mlx-lm's current Qwen3.5-MoE wires `SparseMoeBlock` onto layers
  that should be dense MLPs. That is a potential upstream bug, but
  it would also silently affect any Silica adapter that assumes
  "every layer is MoE when `num_experts > 0`."

The probe did not dump the field's contents (only its key presence).
Flagged as **E-open-2**: verify `mlp_only_layers` contents before E1
wires its adapter forward; if non-empty, either upstream-patch
qwen3_5.DecoderLayer or handle it adapter-side.

### 4.5 Multi-token prediction (MTP) — orthogonal but surfaced

The 35B-A3B config carries `mtp_num_hidden_layers` and
`mtp_use_dedicated_embeddings`. `qwen3_next.sanitize_weights` has
an explicit `has_mtp_weights = any("mtp." in k for k in weights)`
branch (qwen3_next.py:308). If this checkpoint ships MTP weights,
the base-model forward is unchanged but weight loading emits extra
parameters.

E1 impact: probably none — Silica only consumes the base-model
forward. Flagged as **E-open-3** so that a future speculative
decoding track (P-7) that wants to exploit MTP heads does not
silently discover them.

## 5. Proposed E1 / E2 / E3 roadmap

Down-weighted where D-track work already generalises:

| Subunit | Scope | Est. size | Depends on |
| --- | --- | --- | --- |
| **E1-Qwen3_5MoE** | New `silica/models/qwen3_5_moe.py` extending `Qwen3_5Adapter`. Variant guard (`has_moe=True`, per-expert call-path hook per §4.1 option (c)). Factory `_ADAPTERS["qwen3_5_moe"]`. Fake-model unit tests: capabilities / variant guard / factory dispatch. `make_batch_cache` inherits from the dense parent; hybrid DeltaNet + KV routing unchanged from C-track. | ~200 lines + ~10 tests | §4.1 decision |
| **E1-Gemma4MoE** | New `silica/models/gemma4_moe.py` extending `Gemma4Adapter`. Same shape as E1-Qwen3.5-MoE. Factory dispatch via option (i) — local branch inside `_build_gemma4` consulting `enable_moe_block`. `make_batch_cache` inherits the D2/D3 sliding+full hybrid. `_build_kv_layout` populates `bytes_per_token_total` using the dense adapter's per-kind formula (§4.3). Variant guard accepts the MoE config that the dense adapter currently rejects. | ~200 lines + ~10 tests | §4.2 decision, D4 |
| **E2** | D-011 per-expert call-path unit test. Mock `WeightProvider` asserts that MoE forward **dispatches** with per-expert indices (option (c) interpretation), even if the underlying `SwitchGLU` fetches a single stacked tensor. Applies to both E1 adapters. | ~100 lines of test + possibly a `WeightProvider.get_expert` shim on `ResidentWeightProvider` | E1 both |
| **E3-smoke** | Real-model forward smoke via `Engine.generate` (single-request, greedy, `max_tokens=4`) on each MoE target, dual-gated on HF cache + env var. Mirrors the D1.1 + D3 smoke pattern. Both checkpoints are already downloaded (`Qwen3.5-35B-A3B-4bit` 20.4 GB, `gemma-4-26b-a4b-4bit` 15.6 GB). | ~2 tests × 2 families = ~4 tests | E1 both, E2 |

Out of scope for E:

- E1 adapters do **not** add new batched-cache work. The
  `make_batch_cache` factory inherited from the dense parent is
  correct for both MoE families because it only decides per-layer
  KV cache types (SLIDING → `BatchRotatingKVCache`, GLOBAL →
  `BatchKVCache`, HYBRID_DELTANET → `ArraysCache`); the MoE FFN is
  not involved. The FFN itself still runs inside the model's
  forward on both single-request and batched paths — batched does
  not skip it. Batched smoke + parity pinning for MoE is its own
  follow-up (call it E4, mirroring C3d / D3.1 on the attention
  side) and is not planned today because D-011's P-3 ask is
  structural correctness + one forward per family.
- Expert weight streaming (P-6) is explicitly future. E1 / E2
  need only to expose the `(layer_idx, expert_id)` dispatch seam;
  real streaming lands later.

## 6. Open questions (local to E track)

Using the local `E-open-*` convention, not the global `Q-NNN`
space.

### E-open-1 — `WeightProvider.get_expert` return-value semantics

Does `get_expert(layer_idx, expert_id)` return an independent
tensor (option 4.1(b) — required for literal D-011 compliance) or
a view into a larger stacked tensor (option 4.1(c) — preserves
SwitchGLU fast path)? PLAN.md's §D-011 text does not spell this
out; E1 will make the call based on whether the quantized fast
path matters for the P-3 exit criterion. **Recommendation in §4.1:
option (c).** Needs sign-off before E2 writes the unit test.

### E-open-2 — `mlp_only_layers` contents on 35B-A3B checkpoint

The probe shows the key is present in `text_config_keys` but
does not dump its contents. The qwen3_5 DecoderLayer ignores it;
if non-empty, mlx-lm is silently wiring MoE onto layers that
should be dense. Action: rerun the probe with a one-line
addition dumping `_probe_attr(args, "mlp_only_layers")` — or
read it once via a throwaway `python -c` so the 20 GB load pays
for itself twice. If non-empty, file upstream + have E1 guard.

### E-open-3 — MTP weights and base-model forward

If `mtp_num_hidden_layers > 0` and the checkpoint ships `mtp.`
weights, does Silica's base-model forward (which ignores MTP)
produce the same tokens as a reference that includes MTP? For
P-3 this is a question about speculative decoding (P-7), not the
MoE forward. Flag and defer.

### E-open-4 — Gemma4-MoE `sliding_window_pattern` absence

The 26B-A4B config omits `sliding_window_pattern`, which the D1
adapter's `_build_attention_pattern` reads via
`text_config.get("sliding_window_pattern", 0) or 0`. The
authoritative source remains `layer_types`, which is present and
consistent (25 sliding + 5 full). No action needed for E1 beyond
confirming that `Gemma4MoeAdapter` inherits the `layer_types`
primary path, not a synthesised pattern from
`sliding_window_pattern`.

### E-open-5 — `attn_output_gate` and `qwen3_5.Attention`

Qwen3-Next introduced `attn_output_gate` as an additional gate in
the attention path. The 35B-A3B config sets it (presence in
`text_config_keys`). The qwen3_5 `Attention` class may or may
not honour the flag — needs a source check inside qwen3_5.py
before E1 concludes that the dense Qwen3.5 attention forward
applies unchanged to Qwen3.5-MoE. If the flag is honoured,
E1-Qwen3.5-MoE inherits the correct behaviour automatically; if
it is silently dropped, E1 must record the mismatch.

## 7. References

- mlx-lm source (read 2026-04-20):
  - `mlx_lm/models/qwen3_5_moe.py` (53 lines, wrapper)
  - `mlx_lm/models/qwen3_5.py` (DecoderLayer MoE branch at line 223)
  - `mlx_lm/models/qwen3_next.py` (Qwen3NextSparseMoeBlock at line 308)
  - `mlx_lm/models/gemma4_text.py` (Router at line 117, Experts at line 153)
  - `mlx_lm/models/switch_layers.py` (SwitchLinear / SwitchGLU / QuantizedSwitchLinear)
- Probe output (2026-04-20):
  - `mlx-community/Qwen3.5-35B-A3B-4bit` — 20.4 GB, `qwen3_5_moe`,
    load 274 s on M5 Pro.
  - `mlx-community/gemma-4-26b-a4b-4bit` — 15.6 GB, `gemma4`
    (shared key with dense!), load 247 s.
- Companion surveys:
  - `docs/P3_DELTANET_SURVEY.md` — C-track (hybrid DeltaNet)
  - `docs/P3_GEMMA4_SURVEY.md` — D0.2 dense Gemma4
  - `docs/P3_BATCH_ROTATING_KV_SURVEY.md` — D2.0 batched sliding KV
- PLAN references: `docs/PLAN.md` §D-011 (MoE structural correctness),
  §7 P-3-E deliverable list.
