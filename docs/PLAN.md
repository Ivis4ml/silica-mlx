# Silica-MLX Plan

| Field        | Value                                   |
| ------------ | --------------------------------------- |
| Version      | v1.5.0                                  |
| Last updated | 2026-04-16                              |
| Status       | Phase 0 planned                         |
| Maintainer   | Xin Zhou                                |
| Source       | `docs/PLAN.md` (single source of truth) |

> **CRUD convention.** All stable IDs in this document (Phase `P-N`, Decision `D-NNN`, Open Question `Q-NNN`, Milestone `M-N`, Interface `I-N`) are never reused once allocated. When editing, touch only the relevant block. New facts go in the Decisions Log; new questions in Open Questions. A Phase status change updates the `Status` field of that Phase block and appends a line to the Changelog.

---

## Table of Contents

1. [TL;DR](#1-tldr)
2. [Mission](#2-mission)
3. [Scope](#3-scope)
4. [Design Principles](#4-design-principles)
5. [Architecture Overview](#5-architecture-overview)
6. [Core Interfaces (Phase 0 freeze candidate)](#6-core-interfaces-phase-0-freeze-candidate)
7. [Phases](#7-phases)
8. [Priority & Milestones](#8-priority--milestones)
9. [Decisions Log](#9-decisions-log)
10. [Open Questions](#10-open-questions)
11. [Risks](#11-risks)
12. [References](#12-references)
13. [Changelog](#13-changelog)

---

## 1. TL;DR

**Silica-MLX is a single-Mac-chip local LLM inference platform.** MLX-native; vLLM-style core plus a mini-sglang-style outer layer. The goal is to run Qwen3.5-27B / Gemma4-31B reliably on a 48 GB M5 Pro. VQ, weight streaming, and speculative decoding are **native capabilities** actively exploited by the platform — not passively supported third-party extensions. Implementations are swappable, but integration points are fixed (see Principle 9).

**The platform itself is the product.** VQ and similar techniques are weapons, not research subjects (see D-006).

---

## 2. Mission

On a single Apple Silicon Mac, let developers run 27B–31B-class models locally with the same fluidity they get from vLLM in the cloud.

**Target users:** Mac developers who want to run large models locally for apps, experiments, or privacy-sensitive work.

**Not for:** VQ algorithm researchers, distributed-serving system researchers, cloud providers.

**Success criteria (v0.1):**
- Qwen3.5-27B and Gemma4-31B run reliably on a 48 GB M5 Pro.
- Python API + minimal CLI.
- Unified benchmark / runtime path (no separate eval codepath).
- VQ / weight streaming / speculative are native capabilities, built into the main loop as stubs from P-0, progressively replaced with real implementations in P-5 / P-6 / P-7. Integration points fixed, implementations swappable (Principle 9).
- Minimal OpenAI-compatible HTTP API + session available via Phase 8, aligned with §3.1 scope and the D-006 "product face" framing.

---

## 3. Scope

### 3.1 In Scope (v0.1)

- Single Mac chip, single-process local inference.
- MLX-native; no CUDA assumptions.
- Qwen3.5-27B and Gemma4-31B as production dense targets.
- Python API + minimal CLI.
- vLLM-style core: paged KV, continuous batching, prefix cache, memory budget.
- Five frozen interfaces: ModelAdapter / KVManager / KVCodec / WeightProvider / DraftEngine.
- Native capabilities: VQ KV compression (BlockTQ / RaBitQ), weight streaming, draft-target speculative — see Principle 9.
- **MoE + Dense architectural generality (D-011)**; v0.1 must actually run at least one MoE target, not just "interface reserved but untested".
- Unified benchmark path.
- Minimal OpenAI-compatible HTTP API + session (Phase 8).

### 3.2 Non-Goals (v0.1)

- Distributed, multi-node, multi-Mac cooperation.
- PD (prefill/decode) disaggregation.
- Tensor parallelism (not needed on a single chip).
- Hand-rolling native Metal kernels from scratch.
- Non-MLX quantization formats such as GGUF / AWQ.
- Full agent orchestration.
- Explicitly excluded codecs: PQ, OPQ.
- Compressed-domain attention fast path (see D-003, deferred to v0.2).
- Complex speculative schemes (DFlash / EAGLE / Medusa) — deferred to v0.2.
- **PyTorch runtime dependency** (D-009): the inference hot path may not contain `torch.Tensor`; torch is allowed only as an optional dev dependency for offline weight conversion.
- **CUDA / ROCm / XPU / TPU backends** (D-009): `csrc/`, CUDA kernels, and device-specific workers are out of scope.

### 3.3 Target Hardware

| Field        | Value                                              |
| ------------ | -------------------------------------------------- |
| Machine      | Apple M5 Pro                                       |
| Memory       | 48 GB unified memory                               |
| OS           | macOS                                              |
| Acceleration | MLX on Apple Silicon GPU + Neural Engine (via MLX) |

### 3.4 Target Models

| Model             | Parameters                   | Role                             | First used  |
| ----------------- | ---------------------------- | -------------------------------- | ----------- |
| Qwen3-0.6B        | 0.6B                         | Dev / iteration bring-up model   | Phase 1     |
| Qwen3.5-27B       | 27B                          | Dense production target          | Phase 3     |
| Gemma4-31B        | 31B                          | Dense production target          | Phase 3     |
| Qwen3.5-35B-A3B   | 35B total / 3B active        | MoE generality target (D-011)    | Phase 3     |
| gemma-4-26B-A4B   | 26B total / 4B active        | MoE generality target (D-011)    | Phase 3     |

---

## 4. Design Principles

Stable principles. Changing any of them requires a new Decisions Log entry.

1. **Platform as product, VQ as weapon.** Silica-MLX itself is the product. VQ / weight streaming / speculative are means to make the product run large models well, not research subjects. Do not let "VQ is the ultimate deliverable" creep into any design (D-006).

2. **Single Mac chip + Apple unified memory first.** The hard constraint is a single chip; no distribution. But *actively* exploit unified memory — the fact that weights, KV, and activations share one physical pool must be reflected in WeightProvider and KVManager design. Do not abstract the Mac as a generic GPU.

3. **Engine skeleton first, native capabilities integrated progressively.** Phase 0–4 bring up the engine skeleton + baseline + target models (the main loop already contains stubs of every native capability from P-0: `IdentityCodec` / `ResidentWeightProvider` stub / `NoopDraftEngine`). Phase 5–8 progressively replace the stubs with real implementations (VQ / streaming / speculative) and add the serving layer. This is **not** "build the engine, then add plugins" — integration points exist from P-0; P-5..P-7 replace stubs, they do not wire in anything new (see Principle 9).

4. **Bench and runtime share the same path.** A benchmark must be a thin wrapper over `silica.engine.Engine`. There cannot be a second eval codepath.

5. **Native capability contracts are frozen early.** The five core interfaces (ModelAdapter / KVManager / KVCodec / WeightProvider / DraftEngine) are the integration points for native capabilities. They are freeze candidates in Phase 0 and finally frozen at P-0 exit (see §6). The scheduler and engine core are unaware of concrete implementations. Changes go through the Decisions Log. These are **not** "third-party plugin extension points" — they are Silica's own capability boundaries (see Principle 9).

6. **MLX-native hot path (hard constraint).** The inference hot path must be 100% MLX: every tensor is an `mx.array`, every op goes through MLX. `torch.Tensor` / `numpy.ndarray` are **not allowed** in the hot paths of `silica.engine` / `silica.mlx` / `silica.kvcache` / `silica.models` / `silica.scheduler`. Phase 1 may wrap `mlx-lm` **because mlx-lm is itself MLX-native**; replacing it with a torch-based wrapper is forbidden. vllm and transformers are **algorithm references only**, not runtime dependencies. See D-009.

7. **Small over large in early phases.** Phase 1 starts with Qwen3-0.6B; Phase 3 switches to the target large models. Close the loop on small before scaling up.

8. **Savings must be observable.** Any memory / streaming / acceleration saving must be visible to the scheduler (e.g. `KVCodec.logical_bytes` vs `resident_bytes`), otherwise the memory budgeter cannot translate the saving into "admit more requests / longer context".

9. **Native capabilities, swappable implementations.** VQ KV compression, weight-streaming residency, and speculative decoding are Silica-MLX's **native capabilities**, not "third-party plugin extension points". They are built into the engine main loop as stubs from P-0 (`IdentityCodec` / `ResidentWeightProvider` stub / `NoopDraftEngine`) and progressively replaced in P-5 / P-6 / P-7. **Integration points are fixed**: the memory budgeter reads `KVCodec.logical_bytes` / `resident_bytes`; the scheduler reads `WeightProvider` prefetch signals; the decode loop talks to `DraftEngine.propose` / `commit`. **Implementations are swappable**: BlockTQ / RaBitQ / future codecs; Resident / Streaming / future residency strategies; Noop / DraftTarget / future EAGLE / Medusa — all are interchangeable implementations behind the same integration point. Analogy: vLLM's "attention backend" — at the native layer every model goes through a backend, but the concrete backend (FlashAttention / Xformers / Triton) is swappable. We **do not say "plugin"**; we say "backend / codec / implementation". This complements Principle 1 (platform-as-product stance) with the architectural expression.

---

## 5. Architecture Overview

### 5.1 Module Layout

Target layout (the actual repo is the source of truth after Phase 0):

```
silica-mlx/
├── pyproject.toml
├── README.md
├── docs/
│   └── PLAN.md                   # this file
├── python/silica/
│   ├── __init__.py               # re-exports Engine, LLM
│   ├── core/                     # Request, SamplingParams, Context, logging, profiling
│   ├── mlx/                      # MLX array utilities, profiling hooks
│   ├── engine/                   # Engine class, generate() loop
│   ├── scheduler/                # ContinuousBatcher, request lifecycle, memory budget
│   ├── kvcache/                  # PagedKVCache, PrefixCache, KVCodec Protocol
│   ├── models/                   # ModelAdapter Protocol + Qwen3.5 / Gemma4 adapters
│   ├── weights/                  # WeightProvider Protocol + Resident/Streaming impls (residency & prefetch)
│   ├── vq/                       # BlockTQ, RaBitQ codecs
│   ├── speculative/              # DraftEngine Protocol + Noop/DraftTarget impls
│   ├── server/                   # CLI, OpenAI API, session management
│   ├── llm/                      # High-level Python API (LLM class, Phase 8)
│   └── bench/                    # Benchmark runner (thin wrapper over Engine)
└── tests/
```

### 5.2 Data Flow (v0.1 target)

```
User
 → silica.engine.Engine.generate(prompt, sampling_params)
   → Tokenizer
   → Scheduler.admit(request)
     → KVManager.get_computed_blocks / reserve_for_prefill (KVCodec integrates transparently)
   → ModelAdapter.prefill / decode_step (through WeightProvider; KV via kv_handle from KVManager)
     → KVCodec.encode_block / decode_block per layer
   → Sampler (+ optional DraftEngine)
 → token stream
```

Phase 8 adds a `silica.server.openai_api` FastAPI wrapper on top; internally it is still the same Engine instance.

### 5.3 Process Model

v0.1 is **single-process**. No tokenizer worker split, no detokenizer split, no scheduler worker split. Splitting is a v0.2 discussion.

### 5.4 Reference Map to vLLM v1

**Algorithm / architecture reference only**, not a runtime dependency. Local path: `vllm/` (gitignored). vLLM's v0 (the old `engine/llm_engine.py`) is **not** our reference target; we track v1 (`vllm/v1/`).

| Silica module                           | vLLM v1 reference files                                                 | Purpose                                      |
| --------------------------------------- | ----------------------------------------------------------------------- | -------------------------------------------- |
| `silica.engine.Engine`                  | `vllm/v1/engine/llm_engine.py`, `vllm/v1/engine/core.py`                | Engine main-loop structure                   |
| `silica.scheduler.batcher`              | `vllm/v1/core/sched/scheduler.py`                                       | Continuous batching scheduling               |
| `silica.scheduler.budget`               | `vllm/v1/core/kv_cache_manager.py` (budget portion)                     | Memory budget and admission                  |
| `silica.kvcache.paged.PagedKVCache`     | `vllm/v1/core/block_pool.py`, `vllm/v1/core/kv_cache_manager.py`        | Paged / block KV allocator                   |
| `silica.kvcache` (interface)            | `vllm/v1/kv_cache_interface.py`                                         | KV cache spec and layout                     |
| `silica.kvcache.prefix`                 | `vllm/v1/core/kv_cache_manager.py` (prefix portion)                     | Prefix cache hit and reuse                   |
| `silica.core.request.RequestState`      | `vllm/v1/request.py`                                                    | Request state machine                        |
| `silica.core.sampling` / sampler        | `vllm/v1/sample/`                                                       | Sampling reference                           |
| `silica.models.*` attention             | `vllm/v1/attention/backend.py` (interface idea only)                    | Swappable attention backend pattern; **no copying of CUDA implementations** |
| `silica.speculative.*`                  | `vllm/v1/spec_decode/`                                                  | Speculative decoding architecture reference  |

**Not referenced:**
- `vllm/csrc/` — C++ / CUDA kernels.
- `vllm/vllm_flash_attn/` — CUDA flash attention.
- `vllm/v1/worker/gpu_model_runner.py`, `tpu_model_runner.py`, `xpu_model_runner.py`, `cpu_model_runner.py` — device-specific runners; we write only `silica.mlx.runner`.
- `vllm/v1/executor/`, `vllm/distributed/`, `vllm/v1/core/kv_cache_coordinator.py` — multi-process / multi-node coordination, not needed for v0.1.
- vLLM's v0 paths (`vllm/engine/`, top-level `vllm/worker/`) — superseded by v1.

### 5.5 Reference Map to vqbench

**Algorithm + numeric oracle only**, not a runtime dependency. Local path: `vqbench/` (gitignored; includes nested `vqbench/turboquant_plus/`). vqbench is a **NumPy + PyTorch + HF transformers** codebase — under D-009 it is **not a runtime source**. But it carries an empirical Qwen3.5-4B PPL baseline (`BlockTurboQuantMSE B=64` 4-bit K+V near-lossless, see `vqbench/REPORT.md`) that serves as a **correctness oracle** for Silica's MLX-native rewrite in P-5.

| Silica module                | vqbench reference files                                                                                     | Purpose                                                  |
| ---------------------------- | ----------------------------------------------------------------------------------------------------------- | -------------------------------------------------------- |
| `silica.vq.block_tq`         | `vqbench/vqbench/methods/turboquant/block_mse.py`                                                           | BlockTurboQuantMSE algorithm (B=16/20/32/40/64)          |
| `silica.vq.rabitq`           | `vqbench/vqbench/methods/rabitq/rabitq_1bit.py`, `rabitq_ext.py`                                            | RaBitQ 1-bit / extended bits                             |
| `silica.vq` (factory)        | `vqbench/vqbench/torch_wrapper/module.py` (`_get_method_class`)                                             | Codec registry / naming convention reference             |
| `silica.kvcache` (pair layer)| `vqbench/vqbench/kv_cache/compressor.py` (`KVCacheCompressor`)                                              | K/V independent-codec pair pattern (see Q-008)           |
| `silica.bench.scenarios`     | `vqbench/scripts/reproduce_qwen35_4b_headline.py`, `variance_qwen35_4b.py`, `run_qwen35_27b_sweep.py`       | Qwen3.5 bench scenarios + PPL regression reference       |
| P-5 numeric oracle           | `vqbench/REPORT.md` (Qwen3.5-4B result tables), `vqbench/BlockTQ.md`                                        | Empirical baseline + algorithm walkthrough               |

**Explicitly not referenced (forbidden as runtime source):**
- `vqbench/vqbench/torch_wrapper/` — PyTorch `nn.Module` + HF `DynamicCache` subclass, violates D-009; used as "anti-pattern" in D-010 Consequences.
- NumPy implementations in `vqbench/vqbench/methods/*.py` — algorithmic reference only, **not** `mx.array` equivalents; P-5 must rewrite them MLX-native.
- `vqbench/PLAN.md` — vqbench's own historical plan, unrelated to Silica's plan; **not** a stale copy, do not confuse.

---

## 6. Core Interfaces (Phase 0 freeze candidate)

The five core interfaces of v0.1. **Signatures are finalized at P-0 exit**, not the moment this document is written. During Phase 0 we may extend the operation set (for example, I-1 / I-2 gained `prefill`/`decode_step` and `append`/`commit`/`rollback` in this version); only at P-0 exit are they truly frozen. Changing a signature after P-0 requires a new Decisions Log entry. What follows is the contract skeleton; the actual `typing.Protocol` signatures live in code.

### I-1 ModelAdapter

Responsible for model structure, tokenizer, layer execution, attention pattern, and execution semantics (prefill / decode).

```python
class ModelAdapter(Protocol):
    config: ModelConfig

    def build(self, weight_provider: WeightProvider) -> Module: ...
    def kv_layout(self) -> KVLayout: ...                  # num_layers, n_kv_heads, head_dim, dtype
    def attention_pattern(self) -> AttentionPattern: ...  # global / sliding / hybrid per layer
    def tokenizer(self) -> Tokenizer: ...
    def prefill(
        self, tokens: mx.array, kv_handle: KVHandle
    ) -> tuple[mx.array, StateDelta]: ...                 # returns (logits, non-KV state delta)
    def decode_step(
        self, token: mx.array, kv_handle: KVHandle
    ) -> tuple[mx.array, StateDelta]: ...                 # returns (logits, non-KV state delta)
```

**Key constraints:**
1. `attention_pattern()` must be able to express Qwen3.5's hybrid attention — different layers use different cache routing, otherwise Phase 3 will be forced to rewrite the scheduler.
2. **KV mutation ownership belongs to `KVManager`, not the adapter.** `prefill` / `decode_step` read and write KV via `kv_handle` (issued by `KVManager`); they never hold block pointers directly, make residency decisions, or touch the prefix cache structure.
3. `state_delta` carries **non-KV** runtime state only: sampling RNG, MoE router cache, position counter, sliding-window mask cursor, etc. **Counter-examples (forbidden inside `state_delta`)**: KV blocks / cache residency mutations / prefix cache pinning — these must go through `kv_handle` owned by `KVManager`.

### I-2 KVManager

Owns paged / block KV, the prefix cache, the memory budget, and the incremental mutation primitives required by continuous batching and speculative decoding.

```python
class KVManager(Protocol):
    block_size: int

    def reserve_for_prefill(
        self, req_id: str, token_ids: Sequence[int]
    ) -> BlockList: ...                                   # reserve blocks for initial prompt
    def append_slot(self, req_id: str, n: int) -> BlockList: ...  # extend during decode
    def commit(self, req_id: str, n_accepted: int) -> None: ...   # speculative: accept draft
    def rollback(self, req_id: str, n_reject: int) -> None: ...   # speculative: reject draft
    def free(self, req_id: str) -> None: ...
    def get_computed_blocks(
        self, token_ids: Sequence[int]
    ) -> PrefixHit: ...                                   # prefix cache lookup (vLLM v1 naming)
    def available_blocks(self) -> int: ...                # fast path for admission
    def budget(self) -> MemoryBudget: ...                 # logical_bytes, resident_bytes, headroom
```

**Key constraints:**
1. `budget()` must report both logical and resident bytes (Principle 8). The scheduler uses this for admission control.
2. **Incremental semantics.** `reserve_for_prefill` reserves blocks for the initial prompt (*reserve*, not "logically allocate and write"); `append_slot` incrementally extends during decode; `commit` / `rollback` support speculative accept / reject (P-7); `free` releases on request completion. The single-request `SimpleKVCache` in P-1 may implement `commit` / `rollback` as no-ops, but **the signatures must exist from P-0** — otherwise P-7 is forced to break the frozen API.
3. `get_computed_blocks` is the prefix cache lookup (naming aligned with vLLM v1's `kv_cache_manager.get_computed_blocks`); it returns the list of already-computed blocks so the scheduler can reuse and pin them on admission.
4. `available_blocks()` is a fast, block-granular path. It is **not** equivalent to `budget().headroom`: the latter is byte-granular (influenced by the codec's logical/resident ratio), the former is block-granular and gives the scheduler an O(1) "can I admit one more request?" check.

### I-3 KVCodec

Uniform abstraction for KV encoding and decoding. v0.1 **does not include** a compressed-domain attention fast path (D-003).

```python
class KVCodec(Protocol):
    block_size: int
    k_dtype: mx.Dtype
    v_dtype: mx.Dtype

    def encode_block(self, k: mx.array, v: mx.array) -> CodedBlock: ...
    def decode_block(self, block: CodedBlock) -> tuple[mx.array, mx.array]: ...
    def logical_bytes(self, num_tokens: int) -> int: ...   # fp16 baseline equivalent
    def resident_bytes(self, num_blocks: int) -> int: ...  # actual storage
```

**Key constraint:** a codec sees one layer's K/V only; it is unaware of batch, scheduler, or model structure.

### I-4 WeightProvider

Residency / streaming abstraction for weights. Phase 1 uses Resident; Phase 6 wires in Streaming.

```python
class WeightProvider(Protocol):
    def get_layer(self, layer_idx: int) -> LayerWeights: ...      # sync blocking
    def prefetch(self, layer_indices: Sequence[int]) -> None: ...  # hint, may no-op
    def release(self, layer_idx: int) -> None: ...
    def resident_bytes(self) -> int: ...

    # MoE-aware per-expert granularity (D-011)
    def get_expert(self, layer_idx: int, expert_id: int) -> ExpertWeights: ...
    def prefetch_experts(
        self, layer_idx: int, expert_ids: Sequence[int]
    ) -> None: ...
    def release_expert(self, layer_idx: int, expert_id: int) -> None: ...
```

**Key constraints:** `get_layer` blocks synchronously; `prefetch` is a hint and may be a no-op. Callers never need to know where weights physically live. Streaming implementations must exploit unified memory (Principle 2) rather than pretend to be a PCIe device.

**MoE constraints (D-011).** A MoE adapter's FFN execution **must** go through `get_expert` / `prefetch_experts`; it **must not** pull all experts via `get_layer` — otherwise P-6's `StreamingWeightProvider` loses its primary value for MoE (per-expert residency). Dense `WeightProvider` implementations raise `NotImplementedError("dense provider has no per-expert path")` from the three expert methods rather than implementing them as no-ops: this makes a MoE adapter wired to a dense provider fail loudly instead of silently degrading. Wiring a MoE adapter to a dense provider is a model-registry configuration error, not a runtime fallback scenario.

### I-5 DraftEngine

Draft provider for speculative decoding. Phases 1–6 use Noop; Phase 7 wires in DraftTarget.

```python
class DraftEngine(Protocol):
    def propose(self, ctx: RequestState, k: int) -> DraftTokens: ...
    def commit(self, ctx: RequestState, accepted_len: int) -> None: ...
```

---

## 7. Phases

Each Phase uses the same structure: `ID / Goal / Scope / Strategy / Deliverables / Acceptance / Dependencies / Status / Notes`. Edit only the block of the Phase you are changing. Status values: `planned / in-progress / done / blocked / obsolete`.

### P-0 Phase 0 — Skeleton

- **Goal:** stand up the repo, package layout, config, logging, profiling, and interface skeleton.
- **Scope:** `pyproject.toml`, 11 sub-package skeletons, five Protocols, logging, profiler, a minimal interface test.
- **Strategy:**
  - uv + Python 3.12 (D-001).
  - stdlib `logging` + a colored formatter (no structlog).
  - `pydantic` v2 `BaseSettings` for config.
  - Profiler built with `@contextmanager`, a global `MetricsRegistry`, and a unified metrics schema.
- **Deliverables:**
  - [ ] `pyproject.toml` (uv, mlx, mlx-lm, numpy, pydantic, pytest, ruff, mypy; Python 3.12).
  - [ ] 11 sub-package skeletons + `__init__.py`.
  - [ ] `silica.core.request.Request` / `RequestState`.
  - [ ] `silica.core.sampling.SamplingParams`.
  - [ ] `silica.core.logger` + `silica.core.profiler`.
  - [ ] Five Protocols (I-1..I-5) + a stub implementation for each (`StubModelAdapter`, `NullKVManager`, `IdentityCodec`, `ResidentWeightProvider` stub, `NoopDraftEngine`).
  - [ ] `tests/test_interfaces.py` verifying Protocol shape + stub instantiation.
  - [ ] `from silica import Engine` imports cleanly (even if it raises `NotImplementedError`).
- **Acceptance:**
  - [ ] `uv pip install -e .` succeeds.
  - [ ] `pytest tests/test_interfaces.py` passes.
  - [ ] Unified metrics schema includes fields: `ttft_ms`, `prefill_tok_s`, `decode_tok_s`, `resident_mb`, `logical_kv_bytes`.
- **Dependencies:** none.
- **Status:** planned.
- **Notes:** foundation for everything else. Once interfaces are frozen, changes go through the Decisions Log.

### P-1 Phase 1 — Baseline Engine

- **Goal:** a minimal runnable MLX-native inference main loop.
- **Scope:** single request, greedy / temperature / top-p sampling, token streaming, minimal CLI.
- **Strategy:** the Phase 1 `ModelAdapter` borrows from `mlx-lm` for (a) model-structure loading, (b) the tokenizer, (c) the weight loader (safetensors → `mx.array`), but **does not borrow** mlx-lm's rotating KV cache / prompt cache — the ownership boundary is fixed in **D-010** (which supplements D-004). KV is a `SimpleKVCache` (single request, non-paged), injected into the adapter's `prefill` / `decode_step` via `kv_handle`. P-2's `PagedKVCache` is a direct upgrade path, not a replacement for mlx-lm's internal cache. The first task of P-1 is a **day-1 smoke test**: verify that `mlx_lm.generate_step(cache=...)` or an equivalent entry point accepts an external cache object. The result determines the real cost of the remaining P-1 deliverables (see D-010 / R-6).
- **Deliverables:**
  - [ ] **Day-1 gate:** smoke test that `mlx_lm` accepts external cache injection (D-010). Resolve this blocker before expanding the deliverables below.
  - [ ] `silica.engine.Engine.generate(prompt, sampling_params)` returning a token stream.
  - [ ] `silica.mlx.runner` wrapping mlx-lm's forward (external cache injected; mlx-lm's internal cache unused).
  - [ ] `silica.kvcache.simple.SimpleKVCache` (single-request; passed to the adapter as a `kv_handle`).
  - [ ] `silica.server.cli`: `python -m silica run --model Qwen/Qwen3-0.6B --prompt "..."`.
  - [ ] Basic sampling: greedy, temperature, top-p.
  - [ ] `silica.models.qwen3.Qwen3Adapter` (borrows mlx-lm structure, KV self-managed per D-010).
- **Acceptance:**
  - [ ] Generates text reliably.
  - [ ] Greedy decoding is token-for-token identical to the mlx-lm reference implementation (fixed seed, same model).
  - [ ] The profiler produces TTFT, decode tok/s, and resident memory.
- **Dependencies:** P-0.
- **Status:** planned.
- **Notes:** dev-loop model is Qwen3-0.6B. Do not attempt 27B / 31B in Phase 1.

### P-2 Phase 2 — Mini-vLLM Core

- **Goal:** the real engine skeleton — paged KV + continuous batching + prefix cache + memory budget.
- **Scope:** concurrent requests, shared-prefix hits, memory-budget management, request state machine.
- **Strategy:**
  - Default block size 16 tokens (re-evaluated after Phase 4 bench).
  - Radix prefix cache, taking cues from mini-sglang.
  - Chunked prefill is deferred; add it in Phase 3 only if OOM forces it.
- **Deliverables:**
  - [ ] `silica.kvcache.paged.PagedKVCache` implementing KVManager (I-2).
  - [ ] `silica.kvcache.prefix.RadixPrefixCache`.
  - [ ] `silica.core.request.RequestState` state machine: `WAITING → PREFILL → DECODE → DONE/ABORTED`.
  - [ ] `silica.scheduler.batcher.ContinuousBatcher`.
  - [ ] `silica.scheduler.budget.MemoryBudgeter`.
- **Acceptance:**
  - [ ] 8 concurrent requests run stably.
  - [ ] A constructed shared-prefix prompt hits the prefix cache (verifiable via hit counters).
  - [ ] Exceeding the memory budget aborts / queues cleanly without crashing.
- **Dependencies:** P-1.
- **Status:** planned.
- **Notes:** the single most important Phase. The skeleton decides the stub-replacement path for every native capability (VQ / streaming / speculative) that follows (Principle 9).

### P-3 Phase 3 — Model Adapters

- **Goal:** actually run Qwen3.5-27B / Gemma4-31B (dense) + Qwen3.5-35B-A3B / gemma-4-26B-A4B (MoE smoke test).
- **Scope:** four model adapters (2 dense + 2 MoE, D-011), Qwen3.5 hybrid attention, MoE top-k expert routing, MLX-native 4-/8-bit quantization.
- **Strategy:**
  - Quantization rides on mlx-lm's existing 4-bit / 8-bit path (D-005).
  - Weight loading goes through `WeightProvider`, even while this phase still uses `ResidentWeightProvider`.
  - Qwen3.5 hybrid attention is expressed via `AttentionPattern`; the scheduler routes KV by layer index.
  - **MoE adapter FFN execution goes through `WeightProvider.get_expert` / `prefetch_experts` to load the active top-k experts** (D-011 constraint). The Phase 3 `ResidentWeightProvider` still holds everything resident for MoE, but the `get_expert` call path must be real — otherwise P-6's per-expert residency cannot be wired in.
  - MoE adapters are inference-only: top-k routing + gate softmax normalization are computed, but aux-loss / load-balancing (training-only) is not.
- **Deliverables:**
  - [ ] `silica.models.qwen3_5.Qwen35Adapter` (dense, Qwen3.5-27B).
  - [ ] `silica.models.gemma4.Gemma4Adapter` (dense, Gemma4-31B).
  - [ ] `silica.models.qwen3_5_moe.Qwen35MoeAdapter` (MoE, Qwen3.5-35B-A3B).
  - [ ] `silica.models.gemma4_moe.Gemma4MoeAdapter` (MoE, gemma-4-26B-A4B).
  - [ ] Hybrid implementation of `AttentionPattern`.
  - [ ] MoE top-k gating + per-expert FFN aggregation (inference path, aux-loss ignored).
  - [ ] `silica.weights.resident.ResidentWeightProvider` (full residency; the MoE variant exposes `get_expert` per-expert access even if the underlying storage is fully resident).
  - [ ] Model registry: `silica.models.registry` (each entry marks `arch: dense | moe`; MoE entries require a MoE-capable `WeightProvider`).
- **Acceptance:**
  - **Adapter structural correctness (fp16 parity on control model)** — P-3 exit criterion:
    - [ ] On a fp16 control model (Qwen3-0.6B or Qwen3.5-4B), Silica adapter logits match the HuggingFace reference: `max |logit diff| < 1e-3` over the first 50 greedy-decoded tokens under the same tokenizer state and seed. This verifies adapter structural components (attention, RMSNorm, MoE routing, positional encoding) without quantization noise.
    - [ ] Qwen3.5 hybrid attention cache-routing semantics are correct — unit tests cover per-layer dispatch according to `AttentionPattern`.
  - **Quantized big-model correctness** — P-3 exit criterion:
    - [ ] Qwen3.5-27B and Gemma4-31B load and execute at least one forward under the MLX-native 4-bit or 8-bit quantization path (D-005); manual resident caps and small batches are allowed.
    - [ ] **Teacher-forced next-token argmax agreement ≥ 98%** over the first 100 teacher-forced positions, against the `mlx-lm` reference at the same model + same quantization configuration, compared position-by-position on a fixed prefix. We do **not** compare against free-running generated sequences — sequence drift would mask or amplify real differences.
    - [ ] Fallback (if teacher-forced comparison is infeasible on the current toolchain): end-to-end **PPL drift `< 0.1` absolute** vs an `mlx-lm` baseline on the same evaluation corpus at the same quantization configuration.
  - **Product memory-fit target** (conditional, gated on Q-003; ownership by M-4 or M-7, see handoff rules):
    - [ ] Both dense targets sustain ≥ 500 tokens of generation on an M5 Pro 48 GB. If Q-003 is not yet resolved to "int4 fits in 48 GB" at P-3 exit, this item may be satisfied with smaller models (e.g. Qwen3.5-7B) or a manual resident cap, and the real 27B/31B @ 48 GB 500-token validation is handed off to M-7 (see R-1 / Q-003 / M-4 / M-7).
  - **MoE structural correctness** (D-011) — P-3 exit criterion:
    - [ ] On a small MoE control model (an fp16 version of Qwen3.5-35B-A3B or a smaller MLX-community-accessible MoE equivalent, e.g. a tiny Qwen2-57B-A14B variant or OLMoE-1B-7B), Silica MoE adapter logits match the HuggingFace reference: `max |logit diff| < 1e-3` over the first 50 greedy-decoded tokens. This verifies top-k expert routing + gate normalization + expert FFN aggregation without quantization noise.
    - [ ] Qwen3.5-35B-A3B and gemma-4-26B-A4B load and execute at least one forward under the MLX-native 4-bit or 8-bit quantization path; manual resident caps and small batches are allowed; full residency fallback is permitted while P-6 is incomplete (active params are small, so fit risk is far lower than dense — see R-1 MoE mitigation).
    - [ ] **Per-expert call-path unit test**: with a mock `WeightProvider`, the MoE adapter's forward pass assertably invokes `get_expert(layer_idx, expert_id)` for the top-k activated experts, and `get_layer(layer_idx)` is **not** used to load expert weights. This test makes the D-011 constraint regressable before P-6 streaming is wired in.
- **Dependencies:** P-2.
- **Status:** planned.
- **Notes:** risk — Qwen3.5-27B fp16 is ~54 GB, so quantization is required. If 4-bit still doesn't fit, Q-003 fires and P-6 is pulled forward. Acceptance is split into adapter structural correctness (fp16 parity on a dense control model) + quantized big-model correctness (teacher-forced comparison) + product memory-fit target (conditional, dense only) + MoE structural correctness (D-011, independent exit criterion). If dense memory-fit is deferred, the MoE smoke test must still pass — MoE targets have small active params and do not depend on Q-003 resolution.

### P-4 Phase 4 — Bench Unification

- **Goal:** benchmarks run directly through the Engine — no side paths.
- **Scope:** unified bench runner, standard scenarios, unified result format.
- **Strategy:** bench is a thin wrapper over `silica.engine.Engine`.
- **Deliverables:**
  - [ ] `silica.bench.runner.BenchRunner`.
  - [ ] `silica.bench.scenarios`: short-in/long-out; long-in/short-out; concurrent shared-prefix.
  - [ ] Unified metrics schema: TTFT, prefill tok/s, decode tok/s, resident memory, logical KV bytes, quality.
  - [ ] Output: jsonl + markdown report.
  - [ ] `silica.bench.vqbench_baseline`: runs the ready-made vqbench scripts (`reproduce_qwen35_4b_headline.py` etc.) in a separate subprocess to collect PPL as a reference column. D-009 explicitly allows this "separate-process comparison" path; it serves the P-5 numeric cross-check acceptance.
- **Acceptance:**
  - [ ] A single command produces the baseline table (paste-able into README).
  - [ ] No path split between bench and runtime (same Engine instance).
  - [ ] `vqbench_baseline` produces a Qwen3.5-4B PPL number in a separate process as the P-5 comparison column.
- **Dependencies:** P-3.
- **Status:** planned.
- **Notes:** Phase 4 baseline data determines Q-003 (whether Phase 6 is pulled forward).

### P-5 Phase 5 — VQ KV Compression

- **Goal:** replace the P-0 `IdentityCodec` stub with real VQ codecs (Principle 9 stub-to-real replacement), letting the platform admit more requests or longer context within the same memory budget.
- **Scope:** `IdentityCodec`, `BlockTQCodec`, `RaBitQCodec`. **PQ / OPQ stay out of the main line.**
- **Strategy:**
  - No compressed-domain fast path in v0.1 (D-003).
  - `PagedKVCache(codec=...)` injection-based switching.
  - The scheduler reads `KVCodec.logical_bytes` / `resident_bytes` to learn about savings and admits more requests accordingly (Principle 8).
  - **`BlockTQCodec` / `RaBitQCodec` must be rewritten on the `mx.array` hot path**, with resident accounting added. The reference source is **`vqbench/`** (including nested `vqbench/turboquant_plus/`) — vqbench is the **algorithmic reference + Qwen3.5-4B empirical baseline** (`BlockTurboQuantMSE B=64` 4-bit K+V already validated at +0.0% ΔPPL, see `vqbench/REPORT.md`); details in §5.5 Reference Map. vqbench itself is NumPy + PyTorch + HF transformers and is **not imported at runtime** (D-009). The real engineering work in P-5 is "translate the NumPy logic into MLX-native `mx.` ops" + "expose savings as `logical_bytes` / `resident_bytes` to the scheduler". This is **not** "wiring a third-party plugin" — it is replacing a native-capability stub with its real implementation (Principle 9). State this up front so P-5 scope creep is not mistaken for surprise.
  - Whether codec decode overhead signals should enter scheduler admission (avoiding the pathological "saves memory but kills decode throughput" combination) is discussed in **Q-007**; not baked into the interface in v0.1.
- **Deliverables:**
  - [ ] `silica.vq.block_tq.BlockTQCodec`.
  - [ ] `silica.vq.rabitq.RaBitQCodec`.
  - [ ] `silica.kvcache.paged.PagedKVCache` supports codec injection.
  - [ ] Bench gains `--kv-codec {fp16,block_tq,rabitq}`.
  - [ ] Scheduler budget admission policy that exploits codec savings.
- **Acceptance:**
  - [ ] Switching the codec requires no change to the scheduler or model adapter.
  - [ ] For the same scenario set, fp16 vs codec quality delta and memory savings are available from the bench in one command.
  - [ ] With BlockTQ on, the same memory budget admits more requests (quantitatively verifies Principle 8).
  - [ ] **Numeric cross-check against vqbench** — two independent thresholds, both must pass:
    - **(a) Per-block reconstruction error.** Metric: **per-block relative Frobenius error** `||K_decoded - K_original||_F / ||K_original||_F` (same for V), computed block-by-block on a fixed calibration set. Silica MLX-native `BlockTQCodec` on Qwen3.5-0.6B (or larger), compared against the vqbench NumPy reference on the same calibration set under the same metric, must satisfy `ε_recon < 2 × fp16 round-trip baseline noise`, where the baseline is established by vqbench's own fp16 encode→decode round trip (order of magnitude near machine precision). Can be tightened at P-4 exit based on measured baseline noise.
    - **(b) End-to-end PPL drift.** Under the Qwen3.5-4B `BlockTurboQuantMSE B=64` 4-bit K+V configuration, Silica `BlockTQCodec`'s end-to-end perplexity deviates from the `vqbench/REPORT.md` baseline by `ε_ppl < 0.01` absolute.
    - Both must pass; passing only one is insufficient.
- **Dependencies:** P-4.
- **Status:** planned.
- **Notes:** concrete implementation details for BlockTQ / RaBitQ reference `turboquant_plus/` (gitignored reference impl).

### P-6 Phase 6 — Weight Streaming

- **Goal:** weight streaming relieves residency pressure.
- **Scope:** `ResidentWeightProvider` (already in tree) + `StreamingWeightProvider` + scheduler prefetch coordination.
- **Strategy:**
  - Take cues from `mlx-flash`.
  - While layer N computes, prefetch layer N+1 (dense path).
  - **Exploit Apple unified memory** (Principle 2): this is not "GPU pulls weights from disk/CPU"; it is "lifetime management of different regions within the same memory pool".
  - **Dense vs MoE residency granularity (D-011).** Dense streaming uses **layer** granularity (prefetch layer N+1 while computing layer N). MoE streaming uses **expert** granularity — keep only recently-active top-k experts resident; other experts' regions in unified memory may be overwritten. This is streaming's **primary payoff for MoE** (Qwen3.5-35B-A3B: total 35B, active 3B, resident set can shrink to roughly active size + non-FFN weights + headroom). MoE scheduler coordination: as soon as gate logits are available, fire `prefetch_experts(layer_idx, top_k_ids)`; do not wait until the expert FFN is actually invoked.
- **Deliverables:**
  - [ ] `silica.weights.streaming.StreamingWeightProvider` (dense + MoE dual mode).
  - [ ] `silica.weights.prefetch`: scheduler prefetch coordination (layer-granular for dense; expert-granular for MoE).
  - [ ] Unified-memory-aware residency policy (e.g. LRU-over-experts eviction for MoE).
- **Acceptance:**
  - [ ] Under an artificial memory budget (e.g. 24 GB), Qwen3.5-27B int4 does not OOM (dense path).
  - [ ] **Decode tok/s ≥ 70% of the `ResidentWeightProvider` baseline** (dense), with fixed comparison conditions: same machine / same model / same scenario / same sampling config when taking the ratio. If resident cannot run under 24 GB at all (OOM), switch the baseline to an **"uncapped resident reference run"** (no budget, no streaming, same scenario, same decode tok/s) and take the ratio against that.
  - [ ] **MoE per-expert residency takes effect (D-011).** For a MoE target (Qwen3.5-35B-A3B or gemma-4-26B-A4B) under a 24 GB budget, `StreamingWeightProvider.resident_bytes()` ≤ `active_experts × expert_size + non_FFN_weights + expert_residency_headroom` (quantitatively verifies per-expert residency is active rather than silently falling back to full residency). Headroom defaults to ≤ 20% of active size; tightened in P-4 bench after empirical measurement.
  - [ ] MoE decode tok/s ≥ 60% of the MoE `ResidentWeightProvider` baseline — the threshold is below the dense 70%, acknowledging that expert-fetch miss stalls are inherent and the first cut of streaming does not optimize for them.
- **Dependencies:** P-2 (scheduler) + P-3 (model adapter).
- **Status:** planned.
- **Notes:** may be pulled ahead of P-5 out of the T1 tier if P-3 shows 27B/31B won't fit at baseline (Q-003).

### P-7 Phase 7 — Speculative Decoding

- **Goal:** draft-target speculative decoding speeds up decode.
- **Scope:** `NoopDraftEngine` (already in tree) + `DraftTargetEngine` (most basic version).
- **Strategy:**
  - A small model drafts; the large model verifies.
  - EAGLE / Medusa / DFlash complexity is deferred to v0.2.
- **Deliverables:**
  - [ ] `silica.speculative.draft_target.DraftTargetEngine`.
  - [ ] Integration with the decode loop.
  - [ ] Acceptance / rollback metrics.
  - [ ] Bench switch `--speculative {none,draft_target}`.
- **Acceptance:**
  - [ ] Under greedy decoding, with speculative on vs off the token sequences are **identical token by token** (correctness invariant as the baseline gate).
  - [ ] **Decode tok/s ≥ 1.2× the draft-disabled baseline**, measured on a **fixed standard scenario** (e.g. P-4 bench's "long-in / short-out" or an equivalent mixed-batch scenario). **Cherry-picking a best-case workload** (abundant shared prefixes, a specific temperature, or the single most favorable scenario) is not acceptable evidence.
  - [ ] Baseline correctness is preserved (smoke tests pass with the switch in either position).
- **Dependencies:** P-2 + P-3.
- **Status:** planned.
- **Notes:** DFlash / dflash-mlx are v0.2 candidates.

### P-8 Phase 8 — Mini-SGLang Layer

- **Goal:** add a minimal serving layer on top of the engine so the platform is actually "usable".
- **Scope:** OpenAI-compatible HTTP API, session management, prefix-sharing session reuse, a reserved slot for structured output.
- **Strategy:**
  - This is the "product face", not a nicety (D-006).
  - Priority discussion in Q-002 (whether it floats from T2 to the tail of T1).
  - The outer organization follows mini-sglang; CUDA kernels are not copied.
- **Deliverables:**
  - [ ] `silica.server.openai_api`: FastAPI, `/v1/chat/completions`, `/v1/completions`.
  - [ ] `silica.server.session.SessionManager`: session management, cross-request prefix reuse.
  - [ ] An interface slot for structured generation / grammar (unimplemented).
  - [ ] `silica.llm.LLM`: Python-friendly high-level interface.
- **Acceptance:**
  - [ ] The `openai` Python client can stream responses from the silica server.
  - [ ] Cross-request prefix reuse is verifiable (send N shared-prefix requests in one session, check prefix cache hit rate).
  - [ ] Locally behaves like a small serving engine.
- **Dependencies:** P-2 + P-3 (+ P-5 / P-6 optional).
- **Status:** planned.
- **Notes:** this Phase is what upgrades Silica-MLX from "an engine library" to "a usable Mac inference platform".

---

## 8. Priority & Milestones

### 8.1 Priority Tiers

Tier IDs use `T0 / T1 / T2` to avoid visual collision with phase IDs `P-0 / P-1 / P-2`.

| Tier | Phases       | Meaning                                                        |
| ---- | ------------ | -------------------------------------------------------------- |
| T0   | P-0 .. P-4   | Skeleton + baseline engine + target models + bench             |
| T1   | P-5 .. P-6   | VQ KV compression + weight streaming; make big models fit 48GB |
| T2   | P-7 .. P-8   | Speculative + serving layer                                    |

Whether Phase 8 floats in priority: see Q-002. Whether Phase 6 is pulled forward: see Q-003.

### 8.2 Milestones

| ID  | Name                                    | Dependent Phases | Acceptance |
| --- | --------------------------------------- | ---------------- | ---------- |
| M-1 | Skeleton                                | P-0              | Interfaces frozen, stub tests pass |
| M-2 | Single-request gen                      | P-1              | Qwen3-0.6B generates text |
| M-3 | Multi-request core                      | P-2              | 8 concurrent requests + prefix cache hit |
| M-4 | Big models adapter correct              | P-3              | Dense adapter structural correctness (fp16 parity on dense control model, max abs logit diff < 1e-3) + hybrid attention routing + quantized dense correctness (teacher-forced argmax agreement ≥ 98%, or fallback PPL drift < 0.1 absolute) + **MoE smoke test adapter correctness** (D-011: Qwen3.5-35B-A3B / gemma-4-26B-A4B structural correctness + fp16 parity on a MoE control model + per-expert `get_expert` call-path unit test). **Product memory-fit target** (dense 27B/31B @ 48 GB, 500 tokens): validated here **only if** Q-003 resolves to "int4 fits in 48 GB"; otherwise deferred to M-7. MoE memory-fit is not Q-003-gated (small active params, R-1 MoE mitigation). |
| M-5 | Unified bench                           | P-4              | One command produces the baseline table |
| M-6 | VQ on platform                          | P-5              | BlockTQ / RaBitQ wired in, savings quantifiable |
| M-7 | Streaming weights + deferred memory-fit | P-6              | Under a 24 GB budget, Qwen3.5-27B int4 does not OOM; decode tok/s ≥ 70% of the `ResidentWeightProvider` baseline (see P-6 Acceptance). **If M-4 deferred the product memory-fit target** (Q-003 forced P-6 before P-3 exit), this milestone carries the real 27B/31B @ 48 GB 500-token validation. |
| M-8 | Speculative enabled                     | P-7              | Correctness unchanged across switch + speedup |
| M-9 | Platform usable                         | P-8              | OpenAI API + session usable |

---

## 9. Decisions Log

Append-only. New decisions go at the end; old ones are not edited. Revocations / revisions open a new entry referencing the revoked ID.

### D-001 — Package manager & Python version

- **Date:** 2026-04-13.
- **Status:** accepted.
- **Decision:** `uv` + Python 3.12.
- **Rationale:** uv is modern and fast; mini-sglang also uses uv; Python 3.12 is the version MLX / mlx-lm stably support.
- **Consequences:** contributors need uv; Python < 3.12 is unsupported.

### D-002 — vLLM core first, mini-sglang layer later

- **Date:** 2026-04-14.
- **Status:** accepted.
- **Decision:** the engine core follows vLLM's ideas (paged KV, continuous batching, memory budget); the outer serving layer follows mini-sglang's ideas.
- **Rationale:** on a 48 GB M5 Pro the first-order problem is "run stably + save memory + be scalable", which vLLM's paged KV + batching addresses directly; mini-sglang is a better reference for module layering and the serving shell.
- **Consequences:** Phase 0–4 looks like a mini-vLLM; Phase 8 looks like a mini-sglang.

### D-003 — KVCodec v0.1 excludes compressed-domain attention

- **Date:** 2026-04-14.
- **Status:** accepted.
- **Decision:** the v0.1 `KVCodec` interface contains only `encode_block` / `decode_block` / `logical_bytes` / `resident_bytes`; no `attend()` or any compressed-domain fast path.
- **Rationale:** avoid over-committing the interface on day one. v0.1 prioritizes interface simplicity; v0.2 revisits this after Phase 5 bench data.
- **Consequences:** Phase 5 BlockTQ / RaBitQ implementations must follow the "decode then standard attention" path, even if it is suboptimal.

### D-004 — Phase 1 model execution: wrap mlx-lm

- **Date:** 2026-04-14.
- **Status:** accepted.
- **Decision:** the Phase 1 `ModelAdapter` is a thin wrapper over `mlx-lm`; model-execution details are not rewritten in Phase 1.
- **Rationale:** run first, optimize later. Proper adapter work comes in Phase 3.
- **Consequences:** Phase 3 will pay a rewrite cost, but Phase 1 iteration is much faster.

### D-005 — Phase 3 quantization path: MLX-native only

- **Date:** 2026-04-14.
- **Status:** accepted.
- **Decision:** Phase 3 quantization uses mlx-lm's existing 4-bit / 8-bit path; GGUF / AWQ multi-format support is **not** added.
- **Rationale:** 27B / 31B on 48 GB obviously must be quantized; start with what MLX can already run, defer multi-format compatibility to v0.2.
- **Consequences:** users cannot load GGUF / AWQ pre-quantized models; only mlx-lm-supported quantizations.

### D-006 — Platform as product, VQ as means

- **Date:** 2026-04-14.
- **Status:** accepted (user has corrected this framing once, explicitly).
- **Decision:** Silica-MLX itself is the product. VQ / weight streaming / speculative are means to make the platform run big models well; they are not research subjects.
- **Rationale:** the user's goal is "a single-Mac-chip inference platform that exploits VQ-class tech well", not "a benchmark vehicle for VQ research".
- **Consequences:**
  - Target users are Mac developers who want to run big models locally.
  - Phase 8 (the serving layer) gains priority.
  - VQ / weight streaming cannot be mere opt-in flags; the memory budgeter must actively use the savings (Principle 8).
  - Design must be Apple-unified-memory-first (Principle 2).

### D-007 — Plan document structure

- **Date:** 2026-04-14.
- **Status:** accepted.
- **Decision:** `docs/PLAN.md` is the single source of truth. Structure: Meta / TL;DR / Mission / Scope / Principles / Architecture / Interfaces / Phases / Priority / Decisions Log / Open Questions / Risks / References / Changelog, with stable IDs throughout.
- **Rationale:** CRUD-friendly — stable IDs, self-contained phase blocks, append-only decisions log. Lets us locate and modify a single entry without reading the whole document.
- **Consequences:** future plan changes go into the Decisions Log and update the relevant Phase block; the Changelog tracks version bumps.

### D-008 — core / engine boundary: data classes in core, logic classes in engine

- **Date:** 2026-04-14.
- **Status:** accepted.
- **Decision:** data classes (`Request`, `RequestState`, `SamplingParams`, `Context`, ...) live in `silica.core`; runtime logic classes (`Engine`, the runner portion of scheduler state machines) live in `silica.engine`. Follows the mini-sglang convention.
- **Rationale:** Phase 0 needs a clear directory layout. Q-004 Option A is the convention mini-sglang has validated; every Phase block in this document already implicitly uses these paths (e.g. `silica.core.request.Request`, `silica.core.sampling.SamplingParams`). Leaving it undecided would make document and code inconsistent at Phase 0 completion.
- **Consequences:**
  - `silica.core` contains data + observability (logging, profiling, metrics schema); no business logic.
  - `silica.engine.Engine` holds core data classes and drives scheduler / kvcache / model.
  - Resolves Q-004.

### D-009 — MLX-native hot path as hard constraint

- **Date:** 2026-04-14.
- **Status:** accepted (user-required).
- **Decision:** the inference hot path **must** be 100% MLX. Concretely:
  1. All tensors are `mlx.core.array` (`mx.array`). `silica.engine` / `silica.mlx` / `silica.kvcache` / `silica.models` / `silica.scheduler` / `silica.vq` / `silica.weights` / `silica.speculative` **must not** contain `torch.Tensor` or `numpy.ndarray` participating in tensor math (numpy is allowed for config / scalar / list helpers).
  2. **No PyTorch runtime dependency.** `pyproject.toml` may not list `torch` as a runtime dep. torch is allowed only as an **optional** dev / extras dependency for offline weight conversion (e.g. `pip install silica-mlx[convert]`); conversion outputs are MLX-native and torch is never touched again after inference starts.
  3. **vllm and transformers are algorithm / architecture references only, never runtime dependencies.** vllm's `csrc/`, `vllm_flash_attn/`, and GPU/TPU/XPU/CPU model runners are all out of scope.
  4. Phase 1 wrapping `mlx-lm` is legal (D-004) **because mlx-lm is itself MLX-native**. Replacing the runtime path with any torch-based wrapper (including transformers, llama.cpp Python bindings, etc.) is forbidden.
- **Rationale:** user hard requirement of "must be native MLX". Silica-MLX's entire value proposition (D-006 "Mac inference platform") rests on MLX's Apple Silicon performance and unified memory advantages; any torch hot path breaks Principle 2 (unified memory first).
- **Consequences:**
  - All attention / sampling / kvcache internals must use `mx.` ops.
  - In Phase 3, if a target model is missing from mlx-lm, `silica.models.*` must rewrite it MLX-native — **no fallback to transformers**.
  - vllm v1 source code is "how to design" reference only, not "how to call" dependency (see 5.4 Reference Map).
  - If a benchmark needs to cite numbers from another inference engine, that comparison run must execute in a separate process — it may not be mixed into the silica runtime.

### D-010 — Phase 1 mlx-lm borrowing boundary

- **Date:** 2026-04-14.
- **Status:** accepted (pinpointed as the easiest silent-rework hazard during two Codex plan reviews).
- **Decision:** Phase 1 **borrows** from `mlx-lm`:
  1. Model structure loading (class construction + state-dict shapes).
  2. The tokenizer.
  3. The weight loader (safetensors → `mx.array`).

  Phase 1 **does not borrow** mlx-lm's rotating KV cache / prompt cache. Silica manages its own KV from day one. `SimpleKVCache` is an **external cache injected into model forward**, not a layer wrapping mlx-lm's internal cache.
- **Day-1 smoke test** (first task in P-1, before any other deliverable): verify whether `mlx_lm.generate_step(cache=...)` or an equivalent entry point accepts an external cache object.
  - **Accepts** → inject `SimpleKVCache` directly; decoupling is clean; P-1 proceeds as planned.
  - **Rejects** → monkey-patch / fork `mlx_lm.models.*` forward logic; P-1 cost estimate is revised upward (triggers R-6).
- **Rationale:** the ownership boundary between D-004 (wrap mlx-lm) and the P-1 `SimpleKVCache` deliverable was underspecified and is the most likely silent-rework point. It must be fixed before P-1 development. mlx-lm's rotating cache is fine for single-request bring-up but incompatible with Silica's paged / prefix / codec ambitions, so it cannot be the starting point for P-2 — otherwise P-2 becomes "strip mlx-lm's cache, bolt in Silica's", which is rework, not upgrade.
- **Consequences:**
  - P-1 Strategy points at this decision rather than just "thin wrapper". D-004 still stands as the **model-execution** decision; D-010 is the **cache** decision. They are complementary, not conflicting.
  - P-2's `PagedKVCache` is a direct upgrade path from `SimpleKVCache`, not a replacement for mlx-lm's internal cache.
  - R-2 mitigation adjusts: when Phase 3 rewrites the model adapter, the cache integration point is already Silica-owned, so stripping mlx-lm's cache is unnecessary.
  - Adds R-6 for the day-1 smoke test failure risk.
  - **Concrete anti-pattern:** `vqbench/vqbench/torch_wrapper/hook.py`'s `VQBenchCache` (a `transformers.Cache` subclass) is the concrete anti-pattern — that design depends on HF cache lifecycle, which is exactly what mlx-lm's rotating cache looks like in the torch world. If Silica borrowed mlx-lm's cache it would grow into a similar shape (locked to the framework's internal cache layout). D-010 is precisely to avoid this trap.
- **References:** D-004, Principle 6, P-1 Deliverables, R-6, §5.5.

### D-011 — v0.1 architecture generality: MoE + Dense dual support

- **Date:** 2026-04-16.
- **Status:** accepted (user on 2026-04-16 explicitly chose Option B over Option A "interface-only, defer to v0.2").
- **Decision:** Silica-MLX v0.1 architecture **must support both MoE and Dense model families generically** — not "dense-only + expand MoE in v0.2", and not "interface reserved for MoE but untested". Concretely:
  1. **Interface generality.** I-1 `ModelAdapter` `state_delta` is permitted to carry MoE router state; I-4 `WeightProvider` adds three per-expert granularity methods — `get_expert(layer_idx, expert_id)` / `prefetch_experts(layer_idx, expert_ids)` / `release_expert(layer_idx, expert_id)`. Dense implementations raise `NotImplementedError("dense provider has no per-expert path")` rather than being no-ops, so a MoE adapter wired to a dense provider fails loudly.
  2. **At least one MoE target actually runs in v0.1.** The P-3 target-model table expands from 2 dense to **4 targets** (2 dense + 2 MoE smoke test): Qwen3.5-27B (dense), Gemma4-31B (dense), Qwen3.5-35B-A3B (MoE), gemma-4-26B-A4B (MoE). "Supports MoE" must be something we've actually run, not just written in the document.
  3. **MoE streaming is the primary P-6 payoff, not a side case.** P-6 `StreamingWeightProvider` uses layer granularity for dense and expert granularity for MoE — the MoE resident set can shrink close to active-params size, which is the natural sweet spot for a 30B+ model on a 48 GB Mac.
- **Rationale:**
  - D-006's "platform as product" framing demands coverage of mainstream model families. In 2026, MoE is the dominant direction at the open-weight 30B+ scale (Qwen3.5-MoE / Gemma 4 MoE / DeepSeek-V*, etc.); excluding MoE means giving up the product's most natural sweet spot.
  - Principle 2 "Apple unified memory first" + the fact that MoE active params are much smaller than total params = the residency win for MoE on a Mac is theoretically an order of magnitude larger than for dense. Not running MoE means Silica-MLX misses its best-selling scenario.
  - "Interface reserved but untested" is a degenerate middle state: without actually running it, we never find out whether the I-4 per-expert abstraction really covers everything top-k routing / expert eviction / prefetch coordination need. At least one MoE target must actually run to close the feedback loop.
- **Consequences:**
  - **I-4 interface gains three methods** (v1.5.0 amendment; the interface is still in Phase 0 freeze-candidate state, so modification is allowed; frozen at P-0 exit).
  - **P-3 workload grows from 2 adapters to 4**, plus the +1 MoE-family complexity (expert routing + gate normalization + top-k expert loading + aux-loss-ignored forward).
  - **P-6 Strategy / Acceptance gain a per-expert residency path** (MoE-only; dense unchanged).
  - **R-1 MoE mitigation.** MoE active params are small; R-1 does not apply to MoE. An unresolved Q-003 does not block the MoE smoke test.
  - **Q-003 context update.** Q-003 is about dense; MoE fit risk is far lower but does **not** substitute for Q-003 resolution (dense fit is still part of the product promise).
  - **M-4 gains a MoE smoke test adapter correctness item** independent of Q-003 gating.
  - **§3.1 Scope / §3.4 Target Models updated in step.**
- **References:** D-006, Principle 2, Principle 9, I-1, I-4, P-3, P-6, Q-003, R-1, M-4.

---

## 10. Open Questions

Resolved questions are not deleted. Mark `Status: resolved` and append a `Resolution:` block for traceability.

### Q-001 — VQ codec auto-selection vs explicit configuration

- **Raised:** 2026-04-14.
- **Status:** open.
- **Question:** in Phase 5, should the VQ codec be (A) user-selected via a CLI flag, or (B) auto-selected by workload?
- **Context:** D-006 says the platform should use VQ "well", which hints at auto-selection; but the Phase 0 interface does not express this capability.
- **Options:**
  - A. Explicit configuration: simple, user in control.
  - B. Auto-selection: matches D-006 framing, but requires a workload profiler.
- **Blocks:** Phase 5 design finalization.
- **Next step:** decide after Phase 4 bench.

### Q-002 — Should Phase 8 priority float up?

- **Raised:** 2026-04-14.
- **Status:** open.
- **Question:** per D-006 (platform is the product), should Phase 8 (OpenAI API + session) float from T2 up to the tail of T1?
- **Context:** if Phase 8 is the "product face", it should come earlier. But building a serving layer before the engine is stable is risky.
- **Options:**
  - A. Keep it in T2: engine stabilizes first.
  - B. Float to tail of T1: native-capability integration (P-5 / P-6) and serving layer proceed in parallel.
- **Blocks:** actual sequencing of Phase 5–8.
- **Next step:** evaluate after Phase 4.

### Q-003 — Should Phase 6 be pulled forward?

- **Raised:** 2026-04-14.
- **Status:** open.
- **Question:** if Phase 3 finds Qwen3.5-27B / Gemma4-31B still don't fit at 4-bit on 48 GB, is Phase 6 (weight streaming) pulled ahead of Phase 5?
- **Context (v1.5.0 update, D-011):** Q-003 is about **dense targets** (Qwen3.5-27B / Gemma4-31B, where total params = active params). **MoE targets** (Qwen3.5-35B-A3B / gemma-4-26B-A4B) have far lower 48 GB fit risk — active params are only 3–4B, fully resident is under half of a dense target, and it only gets easier with P-6 per-expert streaming. A MoE target can serve as an early scale demonstration while Q-003 is unresolved (the M-4 MoE smoke test is not Q-003-gated), but it does **not** substitute for Q-003 resolution — dense fit is still part of the product promise (D-006); users will reach for `Qwen3.5-27B` directly and will not be consoled by "we have MoE".
- **Blocks:** Phase 5 / 6 ordering.
- **Next step:** decide after Phase 3 produces real residency numbers. The MoE path can close without waiting for Q-003.

### Q-004 — `silica.core` vs `silica.engine` boundary

- **Raised:** 2026-04-14.
- **Status:** resolved.
- **Question:** do `Request`, `SamplingParams`, `RequestState` go in `silica.core` or `silica.engine`?
- **Context:** mini-sglang puts them in `minisgl.core`; but "core" tends to bloat.
- **Options:**
  - A. Data classes in core, logic in engine (mini-sglang style).
  - B. Everything in engine; core only holds logging/profiler.
- **Resolution:** Option A. See D-008. All Phase blocks in this document already implicitly use `silica.core.request.*` paths; fixed here.

### Q-005 — Is MetricsRegistry a global singleton?

- **Raised:** 2026-04-14.
- **Status:** open.
- **Question:** does the Phase 0 profiler use a global `MetricsRegistry`, or one per Engine instance?
- **Context:** a global singleton is simple but collides across multiple Engine instances; per-instance is clean but slightly awkward for CLI / bench access.
- **Next step:** decide when Phase 0 starts.

### Q-006 — Should attention backend be a separate interface?

- **Raised:** 2026-04-14.
- **Status:** open.
- **Question:** do we need a standalone `AttentionBackend` Protocol (akin to vLLM v1's `vllm/v1/attention/backend.py`) so attention implementations can be swapped independently of `ModelAdapter`? Or keep attention hidden inside the Module returned by `ModelAdapter.build()`?
- **Context:** vLLM v1 separates the attention backend to support flashattention / flashinfer / xformers / triton variants; we only have MLX, so short-term we don't need that flexibility. But if the Phase 5 VQ wants a compressed-domain attention fast path (the v0.2 capability D-003 leaves open), a standalone `AttentionBackend` makes wiring cleaner.
- **Options:**
  - A. No standalone in v0.1; attention stays inside ModelAdapter (simple, fits the current 5-interface design).
  - B. Standalone AttentionBackend as a sixth interface (prepares for v0.2 compressed-domain attention).
- **Blocks:** none — can wait until Phase 5 bench results.
- **Next step:** evaluate after Phase 5 (jointly with the D-003 v0.2 upgrade).

### Q-007 — KVCodec decode overhead signal for admission control

- **Raised:** 2026-04-14.
- **Status:** open.
- **Question:** should `KVCodec` expose `decode_overhead_ratio: float` (fp16 baseline = 1.0) so scheduler admission control sees both memory savings and decode cost, avoiding the pathological "saves memory but slows decode" combination?
- **Context:** Principle 8 says savings must be visible to the scheduler; I-3 currently only exposes `logical_bytes` / `resident_bytes`. If the scheduler admits more requests purely on memory savings, a codec with a 2× decode slowdown can tank overall tok/s — savings visible, cost invisible — violating the spirit of Principle 8. Choose between minimal v0.1 interface and completeness of scheduler information.
- **Options:**
  - **A. Don't add in v0.1.** Phase 5 users pick a codec explicitly (as in Q-001 Option A); v0.2 revisits with a profile table. Interface stays minimal, but the scheduler cannot actively use savings (violates the spirit of Principle 8).
  - **B. Add `decode_overhead_ratio: float` as a constant on I-3 in v0.1.** Phase 4 bench backfills the number; the scheduler reads it for admission. One interface line avoids the pathology, but a static constant cannot reflect seq-len / batch-size dependence.
  - **C. Keep it out of the interface;** `silica.bench` produces a per-codec profile table and the scheduler reads it. Most precise but heaviest; may be over-engineering for v0.1.
- **Blocks:** finalization of the P-5 scheduler admission policy; coupled with Q-001 (VQ codec auto-selection).
- **Next step:** decide after Phase 4 bench, with real BlockTQ / RaBitQ decode-overhead data. At resolution, also decide whether `feedback_kvcodec_interface.md` memory is updated.

### Q-008 — KVCodec K/V pair configuration

- **Raised:** 2026-04-14.
- **Status:** open.
- **Question:** should `KVCodec` expose K/V pair configuration at the **interface level** (e.g. `KVCodec(key_method=..., value_method=...)`), letting users explicitly pick different codecs for K and V? Or should it stay hidden inside each codec's constructor?
- **Context:** vqbench explicitly uses different codecs for K and V — K needs unbiased inner-product estimation (`TurboQuantProd` / `QJL`), V needs low-MSE reconstruction (`TurboQuantMSE` / `BlockTurboQuantMSE`); this is the basis of `KVCacheCompressor(key_q, value_q)` in `vqbench/vqbench/kv_cache/compressor.py`. Today's I-3 `encode_block(k, v) -> CodedBlock` takes a single codec object that may internally hold two quantizers, but does not **externally** expose the choice. This becomes concrete when P-5's `BlockTQCodec` constructor is designed.
- **Options:**
  - **A. Internal handling, I-3 contract unchanged.** Each codec (e.g. `BlockTQCodec`) accepts `key_method` / `value_method` as constructor args; I-3's signature doesn't move. **Pro:** I-3 stays minimal, swapping codecs doesn't affect the scheduler. **Con:** users have to learn each codec's constructor.
  - **B. I-3 adds a pair contract.** Split `KVCodec` into `KeyCodec` + `ValueCodec` Protocols with a top-level `KVCodecPair` composer. **Pro:** K/V asymmetry is visible in the types. **Con:** I-3 grows from one interface to three; P-0 freeze complexity rises.
  - **C. `KVCodec.from_pair(key_method, value_method)` class method** — exposes pair intent without changing the signature.
- **Blocks:** design of the concrete P-5 `BlockTQCodec` constructor; may affect Q-001's granularity of auto-selection.
- **Next step:** decide when P-5 starts. Current lean: Option A (most conservative, I-3 minimal).

---

## 11. Risks

| ID  | Description | Triggering phase | Mitigation |
| --- | ----------- | ---------------- | ---------- |
| R-1 | Qwen3.5-27B / Gemma4-31B do not fit 48 GB even at 4-bit (dense only; D-011 MoE targets have small active params, so this risk does not apply to MoE) | P-3 | Q-003: pull P-6 weight streaming forward; MoE targets (Qwen3.5-35B-A3B / gemma-4-26B-A4B) serve as an alternative early scale-demonstration path, independent of Q-003 resolution |
| R-2 | The mlx-lm wrapper's Phase 3 replacement cost exceeds expectations | P-3 | Accepted trade-off from D-004; if the cost is too high, defer the underlying rewrite |
| R-3 | Qwen3.5 hybrid attention cache-routing semantics are complex | P-3 | Unit-test coverage in Phase 3; cross-check against the HF reference implementation |
| R-4 | BlockTQ / RaBitQ decode overhead exceeds that of fp16 attention itself | P-5 | Q-001 auto-selection policy; v0.2 may consider a fast path |
| R-5 | MLX / mlx-lm version churn makes dependency locking painful | all | Pin minimum versions in pyproject.toml + periodic CI upgrades |
| R-6 | `mlx-lm` rejects external cache injection (D-010 day-1 smoke test fails) | P-1 | Monkey-patch / fork `mlx_lm.models.*` forward; worst case, P-1 cost is revised upward and the cache integration point is uniformly refactored before P-2 starts; record in P-1 Strategy Notes |

---

## 12. References

### 12.1 External (reference)

- `mini-sglang`: https://github.com/sgl-project/mini-sglang
- SGLang docs: https://docs.sglang.ai/
- vLLM architecture: https://docs.vllm.ai/en/latest/design/arch_overview/
- vLLM v1 design blog: https://blog.vllm.ai/2025/01/27/v1-alpha-release.html
- MLX: https://github.com/ml-explore/mlx
- mlx-lm: https://github.com/ml-explore/mlx-examples/tree/main/llms/mlx_lm
- mlx-flash: https://github.com/matt-k-wong/mlx-flash
- dflash-mlx: https://github.com/Aryagm/dflash-mlx
- flash-moe: https://github.com/danveloper/flash-moe
- DFlash paper: https://arxiv.org/abs/2602.06036

### 12.2 Local reference checkouts (gitignored)

Local reference implementations sit at the repo root. **Algorithm / architecture reference only, never runtime dependencies** (D-009).

| Directory | Project | Main reference points |
| --------- | ------- | --------------------- |
| `vllm/` | vLLM v1 | See §5.4 Reference Map; focal points `vllm/v1/core/`, `vllm/v1/engine/`, `vllm/v1/kv_cache_interface.py`, `vllm/v1/request.py` |
| `mini-sglang/` | Mini-SGLang | Module layering, serving shell, radix prefix cache |
| `vqbench/` | VQBench (includes nested `vqbench/turboquant_plus/`) | See §5.5 Reference Map; P-5 algorithmic reference + Qwen3.5-4B empirical PPL oracle; NumPy + PyTorch codebase — **not** a runtime import (D-009); `VQBenchCache` is the D-010 anti-pattern |

---

## 13. Changelog

- **v1.5.0** (2026-04-16): architecture scope generalized from dense-only to **MoE + Dense dual support**. On 2026-04-16 the user explicitly chose Option B ("architecture general + v0.1 must actually run at least one MoE target") over Option A ("interface-only, defer MoE testing to v0.2"). Changes are grouped by module; no structural redesign.
  - **New D-011**: v0.1 architecture generality fixed; references D-006 + Principle 2 + Principle 9; consequences make the Interface / Phase / Risk / Milestone impact surface explicit.
  - **Interface (I-4 WeightProvider)**: three per-expert granularity methods added — `get_expert(layer_idx, expert_id)` / `prefetch_experts(layer_idx, expert_ids)` / `release_expert(layer_idx, expert_id)`; dense implementations raise `NotImplementedError` (not a no-op, to prevent a MoE adapter from silently degrading onto a dense provider); key constraints add "MoE adapter FFN must go through `get_expert`; `get_layer` may not be used to pull all experts at once". I-1 / I-2 / I-3 / I-5 untouched.
  - **Scope (§3.1, §3.4)**: In Scope adds "MoE + Dense architectural generality (D-011); v0.1 must actually run at least one MoE target"; Target Models table grows from 3 rows (1 dev + 2 dense prod) to 5 rows, adding Qwen3.5-35B-A3B / gemma-4-26B-A4B as MoE generality targets starting Phase 3. The "production target" label is split into "Dense production target" vs "MoE generality target".
  - **P-3 (Model Adapters)**: Goal / Scope change from "two target models" to "four target models (2 dense + 2 MoE smoke test)"; Strategy adds "MoE adapter FFN goes through `get_expert` / `prefetch_experts`" + "MoE inference-only, aux-loss ignored"; Deliverables add `Qwen35MoeAdapter` / `Gemma4MoeAdapter` + top-k gating + per-expert aggregation + MoE-aware registry; Acceptance adds a fourth group, **MoE structural correctness** — (a) fp16 parity on a small MoE control model (logit max diff < 1e-3 over first 50 greedy tokens); (b) Qwen3.5-35B-A3B / gemma-4-26B-A4B load + forward under the quantized path; (c) per-expert call-path unit test (mock WeightProvider, assert `get_expert` is called and `get_layer` is not used to load experts). The Product memory-fit target is explicitly marked **dense-only**; MoE is not Q-003-gated.
  - **P-6 (Weight Streaming)**: Strategy adds "Dense vs MoE residency granularity" — dense is layer-granular, MoE is expert-granular; MoE scheduler prefetch coordination refined (as soon as gate logits arrive, fire `prefetch_experts(top_k_ids)` rather than waiting for the expert FFN to actually execute). Deliverables note "expert eviction policy (e.g. LRU over experts)" + "dense + MoE dual mode". Acceptance adds a "MoE per-expert residency takes effect" quantitative gate (24 GB budget, `resident_bytes ≤ active_experts × expert_size + non_FFN + headroom ≤ 20%`) and a "MoE decode tok/s ≥ 60% of MoE ResidentWeightProvider baseline" bar (below dense's 70%, acknowledging expert-miss stalls the first cut does not optimize).
  - **Risk / Question / Milestone**: R-1 description adds "MoE active params small, this risk does not apply to MoE" + mitigation note covering MoE as early scale demonstration; Q-003 gains a Context paragraph clarifying "Q-003 is about dense; MoE fit risk is far lower but does not substitute for Q-003 resolution"; M-4 acceptance adds MoE smoke test adapter correctness, independent of Q-003 gating.
  - **Naming normalization**: "Gemma 4 31B" → "Gemma4-31B" across the document, aligned with the user's 2026-04-16 spelling. Qwen3.5-27B / Qwen3.5-35B-A3B / gemma-4-26B-A4B kept as the user-provided IDs.
  - **Review gaps carried forward**: this round only handles the MoE scope decision. The other five freeze-critical gaps raised in the previous Opus review — (1) MLX paged attention risk (proposed R-7 + Q-009); (2) Sampler as I-6; (3) `resident_bytes` canonical measurement (proposed D-012); (4) `RequestState` gains `PREEMPTED`; (5) chunked prefill promoted from "contingent" to deliverable; plus (6) the structured-output / logit-processor middle state — are **not** landed in this version and will be addressed in later v1.5.x rounds to keep this revision's scope contained.
  - **Language pass**: the entire document is now English. Previously mixed Chinese/English prose has been translated and lightly polished; stable IDs, code blocks, tables, URLs, and model names preserved.
- **v1.4.1** (2026-04-14): Codex round-3 review — targeted optimizations, **no structural redesign**.
  - §2 Success criteria gains "minimal OpenAI-compatible HTTP API + session usable (via Phase 8)", aligning with §3.1 scope and D-006.
  - §8.1 Priority Tier labels `P0 / P1 / P2` → **`T0 / T1 / T2`** to avoid visual collision with phase IDs `P-0 / P-1 / P-2` (Q-002 Option B updated in step).
  - **P-3 Acceptance split into two tiers:** **Adapter correctness (hard gate)** — load + one forward + logits max diff < 1e-3 over 50 greedy tokens + hybrid attention routing unit tests; **Product memory-fit target (conditional, gated on Q-003)** — "27B/31B @ 48 GB 500 tokens"; if P-6 is not pulled forward, a smaller model or manual cap is permitted. M-4 milestone updated accordingly so P-3 is not self-blocked before P-6.
  - P-5 Acceptance ε default pinned as `max(2× fp16 baseline noise, 0.01 PPL)`, tightenable at P-4 exit.
  - P-6 Acceptance adds concrete threshold **decode tok/s ≥ 70% of `ResidentWeightProvider` baseline @ 24 GB budget** with fixed same-machine/model/scenario/sampling conditions; if resident OOMs at 24 GB, baseline is "uncapped resident reference run".
  - P-7 Acceptance adds concrete threshold **decode tok/s ≥ 1.2× draft-disabled baseline** on a fixed standard scenario; **cherry-picking a best case is not acceptable**; greedy token-by-token identity as correctness invariant.
  - Module rename **`silica.flash` → `silica.weights`** (avoids overload with FlashAttention, mirrors `silica.kvcache`): §5.1 Module Layout, P-3 Deliverables (`silica.weights.resident.ResidentWeightProvider`), P-6 Deliverables (`silica.weights.streaming.StreamingWeightProvider`, `silica.weights.prefetch`), D-009 module list all synchronized. Principle 1 / D-006 Consequences informal "VQ / flash" phrasing rewritten to "VQ / weight streaming". P-6 phase name "Weight Streaming" was changed in v1.3.0 and is unchanged here. External repo names `mlx-flash` / `vllm_flash_attn` / `flash-moe` kept as is.
  - D-008 typo fix ("不定会导致" → "不定下来会导致"); the then-still-Chinese-heavy document was not pass-translated in this round.
  - **Round 2 correction (pre-commit).** Codex re-reviewed v1.4.1 before commit and caught three points; all fixed before commit.
    1. **P-3 logit threshold vs quantization path unit mismatch.** `max |logit diff| < 1e-3` corresponds to fp16 parity, but D-005 requires 4-bit / 8-bit for big models, where the threshold is unreachable. Split P-3 Acceptance into (a) **fp16 parity on a control model** (Qwen3-0.6B or Qwen3.5-4B) verifying adapter structural components, keeping `max |logit diff| < 1e-3`; (b) **quantized big-model correctness** using **teacher-forced next-token argmax agreement ≥ 98%** over the first 100 teacher-forced positions against `mlx-lm` at the same quantization, compared position-by-position on a fixed prefix; fallback is end-to-end PPL drift `< 0.1` absolute vs the same baseline. We do not compare against free-running sequences, because drift masks or amplifies real differences.
    2. **M-4 deferred product-target ownership gap.** After v1.4.1 narrowed M-4 to adapter correctness, the real `27B/31B @ 48 GB 500 tokens` validation had no milestone to land on. Changed to a **Q-003-gated handoff**: M-4 validates only if Q-003 resolves to "int4 fits in 48 GB"; otherwise the item defers to M-7. No new M-10 — avoids milestone fragmentation.
    3. **P-5 cross-check threshold unit mismatch.** `max(2× fp16 baseline noise, 0.01 PPL)` mixes tensor-space reconstruction error with end-to-end PPL under a single `max` — dimensionally incomparable. Split into two independent thresholds with explicit metrics: `ε_recon < 2 × fp16 round-trip baseline noise`, where the metric is **per-block relative Frobenius error** `||K_decoded - K_original||_F / ||K_original||_F` (same for V), baseline established via vqbench's own fp16 encode→decode round trip; `ε_ppl < 0.01` absolute PPL drift vs `vqbench/REPORT.md` baseline. Both must pass for acceptance.
- **v1.4.0** (2026-04-14): vqbench added as a local reference checkout, bringing integration updates along the VQ path (VQ's framing is promoted from "one P-5 capability" to "a core capability of the final product, whose correctness is now testable against the vqbench empirical baseline").
  - Adds **§5.5 Reference Map to vqbench** — Silica-module ↔ vqbench-file mapping (BlockTQ / RaBitQ algorithms, codec factory, `KVCacheCompressor` pair pattern, Qwen3.5 bench scripts, `REPORT.md` PPL oracle), plus forbidden paths (`torch_wrapper/` / NumPy impls, D-009).
  - §12.2 Local reference checkouts: top-level `turboquant_plus/` replaced by `vqbench/` (turboquant_plus is now nested inside vqbench).
  - P-5 Strategy reference source switched to vqbench, noting the Qwen3.5-4B `B=64` 4-bit +0.0% ΔPPL empirical baseline.
  - P-5 Acceptance gains a **numeric cross-check against vqbench** — Silica `BlockTQCodec` reconstruction error + end-to-end PPL must match the vqbench NumPy reference, making "faithful rewrite" testable.
  - P-4 Deliverables gain `silica.bench.vqbench_baseline` — a separate-subprocess run of vqbench scripts to collect PPL as a reference column, serving the P-5 cross-check via the D-009-permitted "separate-process comparison" path.
  - D-010 Consequences adds `VQBenchCache` (HF `Cache` subclass) as a concrete **anti-pattern** — a living example of what "borrowing mlx-lm's cache" would degenerate into.
  - Adds **Q-008** discussing whether `KVCodec`'s interface layer should expose the K/V pair choice (vqbench shows K wants an unbiased-IP codec while V wants low-MSE; three options A/B/C, lean A).
- **v1.3.0** (2026-04-14): framing unification — plugin → native capability. User clarifies: VQ / weight streaming / speculative decoding are Silica-MLX's **native capabilities**, not "third-party plugin extension points"; built into the main loop as stubs from P-0, progressively replaced in P-5 / P-6 / P-7; integration points fixed, implementations swappable.
  - Adds **Principle 9 — Native capabilities, swappable implementations**, making the architectural stance explicit with the vLLM attention-backend analogy.
  - Principle 3 "Engine first, plugins later" → "Engine skeleton first, native capabilities integrated progressively".
  - Principle 5 "Plugin contracts are frozen early" → "Native capability contracts are frozen early".
  - §1 TL;DR / §2 Mission success criteria / §3.1 In Scope: three "plugin" phrasings rewritten to "native capability".
  - P-5 title "VQ Plugin" → "VQ KV Compression" + Goal / Strategy aligned; P-6 "Flash Plugin" → "Weight Streaming"; P-7 "Speculative Plugin" → "Speculative Decoding".
  - §8.1 P1 tier label / P-2 Notes / Q-002 Option B — scattered "plugin" phrasing removed uniformly. Interface names (`KVCodec` / `WeightProvider` / `DraftEngine`) and P-0 deliverable structure unchanged; this is a framing alignment.
- **v1.2.0** (2026-04-14): Two rounds of Codex plan review reconciliation.
  - §6 header "Frozen for v0.1" → "Phase 0 freeze candidate" — signatures finalized at P-0 exit, not the moment this document is written.
  - I-1 adds `prefill(tokens, kv_handle) -> (logits, StateDelta)` and `decode_step`; KV mutation ownership clarified as `KVManager`-only (via `kv_handle`); `state_delta` carries non-KV runtime state only (counter-examples: KV blocks / cache residency mutations / prefix pinning are not permitted).
  - I-2 `allocate(request_id, num_tokens)` split into `reserve_for_prefill(req_id, token_ids)` + `append_slot`; adds `commit` / `rollback` (for P-7 speculative); `prefix_lookup` renamed `get_computed_blocks` (vLLM v1 naming); adds `available_blocks` as the block-granular fast path for scheduler admission.
  - Adds **D-010** fixing the Phase 1 `mlx-lm` borrowing boundary — borrow model/tokenizer/weight loader, not the rotating KV cache; P-1 day-1 smoke test verifies `mlx_lm.generate_step(cache=...)` external-cache injection.
  - P-1 Strategy / Deliverables aligned to D-010; adds the day-1 gate.
  - P-5 Strategy clarifies that `BlockTQCodec` / `RaBitQCodec` must be rewritten on `mx.array` + resident-accounting; not a reuse of `turboquant_plus` NumPy prototypes; engineering work pre-declared to avoid scope creep.
  - Adds **Q-007** on whether `KVCodec.decode_overhead_ratio` should enter the v0.1 interface (Principle 8 completeness of scheduler information).
  - Adds **R-6** for the day-1 smoke-test failure fallback.
  - §5.2 data flow: `ModelAdapter.forward` → `ModelAdapter.prefill / decode_step (KV via kv_handle from KVManager)`.
- **v1.1.2** (2026-04-14): user pulled vllm as a local reference and confirmed "must be native MLX". Adds D-009 fixing the MLX-native hot-path constraint (no `torch.Tensor`, no PyTorch runtime dep); rewrites Principle 6; §3.2 Non-Goals adds PyTorch / CUDA backend exclusions; adds §5.4 Reference Map to vLLM v1 (what's referenced, what's not); adds Q-006 on whether AttentionBackend should be separate; §12 References splits External / Local checkouts and registers `vllm/`.
- **v1.1.1** (2026-04-14): two CRUD review fixes — §5.1 directory tree adds `llm/` sub-package (aligned with P-8 deliverable); Q-004 marked resolved and D-008 added to fix the core/engine boundary.
- **v1.1** (2026-04-14): Mission and Principle 1 rewritten — platform is the product, VQ is the means (D-006); adds Principle 2 "Apple unified memory first"; adds Principle 8 "savings must be observable"; Phase 5 / 8 framing rewritten; Q-002 opened on Phase 8 priority; Q-005 on MetricsRegistry; Risks table added; whole document organized in CRUD-friendly form (stable IDs P-N / D-NNN / Q-NNN / M-N / I-N, self-contained blocks, append-only logs).
- **v1.0** (2026-04-14): based on Codex polish of v0; unified phase structure.
- **v0** (2026-04-13): initial plan draft (conversational; never committed).
