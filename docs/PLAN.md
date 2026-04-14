# Silica-MLX Plan

| Field        | Value                                  |
| ------------ | -------------------------------------- |
| Version      | v1.4.1                                 |
| Last updated | 2026-04-14                             |
| Status       | Phase 0 planned                        |
| Maintainer   | Xin Zhou                               |
| Source       | `docs/PLAN.md` (single source of truth) |

> **CRUD 约定**: 本文档所有稳定 ID (Phase `P-N`, Decision `D-NNN`, Open Question `Q-NNN`, Milestone `M-N`, Interface `I-N`) 一经分配不再复用。编辑请只改对应 block；新事实走 Decisions Log; 新问题走 Open Questions; Phase 状态变化改对应 Phase block 的 `Status` 字段并在 Changelog 追加一行。

---

## 目录

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

**Silica-MLX 是一个单 Mac 芯片的本地 LLM 推理平台**，MLX-native，vLLM-style core + mini-sglang-style 外层。目标是让 M5 Pro 48GB 能稳定跑 Qwen3.5-27B / Gemma 4 31B。VQ、weight streaming、speculative decoding 作为**原生能力**被平台主动利用（不是被动支持的第三方扩展），实现可换但集成点固定 —— 详见 Principle 9。

**平台本身是产品**。VQ 等技术是武器，不是研究对象。(见 D-006)

---

## 2. Mission

在单台 Apple Silicon Mac 上，让开发者能像在云上用 vLLM 一样流畅地本地跑 27B–31B 量级的大模型。

**目标用户**: Mac 上想本地跑大模型做 app、做实验、做隐私敏感工作的开发者。

**不是**: VQ 算法研究者、分布式 serving 系统研究者、云厂商。

**成功标准 (v0.1)**:
- 在 48GB M5 Pro 上稳定跑通 Qwen3.5-27B 和 Gemma 4 31B
- 提供 Python API + 最小 CLI
- 统一 benchmark / runtime 路径
- VQ / weight streaming / speculative 作为原生能力, 从 P-0 就以 stub 形式内建于主循环, 在 P-5 / P-6 / P-7 逐步替换为真实实现; 集成点固定, 实现可换 (Principle 9)
- 最小 OpenAI-compatible HTTP API + session 可用 (via Phase 8), 与 §3.1 scope 和 D-006 "产品脸面" 定位对齐

---

## 3. Scope

### 3.1 In Scope (v0.1)

- 单 Mac 芯片、单进程本地推理
- MLX-native，不依赖 CUDA 假设
- Qwen3.5-27B、Gemma 4 31B 两个目标模型
- Python API + 最小 CLI
- vLLM-style core: paged KV, continuous batching, prefix cache, memory budget
- 五个冻结接口: ModelAdapter / KVManager / KVCodec / WeightProvider / DraftEngine
- 原生能力 (native capabilities): VQ KV 压缩 (BlockTQ / RaBitQ 实现)、weight streaming、draft-target speculative —— 详见 Principle 9
- 统一 benchmark 路径
- 最小 OpenAI-compatible HTTP API + session (Phase 8)

### 3.2 Non-Goals (v0.1)

- 分布式、多节点、多 Mac 协作
- PD (prefill/decode) 分离
- Tensor parallelism (单芯片不需要)
- 一开始就重写 native Metal kernel
- GGUF / AWQ 等非 MLX 量化格式
- 完整 agent orchestration
- PQ / OPQ 等被明确排除的 codec
- 压缩域直接 attention fast path (见 D-003，留 v0.2)
- DFlash / EAGLE / Medusa 等复杂 speculative 方案 (留 v0.2)
- **PyTorch 运行时依赖** (D-009): inference hot path 不允许出现 `torch.Tensor`; torch 只能作为 offline 权重转换的可选 dev 依赖
- **CUDA / ROCm / XPU / TPU 后端** (D-009): `csrc/`、CUDA kernels、device-specific worker 都不在范围内

### 3.3 Target Hardware

| 字段 | 值 |
| --- | --- |
| 机型 | Apple M5 Pro |
| 内存 | 48 GB unified memory |
| OS | macOS |
| 加速 | MLX on Apple Silicon GPU + Neural Engine (MLX-managed) |

### 3.4 Target Models

| 模型 | 参数量 | 用途 | 起用阶段 |
| --- | --- | --- | --- |
| Qwen3-0.6B | 0.6B | 开发/迭代起步模型 | Phase 1 |
| Qwen3.5-27B | 27B | 生产目标 | Phase 3 |
| Gemma 4 31B | 31B | 生产目标 | Phase 3 |

---

## 4. Design Principles

稳定原则；修改需进 Decisions Log 并开新条目。

1. **Platform as product, VQ as weapon.** Silica-MLX 本身是产品；VQ / weight streaming / speculative 是让产品跑好大模型的手段，不是研究对象。不要把"VQ 是终极 deliverable"写进任何设计。(D-006)

2. **Single Mac chip + Apple unified memory first.** 硬约束是单芯片；不做分布式。但要主动利用 unified memory —— 权重/KV/activations 共享同一物理池的事实必须反映到 WeightProvider 和 KVManager 设计里，不能把 Mac 抽象成一个普通 GPU。

3. **Engine skeleton first, native capabilities integrated progressively.** Phase 0–4 先把 engine 骨架 + baseline + 目标模型跑通 (核心主循环从 P-0 起就包含所有原生能力的 stub: `IdentityCodec` / `ResidentWeightProvider` stub / `NoopDraftEngine`); Phase 5–8 逐步把 stub 替换成真实实现 (VQ / streaming / speculative) 并补 serving 层。**不是**"先做 engine 再加插件"—— 集成点从 P-0 就存在, P-5..P-7 是替换 stub, 不是新接入 (见 Principle 9)。

4. **Bench and runtime share the same path.** Benchmark 必须是 `silica.engine.Engine` 的薄封装。不能有第二条评测路径。

5. **Native capability contracts are frozen early.** 五个核心接口 (ModelAdapter / KVManager / KVCodec / WeightProvider / DraftEngine) 是原生能力的集成点, Phase 0 候选冻结, P-0 退出前最终冻结 (见 §6); scheduler / engine core 不感知具体实现。改动需走 Decisions Log。这些**不是**"第三方插件扩展点", 是 Silica 自己的能力边界 (见 Principle 9)。

6. **MLX-native hot path (hard constraint).** Inference hot path 必须 100% 走 MLX: 所有 tensor 都是 `mx.array`, 所有 ops 都走 MLX。**不允许** `torch.Tensor` / `numpy.ndarray` 出现在 `silica.engine` / `silica.mlx` / `silica.kvcache` / `silica.models` / `silica.scheduler` 的 hot path。Phase 1 可以 wrap `mlx-lm` **因为 mlx-lm 本身就是 MLX-native**，不能替换成 torch-based 的 wrapper。vllm / transformers 仅作为**算法参考**, 不作为运行时依赖。详见 D-009。

7. **Small over large in early phases.** Phase 1 起步用 Qwen3-0.6B，Phase 3 才切到目标大模型。先把闭环跑通再扩大规模。

8. **Savings must be observable.** 任何压缩/流式/加速优化带来的节省必须能被 scheduler 看见 (e.g. `KVCodec.logical_bytes` vs `resident_bytes`)，否则 memory budgeter 没法把节省转化为"多收请求/更长 context"。

9. **Native capabilities, swappable implementations.** VQ KV 压缩、weight streaming residency、speculative decoding 是 Silica-MLX 的**原生能力 (native capabilities)**, 不是"第三方插件扩展点"。从 P-0 开始就以 stub 形式内建在 engine 主循环 (`IdentityCodec` / `ResidentWeightProvider` stub / `NoopDraftEngine`), P-5 / P-6 / P-7 逐步替换为真实实现。**集成点 (integration point) 固定**: memory budgeter 读 `KVCodec.logical_bytes` / `resident_bytes`、scheduler 读 `WeightProvider` prefetch 信号、decode loop 接 `DraftEngine.propose` / `commit`。**实现 (implementation) 可换**: BlockTQ / RaBitQ / 未来 codec; Resident / Streaming / 未来 residency 策略; Noop / DraftTarget / 未来 EAGLE / Medusa —— 都是同一集成点下的可切换实现。类比: vLLM 的 "attention backend" —— native 层面每个模型都走一个 backend, 但具体 backend (FlashAttention / Xformers / Triton) 可换。我们**不说 "plugin"**, 说 "backend / codec / implementation"。与 Principle 1 (platform as product, VQ as weapon) 互补 —— Principle 1 讲产品 stance, Principle 9 讲架构体现。

---

## 5. Architecture Overview

### 5.1 Module Layout

目标布局 (Phase 0 后以实际仓库为 source of truth):

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
     → KVManager.get_computed_blocks / reserve_for_prefill (KVCodec 透明接入)
   → ModelAdapter.prefill / decode_step (through WeightProvider; KV via kv_handle from KVManager)
     → KVCodec.encode_block / decode_block per layer
   → Sampler (+ optional DraftEngine)
 → token stream
```

Phase 8 在最上层加一个 `silica.server.openai_api` FastAPI wrapper，内部还是同一个 Engine 实例。

### 5.3 Process Model

v0.1 **单进程**。无 tokenizer worker 拆分，无 detokenizer 拆分，无 scheduler worker 拆分。未来如果要拆，作为 v0.2 讨论。

### 5.4 Reference Map to vLLM v1

**仅用作算法/架构参考**，不是运行时依赖。vllm 仓库本地路径: `vllm/` (gitignored)。vllm 的 v0 (老的 `engine/llm_engine.py`) **不是**我们的参考目标; 我们对标的是 v1 (`vllm/v1/`)。

| Silica 模块 | vLLM v1 参考文件 | 用途 |
| --- | --- | --- |
| `silica.engine.Engine` | `vllm/v1/engine/llm_engine.py`, `vllm/v1/engine/core.py` | 引擎主循环结构 |
| `silica.scheduler.batcher` | `vllm/v1/core/sched/scheduler.py` | continuous batching 调度 |
| `silica.scheduler.budget` | `vllm/v1/core/kv_cache_manager.py` (budget 部分) | 内存预算与 admission |
| `silica.kvcache.paged.PagedKVCache` | `vllm/v1/core/block_pool.py`, `vllm/v1/core/kv_cache_manager.py` | paged/block KV 分配器 |
| `silica.kvcache` (接口) | `vllm/v1/kv_cache_interface.py` | KV cache spec 与 layout |
| `silica.kvcache.prefix` | `vllm/v1/core/kv_cache_manager.py` (prefix 部分) | prefix cache 命中与复用 |
| `silica.core.request.RequestState` | `vllm/v1/request.py` | 请求状态机 |
| `silica.core.sampling` / sampler | `vllm/v1/sample/` | 采样实现参考 |
| `silica.models.*` attention | `vllm/v1/attention/backend.py` (仅接口思路) | attention backend 可插拔模式; **不 copy CUDA 实现** |
| `silica.speculative.*` | `vllm/v1/spec_decode/` | speculative decoding 架构参考 |

**不参考**的部分:
- `vllm/csrc/` — C++/CUDA 内核
- `vllm/vllm_flash_attn/` — CUDA flash attention
- `vllm/v1/worker/gpu_model_runner.py`, `tpu_model_runner.py`, `xpu_model_runner.py`, `cpu_model_runner.py` — 设备特定 runner; 我们只写一个 `silica.mlx.runner`
- `vllm/v1/executor/`, `vllm/distributed/`, `vllm/v1/core/kv_cache_coordinator.py` — 多进程 / 多节点协调, v0.1 不需要
- vllm 的 v0 路径 (`vllm/engine/`, `vllm/worker/` 顶层) — 已被 v1 取代

### 5.5 Reference Map to vqbench

**仅用作算法 + 数值 oracle**, 不是运行时依赖。vqbench 本地路径: `vqbench/` (gitignored, 含 nested `vqbench/turboquant_plus/`)。vqbench 是 **NumPy + PyTorch + HF transformers** 代码库, 符合 D-009 禁令 —— **不作为 runtime 来源**; 但它有 Qwen3.5-4B 的实证 PPL 基线 (`BlockTurboQuantMSE B=64` 4-bit K+V near-lossless, 见 `vqbench/REPORT.md`), 可作为 Silica P-5 MLX-native 重写的 **correctness oracle**。

| Silica 模块 | vqbench 参考文件 | 用途 |
| --- | --- | --- |
| `silica.vq.block_tq` | `vqbench/vqbench/methods/turboquant/block_mse.py` | BlockTurboQuantMSE 算法 (B=16/20/32/40/64 配置) |
| `silica.vq.rabitq` | `vqbench/vqbench/methods/rabitq/rabitq_1bit.py`, `rabitq_ext.py` | RaBitQ 1-bit / 扩展位 |
| `silica.vq` (工厂) | `vqbench/vqbench/torch_wrapper/module.py` (`_get_method_class`) | codec 注册 / 命名约定参考 |
| `silica.kvcache` (pair 层) | `vqbench/vqbench/kv_cache/compressor.py` (`KVCacheCompressor`) | K/V 独立 codec 的 pair 组合模式 (见 Q-008) |
| `silica.bench.scenarios` | `vqbench/scripts/reproduce_qwen35_4b_headline.py`, `variance_qwen35_4b.py`, `run_qwen35_27b_sweep.py` | Qwen3.5 bench 场景 + PPL 回归参考 |
| P-5 数值 oracle | `vqbench/REPORT.md` (Qwen3.5-4B 结果表), `vqbench/BlockTQ.md` | 实证基线 + 算法 walkthrough |

**不参考 (禁止作为 runtime 来源)**:
- `vqbench/vqbench/torch_wrapper/` — PyTorch `nn.Module` + HF `DynamicCache` 子类, 违反 D-009; 作为"反例精神"写入 D-010 Consequences
- `vqbench/vqbench/methods/*.py` 的 NumPy 具体实现 — 算法思路参考, **不是** `mx.array` 等价物; P-5 必须重写成 MLX-native
- `vqbench/PLAN.md` — vqbench 自己的历史 plan, 和 Silica plan 无关; **不是** stale copy, 不要混用

---

## 6. Core Interfaces (Phase 0 freeze candidate)

v0.1 的五个核心接口, **签名在 P-0 退出前确定** —— 不是"文档一贴就冻"。Phase 0 开发过程中允许补足操作集 (例如 I-1 / I-2 本版新增的 prefill/decode_step 和 append/commit/rollback), P-0 退出那一刻才真正冻结。P-0 之后要动签名需进 Decisions Log 加新条目。下面给的是契约骨架, 实际 `typing.Protocol` 签名以代码为准。

### I-1 ModelAdapter

负责模型结构、tokenizer、layer execution、attention pattern 和执行语义 (prefill / decode)。

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

**关键约束**:
1. `attention_pattern()` 必须能表达 Qwen3.5 的 hybrid attention —— 不同 layer 走不同 cache routing, 否则 Phase 3 会被迫改 scheduler。
2. **KV mutation ownership 归 `KVManager`, 不归 adapter**。`prefill` / `decode_step` 通过 `kv_handle` (由 `KVManager` 颁发) 读写 KV, 不直接持有 block 指针, 不做 residency 决策, 不触碰 prefix cache 结构。
3. `state_delta` 仅承载**非-KV** runtime state: sampling RNG、MoE router 缓存、position counter、sliding-window 掩码游标 等。**反例 (不属于 `state_delta`, 不允许放进去)**: KV blocks / cache residency mutations / prefix cache pinning —— 这些一律通过 `kv_handle` 由 `KVManager` 操作。

### I-2 KVManager

负责 paged/block KV、prefix cache、memory budget, 以及 continuous batching 和 speculative decoding 所需的增量 mutation 原语。

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

**关键约束**:
1. `budget()` 必须同时报告 logical 和 resident 字节 (Principle 8)。Scheduler 基于此做 admission control。
2. **增量语义**: `reserve_for_prefill` 为初始 prompt 预留 block (强调"预留", 不是"逻辑分配完就写入"); `append_slot` 随 decode 增量追加; `commit` / `rollback` 支持 speculative decoding 的 accept/reject (P-7); `free` 在请求结束时释放。P-1 的 `SimpleKVCache` 单请求实现可以把 `commit` / `rollback` 实现为 no-op, 但**签名必须在 P-0 就存在**, 否则 P-7 会被迫改 frozen API。
3. `get_computed_blocks` 是 prefix cache 命中查询 (命名对齐 vLLM v1 `kv_cache_manager.get_computed_blocks`), 返回已计算的 block list, 供 scheduler 在 admit 时复用并 pin。
4. `available_blocks()` 是 block 级粒度的快速路径, **不等同于** `budget().headroom` —— 后者是字节级 (被 codec 的 logical/resident 比影响), 前者是 block 级 (给 scheduler 一个"还能不能再收一个请求"的 O(1) 判断)。

### I-3 KVCodec

KV 编解码的统一抽象。**v0.1 不包含**压缩域直接 attention fast path (D-003)。

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

**关键约束**: codec 只看一个 layer 的 K/V，不感知 batch / scheduler / 模型结构。

### I-4 WeightProvider

权重驻留/流式抽象。Phase 1 用 Resident，Phase 6 接 Streaming。

```python
class WeightProvider(Protocol):
    def get_layer(self, layer_idx: int) -> LayerWeights: ...      # sync blocking
    def prefetch(self, layer_indices: Sequence[int]) -> None: ...  # hint, may no-op
    def release(self, layer_idx: int) -> None: ...
    def resident_bytes(self) -> int: ...
```

**关键约束**: `get_layer` 同步阻塞；`prefetch` 是提示，可以是 no-op。调用方不需要知道权重实际在哪。Streaming 实现要利用 unified memory (Principle 2)，而不是假装是 PCIe 设备。

### I-5 DraftEngine

speculative decoding 的 draft 提供方。Phase 1–6 用 Noop，Phase 7 接 DraftTarget。

```python
class DraftEngine(Protocol):
    def propose(self, ctx: RequestState, k: int) -> DraftTokens: ...
    def commit(self, ctx: RequestState, accepted_len: int) -> None: ...
```

---

## 7. Phases

每个 Phase 的结构统一: `ID / Goal / Scope / Strategy / Deliverables / Acceptance / Dependencies / Status / Notes`。编辑某个 Phase 时只动该 block。状态值: `planned / in-progress / done / blocked / obsolete`。

### P-0 Phase 0 — Skeleton

- **Goal**: 把 repo、包结构、配置、日志、profiling、接口骨架搭起来。
- **Scope**: `pyproject.toml`, 11 个子包骨架, 五个 Protocol, logging, profiler, 最小 interface test。
- **Strategy**:
  - uv + Python 3.12 (D-001)
  - stdlib `logging` + 彩色 formatter (不上 structlog)
  - `pydantic` v2 BaseSettings 做 config
  - profiler 用 `@contextmanager`, 全局 `MetricsRegistry`, 统一 metrics schema
- **Deliverables**:
  - [ ] `pyproject.toml` (uv, mlx, mlx-lm, numpy, pydantic, pytest, ruff, mypy; Python 3.12)
  - [ ] 11 个子包骨架 + `__init__.py`
  - [ ] `silica.core.request.Request` / `RequestState`
  - [ ] `silica.core.sampling.SamplingParams`
  - [ ] `silica.core.logger` + `silica.core.profiler`
  - [ ] 五个 Protocol (I-1..I-5) + 每个的 stub 实现 (`StubModelAdapter`, `NullKVManager`, `IdentityCodec`, `ResidentWeightProvider` stub, `NoopDraftEngine`)
  - [ ] `tests/test_interfaces.py` 验证 Protocol 形状 + stub 实例化
  - [ ] `from silica import Engine` 可 import (即使是 `NotImplementedError` stub)
- **Acceptance**:
  - [ ] `uv pip install -e .` 成功
  - [ ] `pytest tests/test_interfaces.py` 通过
  - [ ] 统一 metrics schema 有字段: `ttft_ms`, `prefill_tok_s`, `decode_tok_s`, `resident_mb`, `logical_kv_bytes`
- **Dependencies**: 无
- **Status**: planned
- **Notes**: 这是所有后续工作的基座；接口一旦冻结，改动需走 Decisions Log。

### P-1 Phase 1 — Baseline Engine

- **Goal**: 最小可跑的 MLX-native 推理主循环。
- **Scope**: 单请求, greedy/temperature/top-p sampling, token streaming, 最小 CLI。
- **Strategy**: Phase 1 的 `ModelAdapter` 借用 `mlx-lm` 的 (a) 模型结构加载 (b) tokenizer (c) 权重 loader, 但**不借用** `mlx-lm` 的 rotating KV cache / prompt cache —— ownership 边界见 **D-010** (补充 D-004)。KV 用 `SimpleKVCache` (单请求, 非 paged), 经 `kv_handle` 注入 adapter 的 `prefill` / `decode_step`; P-2 的 `PagedKVCache` 是它的直接升级路径, 不是替换 mlx-lm 内部 cache。P-1 第一项任务是 **day-1 smoke test**: 验证 `mlx_lm.generate_step(cache=...)` 或等价入口能否接受外部 cache 对象; 结果决定 P-1 剩余 deliverable 的实际成本 (见 D-010 / R-6)。
- **Deliverables**:
  - [ ] **Day-1 gate**: smoke test 验证 `mlx_lm` 接受外部 cache 注入 (D-010); 不通过则先解决该 blocker 再展开下面的 deliverable
  - [ ] `silica.engine.Engine.generate(prompt, sampling_params)` 返回 token 流
  - [ ] `silica.mlx.runner` 包装 mlx-lm 的 forward (经外部 cache 注入, 不使用 mlx-lm 内部 cache)
  - [ ] `silica.kvcache.simple.SimpleKVCache` (单请求版, 以 `kv_handle` 形式传入 adapter)
  - [ ] `silica.server.cli`: `python -m silica run --model Qwen/Qwen3-0.6B --prompt "..."`
  - [ ] 基础采样: greedy, temperature, top-p
  - [ ] `silica.models.qwen3.Qwen3Adapter` (借 mlx-lm 结构, KV 自管, 见 D-010)
- **Acceptance**:
  - [ ] 能稳定生成文本
  - [ ] greedy 下和 mlx-lm 参考实现逐 token 一致 (固定 seed / 同一模型)
  - [ ] 能从 profiler 拿到 TTFT, decode tok/s, resident memory
- **Dependencies**: P-0
- **Status**: planned
- **Notes**: 开发机模型: Qwen3-0.6B。不要在 Phase 1 就上 27B/31B。

### P-2 Phase 2 — Mini-vLLM Core

- **Goal**: 真正的 engine 骨架 —— paged KV + continuous batching + prefix cache + memory budget。
- **Scope**: 多请求并发, 共享前缀命中, 内存预算管理, 请求状态机。
- **Strategy**:
  - Block size 默认 16 tokens (Phase 4 bench 后评估)
  - Radix prefix cache 参考 mini-sglang
  - chunked prefill 先不做, Phase 3 如果被 OOM 逼出来再加
- **Deliverables**:
  - [ ] `silica.kvcache.paged.PagedKVCache` 实现 KVManager (I-2)
  - [ ] `silica.kvcache.prefix.RadixPrefixCache`
  - [ ] `silica.core.request.RequestState` 状态机: `WAITING → PREFILL → DECODE → DONE/ABORTED`
  - [ ] `silica.scheduler.batcher.ContinuousBatcher`
  - [ ] `silica.scheduler.budget.MemoryBudgeter`
- **Acceptance**:
  - [ ] 同时跑 8 个 request 稳定
  - [ ] 构造共享前缀 prompt 能命中 prefix cache (prefix cache hit 计数可验证)
  - [ ] 内存超预算时 abort / 排队不崩
- **Dependencies**: P-1
- **Status**: planned
- **Notes**: 这是最重要的一个 Phase; 骨架决定后续所有原生能力 (VQ / streaming / speculative) 的 stub 替换路径 (Principle 9)。

### P-3 Phase 3 — Model Adapters

- **Goal**: Qwen3.5-27B 和 Gemma 4 31B 真正跑通。
- **Scope**: 两个目标模型的 adapter, Qwen3.5 hybrid attention, MLX-native 4/8-bit 量化。
- **Strategy**:
  - 量化走 mlx-lm 现成的 4-bit / 8-bit 路径 (D-005)
  - 权重加载走 `WeightProvider`, 即使此阶段仍是 `ResidentWeightProvider`
  - Qwen3.5 的 hybrid attention 通过 `AttentionPattern` 表达, scheduler 按 layer idx 路由 KV
- **Deliverables**:
  - [ ] `silica.models.qwen3_5.Qwen35Adapter`
  - [ ] `silica.models.gemma4.Gemma4Adapter`
  - [ ] `AttentionPattern` 的 hybrid 实现
  - [ ] `silica.weights.resident.ResidentWeightProvider` (全量驻留)
  - [ ] 模型 registry: `silica.models.registry`
- **Acceptance**:
  - **Adapter structural correctness (fp16 parity on control model)** — P-3 exit criterion:
    - [ ] 在 fp16 下运行的控制模型 (Qwen3-0.6B 或 Qwen3.5-4B) 上, Silica adapter 输出 logits 与 HuggingFace 参考实现对照: `max |logit diff| < 1e-3` 在前 50 个 greedy 解码 token 上, 同一 tokenizer state 与 seed。此项验证 adapter 结构组件 (attention, RMSNorm, MoE routing, 位置编码) 在无量化噪声条件下的正确性。
    - [ ] Qwen3.5 hybrid attention cache routing 语义正确 —— 单元测试覆盖不同 layer 按 `AttentionPattern` 的分派。
  - **Quantized big-model correctness** — P-3 exit criterion:
    - [ ] Qwen3.5-27B 与 Gemma 4 31B 在 MLX-native 4-bit 或 8-bit 量化路径下 (D-005) 成功加载并至少执行一次前向; 允许人工 resident 限额与小 batch。
    - [ ] **Teacher-forced next-token argmax agreement ≥ 98%** over 前 100 个 teacher-forced positions, 对照 `mlx-lm` 在同一模型 + 同一量化配置下的 reference 输出, 在固定 prefix 下逐位置比较。不使用 free-running 生成序列对照, 以避免序列漂移掩盖或放大真实差异。
    - [ ] Fallback (若 teacher-forced 对照在当前工具链下不可行): 改用 end-to-end **PPL drift `< 0.1` absolute** vs `mlx-lm` 同量化配置 baseline 在同一 evaluation corpus 上的 PPL。
  - **Product memory-fit target** (conditional, gated on Q-003; ownership by M-4 或 M-7, 见 handoff 规则):
    - [ ] 两个目标模型在 M5 Pro 48 GB 上连续生成 ≥ 500 tokens。若 P-3 退出时 Q-003 尚未 resolve 为 "int4 fits in 48 GB", 本条允许用更小模型 (e.g. Qwen3.5-7B) 或人工 resident 限额完成 adapter 路径验证, 真实 27B/31B @ 48 GB 500 token 的验证转交给 M-7 (见 R-1 / Q-003 / M-4 / M-7)。
- **Dependencies**: P-2
- **Status**: planned
- **Notes**: 风险点 —— Qwen3.5-27B fp16 约 54 GB, 必须量化。若 4-bit 仍放不下, Q-003 触发, P-6 提前。Acceptance 拆为 adapter structural correctness (fp16 parity on control model) + quantized big-model correctness (teacher-forced 对照) + product memory-fit target (conditional), 以避免 P-3 在 P-6 尚未完成时被自身阻塞。

### P-4 Phase 4 — Bench Unification

- **Goal**: benchmark 直接走 Engine, 不再有旁路。
- **Scope**: 统一 bench runner, 标准场景, 统一结果格式。
- **Strategy**: bench 是 `silica.engine.Engine` 的薄封装。
- **Deliverables**:
  - [ ] `silica.bench.runner.BenchRunner`
  - [ ] `silica.bench.scenarios`: 短 in/长 out; 长 in/短 out; 多并发共享前缀
  - [ ] 统一指标 schema: TTFT, prefill tok/s, decode tok/s, resident memory, logical KV bytes, quality
  - [ ] 输出: jsonl + markdown 报告
  - [ ] `silica.bench.vqbench_baseline`: 在独立 subprocess 跑 vqbench 现成脚本 (`reproduce_qwen35_4b_headline.py` 等) 收集 PPL 作为参考列 —— D-009 允许这条"独立 process 对照跑"路径; 服务 P-5 numeric cross-check acceptance
- **Acceptance**:
  - [ ] 单条命令跑出 baseline 表 (可贴 README)
  - [ ] bench 和 runtime 无路径分叉 (同一个 Engine 实例)
  - [ ] `vqbench_baseline` 入口能在独立 process 跑出 Qwen3.5-4B 的 PPL 数字作为 P-5 对照列
- **Dependencies**: P-3
- **Status**: planned
- **Notes**: Phase 4 产出的 baseline 数据决定 Q-003 (Phase 6 是否提前)。

### P-5 Phase 5 — VQ KV Compression

- **Goal**: 把 P-0 的 `IdentityCodec` stub 替换为真实 VQ codec (Principle 9 "stub → real" 替换), 让平台在同样内存预算下能容更多请求或更长 context。
- **Scope**: `IdentityCodec`, `BlockTQCodec`, `RaBitQCodec`。**PQ / OPQ 不进主线**。
- **Strategy**:
  - v0.1 不做压缩域 fast path (D-003)
  - `PagedKVCache(codec=...)` 注入式切换
  - scheduler 通过 `KVCodec.logical_bytes` / `resident_bytes` 获得节省信息并据此多收请求 (Principle 8)
  - **`BlockTQCodec` / `RaBitQCodec` 必须按 `mx.array` hot path 重写** + 补 resident accounting。参考源是 **`vqbench/`** (含 nested `vqbench/turboquant_plus/`) —— vqbench 是**算法参考 + Qwen3.5-4B 实证基线** (`BlockTurboQuantMSE B=64` 4-bit K+V 已验证 +0.0% ΔPPL, 见 `vqbench/REPORT.md`); 详见 §5.5 Reference Map。vqbench 本身是 NumPy + PyTorch + HF transformers, **不作为 runtime import** (D-009)。P-5 实际工程量 = "把 NumPy 思路翻译成 `mx.` ops 的 MLX-native 实现" + "把节省以 `logical_bytes` / `resident_bytes` 暴露给 scheduler"; **不是**"接第三方插件", 而是把原生能力的 stub 替换为真实实现 (Principle 9) —— 这点必须前置写出来, 避免 P-5 scope creep 看起来像意外。
  - codec 的 decode 开销信号是否要进 scheduler admission (避开"省内存但拖慢 decode"的病态组合), 见 **Q-007** —— v0.1 先不冻进接口
- **Deliverables**:
  - [ ] `silica.vq.block_tq.BlockTQCodec`
  - [ ] `silica.vq.rabitq.RaBitQCodec`
  - [ ] `silica.kvcache.paged.PagedKVCache` 支持 codec 注入
  - [ ] bench 加 `--kv-codec {fp16,block_tq,rabitq}` 开关
  - [ ] scheduler budget 利用 codec savings 的 admission 策略
- **Acceptance**:
  - [ ] 切 codec 不改 scheduler / model adapter
  - [ ] 同一组场景下 fp16 vs codec 的 quality 差距和内存节省可从 bench 一键拿到
  - [ ] 开 BlockTQ 后同样内存预算能收更多请求 (量化验证 Principle 8)
  - [ ] **Numeric cross-check against vqbench** —— 两个独立阈值, 均须通过:
    - **(a) Per-block reconstruction error**: metric 为 **per-block relative Frobenius error** `||K_decoded - K_original||_F / ||K_original||_F` (V 同理), 在固定 calibration set 上按 block 计算。Silica MLX-native `BlockTQCodec` 在 Qwen3.5-0.6B (或更大) 上的该 metric, 与 vqbench NumPy 参考在同一 calibration set 上的同一 metric 对照, 允许偏差 `ε_recon < 2 × fp16 round-trip baseline noise`, 其中 baseline 由 vqbench 自身执行一次 fp16 encode→decode 轮回得到 (量级接近机器精度)。P-4 exit 时可根据实测 baseline noise 收紧。
    - **(b) End-to-end PPL drift**: Qwen3.5-4B `BlockTurboQuantMSE B=64` 4-bit K+V 配置下, Silica `BlockTQCodec` 的 end-to-end perplexity 与 `vqbench/REPORT.md` 基线偏差 `ε_ppl < 0.01` absolute。
    - 两项均须通过方算 acceptance pass; 单项通过不足。
- **Dependencies**: P-4
- **Status**: planned
- **Notes**: BlockTQ / RaBitQ 的具体实现参考 `turboquant_plus/` (gitignored reference impl)。

### P-6 Phase 6 — Weight Streaming

- **Goal**: weight streaming 解决权重驻留压力。
- **Scope**: `ResidentWeightProvider` (已有) + `StreamingWeightProvider` + scheduler prefetch 协同。
- **Strategy**:
  - 参考 `mlx-flash` 思路
  - 在 layer N 计算时 prefetch layer N+1
  - **利用 Apple unified memory** (Principle 2): 不是"GPU 从 disk/CPU 拉权重", 而是"同一内存池内不同区域的生命周期管理"
- **Deliverables**:
  - [ ] `silica.weights.streaming.StreamingWeightProvider`
  - [ ] `silica.weights.prefetch`: scheduler 预取协同逻辑
  - [ ] unified memory aware 的 residency 策略
- **Acceptance**:
  - [ ] 人为限制内存预算 (如 24GB) 下 Qwen3.5-27B int4 不 OOM
  - [ ] **Decode tok/s ≥ 70% of `ResidentWeightProvider` baseline**, 比较条件固定: 同机型 / 同模型 / 同 scenario / 同 sampling 配置下取 decode tok/s 比值。若 resident 在 24GB 下根本跑不了 (OOM), 对照对象改为**"无人工限额的 resident reference run"** (不限预算、不开 streaming 的 resident 跑同一 scenario 的 decode tok/s), 以此作为 baseline 取比值
- **Dependencies**: P-2 (scheduler) + P-3 (model adapter)
- **Status**: planned
- **Notes**: 可能从 T1 tier 提前到 P-5 之前 —— 如果 P-3 发现 27B/31B 在 baseline 就放不下 (Q-003)。

### P-7 Phase 7 — Speculative Decoding

- **Goal**: draft-target speculative decoding 加速 decode。
- **Scope**: `NoopDraftEngine` (已有) + `DraftTargetEngine` (最基础版)。
- **Strategy**:
  - 用一个小模型做 draft, 大模型做 verify
  - 先不做 EAGLE / Medusa / DFlash 复杂方案 (推到 v0.2)
- **Deliverables**:
  - [ ] `silica.speculative.draft_target.DraftTargetEngine`
  - [ ] 与 decode loop 的集成
  - [ ] acceptance / rollback metrics
  - [ ] bench 开关 `--speculative {none,draft_target}`
- **Acceptance**:
  - [ ] greedy 下开/关 speculative 输出 token 序列**逐 token 一致** (correctness invariant, 作为基础正确性门)
  - [ ] **Decode tok/s ≥ 1.2× draft-disabled baseline**, 必须在**固定标准 scenario** (e.g. P-4 bench 的"长 in / 短 out" 或等价 mixed-batch 场景) 下测; **不允许挑 best-case** (e.g. 大量共享前缀 / 特定温度 / 只跑最有利的 workload) 作为验收证据
  - [ ] 不破坏 baseline correctness (覆盖开关两种模式下 smoke test 通过)
- **Dependencies**: P-2 + P-3
- **Status**: planned
- **Notes**: DFlash / dflash-mlx 作为 v0.2 候选。

### P-8 Phase 8 — Mini-SGLang Layer

- **Goal**: engine 之上补最小 serving 层, 让平台真的"可用"。
- **Scope**: OpenAI-compatible HTTP API, session management, prefix-sharing session reuse, structured output 接口预留。
- **Strategy**:
  - 这是"产品脸面", 不是锦上添花 (D-006)
  - 优先级讨论见 Q-002 (是否从 T2 tier 上浮到 T1 末尾)
  - 外层组织参考 mini-sglang, 但不照搬 CUDA 内核
- **Deliverables**:
  - [ ] `silica.server.openai_api`: FastAPI, `/v1/chat/completions`, `/v1/completions`
  - [ ] `silica.server.session.SessionManager`: 会话管理, 跨请求复用 prefix
  - [ ] structured generation / grammar 的接口位 (不实现)
  - [ ] `silica.llm.LLM`: Python 友好的 high-level 接口
- **Acceptance**:
  - [ ] 用 `openai` Python 客户端对 silica server 发请求能拿到流式返回
  - [ ] 跨请求的 prefix 复用可验证 (同一 session 里发 N 个共享前缀请求, prefix cache hit 率)
  - [ ] 本地可以像一个小型 serving engine 一样工作
- **Dependencies**: P-2 + P-3 (+ P-5 / P-6 optional)
- **Status**: planned
- **Notes**: 本阶段决定 Silica-MLX 从"一个 engine library"升格为"一个可用的 Mac 推理平台"。

---

## 8. Priority & Milestones

### 8.1 Priority Tiers

Tier ID 用 `T0 / T1 / T2`, 避免和 phase ID `P-0 / P-1 / P-2` 视觉撞名。

| Tier | Phases | 含义 |
| --- | --- | --- |
| T0 | P-0 .. P-4 | 骨架 + baseline engine + 目标模型 + bench |
| T1 | P-5 .. P-6 | VQ KV 压缩 + weight streaming; 使大模型在 48GB 真正跑好 |
| T2 | P-7 .. P-8 | speculative + serving 层 |

Phase 8 优先级是否上浮见 Q-002。Phase 6 是否提前见 Q-003。

### 8.2 Milestones

| ID  | 名称                    | 依赖 Phases | 验收 |
| --- | ----------------------- | ----------- | --- |
| M-1 | Skeleton                | P-0          | 接口冻结, stub test 通过 |
| M-2 | Single-request gen      | P-1          | Qwen3-0.6B 能生成文本 |
| M-3 | Multi-request core      | P-2          | 8 并发 + prefix cache 命中 |
| M-4 | Big models adapter correct | P-3 | Adapter 结构正确性 (fp16 parity on control model, logit 最大绝对差 < 1e-3) + hybrid attention routing + quantized 大模型 correctness (teacher-forced next-token argmax 一致率 ≥ 98%, 或 fallback PPL drift < 0.1 absolute)。**Product memory-fit target** (27B/31B @ 48 GB, 500 tokens): **仅当** Q-003 resolve 为 "int4 fits in 48 GB" 时在本 milestone 验证; 否则 defer to M-7。 |
| M-5 | Unified bench           | P-4          | 一条命令出 baseline 表 |
| M-6 | VQ on platform          | P-5          | BlockTQ / RaBitQ 接入, 节省可量化 |
| M-7 | Streaming weights + deferred memory-fit | P-6 | 24 GB 预算下 Qwen3.5-27B int4 不 OOM; decode tok/s ≥ 70% of `ResidentWeightProvider` baseline (见 P-6 Acceptance)。**若 M-4 已 defer product memory-fit target** (Q-003 forced P-6 在 P-3 exit 之前), 本 milestone 承接 27B/31B @ 48 GB 500 token 的真实验证。 |
| M-8 | Speculative enabled     | P-7          | 开关 correctness 不变 + 提速 |
| M-9 | Platform usable         | P-8          | OpenAI API + session 可用 |

---

## 9. Decisions Log

Append-only。新决策往后加，不改老条目。撤销/修订需新开一条并在新条目里引用被撤销的 ID。

### D-001 — Package manager & Python version

- **Date**: 2026-04-13
- **Status**: accepted
- **Decision**: `uv` + Python 3.12
- **Rationale**: uv 现代快; mini-sglang 也用 uv; Python 3.12 是 MLX / mlx-lm 稳定支持的版本。
- **Consequences**: 贡献者需要 uv; Python < 3.12 不支持。

### D-002 — vLLM core first, mini-sglang layer later

- **Date**: 2026-04-14
- **Status**: accepted
- **Decision**: engine 核心按 vLLM 思路 (paged KV, continuous batching, memory budget); 外层服务按 mini-sglang 思路。
- **Rationale**: M5 Pro 48GB 上首要问题是"能跑稳 + 省内存 + 可扩展", vLLM 的 paged KV + batching 正好解决这个; mini-sglang 更适合拿来参考模块分层和 serving 外层。
- **Consequences**: Phase 0–4 像 mini-vLLM; Phase 8 像 mini-sglang。

### D-003 — KVCodec v0.1 excludes compressed-domain attention

- **Date**: 2026-04-14
- **Status**: accepted
- **Decision**: v0.1 的 `KVCodec` 接口只含 `encode_block / decode_block / logical_bytes / resident_bytes`; 不包含 `attend()` 或其他压缩域 fast path。
- **Rationale**: 避免一开始就把接口绑死; v0.1 优先接口简洁, v0.2 再根据 Phase 5 bench 结果决定是否加。
- **Consequences**: BlockTQ / RaBitQ 的 Phase 5 实现必须走"decode 再标准 attention"路径, 即使性能次优。

### D-004 — Phase 1 model execution: wrap mlx-lm

- **Date**: 2026-04-14
- **Status**: accepted
- **Decision**: Phase 1 的 `ModelAdapter` 是对 `mlx-lm` 的 thin wrapper, 不在 Phase 1 重写模型执行细节。
- **Rationale**: 先跑通再优化; Phase 3 做真正的 adapter 时再下沉。
- **Consequences**: Phase 3 重写时会有改动成本, 但 Phase 1 迭代周期缩短。

### D-005 — Phase 3 quantization path: MLX-native only

- **Date**: 2026-04-14
- **Status**: accepted
- **Decision**: Phase 3 量化走 mlx-lm 现成的 4-bit / 8-bit 路径; **不**扩 GGUF / AWQ 多格式兼容。
- **Rationale**: 48GB 上 27B / 31B 必然要量化; 先用 MLX 已经能跑的方案, 多格式兼容推到 v0.2。
- **Consequences**: 用户不能用 GGUF / AWQ 预量化模型; 只能用 mlx-lm 支持的量化。

### D-006 — Platform as product, VQ as means

- **Date**: 2026-04-14
- **Status**: accepted (用户明确纠正过一次)
- **Decision**: Silica-MLX 本身是产品; VQ / weight streaming / speculative 是让平台跑好大模型的手段, 不是研究对象。
- **Rationale**: 用户目标是"单 Mac 芯片的推理平台, 能很好利用 VQ 之类 tech"; 不是"VQ 研究的 benchmark 载体"。
- **Consequences**:
  - 目标用户是 Mac 上想本地跑大模型的开发者
  - Phase 8 (serving 层) 重要性上升
  - VQ / weight streaming 不应只是 opt-in flag, memory budgeter 要主动利用节省 (Principle 8)
  - 设计要 Apple unified memory first (Principle 2)

### D-007 — Plan document structure

- **Date**: 2026-04-14
- **Status**: accepted
- **Decision**: 用 `docs/PLAN.md` 作为单一 source of truth; 结构为 Meta / TL;DR / Mission / Scope / Principles / Architecture / Interfaces / Phases / Priority / Decisions Log / Open Questions / Risks / References / Changelog, 所有条目带稳定 ID。
- **Rationale**: CRUD-friendly —— stable IDs, self-contained phase blocks, append-only decisions log; 允许在不读全文的情况下定位和修改某一条。
- **Consequences**: 未来 plan 变更要进 Decisions Log 并更新对应 Phase block; Changelog 跟版本号。

### D-008 — core / engine 边界: 数据类放 core, 逻辑类放 engine

- **Date**: 2026-04-14
- **Status**: accepted
- **Decision**: 数据类 (`Request`, `RequestState`, `SamplingParams`, `Context` 等) 放 `silica.core`; 运行时逻辑类 (`Engine`, scheduler 内部状态机的 runner 部分) 放 `silica.engine`。遵循 mini-sglang 风格。
- **Rationale**: Phase 0 需要建立明确的目录结构; Q-004 里 Option A 是 mini-sglang 验证过的风格; 本文档所有 Phase block 也已隐式使用此路径 (如 `silica.core.request.Request`, `silica.core.sampling.SamplingParams`), 不定下来会导致 Phase 0 完成时文档和代码不一致。
- **Consequences**:
  - `silica.core` 是数据 + 观测 (logging, profiling, metrics schema), 不含业务逻辑
  - `silica.engine.Engine` 持有 core 的数据类, 驱动 scheduler / kvcache / model
  - 解决 Q-004

### D-009 — MLX-native hot path as hard constraint

- **Date**: 2026-04-14
- **Status**: accepted (用户明确要求)
- **Decision**: Inference hot path **必须** 100% 走 MLX。具体约束:
  1. 所有 tensor 都是 `mlx.core.array` (`mx.array`)。`silica.engine` / `silica.mlx` / `silica.kvcache` / `silica.models` / `silica.scheduler` / `silica.vq` / `silica.weights` / `silica.speculative` 内部 **不允许** 出现 `torch.Tensor` 或 `numpy.ndarray` 参与 tensor math (numpy 仅可用于配置/标量/列表辅助)。
  2. **不依赖 PyTorch 运行时**。`pyproject.toml` 不得把 `torch` 列为 runtime dep。torch 只能作为 offline 权重转换的**可选** dev/extras 依赖 (例如 `pip install silica-mlx[convert]`); 转换结果是 MLX-native 格式, inference 启动后不再碰 torch。
  3. **vllm 和 transformers 只作为算法/架构参考, 不作为运行时依赖**。vllm 的 `csrc/`, `vllm_flash_attn/`, GPU/TPU/XPU/CPU model runner 一律不在范围内。
  4. Phase 1 wrap `mlx-lm` 是合法的 (D-004), 因为 **mlx-lm 本身就是 MLX-native**。禁止替换成任何 torch-based wrapper (包括 transformers、llama.cpp Python 绑定等) 作为运行时路径。
- **Rationale**: 用户硬要求 "必须是原生 mlx"; Silica-MLX 的整个价值命题 (D-006 "Mac 推理平台") 依赖于 MLX 在 Apple Silicon 上的性能和 unified memory 优势; 任何 torch hot path 都会破坏 Principle 2 (unified memory first)。
- **Consequences**:
  - 所有 attention / sampling / kvcache 内部实现必须用 `mx.` ops
  - Phase 3 如果某个目标模型在 mlx-lm 缺失, 需要在 `silica.models.*` 里 MLX-native 重写, **不能回退到 transformers**
  - vllm v1 的源代码仅用作"怎么设计"参考, 不用作"怎么调用"依赖 (见 5.4 Reference Map)
  - Benchmark 里如果要对照其他推理引擎的数字, 那个对照跑必须在独立 process 里, 不能混入 silica 运行时

### D-010 — Phase 1 mlx-lm borrowing boundary

- **Date**: 2026-04-14
- **Status**: accepted (两轮 Codex plan review 共同定位出的最易返工点)
- **Decision**: Phase 1 从 `mlx-lm` **借用**:
  1. 模型结构加载 (模型 class 构造 + state dict 形状)
  2. tokenizer
  3. 权重 loader (safetensors → `mx.array`)

  Phase 1 **不借用** `mlx-lm` 的 rotating KV cache / prompt cache。Silica 从 day-1 自己管 KV, `SimpleKVCache` 是**注入到模型 forward** 的外部 cache, 不是包在 `mlx-lm` 内部 cache 外的一层 wrapper。
- **Day-1 smoke test** (P-1 第一项任务, 先于其他 deliverable): 验证 `mlx_lm.generate_step(cache=...)` 或等价入口是否接受外部 cache 对象。
  - **接受** → `SimpleKVCache` 直接注入, decoupling 干净, P-1 按计划展开
  - **不接受** → monkey-patch / fork `mlx_lm.models.*` 前向逻辑, P-1 cost estimate 上调 (触发 R-6)
- **Rationale**: D-004 (wrap mlx-lm) 和 P-1 deliverable (`SimpleKVCache`) 之间的 ownership 边界没写清楚, 是现阶段 plan 最容易静默返工的点。必须在 P-1 动手前固化"借什么 / 不借什么"。`mlx-lm` 自己的 rotating cache 对单请求开发够用, 但和 Silica 的 paged / prefix / codec 愿景不兼容, 不能作为 P-2 的起点 —— 否则 P-2 会变成"剥 mlx-lm 的 cache 再换上 Silica 的", 这是返工而不是升级。
- **Consequences**:
  - P-1 Strategy 指向此决策, 不再只写 "thin wrapper"。D-004 仍保留作为**模型执行**层面的决策; D-010 是 **cache** 层面的决策, 两者互补不冲突
  - P-2 的 `PagedKVCache` 是 `SimpleKVCache` 的直接升级路径, 不是"替换 mlx-lm 内部 cache"
  - R-2 缓解策略调整: Phase 3 重写模型 adapter 时, cache 接入点已经是 Silica 自己的, 不用再剥 mlx-lm 的 cache
  - 新增 R-6 登记 day-1 smoke test 失败的风险
  - **Concrete precedent (反例)**: `vqbench/vqbench/torch_wrapper/hook.py` 的 `VQBenchCache` (subclass `transformers.Cache`) 是一个具体反例 —— 那个模式依赖 HF cache 生命周期, 正是 mlx-lm 内部 rotating cache 在 torch 世界的对应物。如果 Silica 借 mlx-lm 的 cache 就会长成类似形态 (绑死在 framework 的 internal cache shape 上)。D-010 就是为了避开这条路
- **References**: D-004, Principle 6, P-1 Deliverables, R-6, §5.5

---

## 10. Open Questions

Resolved 的 question 不删, 改 `Status: resolved` 并在条目末尾加 `Resolution:` 段, 方便追溯。

### Q-001 — VQ codec 自动选择 vs 显式配置

- **Raised**: 2026-04-14
- **Status**: open
- **Question**: Phase 5 的 VQ codec 应该 (A) 用户显式选择 via CLI flag, 还是 (B) 按 workload 自动选择?
- **Context**: D-006 说平台要"很好地"利用 VQ, 暗示自动选择; 但 Phase 0 接口不表达这种能力。
- **Options**:
  - A. 显式配置: 简单, 用户控制
  - B. 自动选择: 符合 D-006 framing, 但需要 workload profiler
- **Blocks**: Phase 5 设计定稿
- **Next step**: Phase 4 bench 跑完后再判断。

### Q-002 — Phase 8 优先级是否上浮

- **Raised**: 2026-04-14
- **Status**: open
- **Question**: 根据 D-006 (平台是产品), Phase 8 (OpenAI API + session) 是否应该从 T2 tier 上浮到 T1 末尾?
- **Context**: 如果 Phase 8 是"产品脸面", 应该早做; 但 engine 没稳定前做 serving 层风险大。
- **Options**:
  - A. 保持 T2: engine 先稳定
  - B. 上浮到 T1 末尾: P-5 / P-6 原生能力集成和 serving 层并行
- **Blocks**: Phase 5–8 实际排期
- **Next step**: Phase 4 完成后评估。

### Q-003 — Phase 6 是否提前

- **Raised**: 2026-04-14
- **Status**: open
- **Question**: 如果 Phase 3 发现 Qwen3.5-27B / Gemma 4 31B 在 4-bit 量化下仍放不下 48GB, Phase 6 (weight streaming) 是否提前到 Phase 5 之前?
- **Blocks**: Phase 5 / 6 顺序
- **Next step**: Phase 3 跑出真实驻留数字后决策。

### Q-004 — `silica.core` vs `silica.engine` 边界

- **Raised**: 2026-04-14
- **Status**: resolved
- **Question**: `Request`, `SamplingParams`, `RequestState` 放 `silica.core` 还是 `silica.engine`?
- **Context**: mini-sglang 放 `minisgl.core`; 但 "core" 容易膨胀。
- **Options**:
  - A. 数据类放 core, 逻辑类放 engine (mini-sglang 风格)
  - B. 全放 engine, core 只留 logging/profiler
- **Resolution**: 选 Option A。见 D-008。文档内所有 Phase block 已隐式使用 `silica.core.request.*` 路径, 此处固化。

### Q-005 — MetricsRegistry 是否全局单例

- **Raised**: 2026-04-14
- **Status**: open
- **Question**: Phase 0 的 profiler 用全局 `MetricsRegistry` 还是每个 Engine 实例一个?
- **Context**: 全局单例简单但多 Engine 实例会冲突; per-instance 干净但 CLI/bench 访问略麻烦。
- **Next step**: Phase 0 动手时定。

### Q-006 — Attention backend 是否独立接口

- **Raised**: 2026-04-14
- **Status**: open
- **Question**: 是否需要一个独立的 `AttentionBackend` Protocol (类似 vllm v1 的 `vllm/v1/attention/backend.py`), 让 attention 实现可以在 `ModelAdapter` 之外被替换? 还是继续把 attention 藏在 `ModelAdapter.build()` 返回的 Module 内部?
- **Context**: vllm v1 把 attention backend 独立出来是为了支持 flashattention / flashinfer / xformers / triton 多种 CUDA 实现; 我们只有 MLX 一条路径, 短期不需要这种灵活性。但如果 Phase 5 的 VQ 需要压缩域 attention fast path (D-003 留的 v0.2 能力), 独立 `AttentionBackend` 会让接入更干净。
- **Options**:
  - A. v0.1 不独立, attention 藏在 ModelAdapter 内部 (简单, 契合当前 5 接口设计)
  - B. v0.1 就独立 AttentionBackend 作为第 6 个接口 (为 v0.2 的压缩域 attention 做准备)
- **Blocks**: 无 —— 可延到 Phase 5 bench 出结果后再定
- **Next step**: Phase 5 完成后评估 (与 D-003 的 v0.2 升级一起决定)

### Q-007 — KVCodec decode overhead signal for admission control

- **Raised**: 2026-04-14
- **Status**: open
- **Question**: `KVCodec` 是否需要暴露 `decode_overhead_ratio: float` (fp16 baseline = 1.0), 让 scheduler admission control 同时看内存节省和 decode 成本, 避开"省内存但拖慢 decode"的病态组合?
- **Context**: Principle 8 说 savings 必须被 scheduler 看见; 目前 I-3 只暴露 `logical_bytes` / `resident_bytes`。如果 scheduler 只看内存节省就放行更多请求, 某些 codec (decode 慢 2x) 反而会掉整体 tok/s —— savings 可见但成本不可见, 违反 Principle 8 的精神。保持 v0.1 接口 minimal 和保证 scheduler 信息完备之间要选一个。
- **Options**:
  - **A. v0.1 不加**, Phase 5 由用户显式选 codec (同 Q-001 Option A); v0.2 再考虑 profile table。接口 minimal, 但 scheduler 无法主动利用节省 (违反 Principle 8 的精神)
  - **B. v0.1 在 I-3 加 `decode_overhead_ratio: float` 常量**, Phase 4 bench 回填数值, scheduler 读之做 admission。接口加一行, 能避开病态组合, 但静态常量不能反映 seq-len / batch-size 依赖
  - **C. v0.1 不进接口**, 由 `silica.bench` 产出 per-codec profile table, scheduler 读该 table。最精确但最重, v0.1 可能过度设计
- **Blocks**: P-5 scheduler admission 策略定稿; 和 Q-001 (VQ codec 自动选择) 耦合
- **Next step**: Phase 4 bench 跑完, 有了 BlockTQ / RaBitQ 的真实 decode 开销数据再定; 决议时同步决定 `feedback_kvcodec_interface.md` memory 是否更新。

### Q-008 — KVCodec K/V pair configuration

- **Raised**: 2026-04-14
- **Status**: open
- **Question**: `KVCodec` 是否要在**接口层**暴露 K/V pair 配置 (e.g. `KVCodec(key_method=..., value_method=...)`) 让用户显式指定 K 和 V 用不同 codec, 还是隐藏在具体 codec 实现的 constructor 里?
- **Context**: vqbench 明确 K 和 V 需要不同 codec —— K 需要 unbiased inner product 估计 (`TurboQuantProd` / `QJL`), V 需要低 MSE 重建 (`TurboQuantMSE` / `BlockTurboQuantMSE`); 这是 `vqbench/vqbench/kv_cache/compressor.py` 里 `KVCacheCompressor(key_q, value_q)` 的设计依据。目前 I-3 的 `encode_block(k, v) -> CodedBlock` 是单 codec 对象, 内部可以持有 K/V 两个 quantizer, 但**外部**没有暴露这个选择。P-5 做 `BlockTQCodec` 具体 constructor 时这个决策会立刻变得具体。
- **Options**:
  - **A. 内部处理, I-3 契约不变**: 具体 codec (e.g. `BlockTQCodec`) 的 constructor 接受 `key_method` / `value_method` 两参, I-3 签名不动。**优点**: I-3 保持 minimal, 换 codec 不破坏 scheduler。**缺点**: 用户要理解每个 codec 自己的构造参数
  - **B. I-3 增加 pair contract**: 把 `KVCodec` 拆成 `KeyCodec` + `ValueCodec` 两个 Protocol, 顶层 `KVCodecPair` 做组合。**优点**: K/V 不对称性在类型层可见。**缺点**: I-3 从 1 个接口变成 3 个接口, P-0 冻结复杂度上升
  - **C. `KVCodec.from_pair(key_method, value_method)` class method**: 既不改签名又暴露 pair 意图的折中
- **Blocks**: P-5 `BlockTQCodec` 具体构造函数设计; 可能影响 Q-001 (自动选择 codec) 的选择粒度
- **Next step**: P-5 动手时定 —— 倾向 Option A (最保守, I-3 minimal)

---

## 11. Risks

| ID | 描述 | 触发阶段 | 缓解 |
| --- | --- | --- | --- |
| R-1 | Qwen3.5-27B / Gemma4-31B 在 4-bit 量化下仍放不下 48GB | P-3 | Q-003: 提前 P-6 weight streaming |
| R-2 | mlx-lm 的 wrapper 在 Phase 3 替换成本高于预期 | P-3 | D-004 接受的权衡; 若过高可延后下沉 |
| R-3 | Qwen3.5 hybrid attention cache routing 语义复杂 | P-3 | Phase 3 单元测试覆盖; 对照 HF 参考实现 |
| R-4 | BlockTQ / RaBitQ 的 decode 开销比 fp16 attention 本身大 | P-5 | Q-001 自动选择策略; v0.2 考虑 fast path |
| R-5 | MLX / mlx-lm 版本不稳定导致依赖锁麻烦 | all | pyproject.toml 锁最低版本 + CI 定期升级 |
| R-6 | `mlx-lm` 不接受外部 cache 注入 (D-010 day-1 smoke test 失败) | P-1 | monkey-patch / fork `mlx_lm.models.*` 前向; 最坏情况 P-1 cost 上调并在 P-2 动手前统一重构 cache 接入点; 记入 P-1 Strategy Notes |

---

## 12. References

### 12.1 External (参考)

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

本地放在 repo 根的参考实现，**仅作为算法/架构参考, 非运行时依赖** (D-009):

| 目录 | 项目 | 主要参考点 |
| --- | --- | --- |
| `vllm/` | vLLM v1 | 见 §5.4 Reference Map; 重点 `vllm/v1/core/`, `vllm/v1/engine/`, `vllm/v1/kv_cache_interface.py`, `vllm/v1/request.py` |
| `mini-sglang/` | Mini-SGLang | 模块分层、serving 外层、radix prefix cache 思路 |
| `vqbench/` | VQBench (含 nested `vqbench/turboquant_plus/`) | 见 §5.5 Reference Map; P-5 算法参考 + Qwen3.5-4B 实证 PPL oracle; NumPy + PyTorch 代码库, **不是** runtime import (D-009); `VQBenchCache` 是 D-010 反例 |

---

## 13. Changelog

- **v1.4.1** (2026-04-14): Codex 第 3 轮 review pass targeted optimizations, **no structural redesign**。(1) §2 Success criteria 加 "最小 OpenAI-compatible HTTP API + session 可用 (via Phase 8)", 和 §3.1 scope + D-006 对齐; (2) §8.1 Priority Tier 名 `P0 / P1 / P2` → **`T0 / T1 / T2`**, 避免和 phase ID `P-0 / P-1 / P-2` 视觉撞名 (Q-002 Option B 同步更新); (3) **P-3 Acceptance 拆成两层**: **Adapter correctness (硬门)** —— 加载 + 前向一次, logits max diff < 1e-3 over 50 token greedy, hybrid attention routing 单元测试; **Product memory-fit target (conditional, gated on Q-003)** —— "27B/31B @ 48GB 500 token" 若 P-6 未提前则允许用更小模型或人工限额验证; M-4 milestone 同步 conditional 表述, 避免 P-3 在 P-6 之前被自锁; (4) P-5 Acceptance 的 ε 默认值 pin 为 `max(2× fp16 baseline noise, 0.01 PPL)`, P-4 exit 时可收紧; (5) P-6 Acceptance 加具体阈值 **decode tok/s ≥ 70% of `ResidentWeightProvider` baseline @ 24GB 预算**, 限定同机型/同模型/同 scenario/同 sampling 比较; 若 resident 在 24GB OOM 则对照"无人工限额的 resident reference run"; (6) P-7 Acceptance 加具体阈值 **decode tok/s ≥ 1.2× draft-disabled baseline**, 必须在固定标准 scenario 下测, **不允许挑 best-case**; greedy 逐 token 一致作为 correctness invariant; (7) 模块重命名 **`silica.flash` → `silica.weights`** (避免和 FlashAttention 过载, 和 `silica.kvcache` 命名对仗): §5.1 Module Layout, P-3 Deliverables (`silica.weights.resident.ResidentWeightProvider`), P-6 Deliverables (`silica.weights.streaming.StreamingWeightProvider`, `silica.weights.prefetch`), D-009 模块名单 都同步; Principle 1 / D-006 Consequences 里非正式的 "VQ / flash" 用法改成 "VQ / weight streaming" 保持一致; P-6 phase 名 "Weight Streaming" v1.3.0 已改, 本轮不动; `mlx-flash` / `mlx-flash` / `vllm_flash_attn` / `flash-moe` 作为外部仓库名保留原样; (8) D-008 typo 修 "不定会导致" → "不定下来会导致"。整篇中英混用本轮**不统一**, 待后续 contributor-facing 时再做整篇语言 pass。 **Round 2 correction (pre-commit)**: Codex 在 commit 前对 v1.4.1 本身再 review 一次, 指出三处需要订正, 均在本轮 commit 前修好。(i) **P-3 logit 阈值与量化路径单位不一致**: `max |logit diff| < 1e-3` 对应 fp16 parity, 但 D-005 要求大模型走 4-bit / 8-bit 量化路径, 量化后该阈值不可达。P-3 Acceptance 相应拆成两层: (a) **fp16 parity on control model** (Qwen3-0.6B 或 Qwen3.5-4B) 验证 adapter 结构组件, 保留 `max |logit diff| < 1e-3`; (b) **quantized big-model correctness** 采用 **teacher-forced next-token argmax agreement ≥ 98%** over 前 100 个 teacher-forced positions, 对照 `mlx-lm` 同量化配置 reference, 在固定 prefix 下逐位置比较; fallback 为 end-to-end PPL drift `< 0.1` absolute vs 同一 baseline。不使用 free-running 生成序列对照, 以避免序列漂移掩盖或放大真实差异。(ii) **M-4 deferred product target ownership gap**: v1.4.1 将 M-4 收窄到 adapter correctness 后, `27B/31B @ 48 GB 500 token` 的真实验证缺少 milestone 承接。改为 **Q-003 gated handoff**: M-4 若 Q-003 resolve 为 "int4 fits in 48 GB" 则在此 milestone 完成验证; 否则 defer 给 M-7, 由 M-7 承接 product memory-fit target 的真实验证。不新开 M-10, 避免 milestone 系统碎片化。(iii) **P-5 cross-check 阈值单位不合法**: `max(2× fp16 baseline noise, 0.01 PPL)` 将 reconstruction error (tensor-space 量) 与 PPL (end-to-end quality 量) 用 `max` 组合, 量纲不可比。拆成两个独立阈值并显式定义 metric: `ε_recon < 2 × fp16 round-trip baseline noise`, metric 定义为 **per-block relative Frobenius error** `||K_decoded - K_original||_F / ||K_original||_F` (V 同理), baseline 由 vqbench 自身一次 fp16 encode→decode 轮回确定; `ε_ppl < 0.01` absolute PPL drift vs `vqbench/REPORT.md` 基线; 两项均须通过方算 acceptance pass。
- **v1.4.0** (2026-04-14): vqbench 作为 local reference checkout 拉入 silica-mlx, 带来 VQ path 的集成更新 (VQ 定位从"P-5 的一个能力"升格为"final product 的核心能力, 借 vqbench 实证基线使其正确性可测")。具体: (1) 新增 **§5.5 Reference Map to vqbench** —— Silica 模块 ↔ vqbench 文件的映射 (BlockTQ / RaBitQ 算法, codec 工厂, `KVCacheCompressor` pair 模式, Qwen3.5 bench 脚本, `REPORT.md` PPL oracle) 和 `torch_wrapper/` / NumPy impl 的禁用路径 (D-009); (2) §12.2 Local reference checkouts 顶层 `turboquant_plus/` 替换为 `vqbench/` (turboquant_plus 现 nested 在 vqbench 里); (3) P-5 Strategy reference 源换成 vqbench, 注明 Qwen3.5-4B `B=64` 4-bit +0.0% ΔPPL 实证基线; (4) P-5 Acceptance 加 **numeric cross-check against vqbench** —— Silica `BlockTQCodec` 的重构误差 + end-to-end PPL 必须与 vqbench NumPy 参考吻合, "faithful 重写"可测; (5) P-4 Deliverables 加 `silica.bench.vqbench_baseline` —— 独立 subprocess 跑 vqbench 脚本收集 PPL 参考列, 服务 P-5 cross-check, 走 D-009 允许的"独立 process 对照"路径; (6) D-010 Consequences 加 `VQBenchCache` (HF `Cache` 子类) 作为具体**反例** —— "如果借 mlx-lm 的 cache 就会长成这种形态"的活教材; (7) 新增 **Q-008** 讨论 `KVCodec` 接口层是否暴露 K/V pair 配置 (vqbench 显示 K 需要 unbiased IP codec / V 需要低 MSE codec, 三种可选方案 A/B/C, 倾向 A)。
- **v1.3.0** (2026-04-14): Framing 统一 — plugin → native capability。用户澄清: VQ / weight streaming / speculative decoding 是 Silica-MLX 的**原生能力**, 不是"第三方插件扩展点"; 从 P-0 就以 stub 形式内建, P-5 / P-6 / P-7 逐步替换为真实实现; 集成点固定, 实现可换。具体改动: (1) 新增 **Principle 9 — Native capabilities, swappable implementations**, 明确架构立场 + vLLM attention backend 类比; (2) Principle 3 "Engine first, plugins later" → "Engine skeleton first, native capabilities integrated progressively"; (3) Principle 5 "Plugin contracts are frozen early" → "Native capability contracts are frozen early"; (4) §1 TL;DR / §2 Mission 成功标准 / §3.1 In Scope 三处"插件"措辞重写为"原生能力"; (5) P-5 title "VQ Plugin" → "VQ KV Compression" + Goal/Strategy 同步; P-6 "Flash Plugin" → "Weight Streaming"; P-7 "Speculative Plugin" → "Speculative Decoding"; (6) §8.1 P1 tier label / P-2 Notes / Q-002 Option B 零散"插件"措辞统一清掉。接口名 (`KVCodec` / `WeightProvider` / `DraftEngine`) 和 P-0 deliverable 结构不变, 只是 framing 对齐。
- **v1.2.0** (2026-04-14): 两轮 Codex plan review reconciliation。(1) §6 header 从 "Frozen for v0.1" 改为 "Phase 0 freeze candidate" —— 签名在 P-0 退出前确定, 而不是文档一贴就冻; (2) I-1 补 `prefill(tokens, kv_handle) -> (logits, StateDelta)` 和 `decode_step`, 明确 KV mutation ownership 归 `KVManager` (通过 `kv_handle`), `state_delta` 只承载非-KV runtime state (反例写出: KV blocks / cache residency mutations / prefix pinning 不允许放进 state_delta); (3) I-2 `allocate(request_id, num_tokens)` 拆成 `reserve_for_prefill(req_id, token_ids)` + `append_slot`, 新增 `commit` / `rollback` (P-7 speculative), `prefix_lookup` 改名 `get_computed_blocks` (对齐 vLLM v1), 新增 `available_blocks` 作为 scheduler admission 的 block 级快速路径; (4) 新增 **D-010** 固化 Phase 1 对 `mlx-lm` 的 borrowing boundary —— 借模型/tokenizer/权重 loader, 不借 rotating KV cache; P-1 day-1 smoke test 验证 `mlx_lm.generate_step(cache=...)` 外部 cache 注入; (5) P-1 Strategy / Deliverables 对齐 D-010, 加 day-1 gate; (6) P-5 Strategy 明确 `BlockTQCodec` / `RaBitQCodec` 需 `mx.array` 重写 + resident accounting, 不是复用 `turboquant_plus` NumPy 原型, 工程量前置写出避免 scope creep; (7) 新增 **Q-007** 讨论 `KVCodec.decode_overhead_ratio` 是否进 v0.1 接口 (Principle 8 scheduler 信息完备性); (8) 新增 **R-6** 风险登记 mlx-lm 外部 cache 注入失败的兜底方案; (9) §5.2 数据流更新: `ModelAdapter.forward` → `ModelAdapter.prefill / decode_step (KV via kv_handle from KVManager)`。
- **v1.1.2** (2026-04-14): 用户拉下 vllm 作为本地 reference, 明确 "必须是原生 mlx" —— 新增 D-009 固化 MLX-native hot path 硬约束 (禁 torch.Tensor / 禁 PyTorch runtime dep); Principle 6 重写; 3.2 Non-Goals 加 PyTorch / CUDA 等后端排除项; 新增 §5.4 Reference Map to vLLM v1 (明确参考哪些 / 不参考哪些); 新增 Q-006 讨论是否独立 AttentionBackend; 12 References 拆分 External / Local checkouts, 登记 `vllm/` 本地路径。
- **v1.1.1** (2026-04-14): 修 CRUD review 两处 —— 5.1 目录树补 `llm/` 子包 (和 P-8 deliverable 对齐); Q-004 标 resolved 并新增 D-008 固化 core/engine 边界。
- **v1.1** (2026-04-14): 重写 Mission 与 Principle 1 —— 平台是产品, VQ 是手段 (D-006); 加 Principle 2 "Apple unified memory first"; 加 Principle 8 "savings must be observable"; Phase 5 / 8 的 framing 重写; 新开 Q-002 讨论 Phase 8 优先级, Q-005 讨论 MetricsRegistry; 新增 Risks 表; 结构全部按 CRUD-friendly 格式组织 (stable IDs P-N / D-NNN / Q-NNN / M-N / I-N, self-contained blocks, append-only logs)。
- **v1.0** (2026-04-14): 基于 Codex 对 v0 的润色; 添加 Phase 的统一结构。
- **v0** (2026-04-13): 初版 plan 草稿 (对话形式, 未落盘)。
