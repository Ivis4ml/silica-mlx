"""P-3-E0 probe: can Silica's current mlx-lm path load a Qwen3.5-MoE repo?

**First probe of the P-3-E MoE track, Qwen3.5 family (35B-A3B).** Mirrors
``scripts/probe_gemma4_31b_load.py`` (P-3-D0) structurally but the MoE
specifics differ enough to merit a dedicated script:

  - ``mlx_lm.models.qwen3_5_moe.Model`` is a 53-line wrapper around
    ``qwen3_5.Model``; all MoE logic lives in ``qwen3_5.DecoderLayer``
    behind ``if args.num_experts > 0: self.mlp = SparseMoeBlock(args)``.
  - ``SparseMoeBlock`` is imported from ``qwen3_next.py`` as
    ``Qwen3NextSparseMoeBlock`` — a SwitchGLU-based sparse MLP with
    softmax routing (top-k via ``argpartition``), optional
    normalization, plus a **shared-expert branch** with its own gate
    (non-standard; not present in the Gemma4 MoE flavour).
  - Weights arrive as stacked ``experts.gate_up_proj`` /
    ``experts.down_proj`` tensors; the sanitizer in
    ``qwen3_5_moe.py`` splits them into
    ``switch_mlp.{gate_proj,up_proj,down_proj}`` for SwitchGLU's
    ``gather_mm`` execution path.

Expected outcomes on a fresh run, Silica @ v1.6.3+:

  Phase 1 (``mlx_lm.load``) — succeeds. Produces a
            ``Qwen3_5Moe``-shaped model; fields we specifically
            capture include ``num_experts``,
            ``num_experts_per_tok``, ``moe_intermediate_size``,
            ``shared_expert_intermediate_size``, ``norm_topk_prob``.
            These are read from ``args.text_config`` using the same
            tolerant attribute / dict lookup as the Gemma4 probe.

  Phase 2 (``factory.adapter_for_repo``) — expected to FAIL. No
            ``Qwen3_5MoeAdapter`` is registered in
            ``silica.models.factory._ADAPTERS`` as of the probe
            date; the non-MoE ``Qwen3_5Adapter`` covers dense
            Qwen3.5 (hybrid DeltaNet + GQA) but its structural
            assumptions do not cover a sparse MoE MLP. A failing
            Phase 2 is the invitation to land ``E1`` — a dedicated
            ``Qwen3_5MoeAdapter`` file.

  Phase 3 (``adapter.capabilities``) — skipped if Phase 2 failed,
            which is the expected branch.

  Phase 4 (``Engine.generate`` micro-forward) — **only when
            ``--run-forward`` is passed AND Phase 2 succeeds.**
            Default is metadata-only. Running a forward on a 35B-
            A3B MoE checkpoint is expensive (weights ~20 GB
            quantized, peak memory likely near the 48 GB M5 Pro
            ceiling); the opt-in protects against accidental long
            runs.

D-011 note: the MoE path in mlx-lm is fused via ``SwitchGLU`` +
``gather_mm`` — all experts' weights are stacked in one tensor and
dispatched internally. This conflicts with D-011's per-expert
``WeightProvider.get_expert`` call-path requirement. The survey
that E0 produces will explicitly flag this tension; E1 / E2 will
decide whether to wrap SwitchGLU (keep fused, sacrifice D-011
literal compliance) or replace it with per-expert dispatch (slower
but satisfies D-011 exactly).

Exit code convention (same as other probes):
  0 — probe completed and produced a report, even if individual
      phases failed.
  1 — probe itself crashed (CLI parsing, import failure, etc.).

Usage:
  uv run python scripts/probe_qwen3_5_moe_load.py --repo <hf_repo>
  uv run python scripts/probe_qwen3_5_moe_load.py --repo <hf_repo> --run-forward

No default repo is built in. Candidate name on the HF mlx-community
space is ``mlx-community/Qwen3.5-35B-A3B-4bit`` but verify the
exact repo id before running — quantized 35B MoE downloads are
~20 GB.
"""

from __future__ import annotations

import argparse
import platform
import sys
import time
import traceback
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import mlx.core as mx  # noqa: E402
from mlx_lm.utils import load as _mlx_lm_load  # noqa: E402

from silica import Engine  # noqa: E402
from silica.core.sampling import SamplingParams  # noqa: E402
from silica.models.factory import (  # noqa: E402
    adapter_from_loaded_model,
    supported_model_types,
)


@dataclass
class PhaseResult:
    name: str
    ok: bool
    details: dict[str, Any] = field(default_factory=dict)
    error: str | None = None


def _describe_exception(exc: BaseException) -> str:
    tb = traceback.extract_tb(exc.__traceback__)
    tail = tb[-1] if tb else None
    frame = (
        f" (at {Path(tail.filename).name}:{tail.lineno} in {tail.name})"
        if tail is not None
        else ""
    )
    return f"{type(exc).__name__}: {exc}{frame}"


def _probe_attr(container: Any, name: str) -> Any:
    """Read ``container.<name>`` or ``container[name]`` tolerating the
    three Qwen/Gemma shapes: flat attributes on ``args``, nested object
    under ``args.text_config``, and nested dict under
    ``args.text_config``. Returns ``None`` when absent everywhere.
    """
    if container is None:
        return None
    if hasattr(container, name):
        return getattr(container, name)
    text_config = getattr(container, "text_config", None)
    if text_config is None:
        return None
    if isinstance(text_config, dict):
        return text_config.get(name)
    if hasattr(text_config, name):
        return getattr(text_config, name)
    return None


def _attr_keys(container: Any) -> list[str]:
    if container is None:
        return []
    if isinstance(container, dict):
        return sorted(str(k) for k in container.keys())
    return sorted(
        name
        for name in dir(container)
        if not name.startswith("_") and not callable(getattr(container, name, None))
    )


def _probe_load(repo: str) -> tuple[PhaseResult, Any, Any]:
    t0 = time.perf_counter()
    try:
        model, tokenizer = _mlx_lm_load(repo)  # type: ignore[misc]
    except Exception as exc:  # noqa: BLE001
        return (
            PhaseResult(
                name="mlx_lm.load",
                ok=False,
                details={"repo": repo, "load_s": time.perf_counter() - t0},
                error=_describe_exception(exc),
            ),
            None,
            None,
        )
    elapsed = time.perf_counter() - t0
    args = getattr(model, "args", None)
    text_config = getattr(args, "text_config", None) if args is not None else None
    details: dict[str, Any] = {
        "repo": repo,
        "load_s": elapsed,
        "model_type": getattr(model, "model_type", None),
        # Structural fields shared with dense probes.
        "num_hidden_layers": _probe_attr(args, "num_hidden_layers"),
        "hidden_size": _probe_attr(args, "hidden_size"),
        "num_attention_heads": _probe_attr(args, "num_attention_heads"),
        "num_key_value_heads": _probe_attr(args, "num_key_value_heads"),
        "head_dim": _probe_attr(args, "head_dim"),
        "vocab_size": getattr(tokenizer, "vocab_size", None),
        "tokenizer_type": type(tokenizer).__name__,
        "model_class": type(model).__name__,
        # MoE-specific fields — read by Qwen3.5-MoE's DecoderLayer to
        # decide SparseMoeBlock vs dense MLP.
        "num_experts": _probe_attr(args, "num_experts"),
        "num_experts_per_tok": _probe_attr(args, "num_experts_per_tok"),
        "moe_intermediate_size": _probe_attr(args, "moe_intermediate_size"),
        "shared_expert_intermediate_size": _probe_attr(
            args, "shared_expert_intermediate_size"
        ),
        "norm_topk_prob": _probe_attr(args, "norm_topk_prob"),
        # Hybrid-DeltaNet fields carried over from Qwen3.5 dense: MoE
        # variants inherit the same 3:1 D:G layer pattern and the
        # recurrent-state plumbing lifted by C3c/C3d. A Qwen3.5-MoE
        # adapter therefore needs BOTH the hybrid-DeltaNet handling
        # (C-track) and the per-expert dispatch the MoE track adds.
        "full_attention_interval": _probe_attr(args, "full_attention_interval"),
        "args_attrs": _attr_keys(args),
        "text_config_keys": _attr_keys(text_config),
        "n_layers_from_model_list": len(getattr(model, "layers", []) or []),
    }
    return (
        PhaseResult(name="mlx_lm.load", ok=True, details=details),
        model,
        tokenizer,
    )


def _probe_dispatch(
    repo: str, model: Any, tokenizer: Any
) -> tuple[PhaseResult, Any, Any]:
    """Dispatch against the Phase-1 loaded ``(model, tokenizer)``.

    Deliberately uses ``adapter_from_loaded_model`` rather than
    ``adapter_for_repo`` so a 16-20 GB MoE checkpoint is not loaded
    twice. Double-loading inflates device memory peak, slows the probe
    by tens of seconds, and on a 48 GB M5 Pro can turn an expected
    "variant guard / no adapter" failure into a memory-exhaustion
    failure — exactly the signal the probe is trying to preserve.

    Returns ``(PhaseResult, adapter, kv)`` so Phase 4 can reuse both
    without re-building a second ``SimpleKVCache``.
    """
    try:
        adapter, kv = adapter_from_loaded_model(model, tokenizer)
    except Exception as exc:  # noqa: BLE001
        return (
            PhaseResult(
                name="factory.adapter_for_repo",
                ok=False,
                details={
                    "repo": repo,
                    "supported_model_types": list(supported_model_types()),
                    "next_step_hint": (
                        "Expected failure on first MoE probe: no Silica "
                        "MoE adapter is registered. Land E1 by creating "
                        "silica/models/qwen3_5_moe.py with a "
                        "Qwen3_5MoeAdapter that extends the dense "
                        "Qwen3_5Adapter (inherit hybrid-DeltaNet + KV "
                        "routing) and adds has_moe=True plus the D-011 "
                        "per-expert call-path, then register it in "
                        "silica.models.factory._ADAPTERS keyed on the "
                        "Phase 1 model_type."
                    ),
                },
                error=_describe_exception(exc),
            ),
            None,
            None,
        )
    return (
        PhaseResult(
            name="factory.adapter_for_repo",
            ok=True,
            details={
                "repo": repo,
                "adapter_class": type(adapter).__name__,
                "kv_class": type(kv).__name__,
                "note": (
                    "Dispatch succeeded — either an MoE adapter has been "
                    "registered since this probe was written, or the "
                    "dense Qwen3_5Adapter accepted the checkpoint. In "
                    "the latter case Phase 3 will still show has_moe as "
                    "whatever the dense adapter's capabilities report, "
                    "which is NOT authoritative for an MoE checkpoint."
                ),
            },
        ),
        adapter,
        kv,
    )


def _probe_capabilities(adapter: Any) -> PhaseResult:
    try:
        caps = adapter.capabilities()
        pattern = adapter.attention_pattern()
    except Exception as exc:  # noqa: BLE001
        return PhaseResult(
            name="adapter.capabilities",
            ok=False,
            error=_describe_exception(exc),
        )
    kinds_sorted = sorted(k.value for k in caps.attention_kinds)
    per_layer = tuple(k.value for k in pattern.per_layer)
    return PhaseResult(
        name="adapter.capabilities",
        ok=True,
        details={
            "attention_kinds": kinds_sorted,
            "has_recurrent_state": caps.has_recurrent_state,
            "has_moe": caps.has_moe,
            "num_layers_from_pattern": len(per_layer),
            "first_8_layer_kinds": list(per_layer[:8]),
            "last_8_layer_kinds": list(per_layer[-8:]),
            "distinct_layer_kinds": sorted(set(per_layer)),
        },
    )


def _probe_forward(engine: Engine, adapter: Any) -> PhaseResult:
    try:
        tokenizer = adapter.tokenizer()
        eos_ids = tuple(sorted(getattr(tokenizer, "eos_token_ids", set()) or ()))
        params = SamplingParams(
            temperature=0.0,
            top_p=1.0,
            max_tokens=4,
            stop_token_ids=eos_ids,
        )
        mx.reset_peak_memory()
        t0 = time.perf_counter()
        token_ids = list(engine.generate("Hello", params))
        wall = time.perf_counter() - t0
        snap = engine.metrics.snapshot()
        return PhaseResult(
            name="engine.generate",
            ok=True,
            details={
                "prompt": "Hello",
                "max_tokens": 4,
                "wall_s": wall,
                "output_token_ids": token_ids,
                "decoded": tokenizer.decode(token_ids),
                "ttft_ms": snap.ttft_ms,
                "prefill_tok_s": snap.prefill_tok_s,
                "decode_tok_s": snap.decode_tok_s,
                "resident_mb": snap.resident_mb,
                "peak_memory_mb": mx.get_peak_memory() / 1e6,
            },
        )
    except Exception as exc:  # noqa: BLE001
        return PhaseResult(
            name="engine.generate",
            ok=False,
            error=_describe_exception(exc),
        )


def _render_report(
    phases: list[PhaseResult],
    *,
    repo: str,
    ran_forward: bool,
) -> str:
    lines: list[str] = []
    lines.append(f"# P-3-E0 probe (Qwen3.5-MoE) — {repo}")
    lines.append("")
    lines.append(f"- Host: {platform.platform()}")
    lines.append(
        f"- Timestamp: {time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}"
    )
    lines.append("")
    for ph in phases:
        status = "ok" if ph.ok else "failed"
        lines.append(f"## Phase — {ph.name}: {status}")
        lines.append("")
        for k, v in ph.details.items():
            lines.append(f"- **{k}**: `{v}`")
        if not ph.ok:
            lines.append(f"- **error**: `{ph.error}`")
        lines.append("")
    if not ran_forward:
        lines.append("## Phase — engine.generate: skipped")
        lines.append("")
        lines.append(
            "- **reason**: `--run-forward` was not passed or dispatch "
            "failed; default is metadata-only. Rerun with "
            "`--run-forward` after an E1 adapter file exists and the "
            "35B-A3B weights are cached locally."
        )
        lines.append("")

    load_ok = any(p.name == "mlx_lm.load" and p.ok for p in phases)
    dispatch_ok = any(
        p.name == "factory.adapter_for_repo" and p.ok for p in phases
    )
    lines.append("## Summary")
    lines.append("")
    if load_ok and dispatch_ok:
        lines.append(
            "- `mlx_lm.load` and `factory.adapter_for_repo` both "
            "succeeded. Read Phase 3 `has_moe` before concluding that "
            "the adapter actually handles MoE routing — a dispatch onto "
            "a dense adapter would still report ok at Phase 2."
        )
    elif load_ok and not dispatch_ok:
        lines.append(
            "- `mlx_lm.load` succeeded and `adapter_for_repo` failed "
            "as expected. Phase 1 captured the MoE structural fields "
            "(`num_experts`, `num_experts_per_tok`, "
            "`moe_intermediate_size`, `shared_expert_intermediate_size`, "
            "`norm_topk_prob`) — feed them into the E0 survey and the "
            "upcoming `Qwen3_5MoeAdapter` design. The `args_attrs` / "
            "`text_config_keys` dumps surface any unknown fields that "
            "need attention before E1."
        )
    else:
        lines.append(
            "- `mlx_lm.load` failed. Before writing adapter code, "
            "decide between extending mlx-lm, switching to an "
            "mlx-lm-converted repo, or bringing in mlx-vlm. The error "
            "above is the primary evidence."
        )
    return "\n".join(lines)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--repo",
        required=True,
        help=(
            "HF repo id to probe. REQUIRED — Qwen3.5-35B-A3B quantized "
            "checkpoints are ~20 GB; the probe does not guess a default "
            "to avoid downloading the wrong repository. Typical "
            "candidate: mlx-community/Qwen3.5-35B-A3B-4bit (verify the "
            "exact name on HF)."
        ),
    )
    parser.add_argument(
        "--run-forward",
        action="store_true",
        help=(
            "Run Phase 4 (Engine.generate micro-forward). Default is "
            "metadata-only — rerun with this flag after an E1 adapter "
            "exists so dispatch actually succeeds."
        ),
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    phases: list[PhaseResult] = []

    load_result, model, tokenizer = _probe_load(args.repo)
    phases.append(load_result)

    adapter: Any = None
    kv: Any = None
    if load_result.ok:
        dispatch_result, adapter, kv = _probe_dispatch(
            args.repo, model, tokenizer
        )
        phases.append(dispatch_result)
    else:
        phases.append(
            PhaseResult(
                name="factory.adapter_for_repo",
                ok=False,
                details={"skipped_due_to": "mlx_lm.load failed"},
            )
        )

    if adapter is not None:
        phases.append(_probe_capabilities(adapter))
    else:
        phases.append(
            PhaseResult(
                name="adapter.capabilities",
                ok=False,
                details={"skipped_due_to": "no adapter (earlier phase failed)"},
            )
        )

    ran_forward = args.run_forward and adapter is not None and kv is not None
    if ran_forward:
        engine = Engine(adapter, kv)
        phases.append(_probe_forward(engine, adapter))
    elif args.run_forward and adapter is None:
        phases.append(
            PhaseResult(
                name="engine.generate",
                ok=False,
                details={
                    "skipped_due_to": (
                        "--run-forward requested but no adapter "
                        "(earlier phase failed)"
                    ),
                },
            )
        )

    report = _render_report(
        phases, repo=args.repo, ran_forward=ran_forward
    )
    print(report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
