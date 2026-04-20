"""P-3-E0 probe: can Silica's current mlx-lm path load a Gemma4-MoE repo?

**First probe of the P-3-E MoE track, Gemma4 family (26B-A4B).**
Unlike ``probe_qwen3_5_moe_load.py``, Gemma4's MoE variant does NOT
live in a separate module — ``mlx_lm.models.gemma4_text.py`` contains
BOTH the dense (Gemma4-31B, P-3-D track) and the MoE (Gemma4-26B-A4B,
this track) variants, gated on ``enable_moe_block``. The structural
differences captured by this probe:

  - ``DecoderLayer`` branches at ``self.enable_moe =
    config.enable_moe_block``. When True, a ``Router`` + ``Experts``
    block replaces the dense ``MLP`` (gemma4_text.py:290-302).
  - ``Router`` (gemma4_text.py:117) runs RMSNorm → linear projection
    to ``num_experts`` → ``argpartition`` top-k → softmax →
    ``per_expert_scale`` weighting.
  - ``Experts`` (gemma4_text.py:153) wraps ``SwitchGLU`` with
    ``GeGLU`` activation. All experts' weights are stacked; dispatch
    goes through ``gather_mm`` internally.
  - Config field names: ``enable_moe_block`` (bool),
    ``num_experts`` (int), ``top_k_experts`` (int),
    ``moe_intermediate_size``. Notably **no** ``shared_expert_*``
    fields — Gemma4 does not have the shared-expert branch that
    Qwen3.5-MoE inherits from Qwen3-Next.

Existing Silica guards against MoE Gemma4:

  - ``Gemma4Adapter._validate_supported_variant`` (P-3-D1) raises
    ``NotImplementedError`` if ``enable_moe_block=True`` or
    ``num_experts > 0`` — pointing at P-3-E. This probe will hit
    that guard at Phase 2: ``adapter_from_loaded_model`` dispatches
    on ``model.model_type == "gemma4"`` and calls the ``_build_gemma4``
    factory builder, which constructs ``Gemma4Adapter(...)`` whose
    ``__init__`` runs ``_validate_supported_variant`` first.

Expected outcomes on a fresh run, Silica @ v1.6.3+:

  Phase 1 — ``mlx_lm.load`` succeeds. Captures ``enable_moe_block``,
            ``num_experts``, ``top_k_experts``,
            ``moe_intermediate_size``, plus the dense-side fields
            (``layer_types``, ``sliding_window``, head dims) so the
            report surfaces how the sliding+full hybrid interacts
            with MoE — is every layer MoE, or only full-attention
            layers, or some other scheme?

  Phase 2 — ``factory.adapter_for_repo`` fails loudly via the
            P-3-D1 variant guard. Error message names
            ``enable_moe_block`` / ``num_experts`` which matches
            the guard branch at ``gemma4.py:177-194``.

  Phase 3 — skipped (Phase 2 failed).

  Phase 4 — ``Engine.generate`` micro-forward, ``--run-forward``
            gate (same semantics as other probes). Requires an E1
            ``Gemma4MoeAdapter`` to exist before this can succeed.

D-011 note: identical to Qwen3.5-MoE — mlx-lm's Gemma4 MoE path is
fused via ``SwitchGLU`` + ``gather_mm``. The E0 survey will flag
the same tension. Decision between fused-wrapper vs per-expert
dispatch applies uniformly to both families.

Exit code convention (same as other probes):
  0 — probe completed and produced a report, even if individual
      phases failed.
  1 — probe itself crashed.

Usage:
  uv run python scripts/probe_gemma4_moe_load.py --repo <hf_repo>
  uv run python scripts/probe_gemma4_moe_load.py --repo <hf_repo> --run-forward

No default repo. Candidate name: ``mlx-community/gemma-4-26b-a4b-4bit``
or similar; verify the exact repo id on HF before running. Quantized
26B-A4B downloads are likely ~16 GB.
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

    layer_types = _probe_attr(args, "layer_types")
    layer_types_repr: Any = layer_types
    layer_type_counts: dict[str, int] | None = None
    if isinstance(layer_types, (list, tuple)) and layer_types:
        counts: dict[str, int] = {}
        for lt in layer_types:
            counts[str(lt)] = counts.get(str(lt), 0) + 1
        layer_type_counts = counts
        layer_types_repr = (
            f"len={len(layer_types)}; first_8={list(layer_types[:8])}"
        )

    details: dict[str, Any] = {
        "repo": repo,
        "load_s": elapsed,
        "model_type": getattr(model, "model_type", None),
        # Structural fields shared with the Gemma4 dense probe.
        "num_hidden_layers": _probe_attr(args, "num_hidden_layers"),
        "hidden_size": _probe_attr(args, "hidden_size"),
        "num_attention_heads": _probe_attr(args, "num_attention_heads"),
        "num_key_value_heads": _probe_attr(args, "num_key_value_heads"),
        "num_global_key_value_heads": _probe_attr(
            args, "num_global_key_value_heads"
        ),
        "head_dim": _probe_attr(args, "head_dim"),
        "global_head_dim": _probe_attr(args, "global_head_dim"),
        "sliding_window": _probe_attr(args, "sliding_window"),
        "sliding_window_pattern": _probe_attr(args, "sliding_window_pattern"),
        "attention_k_eq_v": _probe_attr(args, "attention_k_eq_v"),
        "num_kv_shared_layers": _probe_attr(args, "num_kv_shared_layers"),
        "hidden_size_per_layer_input": _probe_attr(
            args, "hidden_size_per_layer_input"
        ),
        "layer_types_summary": layer_types_repr,
        "layer_type_counts": layer_type_counts,
        "vocab_size": getattr(tokenizer, "vocab_size", None),
        "tokenizer_type": type(tokenizer).__name__,
        "model_class": type(model).__name__,
        # MoE-specific fields — gemma4_text.py:43-45.
        "enable_moe_block": _probe_attr(args, "enable_moe_block"),
        "num_experts": _probe_attr(args, "num_experts"),
        "top_k_experts": _probe_attr(args, "top_k_experts"),
        "moe_intermediate_size": _probe_attr(args, "moe_intermediate_size"),
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

    Uses ``adapter_from_loaded_model`` to avoid loading the ~16 GB
    Gemma4-MoE checkpoint a second time — see the matching comment in
    ``probe_qwen3_5_moe_load.py`` for the rationale. Returns
    ``(PhaseResult, adapter, kv)``.
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
                        "Expected failure: the P-3-D1 Gemma4Adapter "
                        "variant guard rejects enable_moe_block=True / "
                        "num_experts>0, pointing at P-3-E. Land E1 by "
                        "creating silica/models/gemma4_moe.py with a "
                        "Gemma4MoeAdapter extending the dense "
                        "Gemma4Adapter's attention/pattern handling and "
                        "adding has_moe=True plus the D-011 per-expert "
                        "call path, then register the appropriate "
                        "model_type in silica.models.factory._ADAPTERS. "
                        "Cross-check Phase 1 for whether Gemma4-MoE "
                        "uses the SAME outer model_type as Gemma4-dense "
                        "(in which case factory keying alone is "
                        "ambiguous and a separate dispatch hook is "
                        "needed)."
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
                    "Dispatch succeeded unexpectedly — either the D1 "
                    "variant guard has been removed / weakened, or an "
                    "E1 adapter has landed since this probe was "
                    "written. Check `adapter_class` for which adapter "
                    "actually accepted the checkpoint."
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
    lines.append(f"# P-3-E0 probe (Gemma4-MoE) — {repo}")
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
            "`--run-forward` after an E1 Gemma4MoeAdapter exists and "
            "the 26B-A4B weights are cached locally."
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
            "succeeded. Read Phase 3 `has_moe` — a success here with "
            "`has_moe=False` would indicate the dense Gemma4Adapter "
            "accepted the checkpoint, which means the D1 variant guard "
            "has regressed and needs to be re-tightened before E1 "
            "lands a proper MoE adapter."
        )
    elif load_ok and not dispatch_ok:
        lines.append(
            "- `mlx_lm.load` succeeded and `adapter_for_repo` failed, "
            "expected via the D1 variant guard. Phase 1 captured the "
            "MoE-specific fields (`enable_moe_block`, `num_experts`, "
            "`top_k_experts`, `moe_intermediate_size`) plus the "
            "dense-side sliding+full hybrid fields — the E0 survey "
            "will combine them to answer 'does MoE apply to every "
            "layer, or is it co-located with a specific attention "
            "kind?'. The `args_attrs` / `text_config_keys` dumps "
            "surface any unknown fields."
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
            "HF repo id to probe. REQUIRED — Gemma4-26B-A4B quantized "
            "checkpoints are ~16 GB; the probe does not guess a default "
            "to avoid downloading the wrong repository. Typical "
            "candidate: mlx-community/gemma-4-26b-a4b-4bit (verify "
            "the exact name on HF)."
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
