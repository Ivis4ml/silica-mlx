"""P-3-D0 probe: can Silica's current mlx-lm path load a Gemma4-31B repo?

**First probe of the P-3 Gemma4 dense track.** Mirrors the layout of
``scripts/probe_qwen3_5_27b_load.py`` (P-3-A / B) but the question
it answers is different: we do **not** yet know whether mlx-lm ships
an architecture handler for Gemma4, what ``model_type`` /
``args.text_config`` shape the checkpoint uses, or whether a new
``Gemma4Adapter`` file is required before ``silica.models.factory``
can dispatch. This probe collects the facts, it does not commit to
an answer. Four phases:

  Phase 1 — ``mlx_lm.load(repo)`` returns ``(model, tokenizer)``.
            Captures ``model_type`` and the structural fields Silica
            adapters read (``num_hidden_layers``, ``hidden_size``,
            ``num_attention_heads``, ``num_key_value_heads``,
            ``head_dim``), plus a full dump of ``args`` /
            ``args.text_config`` attribute names so unknown fields
            (sliding-window, MoE, …) are surfaced without the probe
            having to pre-guess their names.

  Phase 2 — ``adapter_for_repo(repo)``. Expected on a first run to
            fail with ``NotImplementedError: No Silica adapter
            registered for model_type=…`` — that is a valid data
            point. The report names the dispatch outcome together
            with the currently supported model_type list so the
            next step can decide between: (a) register a new
            ``Gemma4Adapter`` in ``silica/models/gemma4.py`` +
            ``factory._ADAPTERS``; (b) reuse an existing family
            adapter if Gemma4's ``model_type`` happens to overlap.

  Phase 3 — ``adapter.capabilities()`` +
            ``adapter.attention_pattern()``. Only runs when Phase 2
            succeeded. Reports ``attention_kinds``,
            ``has_recurrent_state``, ``has_moe`` so the kind of
            scheduler path (plain GLOBAL vs hybrid DeltaNet vs
            sliding-window) is explicit before writing any adapter
            code.

  Phase 4 — ``Engine.generate`` micro-forward (4 tokens, greedy),
            **only when ``--run-forward`` is passed**. Default is
            metadata-only: we want to see Phase 1–3 before paying
            the forward's MLX kernel-compile cost, and a failed
            dispatch makes Phase 4 trivially impossible anyway.

No mlx-vlm fallback. A failure inside ``mlx_lm.load`` is a valid
result and is reported verbatim so the caller can decide whether to
extend the loader, switch to an mlx-lm-converted repo, or bite the
bullet on mlx-vlm.

Exit code convention (same as the Qwen3.5-27B probe):
  0 — probe completed and produced a report, even if individual
      phases failed.
  1 — probe itself crashed (CLI parsing, import failure, etc.).

Usage:
  uv run python scripts/probe_gemma4_31b_load.py --repo <hf_repo>
  uv run python scripts/probe_gemma4_31b_load.py --repo <hf_repo> --run-forward

No default repo is built in: Gemma4-31B checkpoints are tens of GB,
and guessing a wrong ``mlx-community/…`` name would download
something we do not intend.
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
    adapter_for_repo,
    supported_model_types,
)


@dataclass
class PhaseResult:
    name: str
    ok: bool
    details: dict[str, Any] = field(default_factory=dict)
    error: str | None = None


def _describe_exception(exc: BaseException) -> str:
    """Compact, single-string description of an exception for the report."""
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
    three Qwen/Gemma shapes seen in practice: flat attributes on
    ``args``, nested object under ``args.text_config``, and nested
    dict under ``args.text_config``. Returns ``None`` when absent in
    every path.
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
    """Public-looking attribute / key names for diagnostic output.

    For an object we list its non-underscore attributes (skipping
    callables); for a dict we return its keys. Useful when the
    probe cannot pre-guess which fields the checkpoint exposes.
    """
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
        # mlx-lm's load() returns Union[2tuple, 3tuple]; we use the 2tuple form.
        model, tokenizer = _mlx_lm_load(repo)  # type: ignore[misc]
    except Exception as exc:  # noqa: BLE001 — probe captures anything
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
        "num_hidden_layers": _probe_attr(args, "num_hidden_layers"),
        "hidden_size": _probe_attr(args, "hidden_size"),
        "num_attention_heads": _probe_attr(args, "num_attention_heads"),
        "num_key_value_heads": _probe_attr(args, "num_key_value_heads"),
        "head_dim": _probe_attr(args, "head_dim"),
        "vocab_size": getattr(tokenizer, "vocab_size", None),
        "tokenizer_type": type(tokenizer).__name__,
        "model_class": type(model).__name__,
        # Diagnostic dumps: surface unknown fields (sliding-window, MoE
        # routing, RoPE variants, …) without the probe pre-guessing
        # what each family might expose.
        "args_attrs": _attr_keys(args),
        "text_config_keys": _attr_keys(text_config),
        "n_layers_from_model_list": len(getattr(model, "layers", []) or []),
    }
    return (
        PhaseResult(name="mlx_lm.load", ok=True, details=details),
        model,
        tokenizer,
    )


def _probe_dispatch(repo: str) -> tuple[PhaseResult, Any]:
    try:
        adapter, kv = adapter_for_repo(repo)
    except Exception as exc:  # noqa: BLE001
        return (
            PhaseResult(
                name="factory.adapter_for_repo",
                ok=False,
                details={
                    "repo": repo,
                    "supported_model_types": list(supported_model_types()),
                    "next_step_hint": (
                        "On first-run dispatch failure for a new family, "
                        "write silica/models/gemma4.py with a Gemma4Adapter "
                        "class (pattern: silica/models/qwen3.py or "
                        "silica/models/qwen3_5.py) and register it in "
                        "silica.models.factory._ADAPTERS keyed on the "
                        "model_type surfaced by Phase 1."
                    ),
                },
                error=_describe_exception(exc),
            ),
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
                    "Dispatch succeeded on an existing family adapter — "
                    "Gemma4's model_type must overlap with a registered "
                    "family. Double-check Phase 3 capabilities to confirm "
                    "the adapter's attention-pattern reading is sane on "
                    "this checkpoint before treating it as correct."
                ),
            },
        ),
        adapter,
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
    lines.append(f"# P-3-D0 probe — {repo}")
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
            "- **reason**: `--run-forward` was not passed; default is "
            "metadata-only to keep the first run cheap. Rerun with "
            "`--run-forward` after Phases 1–3 look sensible."
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
            "succeeded. Read Phase 3 `attention_kinds` / "
            "`has_recurrent_state` / `has_moe` before concluding that "
            "no new adapter file is needed — a fallback dispatch onto "
            "the wrong family would still report ok at this phase. If "
            "`--run-forward` was set and Phase 4 also green, "
            "``peak_memory_mb`` is the headroom estimate for "
            "subsequent P-3-D adapter / bench work."
        )
    elif load_ok and not dispatch_ok:
        lines.append(
            "- `mlx_lm.load` succeeded but `adapter_for_repo` failed. "
            "Phase 2's `next_step_hint` names the path forward: a new "
            "`silica/models/gemma4.py` adapter file + "
            "`factory._ADAPTERS` registration keyed on the Phase 1 "
            "`model_type`. Compare Phase 1's `args_attrs` / "
            "`text_config_keys` against `silica/models/qwen3_5.py` to "
            "decide how much structural code the adapter needs."
        )
    else:
        lines.append(
            "- `mlx_lm.load` failed — before writing adapter code, "
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
            "HF repo id to probe. REQUIRED — Gemma4-31B checkpoints are "
            "tens of GB; the probe does not guess a default to avoid "
            "downloading the wrong repository. Typical candidates include "
            "mlx-community / Gemma4-31B variants (e.g. 4-bit / 6-bit) "
            "once you have confirmed the exact name on HF."
        ),
    )
    parser.add_argument(
        "--run-forward",
        action="store_true",
        help=(
            "Run Phase 4 (Engine.generate micro-forward). Default is "
            "metadata-only — rerun with this flag after Phases 1–3 "
            "look sensible."
        ),
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    phases: list[PhaseResult] = []

    # ``tokenizer`` from Phase 1 is reached later via ``adapter.tokenizer()``;
    # the raw handle is not needed separately here.
    load_result, model, _ = _probe_load(args.repo)
    phases.append(load_result)

    adapter: Any = None
    if load_result.ok:
        dispatch_result, adapter = _probe_dispatch(args.repo)
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

    ran_forward = args.run_forward and adapter is not None
    if ran_forward:
        from silica.kvcache.simple import SimpleKVCache

        kv = SimpleKVCache.from_model(model)
        adapter._kv_manager = kv  # noqa: SLF001 — probe-only
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
