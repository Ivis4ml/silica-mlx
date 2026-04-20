"""P-3-A probe: can Silica's current mlx-lm path load Qwen3.5-27B 4-bit?

**First probe of the P-3 dense big-model track.** The question this
script answers is deliberately narrow and empirical — the HF model
card for ``mlx-community/Qwen3.5-27B-4bit`` (about 16.1 GB on disk,
converted from ``Qwen/Qwen3.5-27B``) documents usage via ``mlx-vlm``,
not ``mlx-lm``, so there is a real risk that ``mlx_lm.load`` either
fails outright or succeeds with a degraded model. We want primary-source
evidence either way before we touch ``silica/models/`` to add a 27B
adapter.

This probe does NOT fall back to ``mlx-vlm``. A failure inside
``mlx_lm.load`` is a valid result — it tells us whether the next step is
(a) extending the mlx-lm loader, (b) switching to an mlx-lm-converted
repo if one exists, or (c) biting the bullet on mlx-vlm integration.

Four phases (each independent; later phases print ``skipped`` when the
previous phase's precondition failed):

  Phase 1 — ``mlx_lm.load(repo)`` returns ``(model, tokenizer)``.
            Capture model_type and the structural metadata the
            ``Qwen3_5Adapter`` reads (``num_hidden_layers``,
            ``hidden_size``, ``num_attention_heads``,
            ``num_key_value_heads``, ``head_dim`` when present).

  Phase 2 — ``adapter_for_repo(repo)`` dispatches the factory. Reports
            which adapter class was selected (expected: Qwen3_5Adapter
            via ``model_type == "qwen3_5"``).

  Phase 3 — ``adapter.capabilities()`` and ``adapter.attention_pattern()``.
            Expectation from PLAN.md §D-015 + D-011 is hybrid
            DeltaNet + no MoE for the 27B variant ("dense" here means
            **non-MoE**, not "no recurrent state") — but the probe
            **records the observed values** rather than asserting.

  Phase 4 — a single max_tokens=4 greedy forward through
            ``Engine.generate``. Reports generated tokens, TTFT,
            decode throughput, resident KV, and peak device memory.
            Pass ``--skip-forward`` to leave Phase 4 out (metadata-
            only probe).

Exit code convention:
  0 — probe completed end-to-end and produced a report (even if
      individual phases failed; their failure is data).
  1 — probe itself crashed (CLI parsing, import failure, etc.).

Sources (as of 2026-04-19):
  https://huggingface.co/mlx-community/Qwen3.5-27B-4bit
  https://huggingface.co/mlx-community/Qwen3.5-27B-6bit
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

DEFAULT_REPO = "mlx-community/Qwen3.5-27B-4bit"


@dataclass
class PhaseResult:
    name: str
    ok: bool
    details: dict[str, Any] = field(default_factory=dict)
    error: str | None = None


def _describe_exception(exc: BaseException) -> str:
    """Compact, single-string description of an exception.

    Includes type + message + the most recent frame from the traceback
    so readers can tell whether the failure is in mlx-lm's registry, in
    a safetensors / weight file, or in our own factory.
    """
    tb = traceback.extract_tb(exc.__traceback__)
    tail = tb[-1] if tb else None
    frame = (
        f" (at {Path(tail.filename).name}:{tail.lineno} in {tail.name})"
        if tail is not None
        else ""
    )
    return f"{type(exc).__name__}: {exc}{frame}"


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
    }
    return (
        PhaseResult(name="mlx_lm.load", ok=True, details=details),
        model,
        tokenizer,
    )


def _probe_attr(args: Any, name: str) -> Any:
    """Read ``args.<name>`` tolerating three shapes seen across Qwen families.

    - Plain Qwen3 keeps structural fields flat on ``args`` (e.g.
      ``args.hidden_size``).
    - Some checkpoints nest them under ``args.text_config`` as an
      object with attributes (``args.text_config.hidden_size``).
    - Qwen3.5 stores ``args.text_config`` as a **plain dict** — so
      ``hasattr`` returns ``False`` and the attribute reader misses it.
      Probe first ran against Qwen3.5-27B-4bit in this branch and
      reported every field as ``None`` even though the adapter itself
      extracted them correctly via its own ``_text_config_dict``.
    """
    if args is None:
        return None
    if hasattr(args, name):
        return getattr(args, name)
    text_config = getattr(args, "text_config", None)
    if text_config is None:
        return None
    if isinstance(text_config, dict):
        return text_config.get(name)
    if hasattr(text_config, name):
        return getattr(text_config, name)
    return None


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
    # Summarise per-layer rather than dumping 60+ entries: count + first few.
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
    skipped_forward: bool,
) -> str:
    lines: list[str] = []
    lines.append(f"# P-3-A probe — {repo}")
    lines.append("")
    lines.append(f"- Host: {platform.platform()}")
    lines.append(f"- Timestamp: {time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}")
    lines.append("")
    for ph in phases:
        status = "ok" if ph.ok else "failed"
        lines.append(f"## Phase — {ph.name}: {status}")
        lines.append("")
        if ph.ok:
            for k, v in ph.details.items():
                lines.append(f"- **{k}**: `{v}`")
        else:
            for k, v in ph.details.items():
                lines.append(f"- **{k}**: `{v}`")
            lines.append(f"- **error**: `{ph.error}`")
        lines.append("")
    if skipped_forward:
        lines.append("## Phase — engine.generate: skipped (--skip-forward)")
        lines.append("")
    # Summary verdict.
    load_ok = any(p.name == "mlx_lm.load" and p.ok for p in phases)
    lines.append("## Summary")
    lines.append("")
    if load_ok:
        lines.append(
            "- `mlx_lm.load` succeeded — Silica's current loader path "
            "accepts this repo. Next step: read the Phase 3 "
            "`attention_kinds` output. If it includes `hybrid_deltanet`, "
            "the capability gate will still reject batched execution "
            "until P-3-C lands DeltaNet plumbing; single-request "
            "`Engine.generate` works regardless."
        )
    else:
        lines.append(
            "- `mlx_lm.load` failed — probe's primary question is "
            "answered: Silica's current loader cannot ingest this repo "
            "as-is. Read the error to decide between (a) extending "
            "mlx-lm, (b) switching to an mlx-lm-converted repo if one "
            "exists, or (c) mlx-vlm integration."
        )
    return "\n".join(lines)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", default=DEFAULT_REPO)
    parser.add_argument(
        "--skip-forward",
        action="store_true",
        help="Skip Phase 4 (engine.generate); only produce metadata.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    phases: list[PhaseResult] = []

    # ``tokenizer`` is consumed via ``adapter.tokenizer()`` in Phase 4;
    # the raw handle from ``mlx_lm.load`` is not needed separately.
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

    if args.skip_forward:
        pass
    elif adapter is not None:
        # Build Engine with the kv that adapter_for_repo already produced.
        # The factory function returns (adapter, kv) but _probe_dispatch
        # dropped the kv — rebuild Engine using SimpleKVCache.from_model
        # on the loaded model, same shape as factory produces.
        from silica.kvcache.simple import SimpleKVCache

        kv = SimpleKVCache.from_model(model)
        # Rewire adapter to use this kv (adapters hold their own ref;
        # match the factory's invariant so cache_list(req_id) works).
        adapter._kv_manager = kv  # noqa: SLF001 — probe-only
        engine = Engine(adapter, kv)
        phases.append(_probe_forward(engine, adapter))
    else:
        phases.append(
            PhaseResult(
                name="engine.generate",
                ok=False,
                details={"skipped_due_to": "no adapter (earlier phase failed)"},
            )
        )

    report = _render_report(
        phases, repo=args.repo, skipped_forward=args.skip_forward
    )
    print(report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
