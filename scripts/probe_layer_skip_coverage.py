"""Probe: layer-skip self-spec coverage on Qwen3.5-27B-4bit.

For each decode position in a teacher-forced corpus, compute target argmax
(full forward) and drafter argmax (early-exit forward stopping at layer L,
then applying final norm + lm_head). Measure agreement rate (= accept rate
proxy) across multiple skip schedules.

This is the empirical equivalent of the C.5 β.2 top-b coverage probe but
for a self-speculative drafter rather than an external small model. The
drafter cost is `(L / num_layers) × full_forward_cost`; the gating
question is "does early-exit produce target-matching argmax often enough
that spec amortisation pays for the drafter cost?"

Per AR.md: read-only probe, no model fine-tuning, uses existing
``decode_step_multi_with_capture`` infrastructure (D-021 step 6 αβ.1).

Decision matrix:
    agreement >= 0.40 → opens layer-skip self-spec implementation track
    agreement in [0.20, 0.40) → escalate; depends on drafter-cost / kernel evidence
    agreement < 0.20 → retire layer-skip path

Production target: ``mlx-community/Qwen3.5-27B-4bit``. 64 layers, hybrid 48/16.

Usage:
    SILICA_REAL_QWEN3_5_27B=1 \\
        uv run python -m scripts.probe_layer_skip_coverage \\
            --skip-from-end 8 16 24 32 \\
            --out plans/P6_AUTORESEARCH/layer_skip_coverage.jsonl
"""

from __future__ import annotations

import argparse
import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import mlx.core as mx
from mlx_lm.models import cache as mlx_cache


def _short_corpus_tokens(tokenizer: Any, max_tokens: int) -> list[int]:
    text = (
        "Memory bandwidth has emerged as the dominant constraint in "
        "single-stream autoregressive decoding for large language models "
        "on consumer hardware platforms. Each decode step must read the "
        "entire active parameter set from unified memory before any "
        "computation can begin. On Apple Silicon the available bandwidth "
        "caps the achievable tokens per second well below what arithmetic "
        "throughput would otherwise allow. As model parameter counts "
        "continue to grow, this bottleneck becomes more pronounced. "
        "Hardware vendors are responding with wider memory interfaces "
        "and dedicated neural acceleration units, while software "
        "frameworks such as MLX try to extract every available cycle "
        "through aggressive kernel fusion and tensor reuse strategies."
    )
    ids = tokenizer.encode(text)
    return ids[:max_tokens]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", default="mlx-community/Qwen3.5-27B-4bit")
    parser.add_argument(
        "--skip-from-end",
        type=int,
        nargs="+",
        default=[8, 16, 24, 32],
        help="Number of layers to skip from the end (drafter exits early).",
    )
    parser.add_argument("--max-tokens", type=int, default=64)
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("plans/P6_AUTORESEARCH/layer_skip_coverage.jsonl"),
    )
    args = parser.parse_args()

    if not os.environ.get("SILICA_REAL_QWEN3_5_27B"):
        raise SystemExit("SILICA_REAL_QWEN3_5_27B=1 required.")

    from silica.models.factory import adapter_for_repo

    print(f"[layer_skip] loading {args.repo}...")
    adapter, _kv = adapter_for_repo(args.repo)
    model = adapter._model  # noqa: SLF001
    n_layers = len(model.layers)
    print(f"[layer_skip] num_layers={n_layers}")

    tokens = _short_corpus_tokens(adapter.tokenizer(), args.max_tokens)
    print(f"[layer_skip] corpus tokens: {len(tokens)}")

    rows: list[dict[str, Any]] = []
    ts = datetime.now(timezone.utc).isoformat()

    # Step 1 — full forward, capture every layer's output and the final logits.
    capture_ids = frozenset(range(n_layers + 1))  # 0=embedding output; 1..n_layers=layer outputs
    cache_list_full = mlx_cache.make_prompt_cache(model)
    tokens_arr = mx.array(tokens, dtype=mx.int32)
    from silica.models.hidden_capture import run_qwen3_5_forward_with_capture
    full_logits, captured = run_qwen3_5_forward_with_capture(
        model, tokens_arr, cache_list_full, capture_ids,
    )
    mx.eval(full_logits)
    for cid in captured:
        mx.eval(captured[cid])

    target_argmax = mx.argmax(full_logits, axis=-1)
    mx.eval(target_argmax)
    target_argmax_list = target_argmax.tolist()
    print(f"[layer_skip] target argmax over {len(target_argmax_list)} positions captured")

    # Qwen3.5 wrapper: model.language_model.{model.norm, lm_head}
    if hasattr(model, "language_model"):
        text_model = model.language_model  # TextModel
        final_norm = text_model.model.norm
        lm_head = text_model.lm_head if hasattr(text_model, "lm_head") else None
        if lm_head is None:
            # tied embeddings path
            embed = text_model.model.embed_tokens
            def lm_head(h: mx.array) -> mx.array:  # type: ignore[no-redef]
                return embed.as_linear(h)
    else:
        final_norm = getattr(getattr(model, "model", model), "norm", None)
        lm_head = getattr(model, "lm_head", None)
    if final_norm is None or lm_head is None:
        raise RuntimeError(
            f"could not locate final_norm or lm_head on model; got "
            f"final_norm={final_norm}, lm_head={lm_head}"
        )

    for skip in args.skip_from_end:
        if skip <= 0 or skip >= n_layers:
            print(f"[layer_skip] skipping invalid skip-from-end value {skip}")
            continue
        exit_layer = n_layers - skip
        # captured[exit_layer] = output of model.layers[exit_layer - 1]
        # (per the convention in run_qwen3_5_forward_with_capture: layer-id i+1 = output of layers[i])
        if exit_layer not in captured:
            print(f"[layer_skip] no captured hidden at exit_layer={exit_layer}")
            continue
        h_at_exit = captured[exit_layer]
        # h_at_exit shape: (1, T, hidden) — drop batch dim before final norm + lm_head.
        if h_at_exit.ndim == 3:
            h_at_exit = h_at_exit[0]
        h_normed = final_norm(h_at_exit)
        drafter_logits = lm_head(h_normed)
        mx.eval(drafter_logits)
        # drafter_logits is now (T, V); argmax → (T,)
        drafter_argmax = mx.argmax(drafter_logits, axis=-1)
        mx.eval(drafter_argmax)
        drafter_argmax_list = drafter_argmax.tolist()

        # Compare against target_argmax over [0, len-1] (last position has no next-token target)
        n = min(len(target_argmax_list), len(drafter_argmax_list))
        match = sum(
            1 for i in range(n)
            if target_argmax_list[i] == drafter_argmax_list[i]
        )
        agreement = match / n if n > 0 else 0.0

        print(f"[layer_skip] skip-from-end={skip}, exit_layer={exit_layer}: "
              f"agreement = {agreement:.4f} ({match}/{n})")
        rows.append({
            "kind": "skip_schedule",
            "skip_from_end": skip,
            "exit_layer": exit_layer,
            "n_positions": n,
            "matches": match,
            "agreement": agreement,
            "ts": ts,
        })

    # Aggregate summary
    summary = {
        "kind": "summary",
        "ts": ts,
        "repo": args.repo,
        "n_layers": n_layers,
        "n_positions_captured": len(target_argmax_list),
        "skip_schedules_tested": args.skip_from_end,
        "agreements_by_skip": {
            r["skip_from_end"]: r["agreement"] for r in rows if r["kind"] == "skip_schedule"
        },
    }
    rows.append(summary)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w") as fh:
        for row in rows:
            fh.write(json.dumps(row) + "\n")

    print(f"[layer_skip] wrote {len(rows)} rows to {args.out}")
    print(f"[layer_skip] decision-matrix readings:")
    for r in rows:
        if r["kind"] != "skip_schedule":
            continue
        a = r["agreement"]
        verdict = (
            "OPEN_IMPL" if a >= 0.40
            else "ESCALATE" if a >= 0.20
            else "RETIRE"
        )
        print(f"  skip={r['skip_from_end']:2d} (exit_layer={r['exit_layer']:2d})  "
              f"agreement={a:.4f}  verdict={verdict}")


if __name__ == "__main__":
    main()
